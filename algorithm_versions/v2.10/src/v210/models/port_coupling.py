"""Structured direction-independent complex mixing between physical array ports."""

from __future__ import annotations

import torch
from torch import nn


def geometry_neighbour_mask() -> torch.Tensor:
    mask = torch.zeros(16, 16, dtype=torch.float32)
    for source in range(16):
        source_row, source_col = divmod(source, 4)
        for destination in range(16):
            destination_row, destination_col = divmod(destination, 4)
            distance = max(
                abs(source_row - destination_row),
                abs(source_col - destination_col),
            )
            if distance == 1:
                mask[source, destination] = 1.0
    return mask


class StructuredPortCoupling(nn.Module):
    """C = I + P_off(mask*K + U*V^T), applied as h = C^T b."""

    def __init__(self, *, low_rank: int = 1) -> None:
        super().__init__()
        if low_rank < 0:
            raise ValueError("low_rank must be non-negative")
        self.low_rank = int(low_rank)
        self.correction_real = nn.Parameter(torch.zeros(16, 16))
        self.correction_imag = nn.Parameter(torch.zeros(16, 16))
        self.register_buffer("sparse_mask", geometry_neighbour_mask())
        self.register_buffer("off_diagonal_mask", 1.0 - torch.eye(16))
        if self.low_rank:
            generator = torch.Generator().manual_seed(29001)
            initial = 0.01 * torch.randn(16, self.low_rank, generator=generator)
            self.low_rank_u_real = nn.Parameter(initial)
            self.low_rank_u_imag = nn.Parameter(torch.zeros(16, self.low_rank))
            self.low_rank_v_real = nn.Parameter(torch.zeros(16, self.low_rank))
            self.low_rank_v_imag = nn.Parameter(torch.zeros(16, self.low_rank))

    def matrix_iq(self) -> torch.Tensor:
        real = self.sparse_mask * self.correction_real
        imag = self.sparse_mask * self.correction_imag
        if self.low_rank:
            low_real = self.low_rank_u_real @ self.low_rank_v_real.T - self.low_rank_u_imag @ self.low_rank_v_imag.T
            low_imag = self.low_rank_u_real @ self.low_rank_v_imag.T + self.low_rank_u_imag @ self.low_rank_v_real.T
            real = real + low_real
            imag = imag + low_imag
        real = real * self.off_diagonal_mask + torch.eye(16, device=real.device, dtype=real.dtype)
        imag = imag * self.off_diagonal_mask
        return torch.stack((real, imag), dim=0)

    def forward(self, operator_iq: torch.Tensor) -> torch.Tensor:
        if operator_iq.ndim != 5 or operator_iq.shape[-3:] != (2, 4, 4):
            raise ValueError("operator_iq must have shape [B,Q,2,4,4]")
        matrix_real, matrix_imag = self.matrix_iq().unbind(dim=0)
        operator_real, operator_imag = operator_iq.unbind(dim=2)
        operator_real = operator_real.flatten(start_dim=-2)
        operator_imag = operator_imag.flatten(start_dim=-2)
        result_real = torch.einsum("bqm,mn->bqn", operator_real, matrix_real) - torch.einsum(
            "bqm,mn->bqn", operator_imag, matrix_imag
        )
        result_imag = torch.einsum("bqm,mn->bqn", operator_real, matrix_imag) + torch.einsum(
            "bqm,mn->bqn", operator_imag, matrix_real
        )
        return torch.stack(
            (result_real.reshape(*result_real.shape[:2], 4, 4), result_imag.reshape(*result_imag.shape[:2], 4, 4)),
            dim=2,
        )

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        real, imag = self.matrix_iq().unbind(dim=0)
        identity = torch.eye(16, device=real.device, dtype=real.dtype)
        delta_real = real - identity
        coupling_l2 = (delta_real.square() + imag.square()).mean()
        reciprocity = ((real - real.T).square() + (imag - imag.T).square()).mean()
        return {"coupling_l2": coupling_l2, "reciprocity": reciprocity}
