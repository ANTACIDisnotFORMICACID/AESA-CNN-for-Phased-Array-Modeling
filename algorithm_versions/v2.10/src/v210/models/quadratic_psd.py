"""V2.10 controlled vector-field baseline and low-rank PSD models.

The PSD model predicts an equivalent factor ``L(q)`` and exposes only the
identifiable total-power quadratic form ``Q(q) = L(q)^H L(q) / 60``. The
factor rows are not interpreted as physical polarization components.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..contracts import ArrayGeometry
from .blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder
from .nominal_pattern import _geometry_maps


def quadratic_form(factor: torch.Tensor) -> torch.Tensor:
    """Return Hermitian PSD matrices for ``factor[..., rank, port]``."""

    if not torch.is_complex(factor) or factor.shape[-1] != 16:
        raise ValueError("factor must be complex with trailing port dimension 16")
    return torch.einsum("...rn,...rm->...nm", factor.conj(), factor) / 60.0


def gains_from_factor(factor: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """Evaluate normalized gain for factors ``[Q,R,16]`` and commands ``[C,16]``."""

    field = torch.einsum("qrn,cn->cqr", factor, command)
    denominator = command.abs().square().sum(-1).clamp_min(1e-12)
    return field.abs().square().sum(-1) / 60.0 / denominator[:, None]


class _ApertureFactorModel(nn.Module):
    """Shared direction/geometry encoder with analytic per-port propagation."""

    def __init__(self, *, output_rows: int, project_transverse: bool) -> None:
        super().__init__()
        self.output_rows = int(output_rows)
        self.project_transverse = bool(project_transverse)
        self.encoder = DirectionFourierEncoder(4)
        self.branch = ApertureCorrectionBranch(
            input_channels=31,
            width=24,
            depth=2,
            output_channels=2 * self.output_rows,
        )
        with torch.no_grad():
            self.branch.head.bias[0] = 1.0
            if self.output_rows > 1:
                self.branch.head.bias[2] = 0.25
        geometry = ArrayGeometry()
        self.register_buffer("geometry", _geometry_maps())
        self.register_buffer(
            "positions",
            torch.tensor(geometry.positions_m.reshape(16, 3), dtype=torch.float32),
        )
        self.wavelength_m = float(geometry.wavelength_m)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 2 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [Q,3]")
        feature = self.encoder(unit_direction[None])[0, :, :, None, None]
        feature = feature.expand(-1, -1, 4, 4)
        geometry = self.geometry[None].expand(len(unit_direction), -1, -1, -1)
        raw = self.branch(torch.cat((feature, geometry), dim=1))
        raw = raw.reshape(-1, self.output_rows, 2, 16)
        factor = torch.complex(raw[:, :, 0], raw[:, :, 1])
        if self.project_transverse:
            factor = factor - unit_direction[:, :, None] * torch.sum(
                unit_direction[:, :, None] * factor,
                dim=1,
                keepdim=True,
            )
        phase = (2.0 * math.pi / self.wavelength_m) * (
            unit_direction @ self.positions.T
        )
        return factor * torch.exp(1j * phase)[:, None, :]


class V210VectorBaseline(_ApertureFactorModel):
    """A10: exact migrated V2.9 Cartesian transverse architecture."""

    def __init__(self) -> None:
        super().__init__(output_rows=3, project_transverse=True)


class V210LowRankPSD(_ApertureFactorModel):
    """B10/C10: rank-2 factor for an identifiable total-power PSD operator."""

    def __init__(self) -> None:
        super().__init__(output_rows=2, project_transverse=False)

    def q_matrix(self, unit_direction: torch.Tensor) -> torch.Tensor:
        return quadratic_form(self(unit_direction))


def build_experiment_model(recipe: str) -> nn.Module:
    if recipe == "A10":
        return V210VectorBaseline()
    if recipe in {"B10", "C10"}:
        return V210LowRankPSD()
    raise ValueError(f"unknown V2.10 recipe: {recipe}")

