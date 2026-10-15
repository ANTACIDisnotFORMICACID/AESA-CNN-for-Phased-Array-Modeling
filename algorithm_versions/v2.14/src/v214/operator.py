"""Parameter-matched cross-branch fusion experiment built on frozen V2.12 physics."""

from __future__ import annotations

import torch
from torch import nn

from v210.contracts import QueryBatch
from v212.operator import CoherentRank1Operator


def _geometry_maps() -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


class ParameterMatchedFusionAdapter(nn.Module):
    """Zero-start 1x1 adapter; both controls have exactly the same parameters."""

    def __init__(self, *, cross_branch_access: bool, hidden_width: int = 4) -> None:
        super().__init__()
        self.cross_branch_access = bool(cross_branch_access)
        self.stem = nn.Conv2d(8, hidden_width, kernel_size=1)
        self.activation = nn.SiLU()
        self.head = nn.Conv2d(hidden_width, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("geometry_maps", _geometry_maps())

    def forward(
        self,
        ssh_iq: torch.Tensor,
        fourier_iq: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if ssh_iq.shape != fourier_iq.shape or ssh_iq.shape[-3:] != (2, 4, 4):
            raise ValueError("branch IQ tensors must share shape [B,Q,2,4,4]")
        batch, queries = ssh_iq.shape[:2]
        safe_scale = scale.to(ssh_iq).clamp_min(1e-12)
        ssh_input = ssh_iq / safe_scale
        if not self.cross_branch_access:
            ssh_input = torch.zeros_like(ssh_input)
        fourier_input = fourier_iq / safe_scale
        geometry = self.geometry_maps.to(ssh_iq)[None, None].expand(
            batch, queries, -1, -1, -1
        )
        features = torch.cat((ssh_input, fourier_input, geometry), dim=2)
        raw = self.head(self.activation(self.stem(features.reshape(batch * queries, 8, 4, 4))))
        return safe_scale * torch.tanh(raw).reshape(batch, queries, 2, 4, 4)


class CrossBranchFusionOperator(CoherentRank1Operator):
    """V2.12 operator plus a zero-start, parameter-matched fusion adapter."""

    def __init__(self, *args, cross_branch_access: bool, fusion_hidden_width: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion = ParameterMatchedFusionAdapter(
            cross_branch_access=cross_branch_access,
            hidden_width=fusion_hidden_width,
        )
        self.cross_branch_access = bool(cross_branch_access)

    def envelope(self, query: QueryBatch, ablation=None):
        base_envelope, terms = super().envelope(query, ablation)
        delta_iq = self.fusion(
            terms["ssh_iq"], terms["fourier_residual_iq"], self.envelope_scale
        )
        delta = torch.complex(delta_iq[:, :, 0], delta_iq[:, :, 1]).reshape(
            *query.unit_direction.shape[:2], 16
        )
        return base_envelope + delta, {**terms, "fusion_delta_iq": delta_iq}

    def forward_factorized_grid(
        self,
        commands: torch.Tensor,
        envelope_direction: torch.Tensor,
        propagation_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Use perturbed envelope coordinates while retaining nominal physical phase."""
        if envelope_direction.shape != propagation_direction.shape:
            raise ValueError("envelope and propagation directions must match in shape")
        envelope, _ = self.envelope(QueryBatch(envelope_direction[None]))
        phase = self.wavenumber_rad_m * torch.einsum(
            "qk,nk->qn", propagation_direction, self.positions_m.to(propagation_direction)
        )
        propagation = torch.exp(torch.complex(torch.zeros_like(phase), phase))
        operator = envelope[0] * propagation
        return torch.einsum("qn,cn->cq", operator, commands.to(operator)).abs()
