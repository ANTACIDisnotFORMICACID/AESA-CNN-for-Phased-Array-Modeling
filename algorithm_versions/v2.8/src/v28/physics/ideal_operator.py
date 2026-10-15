"""Ideal 4x4 far-field Green/operator tensor in Cartesian IQ form."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..contracts import ArrayGeometry, QueryBatch
from ..complex_ops import project_transverse_operator_iq


class IdealFarFieldOperator(nn.Module):
    """Build per-element ideal responses without consuming excitation commands.

    Output axes are [batch, query, xyz, IQ, row, col].  Keeping the source
    element axes intact allows every later 3x3 operation to represent a real
    neighbourhood in the 4x4 aperture.
    """

    def __init__(self, geometry: ArrayGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or ArrayGeometry()
        self.register_buffer(
            "positions_m",
            torch.as_tensor(self.geometry.positions_m, dtype=torch.float32),
            persistent=True,
        )
        # Existing HFSS evidence makes +Y the provisional co-polar axis.
        self.register_buffer("nominal_polarization", torch.tensor([0.0, 1.0, 0.0]))
        self.wavenumber_rad_m = 2.0 * np.pi / self.geometry.wavelength_m

    def forward(self, query: QueryBatch) -> torch.Tensor:
        direction = query.unit_direction
        positions = self.positions_m.to(device=direction.device, dtype=direction.dtype)
        phase = self.wavenumber_rad_m * torch.einsum("bqk,rck->bqrc", direction, positions)
        phase_iq = torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)

        polarization = self.nominal_polarization.to(device=direction.device, dtype=direction.dtype)
        transverse = polarization - torch.einsum("bqk,k->bq", direction, polarization)[..., None] * direction
        operator = transverse[:, :, :, None, None, None] * phase_iq[:, :, None, :, :, :]
        return project_transverse_operator_iq(operator, direction)
