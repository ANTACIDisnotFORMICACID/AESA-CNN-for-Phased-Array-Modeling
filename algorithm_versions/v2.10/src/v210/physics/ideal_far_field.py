"""Analytic scalar far-field propagation phase for the fixed 4x4 aperture."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..contracts import ArrayGeometry, QueryBatch


class IdealScalarFarFieldOperator(nn.Module):
    """Return exp(+j k q.r) in [B,Q,IQ,4,4] transmit convention."""

    def __init__(self, geometry: ArrayGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.register_buffer(
            "positions_m",
            torch.as_tensor(geometry.positions_m, dtype=torch.float32),
        )
        self.wavenumber_rad_m = 2.0 * np.pi / geometry.wavelength_m

    def forward(self, query: QueryBatch) -> torch.Tensor:
        direction = query.unit_direction
        positions = self.positions_m.to(direction)
        phase = self.wavenumber_rad_m * torch.einsum(
            "bqk,rck->bqrc", direction, positions
        )
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)
