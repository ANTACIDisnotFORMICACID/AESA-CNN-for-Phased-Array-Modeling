"""Single capacity-limited direction-dependent residual over the 4x4 aperture."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import QueryBatch
from .blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder
from .nominal_pattern import _geometry_maps


class AngularOperatorResidual(nn.Module):
    def __init__(
        self,
        *,
        fourier_bands: int = 4,
        width: int = 12,
        depth: int = 1,
        maximum_magnitude: float = 2.0,
    ) -> None:
        super().__init__()
        if maximum_magnitude <= 0:
            raise ValueError("maximum_magnitude must be positive")
        self.direction_encoder = DirectionFourierEncoder(fourier_bands)
        self.branch = ApertureCorrectionBranch(
            self.direction_encoder.output_channels + 4,
            width,
            depth=depth,
            output_channels=2,
        )
        self.maximum_magnitude = float(maximum_magnitude)
        self.register_buffer("geometry_maps", _geometry_maps())

    def forward(self, query: QueryBatch) -> torch.Tensor:
        batch, queries = query.unit_direction.shape[:2]
        encoded = self.direction_encoder(query.unit_direction)
        direction_maps = encoded[:, :, :, None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(query.unit_direction)[None, None].expand(
            batch, queries, -1, -1, -1
        )
        features = torch.cat((direction_maps, geometry), dim=2)
        raw = self.branch(features.reshape(batch * queries, features.shape[2], 4, 4))
        raw = raw.reshape(batch, queries, 2, 4, 4)
        return self.maximum_magnitude * torch.tanh(raw / self.maximum_magnitude)
