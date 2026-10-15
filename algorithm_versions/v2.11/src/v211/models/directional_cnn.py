"""Isolated V2.10 direction-to-aperture CNN front end for V2.11 ablations."""

from __future__ import annotations

import torch
from torch import nn

from .blocks_2d import ApertureCorrectionBranch


def geometry_maps() -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


class DirectionalApertureCNN(nn.Module):
    """Keep the V2.10 27+4=31 channel CNN contract independent of encoder type."""

    def __init__(
        self,
        direction_encoder: nn.Module,
        *,
        width: int = 24,
        depth: int = 2,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        encoder_channels = int(getattr(direction_encoder, "output_channels"))
        if encoder_channels != 27:
            raise ValueError("direction encoder must emit exactly 27 channels")
        self.direction_encoder = direction_encoder
        self.head = ApertureCorrectionBranch(
            encoder_channels + 4,
            width,
            depth=depth,
            output_channels=output_channels,
        )
        self.output_channels = int(output_channels)
        self.register_buffer("geometry_maps", geometry_maps())

    def build_feature_maps(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, queries = unit_direction.shape[:2]
        encoded = self.direction_encoder(unit_direction)
        direction = encoded[:, :, :, None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(unit_direction)[None, None].expand(
            batch, queries, -1, -1, -1
        )
        return torch.cat((direction, geometry), dim=2)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        batch, queries = unit_direction.shape[:2]
        features = self.build_feature_maps(unit_direction)
        raw = self.head(features.reshape(batch * queries, 31, 4, 4))
        return raw.reshape(batch, queries, self.output_channels, 4, 4)
