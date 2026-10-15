"""Small 3x3 blocks whose spatial axes are always physical array row/column."""

from __future__ import annotations

import math

import torch
from torch import nn


class DirectionFourierEncoder(nn.Module):
    """Encode direction globally; output has no element index."""

    def __init__(self, bands: int = 4) -> None:
        super().__init__()
        if bands < 0:
            raise ValueError("bands must be non-negative")
        self.bands = bands
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    @property
    def output_channels(self) -> int:
        return 3 * (1 + 2 * self.bands)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        if self.bands == 0:
            return unit_direction
        angles = unit_direction[..., :, None] * self.frequencies.to(unit_direction)
        sine = torch.sin(angles).reshape(*unit_direction.shape[:2], -1)
        cosine = torch.cos(angles).reshape(*unit_direction.shape[:2], -1)
        return torch.cat((unit_direction, sine, cosine), dim=-1)


class ResidualBlock2D(nn.Module):
    """Two local 3x3 aperture operations; second starts at zero."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.conv2(self.activation(self.conv1(inputs)))


class ApertureCorrectionBranch(nn.Module):
    """Map physical feature maps to a six-channel Cartesian-IQ correction."""

    def __init__(
        self,
        input_channels: int,
        width: int,
        depth: int = 1,
        output_channels: int = 6,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.blocks = nn.Sequential(*(ResidualBlock2D(width) for _ in range(depth)))
        self.head = nn.Conv2d(width, output_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.stem(feature_maps))
        return self.head(self.blocks(hidden))
