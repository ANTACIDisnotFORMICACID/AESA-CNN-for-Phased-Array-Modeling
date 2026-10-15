"""V2.10-compatible 3x3 aperture blocks migrated for the encoding ablation."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock2D(nn.Module):
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
