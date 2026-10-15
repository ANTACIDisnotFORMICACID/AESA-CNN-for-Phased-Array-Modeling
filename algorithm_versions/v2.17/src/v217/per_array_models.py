"""Geometry-specific residual candidates for the V2.17 feasibility pilot.

The four arrays share a physical idea and an audit contract, not parameters or
checkpoints.  This module contains only the residual CNNs; the coherent
low-rank envelope and analytic phase backbone are fitted per array later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


LIGHT_SPEED_M_S = 299_792_458.0


class DirectionFourierEncoder(nn.Module):
    def __init__(self, bands: int = 4) -> None:
        super().__init__()
        if bands < 0:
            raise ValueError("bands must be non-negative")
        self.bands = int(bands)
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(bands, dtype=torch.float32) * math.pi
        )

    @property
    def output_channels(self) -> int:
        return 3 * (1 + 2 * self.bands)

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        if direction.ndim != 3 or direction.shape[-1] != 3:
            raise ValueError("direction must have shape [B,Q,3]")
        if self.bands == 0:
            return direction
        angles = direction[..., :, None] * self.frequencies.to(direction)
        return torch.cat(
            (
                direction,
                torch.sin(angles).flatten(start_dim=-2),
                torch.cos(angles).flatten(start_dim=-2),
            ),
            dim=-1,
        )


def geometry_maps(rows: int, cols: int) -> torch.Tensor:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    row_axis = torch.zeros(rows) if rows == 1 else torch.linspace(-1.0, 1.0, rows)
    col_axis = torch.zeros(cols) if cols == 1 else torch.linspace(-1.0, 1.0, cols)
    row = row_axis[:, None].expand(rows, cols)
    col = col_axis[None, :].expand(rows, cols)
    row_edge = torch.zeros_like(row, dtype=torch.bool) if rows == 1 else row.abs() == 1.0
    col_edge = torch.zeros_like(col, dtype=torch.bool) if cols == 1 else col.abs() == 1.0
    edge = row_edge | col_edge
    corner = col_edge if rows == 1 else row_edge if cols == 1 else row_edge & col_edge
    return torch.stack((row, col, edge.float(), corner.float()), dim=0)


class ColumnResidualBlock(nn.Module):
    def __init__(self, channels: int, branch_width: int, dilation: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(
            channels, branch_width, kernel_size=(1, 3),
            padding=(0, dilation), dilation=(1, dilation),
        )
        self.second = nn.Conv2d(
            branch_width, channels, kernel_size=(1, 3),
            padding=(0, dilation), dilation=(1, dilation),
        )
        self.activation = nn.SiLU()
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, inputs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        branch = self.second(self.activation(self.first(inputs)))
        return inputs + gate[:, None, None, None] * branch


class RowPairColumnResidualBlock(nn.Module):
    def __init__(self, channels: int, row_width: int, col_width: int) -> None:
        super().__init__()
        self.row_first = nn.Conv2d(channels, row_width, kernel_size=(2, 1))
        self.row_second = nn.ConvTranspose2d(row_width, channels, kernel_size=(2, 1))
        self.col_first = nn.Conv2d(channels, col_width, kernel_size=(1, 3), padding=(0, 1))
        self.col_second = nn.Conv2d(col_width, channels, kernel_size=(1, 3), padding=(0, 1))
        self.activation = nn.SiLU()
        for layer in (self.row_second, self.col_second):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        row = self.row_second(self.activation(self.row_first(inputs)))
        col = self.col_second(self.activation(self.col_first(inputs)))
        return (
            inputs
            + gates[:, 0, None, None, None] * row
            + gates[:, 1, None, None, None] * col
        )


class AxialResidualBlock(nn.Module):
    def __init__(self, channels: int, row_width: int, col_width: int) -> None:
        super().__init__()
        self.row_first = nn.Conv2d(channels, row_width, kernel_size=(3, 1), padding=(1, 0))
        self.row_second = nn.Conv2d(row_width, channels, kernel_size=(3, 1), padding=(1, 0))
        self.col_first = nn.Conv2d(channels, col_width, kernel_size=(1, 3), padding=(0, 1))
        self.col_second = nn.Conv2d(col_width, channels, kernel_size=(1, 3), padding=(0, 1))
        self.activation = nn.SiLU()
        for layer in (self.row_second, self.col_second):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        row = self.row_second(self.activation(self.row_first(inputs)))
        col = self.col_second(self.activation(self.col_first(inputs)))
        return (
            inputs
            + gates[:, 0, None, None, None] * row
            + gates[:, 1, None, None, None] * col
        )


class IsotropicResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.second = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.second(self.activation(self.first(inputs)))


@dataclass(frozen=True)
class ResidualConfig:
    array_id: str
    rows: int
    cols: int
    family: str
    width: int
    depth: int
    branch_widths: tuple[tuple[int, int], ...]
    dilations: tuple[int, ...]


CONFIGS = {
    "a1x4": ResidualConfig("a1x4", 1, 4, "column", 28, 2, ((28, 28),) * 2, (1, 1)),
    "a1x8": ResidualConfig("a1x8", 1, 8, "column", 28, 3, ((28, 28),) * 3, (1, 2, 1)),
    "a2x4": ResidualConfig("a2x4", 2, 4, "row_pair_column", 28, 2, ((28, 28),) * 2, (1, 1)),
    "a4x4": ResidualConfig("a4x4", 4, 4, "axial", 24, 2, ((36, 35), (35, 36)), (1, 1)),
}


class GeometryMatchedResidual(nn.Module):
    def __init__(self, array_id: str, arm: str) -> None:
        super().__init__()
        if array_id not in CONFIGS:
            raise ValueError(f"unknown array: {array_id}")
        if arm not in {"control", "candidate"}:
            raise ValueError("arm must be control or candidate")
        config = CONFIGS[array_id]
        self.config = config
        self.arm = arm
        self.encoder = DirectionFourierEncoder(4)
        kernel = (1, 3) if config.rows == 1 else (3, 3)
        padding = (0, 1) if config.rows == 1 else (1, 1)
        self.stem = nn.Conv2d(31, config.width, kernel_size=kernel, padding=padding)
        self.activation = nn.SiLU()
        if config.family == "column":
            self.blocks = nn.ModuleList(
                ColumnResidualBlock(config.width, widths[1], dilation)
                for widths, dilation in zip(config.branch_widths, config.dilations)
            )
            gate_count = config.depth
        elif config.family == "row_pair_column":
            self.blocks = nn.ModuleList(
                RowPairColumnResidualBlock(config.width, *widths)
                for widths in config.branch_widths
            )
            gate_count = 2 * config.depth
        elif arm == "control":
            self.blocks = nn.ModuleList(
                IsotropicResidualBlock(config.width) for _ in range(config.depth)
            )
            gate_count = 0
        else:
            self.blocks = nn.ModuleList(
                AxialResidualBlock(config.width, *widths) for widths in config.branch_widths
            )
            gate_count = 2 * config.depth
        self.direction_gate = (
            nn.Linear(self.encoder.output_channels, gate_count)
            if arm == "candidate" else None
        )
        self.head = nn.Conv2d(config.width, 2, kernel_size=kernel, padding=padding)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("geometry", geometry_maps(config.rows, config.cols))

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(direction)
        batch, queries = direction.shape[:2]
        direction_maps = encoded[..., None, None].expand(
            -1, -1, -1, self.config.rows, self.config.cols
        )
        geometry = self.geometry.to(direction)[None, None].expand(batch, queries, -1, -1, -1)
        features = torch.cat((direction_maps, geometry), dim=2)
        hidden = self.activation(
            self.stem(features.reshape(batch * queries, 31, self.config.rows, self.config.cols))
        )
        if self.arm == "candidate":
            gates = torch.tanh(self.direction_gate(encoded)).reshape(batch * queries, -1)
        else:
            gate_axes = 1 if self.config.family == "column" else 2
            gates = torch.ones(
                batch * queries, gate_axes * self.config.depth,
                dtype=hidden.dtype, device=hidden.device,
            )
        for index, block in enumerate(self.blocks):
            if isinstance(block, IsotropicResidualBlock):
                hidden = block(hidden)
            elif isinstance(block, ColumnResidualBlock):
                hidden = block(hidden, gates[:, index])
            else:
                hidden = block(hidden, gates[:, 2 * index : 2 * index + 2])
        return self.head(hidden).reshape(
            batch, queries, 2, self.config.rows, self.config.cols
        )

    def metadata(self) -> dict[str, object]:
        implemented_family = (
            "isotropic_3x3" if self.config.array_id == "a4x4" and self.arm == "control"
            else self.config.family
        )
        return {
            "array_id": self.config.array_id,
            "arm": self.arm,
            "logical_shape": [self.config.rows, self.config.cols],
            "family": implemented_family,
            "direction_gated": self.arm == "candidate",
            "width": self.config.width,
            "depth": self.config.depth,
            "branch_widths": [list(value) for value in self.config.branch_widths],
            "dilations": list(self.config.dilations),
            "shared_parameters_across_arrays": False,
            "analytic_phase_backbone_required": True,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def centered_positions(array_id: str, row_pitch_m: float = 0.015, col_pitch_m: float = 0.0068) -> torch.Tensor:
    config = CONFIGS[array_id]
    row = torch.zeros(config.rows) if config.rows == 1 else (
        torch.arange(config.rows, dtype=torch.float32) - (config.rows - 1) / 2.0
    ) * row_pitch_m
    col = torch.zeros(config.cols) if config.cols == 1 else (
        torch.arange(config.cols, dtype=torch.float32) - (config.cols - 1) / 2.0
    ) * col_pitch_m
    positions = torch.zeros(config.rows, config.cols, 3)
    positions[:, :, 1] = col[None, :]
    positions[:, :, 2] = row[:, None]
    return positions.reshape(-1, 3)


class GeometryMatchedOperator(nn.Module):
    """Independent per-array coherent rank-1 backbone plus complex CNN residual."""

    def __init__(
        self,
        array_id: str,
        arm: str,
        *,
        positions_m: torch.Tensor | None = None,
        frequency_hz: float = 47.0e9,
        initialization_seed: int = 21720,
        residual_maximum: float = 2.0,
    ) -> None:
        super().__init__()
        if frequency_hz <= 0 or residual_maximum <= 0:
            raise ValueError("frequency_hz and residual_maximum must be positive")
        self.array_id = array_id
        self.arm = arm
        self.frequency_hz = float(frequency_hz)
        self.residual_maximum = float(residual_maximum)
        config = CONFIGS[array_id]
        positions = centered_positions(array_id) if positions_m is None else torch.as_tensor(
            positions_m, dtype=torch.float32
        )
        if positions.shape != (config.rows * config.cols, 3):
            raise ValueError("positions_m shape does not match the selected array")
        self.register_buffer("positions_m", positions.clone())
        self.wavenumber_rad_m = 2.0 * math.pi * self.frequency_hz / LIGHT_SPEED_M_S
        self.encoder = DirectionFourierEncoder(4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.angular_hidden = nn.Linear(self.encoder.output_channels, 64)
        self.angular_output = nn.Linear(64, 2)
        nn.init.zeros_(self.angular_output.weight)
        with torch.no_grad():
            self.angular_output.bias.copy_(torch.tensor([1.0, 0.0]))
        self.element_gain_real = nn.Parameter(torch.ones(config.rows * config.cols))
        self.element_gain_imag = nn.Parameter(torch.zeros(config.rows * config.cols))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed + 1)
            self.residual = GeometryMatchedResidual(array_id, arm)

    def envelope(self, direction: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(direction)
        angular_iq = self.angular_output(torch.nn.functional.silu(self.angular_hidden(encoded)))
        angular = torch.complex(angular_iq[..., 0], angular_iq[..., 1])
        gain = torch.complex(self.element_gain_real, self.element_gain_imag)
        raw_residual = self.residual(direction)
        bounded_iq = self.residual_maximum * torch.tanh(raw_residual / self.residual_maximum)
        bounded_residual = torch.complex(
            bounded_iq[:, :, 0], bounded_iq[:, :, 1]
        ).flatten(start_dim=2)
        return angular[..., None] * gain + bounded_residual

    def operator(self, direction: torch.Tensor) -> torch.Tensor:
        phase = self.wavenumber_rad_m * torch.einsum(
            "bqk,nk->bqn", direction, self.positions_m.to(direction)
        )
        return self.envelope(direction) * torch.exp(1j * phase)

    def forward(self, direction: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        if commands.ndim != 3 or commands.shape[0] != direction.shape[0]:
            raise ValueError("commands must have shape [B,C,N]")
        if commands.shape[-1] != self.positions_m.shape[0]:
            raise ValueError("command element count does not match the array")
        field = torch.einsum("bqn,bcn->bcq", self.operator(direction), commands)
        return torch.abs(field)

    def metadata(self) -> dict[str, object]:
        return {
            **self.residual.metadata(),
            "operator_family": "per-array-coherent-rank1-plus-geometry-residual",
            "rank1_angular_model": "trainable-fourier27-mlp64-complex-scalar",
            "rank1_element_model": "trainable-static-complex-gain-per-element",
            "analytic_phase": True,
            "frequency_hz": self.frequency_hz,
        }
