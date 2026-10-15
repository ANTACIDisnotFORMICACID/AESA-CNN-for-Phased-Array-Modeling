"""Direction-conditioned axial aperture residuals for V2.16."""

from __future__ import annotations

import torch
from torch import nn

from v212.encoding import DirectionFourierEncoder
from v212.operator import CoherentRank1Operator


AXIAL_MODES = ("axial_static", "axial_gated")
CONTROLLED_DCAR_MODES = ("static", "gated")


def _coordinate_maps() -> torch.Tensor:
    """Return normalized row(+Z) and column(+Y) coordinates on the 4x4 aperture."""
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    return torch.stack((row, col), dim=0)


class AxialEnvelopeResidual(nn.Module):
    """Only row/column axial convolutions may exchange element information."""

    def __init__(
        self,
        *,
        mode: str,
        width: int = 68,
        maximum_component: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in AXIAL_MODES:
            raise ValueError(f"unknown axial mode: {mode}")
        if width <= 0 or maximum_component <= 0:
            raise ValueError("width and maximum_component must be positive")
        self.encoder = DirectionFourierEncoder(4)
        self.stem = nn.Conv2d(29, width, kernel_size=1)
        self.activation = nn.SiLU()
        self.row_operator = nn.Conv2d(
            width, width, kernel_size=(3, 1), padding=(1, 0)
        )
        self.col_operator = nn.Conv2d(
            width, width, kernel_size=(1, 3), padding=(0, 1)
        )
        self.direction_gate = (
            nn.Linear(self.encoder.output_channels, 2)
            if mode == "axial_gated" else None
        )
        self.head = nn.Conv2d(width, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("coordinate_maps", _coordinate_maps())
        self.mode = mode
        self.width = int(width)
        self.maximum_component = float(maximum_component)

    def gate_values(self, unit_direction: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(unit_direction)
        if self.direction_gate is None:
            return torch.ones(*encoded.shape[:2], 2, dtype=encoded.dtype, device=encoded.device)
        return torch.tanh(self.direction_gate(encoded))

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, query_count = unit_direction.shape[:2]
        encoded = self.encoder(unit_direction)
        direction_maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
        coordinates = self.coordinate_maps.to(unit_direction)[None, None].expand(
            batch, query_count, -1, -1, -1
        )
        features = torch.cat((direction_maps, coordinates), dim=2)
        hidden = self.activation(
            self.stem(features.reshape(batch * query_count, 29, 4, 4))
        )
        row_delta = self.row_operator(hidden)
        col_delta = self.col_operator(hidden)
        gates = self.gate_values(unit_direction).reshape(batch * query_count, 2)
        mixed = (
            hidden
            + gates[:, 0, None, None, None] * row_delta
            + gates[:, 1, None, None, None] * col_delta
        )
        raw = self.head(self.activation(mixed))
        raw = raw.reshape(batch, query_count, 2, 4, 4)
        scale = self.maximum_component
        return scale * torch.tanh(raw / scale)


class IsotropicEnvelopeResidual(nn.Module):
    """Capacity-matched isotropic control with one 3x3 spatial operator."""

    def __init__(self, *, width: int = 56, maximum_component: float = 2.0) -> None:
        super().__init__()
        if width <= 0 or maximum_component <= 0:
            raise ValueError("width and maximum_component must be positive")
        self.encoder = DirectionFourierEncoder(4)
        self.stem = nn.Conv2d(29, width, kernel_size=1)
        self.activation = nn.SiLU()
        self.spatial_operator = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.head = nn.Conv2d(width, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("coordinate_maps", _coordinate_maps())
        self.width = int(width)
        self.maximum_component = float(maximum_component)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, query_count = unit_direction.shape[:2]
        encoded = self.encoder(unit_direction)
        direction_maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
        coordinates = self.coordinate_maps.to(unit_direction)[None, None].expand(
            batch, query_count, -1, -1, -1
        )
        features = torch.cat((direction_maps, coordinates), dim=2)
        hidden = self.activation(
            self.stem(features.reshape(batch * query_count, 29, 4, 4))
        )
        mixed = hidden + self.spatial_operator(hidden)
        raw = self.head(self.activation(mixed))
        raw = raw.reshape(batch, query_count, 2, 4, 4)
        scale = self.maximum_component
        return scale * torch.tanh(raw / scale)


class V216AxialOperator(CoherentRank1Operator):
    """V2.12 physical operator with only the V2.16 axial residual replaced."""

    def __init__(self, *args, axial_mode: str, axial_width: int = 68, **kwargs) -> None:
        super().__init__(*args, residual_enabled=False, **kwargs)
        self.residual = AxialEnvelopeResidual(mode=axial_mode, width=axial_width)
        self.axial_mode = axial_mode
        self.axial_width = int(axial_width)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "family": "v216-direction-conditioned-axial-residual",
            "mode": self.axial_mode,
            "width": self.axial_width,
            "input_maps": "fourier27+row_coordinate+column_coordinate",
            "spatial_operators": ["row_3x1", "column_1x3"],
            "head_kernel": "1x1",
            "analytic_phase_preserved": True,
            "graph_or_s_matrix_used": False,
            "zero_start_output_head": True,
        }


class V216IsotropicOperator(CoherentRank1Operator):
    """V2.12 physical operator with a clean, capacity-matched 3x3 control."""

    def __init__(self, *args, isotropic_width: int = 56, **kwargs) -> None:
        super().__init__(*args, residual_enabled=False, **kwargs)
        self.residual = IsotropicEnvelopeResidual(width=isotropic_width)
        self.isotropic_width = int(isotropic_width)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "family": "v216-capacity-matched-isotropic-residual",
            "width": self.isotropic_width,
            "input_maps": "fourier27+row_coordinate+column_coordinate",
            "spatial_operators": ["isotropic_3x3"],
            "head_kernel": "1x1",
            "analytic_phase_preserved": True,
            "graph_or_s_matrix_used": False,
            "zero_start_output_head": True,
        }


def _v212_geometry_maps() -> torch.Tensor:
    """Reproduce the V2.12 row/column/edge/corner input contract exactly."""
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


class ControlledAxialResidualBlock(nn.Module):
    """Depth-matched V2.12 block with row/column propagation as the only change."""

    def __init__(
        self,
        channels: int,
        *,
        row_branch_width: int | None = None,
        col_branch_width: int | None = None,
    ) -> None:
        super().__init__()
        row_width = channels if row_branch_width is None else int(row_branch_width)
        col_width = channels if col_branch_width is None else int(col_branch_width)
        if row_width <= 0 or col_width <= 0:
            raise ValueError("axial branch widths must be positive")
        self.row_conv1 = nn.Conv2d(channels, row_width, kernel_size=(3, 1), padding=(1, 0))
        self.row_conv2 = nn.Conv2d(row_width, channels, kernel_size=(3, 1), padding=(1, 0))
        self.col_conv1 = nn.Conv2d(channels, col_width, kernel_size=(1, 3), padding=(0, 1))
        self.col_conv2 = nn.Conv2d(col_width, channels, kernel_size=(1, 3), padding=(0, 1))
        self.activation = nn.SiLU()
        nn.init.zeros_(self.row_conv2.weight)
        nn.init.zeros_(self.row_conv2.bias)
        nn.init.zeros_(self.col_conv2.weight)
        nn.init.zeros_(self.col_conv2.bias)

    def forward(self, inputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        if gates.ndim != 2 or gates.shape != (inputs.shape[0], 2):
            raise ValueError("gates must have shape [N,2]")
        row = self.row_conv2(self.activation(self.row_conv1(inputs)))
        col = self.col_conv2(self.activation(self.col_conv1(inputs)))
        return (
            inputs
            + gates[:, 0, None, None, None] * row
            + gates[:, 1, None, None, None] * col
        )


class ControlledDCAREnvelopeResidual(nn.Module):
    """V2.12-matched residual context with only its residual blocks axialized."""

    def __init__(
        self,
        *,
        mode: str,
        width: int,
        depth: int = 2,
        branch_widths: tuple[tuple[int, int], ...] | None = None,
        maximum_component: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in CONTROLLED_DCAR_MODES:
            raise ValueError(f"unknown controlled DCAR mode: {mode}")
        if width <= 0 or depth <= 0 or maximum_component <= 0:
            raise ValueError("width, depth and maximum_component must be positive")
        self.encoder = DirectionFourierEncoder(4)
        self.stem = nn.Conv2d(31, width, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        if branch_widths is None:
            branch_widths = tuple((width, width) for _ in range(depth))
        if len(branch_widths) != depth:
            raise ValueError("branch_widths must contain one row/column pair per block")
        self.blocks = nn.ModuleList(
            ControlledAxialResidualBlock(
                width, row_branch_width=row_width, col_branch_width=col_width
            )
            for row_width, col_width in branch_widths
        )
        self.direction_gate = (
            nn.Linear(self.encoder.output_channels, 2 * depth) if mode == "gated" else None
        )
        self.head = nn.Conv2d(width, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("geometry_maps", _v212_geometry_maps())
        self.mode = mode
        self.width = int(width)
        self.depth = int(depth)
        self.branch_widths = tuple((int(row), int(col)) for row, col in branch_widths)
        self.maximum_component = float(maximum_component)

    def gate_values(self, unit_direction: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(unit_direction)
        if self.direction_gate is None:
            return torch.ones(
                *encoded.shape[:2], self.depth, 2,
                dtype=encoded.dtype, device=encoded.device,
            )
        return torch.tanh(self.direction_gate(encoded)).reshape(
            *encoded.shape[:2], self.depth, 2
        )

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, query_count = unit_direction.shape[:2]
        encoded = self.encoder(unit_direction)
        direction_maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(unit_direction)[None, None].expand(
            batch, query_count, -1, -1, -1
        )
        features = torch.cat((direction_maps, geometry), dim=2)
        hidden = self.activation(
            self.stem(features.reshape(batch * query_count, 31, 4, 4))
        )
        gates = self.gate_values(unit_direction).reshape(
            batch * query_count, self.depth, 2
        )
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, gates[:, index])
        raw = self.head(hidden).reshape(batch, query_count, 2, 4, 4)
        scale = self.maximum_component
        return scale * torch.tanh(raw / scale)


class V216ControlledDCAROperator(CoherentRank1Operator):
    """V2.12 operator with only its residual-block spatial propagation changed."""

    def __init__(
        self,
        *args,
        dcar_mode: str,
        residual_width: int = 24,
        residual_depth: int = 2,
        branch_widths: tuple[tuple[int, int], ...] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, residual_enabled=False, **kwargs)
        self.residual = ControlledDCAREnvelopeResidual(
            mode=dcar_mode,
            width=residual_width,
            depth=residual_depth,
            branch_widths=branch_widths,
        )
        self.dcar_mode = dcar_mode
        self.residual_width = int(residual_width)
        self.residual_depth = int(residual_depth)

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "family": "v216-controlled-depth-matched-dcar",
            "mode": self.dcar_mode,
            "width": self.residual_width,
            "depth": self.residual_depth,
            "branch_widths": self.residual.branch_widths,
            "input_maps": "v212-fourier27+row+column+edge+corner",
            "stem_and_head": "v212-matched-3x3",
            "residual_block_change_only": "parallel-row-and-column-two-convolution-branches",
            "analytic_phase_preserved": True,
            "graph_or_s_matrix_used": False,
            "zero_start_output_head_and_second_branch_convolutions": True,
        }
