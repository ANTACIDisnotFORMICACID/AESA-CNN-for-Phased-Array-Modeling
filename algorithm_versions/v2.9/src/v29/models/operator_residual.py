"""Hierarchical correction of an ideal operator using physical 3x3 branches."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from ..complex_ops import complex_multiply_iq, project_transverse_operator_iq
from ..contracts import ARRAY_COLS, ARRAY_ROWS, OperatorPrediction, QueryBatch
from .blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder


def _operator_to_feature_maps(operator_iq: torch.Tensor) -> torch.Tensor:
    """Merge component/IQ into channels while preserving row and column."""
    batch, queries = operator_iq.shape[:2]
    return operator_iq.reshape(batch, queries, 6, ARRAY_ROWS, ARRAY_COLS)


def _feature_maps_to_operator(feature_maps: torch.Tensor) -> torch.Tensor:
    batch, queries = feature_maps.shape[:2]
    return feature_maps.reshape(batch, queries, 3, 2, ARRAY_ROWS, ARRAY_COLS)


class ComplexElementGain(nn.Module):
    """L1: one trainable complex calibration per physical source element."""

    def __init__(self) -> None:
        super().__init__()
        gain = torch.zeros(2, ARRAY_ROWS, ARRAY_COLS)
        gain[0] = 1.0
        self.gain_iq = nn.Parameter(gain)

    def forward(self, operator_iq: torch.Tensor) -> torch.Tensor:
        gain = self.gain_iq[None, None, None, :, :, :]
        return complex_multiply_iq(operator_iq, gain, iq_dim=3)


class ComplexLocalCoupling3x3(nn.Module):
    """L2: shared nearest-neighbour complex coupling; self coefficient is masked."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel_real = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.kernel_imag = nn.Parameter(torch.zeros(1, 1, 3, 3))
        mask = torch.ones(1, 1, 3, 3)
        mask[:, :, 1, 1] = 0.0
        self.register_buffer("neighbour_mask", mask)

    def forward(self, operator_iq: torch.Tensor) -> torch.Tensor:
        batch, queries, components = operator_iq.shape[:3]
        real = operator_iq[:, :, :, 0].reshape(batch * queries * components, 1, ARRAY_ROWS, ARRAY_COLS)
        imag = operator_iq[:, :, :, 1].reshape(batch * queries * components, 1, ARRAY_ROWS, ARRAY_COLS)
        kernel_real = self.kernel_real * self.neighbour_mask
        kernel_imag = self.kernel_imag * self.neighbour_mask
        out_real = functional.conv2d(real, kernel_real, padding=1) - functional.conv2d(
            imag, kernel_imag, padding=1
        )
        out_imag = functional.conv2d(real, kernel_imag, padding=1) + functional.conv2d(
            imag, kernel_real, padding=1
        )
        output = torch.stack((out_real[:, 0], out_imag[:, 0]), dim=2)
        return output.reshape(batch, queries, components, 2, ARRAY_ROWS, ARRAY_COLS)


class OperatorResidualCNN(nn.Module):
    """Correct an ideal operator while remaining linear in excitation command.

    Physical hierarchy:
      L1 complex element gains -> L2 local mutual coupling -> L3 direction and
      polarization correction -> L4 edge/corner correction -> optional L5
      multi-hop residual.  All spatial learned operators are 3x3 Conv2d.
    """

    def __init__(
        self,
        *,
        fourier_bands: int = 4,
        directional_width: int = 24,
        edge_width: int = 16,
        multihop_width: int = 16,
        enable_multihop: bool = False,
    ) -> None:
        super().__init__()
        self.direction_encoder = DirectionFourierEncoder(fourier_bands)
        direction_channels = self.direction_encoder.output_channels
        self.element_gain = ComplexElementGain()
        self.local_coupling = ComplexLocalCoupling3x3()
        self.directional_branch = ApertureCorrectionBranch(6 + direction_channels + 2, directional_width)
        self.edge_branch = ApertureCorrectionBranch(6 + direction_channels + 4, edge_width)
        self.enable_multihop = enable_multihop
        self.multihop_branch = ApertureCorrectionBranch(
            6 + direction_channels + 4,
            multihop_width,
            depth=2,
        )
        self.register_buffer("geometry_maps", self._build_geometry_maps())

    @staticmethod
    def _build_geometry_maps() -> torch.Tensor:
        row = torch.linspace(-1.0, 1.0, ARRAY_ROWS)[:, None].expand(ARRAY_ROWS, ARRAY_COLS)
        col = torch.linspace(-1.0, 1.0, ARRAY_COLS)[None, :].expand(ARRAY_ROWS, ARRAY_COLS)
        edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
        corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
        return torch.stack((row, col, edge, corner), dim=0)

    @staticmethod
    def _run_branch(branch: nn.Module, features: torch.Tensor) -> torch.Tensor:
        batch, queries, channels, rows, cols = features.shape
        output = branch(features.reshape(batch * queries, channels, rows, cols))
        return output.reshape(batch, queries, 6, rows, cols)

    def _broadcast_context(self, query: QueryBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch, queries = query.unit_direction.shape[:2]
        direction = self.direction_encoder(query.unit_direction)
        direction_maps = direction[:, :, :, None, None].expand(-1, -1, -1, ARRAY_ROWS, ARRAY_COLS)
        geometry = self.geometry_maps.to(query.unit_direction)
        geometry = geometry[None, None].expand(batch, queries, -1, -1, -1)
        return direction_maps, geometry

    def forward(self, ideal_operator_iq: torch.Tensor, query: QueryBatch) -> OperatorPrediction:
        expected = (*query.unit_direction.shape[:2], 3, 2, ARRAY_ROWS, ARRAY_COLS)
        if ideal_operator_iq.shape != expected:
            raise ValueError(f"ideal_operator_iq must have shape {expected}")

        calibrated = self.element_gain(ideal_operator_iq)
        self_gain_delta = calibrated - ideal_operator_iq
        coupling_delta = self.local_coupling(calibrated)
        after_coupling = calibrated + coupling_delta

        direction_maps, geometry = self._broadcast_context(query)
        current_features = _operator_to_feature_maps(after_coupling)
        directional_input = torch.cat((current_features, direction_maps, geometry[:, :, :2]), dim=2)
        directional_delta = _feature_maps_to_operator(
            self._run_branch(self.directional_branch, directional_input)
        )

        edge_input = torch.cat((current_features, direction_maps, geometry), dim=2)
        edge_delta = _feature_maps_to_operator(self._run_branch(self.edge_branch, edge_input))
        edge_delta = edge_delta * geometry[:, :, 2:3, None]

        before_multihop = after_coupling + directional_delta + edge_delta
        if self.enable_multihop:
            multihop_input = torch.cat(
                (_operator_to_feature_maps(before_multihop), direction_maps, geometry),
                dim=2,
            )
            multihop_delta = _feature_maps_to_operator(
                self._run_branch(self.multihop_branch, multihop_input)
            )
        else:
            multihop_delta = torch.zeros_like(ideal_operator_iq)

        effective = project_transverse_operator_iq(
            before_multihop + multihop_delta,
            query.unit_direction,
        )
        return OperatorPrediction(
            effective_iq=effective,
            terms_iq={
                "ideal": ideal_operator_iq,
                "element_gain_delta": self_gain_delta,
                "local_coupling_delta": coupling_delta,
                "directional_delta": directional_delta,
                "edge_corner_delta": edge_delta,
                "multihop_delta": multihop_delta,
            },
        )

