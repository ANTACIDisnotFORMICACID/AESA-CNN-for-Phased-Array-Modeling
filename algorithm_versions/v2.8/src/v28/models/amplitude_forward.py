"""Physics-structured latent complex channel trained from amplitude only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from ..contracts import ARRAY_COLS, ARRAY_ROWS, ArrayGeometry, QueryBatch
from .blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder


@dataclass(frozen=True)
class AmplitudePrediction:
    amplitude: torch.Tensor
    power: torch.Tensor
    field_iq: torch.Tensor
    operator_iq: torch.Tensor
    terms_iq: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amplitude.ndim != 2:
            raise ValueError("amplitude must have shape [B,Q]")
        if self.power.shape != self.amplitude.shape:
            raise ValueError("power must have shape [B,Q]")
        if self.field_iq.shape != (*self.amplitude.shape, 2):
            raise ValueError("field_iq must have shape [B,Q,2]")
        if self.operator_iq.shape != (*self.amplitude.shape, 2, 4, 4):
            raise ValueError("operator_iq must have shape [B,Q,2,4,4]")


def contract_scalar_operator_iq(operator_iq: torch.Tensor, command_iq: torch.Tensor) -> torch.Tensor:
    if operator_iq.ndim != 5 or operator_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("operator_iq must have shape [B,Q,2,4,4]")
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [B,2,4,4]")
    if operator_iq.shape[0] != command_iq.shape[0]:
        raise ValueError("operator and command batch sizes must match")
    op_real, op_imag = operator_iq.unbind(dim=2)
    cmd_real, cmd_imag = command_iq.unbind(dim=1)
    real = torch.einsum("bqrc,brc->bq", op_real, cmd_real) - torch.einsum(
        "bqrc,brc->bq", op_imag, cmd_imag
    )
    imag = torch.einsum("bqrc,brc->bq", op_real, cmd_imag) + torch.einsum(
        "bqrc,brc->bq", op_imag, cmd_real
    )
    return torch.stack((real, imag), dim=-1)


def contract_scalar_operator_grid_iq(
    operator_iq: torch.Tensor,
    command_iq: torch.Tensor,
) -> torch.Tensor:
    """Contract one direction grid with many known commands without duplicating the CNN pass."""
    if operator_iq.ndim != 4 or operator_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("operator_iq must have shape [Q,2,4,4]")
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [C,2,4,4]")
    op_real, op_imag = operator_iq.unbind(dim=1)
    cmd_real, cmd_imag = command_iq.unbind(dim=1)
    real = torch.einsum("qij,kij->kq", op_real, cmd_real) - torch.einsum(
        "qij,kij->kq", op_imag, cmd_imag
    )
    imag = torch.einsum("qij,kij->kq", op_real, cmd_imag) + torch.einsum(
        "qij,kij->kq", op_imag, cmd_real
    )
    return torch.stack((real, imag), dim=-1)


class IdealScalarFarFieldOperator(nn.Module):
    """Ideal scalar receive-channel steering operator [B,Q,IQ,4,4]."""

    def __init__(self, geometry: ArrayGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.register_buffer("positions_m", torch.as_tensor(geometry.positions_m, dtype=torch.float32))
        self.wavenumber_rad_m = 2.0 * np.pi / geometry.wavelength_m

    def forward(self, query: QueryBatch) -> torch.Tensor:
        direction = query.unit_direction
        positions = self.positions_m.to(direction)
        phase = self.wavenumber_rad_m * torch.einsum("bqk,rck->bqrc", direction, positions)
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=2)


class ScalarComplexElementGain(nn.Module):
    """Per-element latent complex calibration; initialized to identity."""

    def __init__(self) -> None:
        super().__init__()
        gain = torch.zeros(2, 4, 4)
        gain[0] = 1.0
        self.gain_iq = nn.Parameter(gain)

    def forward(self, operator_iq: torch.Tensor) -> torch.Tensor:
        op_real, op_imag = operator_iq.unbind(dim=2)
        gain_real, gain_imag = self.gain_iq.unbind(dim=0)
        real = op_real * gain_real - op_imag * gain_imag
        imag = op_real * gain_imag + op_imag * gain_real
        return torch.stack((real, imag), dim=2)


class ScalarComplexLocalCoupling3x3(nn.Module):
    """Nearest-neighbour latent complex coupling with a zeroed self coefficient."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel_real = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.kernel_imag = nn.Parameter(torch.zeros(1, 1, 3, 3))
        mask = torch.ones(1, 1, 3, 3)
        mask[:, :, 1, 1] = 0.0
        self.register_buffer("neighbour_mask", mask)

    def forward(self, operator_iq: torch.Tensor) -> torch.Tensor:
        batch, queries = operator_iq.shape[:2]
        real = operator_iq[:, :, 0].reshape(batch * queries, 1, 4, 4)
        imag = operator_iq[:, :, 1].reshape(batch * queries, 1, 4, 4)
        kernel_real = self.kernel_real * self.neighbour_mask
        kernel_imag = self.kernel_imag * self.neighbour_mask
        out_real = functional.conv2d(real, kernel_real, padding=1) - functional.conv2d(
            imag, kernel_imag, padding=1
        )
        out_imag = functional.conv2d(real, kernel_imag, padding=1) + functional.conv2d(
            imag, kernel_real, padding=1
        )
        return torch.stack((out_real[:, 0], out_imag[:, 0]), dim=1).reshape(batch, queries, 2, 4, 4)


class AmplitudeOperatorResidualCNN(nn.Module):
    """Predict a latent complex scalar channel; measured phase is never an input or label."""

    def __init__(
        self,
        *,
        fourier_bands: int = 4,
        directional_width: int = 24,
        edge_width: int = 16,
        element_pattern_width: int = 20,
        multihop_width: int = 16,
        enable_element_pattern: bool = True,
        enable_multihop: bool = False,
    ) -> None:
        super().__init__()
        self.direction_encoder = DirectionFourierEncoder(fourier_bands)
        direction_channels = self.direction_encoder.output_channels
        self.element_gain = ScalarComplexElementGain()
        self.enable_element_pattern = enable_element_pattern
        if enable_element_pattern:
            self.element_pattern_branch = ApertureCorrectionBranch(
                direction_channels + 4,
                element_pattern_width,
                depth=2,
                output_channels=2,
            )
        self.local_coupling = ScalarComplexLocalCoupling3x3()
        self.directional_branch = ApertureCorrectionBranch(
            2 + direction_channels + 2,
            directional_width,
            output_channels=2,
        )
        self.edge_branch = ApertureCorrectionBranch(
            2 + direction_channels + 4,
            edge_width,
            output_channels=2,
        )
        self.multihop_branch = ApertureCorrectionBranch(
            2 + direction_channels + 4,
            multihop_width,
            depth=2,
            output_channels=2,
        )
        self.enable_multihop = enable_multihop
        self.register_buffer("geometry_maps", self._build_geometry_maps())

    @staticmethod
    def _build_geometry_maps() -> torch.Tensor:
        row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
        col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
        edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
        corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
        return torch.stack((row, col, edge, corner), dim=0)

    @staticmethod
    def _run_branch(branch: nn.Module, features: torch.Tensor) -> torch.Tensor:
        batch, queries, channels, rows, cols = features.shape
        output = branch(features.reshape(batch * queries, channels, rows, cols))
        return output.reshape(batch, queries, 2, rows, cols)

    @staticmethod
    def _fix_global_phase(operator_iq: torch.Tensor, ideal_operator_iq: torch.Tensor) -> torch.Tensor:
        """Align the latent channel to the whole ideal aperture, avoiding a fragile one-element gauge."""
        real, imag = operator_iq.unbind(dim=2)
        ideal_real, ideal_imag = ideal_operator_iq.unbind(dim=2)
        inner_real = (real * ideal_real + imag * ideal_imag).sum(dim=(-2, -1))
        inner_imag = (imag * ideal_real - real * ideal_imag).sum(dim=(-2, -1))
        norm = torch.sqrt(inner_real.square() + inner_imag.square())
        safe = norm >= 1e-8
        rotation_real = torch.where(safe, inner_real / norm.clamp_min(1e-8), torch.ones_like(norm))
        rotation_imag = torch.where(safe, -inner_imag / norm.clamp_min(1e-8), torch.zeros_like(norm))
        rotated_real = real * rotation_real[..., None, None] - imag * rotation_imag[..., None, None]
        rotated_imag = real * rotation_imag[..., None, None] + imag * rotation_real[..., None, None]
        return torch.stack((rotated_real, rotated_imag), dim=2)

    def forward(self, ideal_operator_iq: torch.Tensor, query: QueryBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (*query.unit_direction.shape[:2], 2, 4, 4)
        if ideal_operator_iq.shape != expected:
            raise ValueError(f"ideal_operator_iq must have shape {expected}")
        batch, queries = query.unit_direction.shape[:2]
        direction = self.direction_encoder(query.unit_direction)
        direction_maps = direction[:, :, :, None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(query.unit_direction)[None, None].expand(batch, queries, -1, -1, -1)
        calibrated = self.element_gain(ideal_operator_iq)
        gain_delta = calibrated - ideal_operator_iq
        if self.enable_element_pattern:
            pattern_delta = self._run_branch(
                self.element_pattern_branch,
                torch.cat((direction_maps, geometry), dim=2),
            )
            calibrated_real, calibrated_imag = calibrated.unbind(dim=2)
            multiplier_real = 1.0 + pattern_delta[:, :, 0]
            multiplier_imag = pattern_delta[:, :, 1]
            patterned_real = calibrated_real * multiplier_real - calibrated_imag * multiplier_imag
            patterned_imag = calibrated_real * multiplier_imag + calibrated_imag * multiplier_real
            patterned = torch.stack((patterned_real, patterned_imag), dim=2)
        else:
            pattern_delta = torch.zeros_like(calibrated)
            patterned = calibrated
        coupling_delta = self.local_coupling(patterned)
        current = patterned + coupling_delta

        directional_delta = self._run_branch(
            self.directional_branch,
            torch.cat((current, direction_maps, geometry[:, :, :2]), dim=2),
        )
        edge_delta = self._run_branch(
            self.edge_branch,
            torch.cat((current, direction_maps, geometry), dim=2),
        ) * geometry[:, :, 2:3]
        current = current + directional_delta + edge_delta
        if self.enable_multihop:
            multihop_delta = self._run_branch(
                self.multihop_branch,
                torch.cat((current, direction_maps, geometry), dim=2),
            )
        else:
            multihop_delta = torch.zeros_like(current)
        effective = self._fix_global_phase(current + multihop_delta, ideal_operator_iq)
        return effective, {
            "ideal": ideal_operator_iq,
            "element_gain_delta": gain_delta,
            "element_pattern_delta": pattern_delta,
            "local_coupling_delta": coupling_delta,
            "directional_delta": directional_delta,
            "edge_corner_delta": edge_delta,
            "multihop_delta": multihop_delta,
        }


class AmplitudeOnlyForwardModel(nn.Module):
    """Known command and coordinate to scalar amplitude/power, with latent complex physics."""

    def __init__(
        self,
        geometry: ArrayGeometry | None = None,
        correction: AmplitudeOperatorResidualCNN | None = None,
        *,
        reference_distance_m: float = 1.0,
    ) -> None:
        super().__init__()
        if reference_distance_m <= 0:
            raise ValueError("reference_distance_m must be positive")
        self.geometry = geometry or ArrayGeometry()
        self.reference_distance_m = float(reference_distance_m)
        self.ideal_operator = IdealScalarFarFieldOperator(self.geometry)
        self.correction = correction or AmplitudeOperatorResidualCNN()

    def operator(self, query: QueryBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.correction(self.ideal_operator(query), query)

    def forward(self, command_iq: torch.Tensor, query: QueryBatch) -> AmplitudePrediction:
        operator, terms = self.operator(query)
        field = contract_scalar_operator_iq(operator, command_iq)
        amplitude = torch.linalg.vector_norm(field, dim=-1)
        if query.range_m is not None:
            amplitude = amplitude * self.reference_distance_m / query.range_m.clamp_min(1e-8)
            field = field * (self.reference_distance_m / query.range_m.clamp_min(1e-8))[..., None]
        return AmplitudePrediction(
            amplitude=amplitude,
            power=amplitude.square(),
            field_iq=field,
            operator_iq=operator,
            terms_iq=terms,
        )

    def forward_command_grid(
        self,
        command_iq: torch.Tensor,
        unit_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Predict [command,direction] scalar amplitudes with one operator evaluation per direction."""
        if unit_direction.ndim != 2 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [Q,3]")
        operator, _ = self.operator(QueryBatch(unit_direction[None]))
        field = contract_scalar_operator_grid_iq(operator[0], command_iq)
        return torch.linalg.vector_norm(field, dim=-1)
