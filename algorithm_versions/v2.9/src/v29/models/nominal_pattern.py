"""Direction-dependent nominal element response with separate amplitude/phase heads."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import QueryBatch
from .blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder


def _geometry_maps() -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


class NominalElementPattern(nn.Module):
    """Learn g0(q) as bounded log-amplitude and a normalized complex phase factor."""

    def __init__(
        self,
        *,
        fourier_bands: int = 4,
        width: int = 24,
        depth: int = 2,
        max_log_amplitude: float = 3.0,
    ) -> None:
        super().__init__()
        if max_log_amplitude <= 0:
            raise ValueError("max_log_amplitude must be positive")
        self.direction_encoder = DirectionFourierEncoder(fourier_bands)
        self.head = ApertureCorrectionBranch(
            self.direction_encoder.output_channels + 4,
            width,
            depth=depth,
            output_channels=3,
        )
        self.max_log_amplitude = float(max_log_amplitude)
        self.register_buffer("geometry_maps", _geometry_maps())

    def forward(
        self,
        ideal_operator_iq: torch.Tensor,
        query: QueryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (*query.unit_direction.shape[:2], 2, 4, 4)
        if ideal_operator_iq.shape != expected:
            raise ValueError(f"ideal_operator_iq must have shape {expected}")
        batch, queries = query.unit_direction.shape[:2]
        encoded = self.direction_encoder(query.unit_direction)
        direction_maps = encoded[:, :, :, None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(query.unit_direction)[None, None].expand(
            batch, queries, -1, -1, -1
        )
        features = torch.cat((direction_maps, geometry), dim=2)
        raw = self.head(features.reshape(batch * queries, features.shape[2], 4, 4))
        raw = raw.reshape(batch, queries, 3, 4, 4)

        log_amplitude = self.max_log_amplitude * torch.tanh(
            raw[:, :, 0] / self.max_log_amplitude
        )
        phase_real = 1.0 + raw[:, :, 1]
        phase_imag = raw[:, :, 2]
        phase_norm = torch.sqrt(phase_real.square() + phase_imag.square()).clamp_min(1e-6)
        multiplier_real = torch.exp(log_amplitude) * phase_real / phase_norm
        multiplier_imag = torch.exp(log_amplitude) * phase_imag / phase_norm

        ideal_real, ideal_imag = ideal_operator_iq.unbind(dim=2)
        patterned_real = ideal_real * multiplier_real - ideal_imag * multiplier_imag
        patterned_imag = ideal_real * multiplier_imag + ideal_imag * multiplier_real
        patterned = torch.stack((patterned_real, patterned_imag), dim=2)
        multiplier = torch.stack((multiplier_real, multiplier_imag), dim=2)
        return patterned, {
            "log_amplitude": log_amplitude,
            "phase_factor_iq": multiplier,
        }
