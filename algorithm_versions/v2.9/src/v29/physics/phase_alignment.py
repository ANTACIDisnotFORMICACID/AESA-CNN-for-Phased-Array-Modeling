"""Gauge alignment used only by diagnostics, never by the prediction path."""

from __future__ import annotations

import torch


def align_global_phase_iq(
    operator_iq: torch.Tensor,
    reference_iq: torch.Tensor,
) -> torch.Tensor:
    if operator_iq.shape != reference_iq.shape or operator_iq.shape[-3] != 2:
        raise ValueError("operator and reference must share [...,2,4,4] shape")
    real, imag = operator_iq.unbind(dim=-3)
    ref_real, ref_imag = reference_iq.unbind(dim=-3)
    inner_real = (real * ref_real + imag * ref_imag).sum(dim=(-2, -1))
    inner_imag = (imag * ref_real - real * ref_imag).sum(dim=(-2, -1))
    norm = torch.sqrt(inner_real.square() + inner_imag.square())
    safe = norm >= 1e-8
    rotation_real = torch.where(safe, inner_real / norm.clamp_min(1e-8), torch.ones_like(norm))
    rotation_imag = torch.where(safe, -inner_imag / norm.clamp_min(1e-8), torch.zeros_like(norm))
    rotated_real = real * rotation_real[..., None, None] - imag * rotation_imag[..., None, None]
    rotated_imag = real * rotation_imag[..., None, None] + imag * rotation_real[..., None, None]
    return torch.stack((rotated_real, rotated_imag), dim=-3)
