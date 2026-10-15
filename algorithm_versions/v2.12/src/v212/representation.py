"""Latent representation diagnostics for V2.12-S2.

These helpers deliberately operate on ``a(q)`` branch outputs only.  They are
architectural diagnostics and must not be interpreted as identifiable field
quantities; physical claims remain attached to gauge-invariant ``Q_tilde``.
"""

from __future__ import annotations

import torch


def weighted_projection(
    basis: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Project values onto a sampled basis using the actual weighted measure."""
    if basis.ndim != 2 or values.shape[0] != basis.shape[0]:
        raise ValueError("basis and values must share a leading sample dimension")
    count = basis.shape[0]
    if weights is None:
        weights = torch.full(
            (count,), 1.0 / count, dtype=basis.dtype, device=basis.device
        )
    else:
        weights = torch.as_tensor(weights, dtype=basis.dtype, device=basis.device).reshape(-1)
        if weights.shape != (count,) or torch.any(weights < 0):
            raise ValueError("weights must be non-negative and have length Q")
        weights = weights / weights.sum().clamp_min(epsilon)
    target = values.reshape(count, -1)
    working_dtype = torch.complex128 if (
        basis.dtype == torch.float64 or target.dtype == torch.complex128
    ) else torch.complex64
    design = basis.to(working_dtype)
    target = target.to(working_dtype)
    root_weight = weights.sqrt()[:, None].to(working_dtype)
    coefficient = torch.linalg.lstsq(root_weight * design, root_weight * target).solution
    return (design @ coefficient).reshape_as(target)


def branch_energy_decomposition(
    basis: torch.Tensor,
    ssh: torch.Tensor,
    fourier: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Return absolute branch energies and their coherent low-order interaction."""
    if ssh.shape != fourier.shape or ssh.shape[0] != basis.shape[0]:
        raise ValueError("ssh, fourier and basis samples must match")
    count = basis.shape[0]
    if weights is None:
        weight = torch.full((count,), 1.0 / count, dtype=torch.float64)
    else:
        weight = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if weight.shape != (count,) or torch.any(weight < 0):
            raise ValueError("weights must be non-negative and have length Q")
        weight = weight / weight.sum().clamp_min(epsilon)
    ssh_flat = ssh.reshape(count, -1).to(torch.complex128)
    fourier_flat = fourier.reshape(count, -1).to(torch.complex128)
    low = weighted_projection(basis.to(torch.float64), fourier_flat, weight)
    high = fourier_flat - low

    def energy(value: torch.Tensor) -> torch.Tensor:
        return (weight[:, None] * value.abs().square()).sum().real

    ssh_energy = energy(ssh_flat)
    low_energy = energy(low)
    fourier_energy = energy(fourier_flat)
    coherent_cross = 2.0 * (
        weight[:, None] * (ssh_flat.conj() * low).real
    ).sum()
    return {
        "ssh_energy": float(ssh_energy),
        "fourier_total_energy": float(fourier_energy),
        "fourier_low_energy": float(low_energy),
        "fourier_high_energy": float(energy(high)),
        "fourier_low_fraction": float(low_energy / fourier_energy.clamp_min(epsilon)),
        "ssh_fourier_low_cross_energy": float(coherent_cross),
        "combined_low_energy": float(energy(ssh_flat + low)),
    }


def tangent_frame(unit_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a deterministic orthonormal tangent frame for unit directions."""
    if unit_direction.ndim != 2 or unit_direction.shape[1] != 3:
        raise ValueError("unit_direction must have shape [Q,3]")
    q = unit_direction / unit_direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device).expand_as(q)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=q.dtype, device=q.device).expand_as(q)
    anchor = torch.where((q[:, 2].abs() < 0.9)[:, None], z, y)
    first = torch.linalg.cross(q, anchor, dim=1)
    first = first / first.norm(dim=1, keepdim=True).clamp_min(1e-12)
    second = torch.linalg.cross(q, first, dim=1)
    second = second / second.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return first, second


def tangent_perturbations(
    unit_direction: torch.Tensor,
    *,
    step_rad: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return central angular perturbations along two tangent directions."""
    if step_rad <= 0:
        raise ValueError("step_rad must be positive")
    q = unit_direction / unit_direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    first, second = tangent_frame(q)
    cosine = torch.cos(torch.as_tensor(step_rad, dtype=q.dtype, device=q.device))
    sine = torch.sin(torch.as_tensor(step_rad, dtype=q.dtype, device=q.device))
    return (
        cosine * q + sine * first,
        cosine * q - sine * first,
        cosine * q + sine * second,
        cosine * q - sine * second,
    )


def tangent_jacobian_norm(
    plus_first: torch.Tensor,
    minus_first: torch.Tensor,
    plus_second: torch.Tensor,
    minus_second: torch.Tensor,
    center: torch.Tensor,
    *,
    step_rad: float,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference tangent Jacobian Frobenius norm and scale-normalized norm."""
    if not (
        plus_first.shape == minus_first.shape == plus_second.shape == minus_second.shape == center.shape
    ):
        raise ValueError("all branch samples must have the same shape")
    if step_rad <= 0:
        raise ValueError("step_rad must be positive")
    count = center.shape[0]
    derivative_first = (plus_first - minus_first).reshape(count, -1) / (2.0 * step_rad)
    derivative_second = (plus_second - minus_second).reshape(count, -1) / (2.0 * step_rad)
    squared = derivative_first.square().sum(dim=1) + derivative_second.square().sum(dim=1)
    raw = squared.clamp_min(0).sqrt()
    scale = center.reshape(count, -1).square().sum(dim=1).clamp_min(epsilon).sqrt()
    return raw, raw / scale


def per_direction_nmse_db(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Direction-wise command-aggregated amplitude NMSE in dB."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have shape [Q,C]")
    ratio = (prediction - target).square().sum(dim=1) / target.square().sum(dim=1).clamp_min(epsilon)
    return 10.0 * torch.log10(ratio.clamp_min(epsilon))
