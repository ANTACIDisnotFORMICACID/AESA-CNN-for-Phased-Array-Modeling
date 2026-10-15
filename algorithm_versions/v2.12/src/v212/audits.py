"""Gauge-invariant operator and decomposition diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .basis import real_spherical_harmonics, ssh_mode_index


def rank_one_q_matrix(g: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(g) or g.shape[-1] != 16:
        raise ValueError("g must be complex with trailing dimension 16")
    return g.conj()[..., :, None] * g[..., None, :]


def dephase_q_matrix(
    q_matrix: torch.Tensor,
    unit_direction: torch.Tensor,
    positions_m: torch.Tensor,
    wavenumber_rad_m: float,
) -> torch.Tensor:
    """Return D_r Q D_r^H, removing exp(+j k q.r) from both Q axes."""
    if not torch.is_complex(q_matrix) or q_matrix.shape[-2:] != (16, 16):
        raise ValueError("q_matrix must be complex with shape [...,16,16]")
    if unit_direction.shape[-1] != 3 or positions_m.shape != (16, 3):
        raise ValueError("direction/position shapes are invalid")
    phase = float(wavenumber_rad_m) * torch.einsum(
        "...k,nk->...n", unit_direction, positions_m.to(unit_direction)
    )
    diagonal = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    return diagonal[..., :, None] * q_matrix * diagonal.conj()[..., None, :]


def fourier_low_order_leakage(
    filtration_basis: torch.Tensor,
    fourier_residual: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Report low-filtration energy in a latent Fourier residual; never a loss."""
    if filtration_basis.ndim != 2 or fourier_residual.shape[0] != filtration_basis.shape[0]:
        raise ValueError("basis and residual must share a leading Q dimension")
    count = filtration_basis.shape[0]
    if weights is None:
        weights = torch.full(
            (count,), 1.0 / count, dtype=filtration_basis.dtype, device=filtration_basis.device
        )
    else:
        weights = torch.as_tensor(weights, device=filtration_basis.device, dtype=filtration_basis.dtype)
        weights = weights / weights.sum().clamp_min(epsilon)
    residual = fourier_residual.reshape(count, -1)
    working_dtype = (
        torch.complex128
        if filtration_basis.dtype == torch.float64 or residual.dtype == torch.complex128
        else torch.complex64
    )
    residual = residual.to(working_dtype)
    weights = weights.to(residual.real.dtype)
    square_root_weight = weights.sqrt()[:, None]
    design = square_root_weight.to(working_dtype) * filtration_basis.to(working_dtype)
    target = square_root_weight.to(working_dtype) * residual
    coefficients = torch.linalg.lstsq(design, target).solution
    projection = design @ coefficients
    numerator = projection.abs().square().sum()
    denominator = target.abs().square().sum().clamp_min(epsilon)
    return numerator / denominator


@dataclass(frozen=True)
class GaugeInvariantSpectrum:
    degree_energy: torch.Tensor
    cumulative_fraction: dict[int, float]
    normalized_gradient_energy: float
    coefficients: torch.Tensor


@dataclass(frozen=True)
class NestedFiltrationSpectrum:
    cumulative_fraction: torch.Tensor
    shell_fraction: torch.Tensor
    normalized_gradient_energy: float
    projected: tuple[torch.Tensor, ...]
    retained_ranks: tuple[int, ...]
    transformed_condition: float


@dataclass(frozen=True)
class NestedOperatorComparison:
    prediction: NestedFiltrationSpectrum
    truth: NestedFiltrationSpectrum
    spectral_distance: float
    shell_error_fraction: torch.Tensor
    tail_error_fraction: float


def _normalised_weights(
    count: int,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return torch.full((count,), 1.0 / count, dtype=torch.float64)
    value = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if value.shape != (count,) or torch.any(value < 0) or float(value.sum()) <= 0:
        raise ValueError("weights must be non-negative and length Q")
    return value / value.sum()


def nested_filtration_spectrum(
    q_tilde: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    degree: int = 12,
    solid_angle_weights: torch.Tensor | None = None,
    epsilon: float = 1e-12,
    truncation_ratio: float = 1e-4,
) -> NestedFiltrationSpectrum:
    """Basis-invariant energy of nested restricted-SSH subspaces."""
    if not torch.is_complex(q_tilde) or q_tilde.ndim != 3 or q_tilde.shape[-2:] != (16, 16):
        raise ValueError("q_tilde must be complex with shape [Q,16,16]")
    if unit_direction.shape != (len(q_tilde), 3):
        raise ValueError("unit_direction must have shape [Q,3]")
    direction = unit_direction.detach().to(dtype=torch.float64, device="cpu")
    weights = _normalised_weights(len(direction), solid_angle_weights)
    basis_weighted, ranks, condition = _stable_nested_basis(
        direction, weights, degree, truncation_ratio
    )
    return _nested_spectrum_with_basis(
        q_tilde, weights, basis_weighted, ranks, condition, epsilon
    )


def _stable_nested_basis(
    direction: torch.Tensor,
    weights: torch.Tensor,
    degree: int,
    truncation_ratio: float,
) -> tuple[torch.Tensor, tuple[int, ...], float]:
    """Build stable nested quotient spaces directly in weighted sample coordinates.

    Two-pass reorthogonalization prevents the high-degree loss of orthogonality
    seen when tiny hemisphere residuals are converted back through ``1/sigma``.
    The cutoff is referenced to the full design's largest singular value, so the
    retained global design condition is bounded by ``1 / truncation_ratio``.
    """
    raw = real_spherical_harmonics(direction, degree)
    weighted_raw = weights.sqrt()[:, None] * raw
    reference_sigma = torch.linalg.svdvals(weighted_raw).max()
    threshold_squared = truncation_ratio * reference_sigma.square()
    basis = torch.empty((len(direction), 0), dtype=torch.float64)
    ranks: list[int] = []
    start = 0
    for l_value in range(degree + 1):
        width = 2 * l_value + 1
        residual = weighted_raw[:, start : start + width]
        for _ in range(2):
            if basis.shape[1]:
                residual = residual - basis @ (basis.T @ residual)
        u_value, sigma, _ = torch.linalg.svd(residual, full_matrices=False)
        keep = sigma.square() >= threshold_squared
        block = u_value[:, keep]
        if block.shape[1]:
            # Reorthogonalize the accepted vectors as well, then use their
            # weighted-space left singular vectors rather than a fragile 1/sigma map.
            for _ in range(2):
                if basis.shape[1]:
                    block = block - basis @ (basis.T @ block)
            block, block_sigma, _ = torch.linalg.svd(block, full_matrices=False)
            block = block[:, block_sigma > 1e-12]
            basis = torch.cat((basis, block), dim=1)
        ranks.append(int(block.shape[1]))
        start += width
    if basis.shape[1] == 0:
        raise ValueError("all nested SSH modes were truncated")
    gram = basis.T @ basis
    eigenvalues = torch.linalg.eigvalsh(gram)
    condition = float((eigenvalues.max() / eigenvalues.min()).cpu())
    return basis, tuple(ranks), condition


def _nested_spectrum_with_basis(
    q_tilde: torch.Tensor,
    weights: torch.Tensor,
    basis_weighted: torch.Tensor,
    ranks: tuple[int, ...],
    transformed_condition: float,
    epsilon: float,
) -> NestedFiltrationSpectrum:
    target = q_tilde.detach().to(dtype=torch.complex128, device="cpu").reshape(len(q_tilde), -1)
    square_root_weight = weights.sqrt()[:, None]
    weighted_target = square_root_weight.to(torch.complex128) * target
    total_energy = weighted_target.abs().square().sum().clamp_min(epsilon)
    cumulative: list[torch.Tensor] = []
    projections: list[torch.Tensor] = []
    for l_value in range(len(ranks)):
        retained = sum(ranks[: l_value + 1])
        design = basis_weighted[:, :retained].to(torch.complex128)
        projection_weighted = design @ (design.mH @ weighted_target)
        projection = torch.zeros_like(target)
        positive = square_root_weight[:, 0] > 0
        projection[positive] = projection_weighted[positive] / square_root_weight[positive]
        projections.append(projection.reshape_as(q_tilde.cpu()))
        cumulative.append(projection_weighted.abs().square().sum().real / total_energy)
    cumulative_tensor = torch.stack(cumulative)
    shell = torch.cat((cumulative_tensor[:1], cumulative_tensor[1:] - cumulative_tensor[:-1]))
    shell = shell.clamp_min(0.0)
    degrees = torch.arange(len(ranks), dtype=torch.float64)
    captured = shell.sum().clamp_min(epsilon)
    gradient = float((degrees * (degrees + 1) * shell).sum() / captured)
    return NestedFiltrationSpectrum(
        cumulative_tensor,
        shell,
        gradient,
        tuple(projections),
        ranks,
        transformed_condition,
    )


def nested_operator_comparison(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    degree: int = 12,
    solid_angle_weights: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> NestedOperatorComparison:
    direction = unit_direction.detach().to(dtype=torch.float64, device="cpu")
    weights = _normalised_weights(len(unit_direction), solid_angle_weights)
    basis_weighted, ranks, condition = _stable_nested_basis(
        direction, weights, degree, 1e-4
    )
    predicted = _nested_spectrum_with_basis(
        prediction, weights, basis_weighted, ranks, condition, epsilon
    )
    reference = _nested_spectrum_with_basis(
        truth, weights, basis_weighted, ranks, condition, epsilon
    )
    truth_energy = (
        weights[:, None, None] * truth.detach().cpu().to(torch.complex128).abs().square()
    ).sum().clamp_min(epsilon)
    error = (prediction - truth).detach().cpu().to(torch.complex128)
    error_projections = _nested_spectrum_with_basis(
        error, weights, basis_weighted, ranks, condition, epsilon
    ).projected
    shell_error: list[torch.Tensor] = []
    previous = torch.zeros_like(error_projections[0])
    for projection in error_projections:
        increment = projection - previous
        shell_error.append(
            (weights[:, None, None] * increment.abs().square()).sum().real / truth_energy
        )
        previous = projection
    tail = error - error_projections[-1]
    tail_error = float((weights[:, None, None] * tail.abs().square()).sum().real / truth_energy)
    spectral_distance = float(0.5 * (predicted.shell_fraction - reference.shell_fraction).abs().sum())
    return NestedOperatorComparison(
        prediction=predicted,
        truth=reference,
        spectral_distance=spectral_distance,
        shell_error_fraction=torch.stack(shell_error),
        tail_error_fraction=tail_error,
    )


def operator_graph_roughness(
    q_tilde: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    neighbors: int = 6,
    chunk_size: int = 512,
    epsilon: float = 1e-12,
) -> float:
    """Normalized six-neighbor angular finite-difference energy."""
    if len(q_tilde) <= neighbors:
        raise ValueError("more directions than neighbors are required")
    direction = unit_direction.detach().to(dtype=torch.float64, device="cpu")
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(epsilon)
    matrix = q_tilde.detach().to(dtype=torch.complex128, device="cpu")
    numerator = torch.zeros((), dtype=torch.float64)
    edge_count = 0
    for start in range(0, len(direction), chunk_size):
        stop = min(start + chunk_size, len(direction))
        cosine = direction[start:stop] @ direction.T
        row = torch.arange(stop - start)
        cosine[row, torch.arange(start, stop)] = -2.0
        values, indices = torch.topk(cosine, neighbors, dim=1)
        angle = torch.acos(values.clamp(-1.0, 1.0)).clamp_min(1e-8)
        difference = matrix[start:stop, None] - matrix[indices]
        numerator += (difference.abs().square().sum(dim=(-1, -2)) / angle.square()).sum()
        edge_count += difference.shape[0] * difference.shape[1]
    mean_difference = numerator / max(edge_count, 1)
    energy = matrix.abs().square().sum(dim=(-1, -2)).mean().clamp_min(epsilon)
    return float(mean_difference / energy)


def gauge_invariant_spectrum(
    q_tilde: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    degree: int = 12,
    solid_angle_weights: torch.Tensor | None = None,
    cumulative_degrees: tuple[int, ...] = (4, 5, 10),
    epsilon: float = 1e-12,
) -> GaugeInvariantSpectrum:
    """Weighted raw-SSH least-squares spectrum of gauge-invariant Q-tilde."""
    if not torch.is_complex(q_tilde) or q_tilde.ndim != 3 or q_tilde.shape[-2:] != (16, 16):
        raise ValueError("q_tilde must be complex with shape [Q,16,16]")
    if unit_direction.shape != (q_tilde.shape[0], 3):
        raise ValueError("unit_direction must have shape [Q,3]")
    direction = unit_direction.detach().to(dtype=torch.float64, device="cpu")
    matrix = q_tilde.detach().to(dtype=torch.complex128, device="cpu")
    count = len(direction)
    if solid_angle_weights is None:
        weights = torch.full((count,), 1.0 / count, dtype=torch.float64)
    else:
        weights = torch.as_tensor(solid_angle_weights, dtype=torch.float64).reshape(-1)
        if weights.shape != (count,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("solid_angle_weights must be non-negative and length Q")
        weights = weights / weights.sum()
    raw = real_spherical_harmonics(direction, degree)
    square_root = weights.sqrt()[:, None]
    design = (square_root * raw).to(torch.complex128)
    target = square_root.to(torch.complex128) * matrix.reshape(count, -1)
    coefficients = torch.linalg.lstsq(design, target).solution.reshape(-1, 16, 16)
    degrees, _ = ssh_mode_index(degree)
    energy = torch.zeros(degree + 1, dtype=torch.float64)
    for l_value in range(degree + 1):
        energy[l_value] = coefficients[degrees == l_value].abs().square().sum().real
    total = energy.sum().clamp_min(epsilon)
    cumulative = {
        cutoff: float(energy[: min(cutoff, degree) + 1].sum() / total)
        for cutoff in cumulative_degrees
    }
    l_values = torch.arange(degree + 1, dtype=torch.float64)
    gradient = float((l_values * (l_values + 1.0) * energy).sum() / total)
    return GaugeInvariantSpectrum(
        degree_energy=energy,
        cumulative_fraction=cumulative,
        normalized_gradient_energy=gradient,
        coefficients=coefficients,
    )
