"""Non-learning physics-closure metrics for the V2.13 validation campaign."""

from __future__ import annotations

import math

import torch

from v212.basis import real_spherical_harmonics


def rank_one_q(g: torch.Tensor) -> torch.Tensor:
    """Return Q=g* g^T so that w^H Q w = |g^T w|^2."""
    if not torch.is_complex(g) or g.ndim != 2 or g.shape[1] != 16:
        raise ValueError("g must be complex with shape [Q,16]")
    return g.conj()[:, :, None] * g[:, None, :]


def operator_contract_metrics(
    g: torch.Tensor,
    commands: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Numerically verify Hermitian/PSD/rank-one/quadratic identities."""
    q_matrix = rank_one_q(g).to(torch.complex128)
    commands = torch.as_tensor(commands).to(torch.complex128)
    scale = q_matrix.abs().amax().clamp_min(epsilon)
    hermitian = (q_matrix - q_matrix.mH).abs().amax() / scale
    eigenvalues = torch.linalg.eigvalsh(q_matrix).real
    largest = eigenvalues[:, -1].clamp_min(epsilon)
    second = eigenvalues[:, -2].abs()
    minimum = eigenvalues[:, 0]
    direct = (g.to(torch.complex128) @ commands.T).abs().square()
    quadratic = torch.einsum(
        "cn,qnm,cm->qc", commands.conj(), q_matrix, commands
    ).real
    identity = (quadratic - direct).abs()
    return {
        "hermitian_relative_max": float(hermitian),
        "minimum_eigenvalue_relative_min": float((minimum / largest).min()),
        "lambda2_over_lambda1_p95": float(torch.quantile(second / largest, 0.95)),
        "quadratic_identity_relative_l2": float(
            torch.linalg.vector_norm(identity)
            / torch.linalg.vector_norm(direct).clamp_min(epsilon)
        ),
        "quadratic_identity_relative_max": float(
            (identity / direct.abs().clamp_min(epsilon)).amax()
        ),
    }


def compare_operators(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    if prediction.shape != truth.shape or not torch.is_complex(prediction):
        raise ValueError("prediction and truth must be matching complex operators")
    error_energy = (prediction - truth).abs().square().sum(dim=(-2, -1))
    truth_energy = truth.abs().square().sum(dim=(-2, -1)).clamp_min(epsilon)
    relative = error_energy / truth_energy
    aggregate = error_energy.sum() / truth_energy.sum().clamp_min(epsilon)
    return {
        "aggregate_nmse": float(aggregate),
        "aggregate_nmse_db": float(10.0 * torch.log10(aggregate.clamp_min(epsilon))),
        "per_direction_relative_error_p50": float(torch.quantile(relative, 0.50)),
        "per_direction_relative_error_p95": float(torch.quantile(relative, 0.95)),
        "per_direction_relative_error_max": float(relative.max()),
    }


def deembed_geometric_phase(
    g: torch.Tensor,
    unit_direction: torch.Tensor,
    positions_m: torch.Tensor,
    wavenumber_rad_m: float,
) -> torch.Tensor:
    if g.shape != (len(unit_direction), len(positions_m)):
        raise ValueError("g, directions and positions do not align")
    phase = float(wavenumber_rad_m) * torch.einsum(
        "qk,nk->qn", unit_direction.to(torch.float64), positions_m.to(torch.float64)
    )
    return g.to(torch.complex128) * torch.exp(torch.complex(torch.zeros_like(phase), -phase))


def _normalised_weights(count: int, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return torch.full((count,), 1.0 / count, dtype=torch.float64)
    value = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if value.shape != (count,) or torch.any(value < 0) or float(value.sum()) <= 0:
        raise ValueError("weights must be non-negative and have length Q")
    return value / value.sum()


def nested_vector_spectrum(
    values: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    degree: int = 12,
    weights: torch.Tensor | None = None,
    truncation_ratio: float = 1e-4,
    epsilon: float = 1e-12,
) -> dict[str, object]:
    """Stable hemisphere filtration spectrum for privileged complex vector fields."""
    if not torch.is_complex(values) or values.ndim != 2 or len(values) != len(unit_direction):
        raise ValueError("values must be complex [Q,D] and align with directions")
    direction = unit_direction.detach().cpu().to(torch.float64)
    target = values.detach().cpu().to(torch.complex128)
    weight = _normalised_weights(len(direction), weights)
    raw = real_spherical_harmonics(direction, degree).to(torch.float64)
    weighted_raw = weight.sqrt()[:, None] * raw
    reference_sigma = torch.linalg.svdvals(weighted_raw).max()
    threshold_squared = truncation_ratio * reference_sigma.square()
    target_weighted = weight.sqrt()[:, None].to(torch.complex128) * target
    total = target_weighted.abs().square().sum().real.clamp_min(epsilon)
    basis = torch.empty((len(direction), 0), dtype=torch.float64)
    ranks: list[int] = []
    cumulative: list[torch.Tensor] = []
    start = 0
    for l_value in range(degree + 1):
        width = 2 * l_value + 1
        residual = weighted_raw[:, start : start + width]
        for _ in range(2):
            if basis.shape[1]:
                residual = residual - basis @ (basis.T @ residual)
        u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        retained = singular.square() >= threshold_squared
        block = u[:, retained]
        if block.shape[1]:
            for _ in range(2):
                if basis.shape[1]:
                    block = block - basis @ (basis.T @ block)
            block, _ = torch.linalg.qr(block, mode="reduced")
            basis = torch.cat((basis, block), dim=1)
        ranks.append(block.shape[1])
        coefficient = basis.T.to(torch.complex128) @ target_weighted
        cumulative.append(coefficient.abs().square().sum().real / total)
        start += width
    cumulative_tensor = torch.stack(cumulative).clamp(0.0, 1.0)
    shell = torch.cat((cumulative_tensor[:1], cumulative_tensor[1:] - cumulative_tensor[:-1]))
    degree_index = torch.arange(degree + 1, dtype=torch.float64)
    effective_gradient = (degree_index * (degree_index + 1.0) * shell).sum()
    return {
        "cumulative_fraction": {str(i): float(cumulative_tensor[i]) for i in range(degree + 1)},
        "shell_fraction": shell.tolist(),
        "retained_ranks": ranks,
        "projected_fraction_l12": float(cumulative_tensor[-1]),
        "tail_fraction": float((1.0 - cumulative_tensor[-1]).clamp_min(0.0)),
        "normalized_filtration_gradient": float(effective_gradient),
    }


def vector_graph_roughness(
    values: torch.Tensor,
    unit_direction: torch.Tensor,
    *,
    neighbors: int = 6,
    chunk_size: int = 512,
    epsilon: float = 1e-12,
) -> float:
    if len(values) <= neighbors or values.ndim != 2:
        raise ValueError("values must be [Q,D] with Q greater than neighbors")
    direction = unit_direction.detach().cpu().to(torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(epsilon)
    field = values.detach().cpu().to(torch.complex128)
    numerator = torch.zeros((), dtype=torch.float64)
    edge_count = 0
    for start in range(0, len(direction), chunk_size):
        stop = min(start + chunk_size, len(direction))
        cosine = direction[start:stop] @ direction.T
        row = torch.arange(stop - start)
        cosine[row, torch.arange(start, stop)] = -2.0
        cosine_value, index = torch.topk(cosine, neighbors, dim=1)
        angle = torch.acos(cosine_value.clamp(-1.0, 1.0)).clamp_min(1e-8)
        difference = field[start:stop, None] - field[index]
        numerator += (difference.abs().square().sum(-1) / angle.square()).sum()
        edge_count += difference.shape[0] * difference.shape[1]
    energy = field.abs().square().sum(-1).mean().clamp_min(epsilon)
    return float((numerator / edge_count) / energy)


def gradient_alignment(
    prediction_q: torch.Tensor,
    truth_q: torch.Tensor,
    commands: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare Wirtinger power gradients Qw in the real complex-vector metric."""
    if prediction_q.shape != truth_q.shape or prediction_q.ndim != 3:
        raise ValueError("prediction_q and truth_q must match [Q,N,N]")
    command = torch.as_tensor(commands).to(prediction_q.dtype)
    predicted = torch.einsum("qnm,cm->qcn", prediction_q, command)
    reference = torch.einsum("qnm,cm->qcn", truth_q, command)
    inner = (predicted.conj() * reference).sum(-1).real
    predicted_norm = predicted.abs().square().sum(-1).sqrt()
    reference_norm = reference.abs().square().sum(-1).sqrt()
    cosine = inner / (predicted_norm * reference_norm).clamp_min(epsilon)
    relative = (predicted - reference).abs().square().sum(-1).sqrt() / reference_norm.clamp_min(epsilon)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(torch.quantile(cosine, 0.05)),
        "cosine_median": float(torch.quantile(cosine, 0.50)),
        "relative_gradient_error_mean": float(relative.mean()),
        "relative_gradient_error_p95": float(torch.quantile(relative, 0.95)),
    }


def phase_only_maximin(
    q_matrix: torch.Tensor,
    *,
    starts: int = 8,
    steps: int = 200,
    learning_rate: float = 0.05,
    temperature: float = 0.02,
    seed: int = 21301,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically optimize constant-modulus weights for batched K-point maximin power."""
    if not torch.is_complex(q_matrix) or q_matrix.ndim != 4 or q_matrix.shape[-2:] != (16, 16):
        raise ValueError("q_matrix must be complex [B,K,16,16]")
    if starts <= 0 or steps <= 0:
        raise ValueError("starts and steps must be positive")
    batch, targets, elements, _ = q_matrix.shape
    scale = q_matrix.diagonal(dim1=-2, dim2=-1).real.sum(-1).mean(-1).clamp_min(1e-12)
    normalized = q_matrix / scale[:, None, None, None]
    generator = torch.Generator(device=q_matrix.device).manual_seed(seed)
    theta = 2.0 * math.pi * torch.rand(
        (batch, starts, elements), generator=generator, device=q_matrix.device,
        dtype=q_matrix.real.dtype,
    )
    for target_index in range(min(targets, starts)):
        _, eigenvector = torch.linalg.eigh(normalized[:, target_index])
        theta[:, target_index] = torch.angle(eigenvector[:, :, -1])
    theta.requires_grad_(True)
    optimizer = torch.optim.Adam((theta,), lr=learning_rate)
    for _ in range(steps):
        weight = torch.exp(torch.complex(torch.zeros_like(theta), theta)) / math.sqrt(elements)
        power = torch.einsum(
            "bsn,bknm,bsm->bsk", weight.conj(), normalized, weight
        ).real
        soft_minimum = -temperature * torch.logsumexp(-power / temperature, dim=-1)
        loss = -soft_minimum.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weight = torch.exp(torch.complex(torch.zeros_like(theta), theta)) / math.sqrt(elements)
        power = torch.einsum("bsn,bknm,bsm->bsk", weight.conj(), q_matrix, weight).real
        objective = power.min(-1).values
        best = objective.argmax(-1)
        row = torch.arange(batch, device=q_matrix.device)
        return weight[row, best], objective[row, best]
