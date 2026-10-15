"""Non-learning rank-one/rank-two inversion and identifiability audit."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .audits import rank_one_q_matrix
from .codebook import MeasurementCodebook, recover_pair_entry


@dataclass(frozen=True)
class InversionResult:
    q_matrix: torch.Tensor
    factor: torch.Tensor
    predicted_power: torch.Tensor
    normalized_residual: float
    minimum_jacobian_singular_value: float
    jacobian_condition: float


def powers_from_factor(factor: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """factor is [rank,16] with Q=sum_r g_r* g_r^T."""
    if not torch.is_complex(factor) or factor.ndim != 2 or factor.shape[-1] != 16:
        raise ValueError("factor must be complex with shape [rank,16]")
    if not torch.is_complex(commands) or commands.ndim != 2 or commands.shape[-1] != 16:
        raise ValueError("commands must be complex with shape [M,16]")
    fields = torch.einsum("mn,rn->mr", commands, factor)
    return fields.abs().square().sum(dim=-1)


def _powers_from_factor_batch(
    factor: torch.Tensor, commands: torch.Tensor
) -> torch.Tensor:
    """Batched counterpart with factor [B,rank,16] and output [B,M]."""
    if not torch.is_complex(factor) or factor.ndim != 3 or factor.shape[-1] != 16:
        raise ValueError("factor must be complex with shape [B,rank,16]")
    fields = torch.einsum("mn,brn->bmr", commands, factor)
    return fields.abs().square().sum(dim=-1)


def _spectral_initialization(commands: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    backprojection = torch.einsum(
        "m,mi,mj->ij", powers, commands, commands.conj()
    ) / len(commands)
    eigenvalues, eigenvectors = torch.linalg.eigh(backprojection)
    h = eigenvectors[:, -1] * eigenvalues[-1].real.clamp_min(1e-8).sqrt()
    g = h.conj()
    predicted = powers_from_factor(g[None], commands)
    scale = torch.sqrt(
        powers.sum().clamp_min(1e-12) / predicted.sum().clamp_min(1e-12)
    )
    return g * scale


def _spectral_initialization_batch(
    commands: torch.Tensor, powers: torch.Tensor
) -> torch.Tensor:
    backprojection = torch.einsum(
        "bm,mi,mj->bij", powers, commands, commands.conj()
    ) / len(commands)
    eigenvalues, eigenvectors = torch.linalg.eigh(backprojection)
    h = eigenvectors[:, :, -1] * eigenvalues[:, -1].real.clamp_min(1e-8).sqrt()[:, None]
    g = h.conj()
    predicted = _powers_from_factor_batch(g[:, None], commands)
    scale = torch.sqrt(
        powers.sum(dim=1).clamp_min(1e-12)
        / predicted.sum(dim=1).clamp_min(1e-12)
    )
    return g * scale[:, None]


def _fix_global_phase(factor: torch.Tensor) -> None:
    with torch.no_grad():
        reference = int(factor[0].abs().argmax())
        rotation = torch.exp(-1j * torch.angle(factor[0, reference]))
        factor.mul_(rotation)


def _fix_global_phase_batch(factor: torch.Tensor) -> None:
    with torch.no_grad():
        anchor = factor[:, 0].abs().argmax(dim=1)
        reference = factor[:, 0].gather(1, anchor[:, None]).squeeze(1)
        rotation = torch.exp(-1j * torch.angle(reference))
        factor.mul_(rotation[:, None, None])


def measurement_jacobian(g: torch.Tensor, commands: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Real Jacobian of |g^T w|^2 after removing one U(1) gauge column."""
    if not torch.is_complex(g) or g.shape != (16,):
        raise ValueError("g must be complex with shape [16]")
    if not torch.is_complex(commands) or commands.ndim != 2 or commands.shape[-1] != 16:
        raise ValueError("commands must be complex with shape [M,16]")
    anchor = int(g.abs().argmax())
    aligned = g * torch.exp(-1j * torch.angle(g[anchor]))
    field = commands @ aligned
    differential = field.conj()[:, None] * commands
    real_part = 2.0 * differential.real
    imaginary_part = -2.0 * differential.imag
    full = torch.cat((real_part, imaginary_part), dim=1)
    keep = torch.ones(32, dtype=torch.bool, device=full.device)
    keep[16 + anchor] = False
    return full[:, keep], anchor


def jacobian_diagnostics(
    g: torch.Tensor,
    commands: torch.Tensor,
    noise_standard_deviation: torch.Tensor | None = None,
    *,
    tolerance: float = 1e-10,
) -> dict[str, float | int | torch.Tensor]:
    jacobian, anchor = measurement_jacobian(g, commands)
    if noise_standard_deviation is not None:
        noise = torch.as_tensor(
            noise_standard_deviation, dtype=jacobian.dtype, device=jacobian.device
        ).reshape(-1)
        if noise.shape != (len(commands),) or torch.any(noise <= 0):
            raise ValueError("noise_standard_deviation must be positive with shape [M]")
        jacobian = jacobian / noise[:, None]
    singular = torch.linalg.svdvals(jacobian)
    retained = singular[singular > tolerance * singular.max().clamp_min(tolerance)]
    minimum = retained.min() if retained.numel() else torch.zeros((), device=singular.device)
    condition = singular.max() / minimum if float(minimum) > 0 else torch.tensor(float("inf"))
    fisher = jacobian.T @ jacobian
    return {
        "anchor": anchor,
        "minimum_singular_value": float(minimum.detach().cpu()),
        "condition": float(condition.detach().cpu()),
        "singular_values": singular.detach(),
        "fisher": fisher.detach(),
    }


def _fit_factor(
    commands: torch.Tensor,
    powers: torch.Tensor,
    *,
    rank: int,
    steps: int,
    learning_rate: float,
    seed: int,
    initial_g: torch.Tensor | None = None,
) -> torch.Tensor:
    commands = commands.detach()
    powers = powers.detach().to(commands.real.dtype)
    if initial_g is None:
        initial_g = _spectral_initialization(commands, powers)
    else:
        initial_g = initial_g.detach().to(device=commands.device, dtype=commands.dtype)
        if initial_g.shape != (16,):
            raise ValueError("initial_g must have shape [16]")
    generator = torch.Generator(device=commands.device).manual_seed(seed)
    factor = torch.zeros(rank, 16, dtype=commands.dtype, device=commands.device)
    factor[0] = initial_g
    if rank > 1:
        noise_real = torch.randn(rank - 1, 16, generator=generator, device=commands.device)
        noise_imag = torch.randn(rank - 1, 16, generator=generator, device=commands.device)
        factor[1:] = 1e-3 * torch.complex(noise_real, noise_imag).to(commands.dtype)
    factor.requires_grad_(True)
    optimizer = torch.optim.Adam([factor], lr=learning_rate)
    target_scale = powers.median().clamp_min(1e-8)
    best_loss = float("inf")
    best_factor = factor.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = powers_from_factor(factor, commands)
        residual = (predicted - powers) / target_scale
        absolute = residual.abs()
        huber = torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)
        loss = huber.mean()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_factor = factor.detach().clone()
        loss.backward()
        optimizer.step()
        _fix_global_phase(factor)
    return best_factor


def _fit_factor_batch(
    commands: torch.Tensor,
    powers: torch.Tensor,
    *,
    rank: int,
    steps: int,
    learning_rate: float,
    seed: int,
    initial_g: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit independent directions together; gradients remain direction-separable."""
    commands = commands.detach()
    powers = powers.detach().to(commands.real.dtype)
    if powers.ndim != 2 or powers.shape[1] != len(commands):
        raise ValueError("powers must have shape [B,M]")
    batch = powers.shape[0]
    if initial_g is None:
        initial_g = _spectral_initialization_batch(commands, powers)
    else:
        initial_g = initial_g.detach().to(device=commands.device, dtype=commands.dtype)
        if initial_g.shape != (batch, 16):
            raise ValueError("initial_g must have shape [B,16]")
    generator = torch.Generator(device=commands.device).manual_seed(seed)
    factor = torch.zeros(batch, rank, 16, dtype=commands.dtype, device=commands.device)
    factor[:, 0] = initial_g
    if rank > 1:
        noise_real = torch.randn(
            batch, rank - 1, 16, generator=generator, device=commands.device
        )
        noise_imag = torch.randn(
            batch, rank - 1, 16, generator=generator, device=commands.device
        )
        factor[:, 1:] = 1e-3 * torch.complex(noise_real, noise_imag).to(commands.dtype)
    factor.requires_grad_(True)
    optimizer = torch.optim.Adam([factor], lr=learning_rate)
    target_scale = powers.median(dim=1).values.clamp_min(1e-8)
    best_loss = torch.full((batch,), float("inf"), device=commands.device)
    best_factor = factor.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = _powers_from_factor_batch(factor, commands)
        residual = (predicted - powers) / target_scale[:, None]
        absolute = residual.abs()
        huber = torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)
        per_direction = huber.mean(dim=1)
        improved = per_direction.detach() < best_loss
        if torch.any(improved):
            best_loss[improved] = per_direction.detach()[improved]
            best_factor[improved] = factor.detach()[improved]
        per_direction.sum().backward()
        optimizer.step()
        _fix_global_phase_batch(factor)
    return best_factor


def recover_rank_one(
    commands: torch.Tensor,
    powers: torch.Tensor,
    *,
    steps: int = 400,
    learning_rate: float = 3e-2,
    seed: int = 21280,
    initial_g: torch.Tensor | None = None,
) -> InversionResult:
    if powers.shape != (len(commands),) or torch.any(powers < 0):
        raise ValueError("powers must be non-negative with shape [M]")
    factor = _fit_factor(
        commands,
        powers,
        rank=1,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        initial_g=initial_g,
    )
    predicted = powers_from_factor(factor, commands)
    residual = float(
        ((predicted - powers).square().sum() / powers.square().sum().clamp_min(1e-12)).cpu()
    )
    diagnostics = jacobian_diagnostics(factor[0], commands)
    return InversionResult(
        q_matrix=rank_one_q_matrix(factor[0]),
        factor=factor,
        predicted_power=predicted,
        normalized_residual=residual,
        minimum_jacobian_singular_value=float(diagnostics["minimum_singular_value"]),
        jacobian_condition=float(diagnostics["condition"]),
    )


def graph_synchronization_initialization(
    powers: torch.Tensor,
    codebook: MeasurementCodebook,
    *,
    split: str = "train",
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Initialize g from singles and normalized 0/90-degree pair probes."""
    if powers.shape != (96,) or torch.any(powers < 0):
        raise ValueError("powers must be non-negative with shape [96]")
    active = set(codebook.splits[split])
    single_by_element: dict[int, int] = {}
    pair_by_edge: dict[tuple[int, int], dict[str, int]] = {}
    for index in active:
        kind = codebook.kinds[index]
        if kind == "single":
            element = int(codebook.weights[index].abs().argmax())
            single_by_element[element] = index
        elif kind in {"pair_0", "pair_90"}:
            edge = codebook.edges[index]
            if edge is None:
                raise ValueError("pair command is missing its edge metadata")
            pair_by_edge.setdefault(edge, {})[kind] = index
    if set(single_by_element) != set(range(16)):
        raise ValueError("graph initialization requires all 16 single commands")

    power64 = powers.to(torch.float64)
    magnitude = torch.stack(
        [power64[single_by_element[element]].clamp_min(0).sqrt() for element in range(16)]
    )
    connection = torch.zeros(16, 16, dtype=torch.complex128, device=powers.device)
    for (left, right), pair in pair_by_edge.items():
        if set(pair) != {"pair_0", "pair_90"}:
            raise ValueError("each graph edge requires both 0 and 90-degree commands")
        q_entry = recover_pair_entry(
            power64[single_by_element[left]],
            power64[single_by_element[right]],
            power64[pair["pair_0"]],
            power64[pair["pair_90"]],
        )
        weight = q_entry.abs()
        if float(weight) > epsilon:
            unit = q_entry / weight
            connection[left, right] = weight * unit
            connection[right, left] = weight * unit.conj()
    if not torch.any(connection.abs() > 0):
        indices = torch.tensor(sorted(active), device=powers.device)
        return _spectral_initialization(
            codebook.weights.to(powers.device)[indices], power64[indices]
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(connection)
    phase = eigenvectors[:, eigenvalues.argmax()].conj()
    phase = phase / phase.abs().clamp_min(epsilon)
    estimate = magnitude.to(phase.dtype) * phase
    anchor = int(magnitude.argmax())
    estimate = estimate * torch.exp(-1j * torch.angle(estimate[anchor]))
    return estimate.to(codebook.weights.dtype)


def _graph_synchronization_initialization_batch(
    powers: torch.Tensor,
    codebook: MeasurementCodebook,
    *,
    split: str = "train",
    epsilon: float = 1e-10,
) -> torch.Tensor:
    if powers.ndim != 2 or powers.shape[1] != 96 or torch.any(powers < 0):
        raise ValueError("powers must be non-negative with shape [B,96]")
    active = set(codebook.splits[split])
    single_by_element: dict[int, int] = {}
    pair_by_edge: dict[tuple[int, int], dict[str, int]] = {}
    for index in active:
        kind = codebook.kinds[index]
        if kind == "single":
            element = int(codebook.weights[index].abs().argmax())
            single_by_element[element] = index
        elif kind in {"pair_0", "pair_90"}:
            edge = codebook.edges[index]
            if edge is None:
                raise ValueError("pair command is missing its edge metadata")
            pair_by_edge.setdefault(edge, {})[kind] = index
    if set(single_by_element) != set(range(16)):
        raise ValueError("graph initialization requires all 16 single commands")

    power64 = powers.to(torch.float64)
    single_indices = torch.tensor(
        [single_by_element[element] for element in range(16)],
        dtype=torch.long,
        device=powers.device,
    )
    magnitude = power64[:, single_indices].clamp_min(0).sqrt()
    connection = torch.zeros(
        len(powers), 16, 16, dtype=torch.complex128, device=powers.device
    )
    for (left, right), pair in pair_by_edge.items():
        if set(pair) != {"pair_0", "pair_90"}:
            raise ValueError("each graph edge requires both 0 and 90-degree commands")
        q_entry = recover_pair_entry(
            power64[:, single_by_element[left]],
            power64[:, single_by_element[right]],
            power64[:, pair["pair_0"]],
            power64[:, pair["pair_90"]],
        )
        connection[:, left, right] = q_entry
        connection[:, right, left] = q_entry.conj()
    eigenvalues, eigenvectors = torch.linalg.eigh(connection)
    phase = eigenvectors[:, :, -1].conj()
    phase = phase / phase.abs().clamp_min(epsilon)
    estimate = magnitude.to(phase.dtype) * phase
    anchor = magnitude.argmax(dim=1)
    reference = estimate.gather(1, anchor[:, None]).squeeze(1)
    estimate = estimate * torch.exp(-1j * torch.angle(reference))[:, None]

    disconnected = connection.abs().sum(dim=(1, 2)) <= epsilon
    if torch.any(disconnected):
        indices = torch.tensor(sorted(active), device=powers.device)
        estimate[disconnected] = _spectral_initialization_batch(
            codebook.weights.to(powers.device)[indices], power64[disconnected][:, indices]
        )
    return estimate.to(codebook.weights.dtype)


def _jacobian_diagnostics_batch(
    g: torch.Tensor,
    commands: torch.Tensor,
    noise_standard_deviation: torch.Tensor,
    *,
    tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return minimum singular value, condition and Fisher for [B,16] factors."""
    if not torch.is_complex(g) or g.ndim != 2 or g.shape[1] != 16:
        raise ValueError("g must be complex with shape [B,16]")
    anchor = g.abs().argmax(dim=1)
    reference = g.gather(1, anchor[:, None]).squeeze(1)
    aligned = g * torch.exp(-1j * torch.angle(reference))[:, None]
    field = torch.einsum("mn,bn->bm", commands, aligned)
    differential = field.conj()[:, :, None] * commands[None]
    full = torch.cat((2.0 * differential.real, -2.0 * differential.imag), dim=2)
    all_columns = torch.arange(32, device=g.device)[None].expand(len(g), -1)
    keep = all_columns != (16 + anchor)[:, None]
    retained_columns = all_columns[keep].reshape(len(g), 31)
    jacobian = full.gather(
        2, retained_columns[:, None, :].expand(-1, full.shape[1], -1)
    )
    noise = noise_standard_deviation.to(jacobian).reshape(len(g), len(commands))
    if torch.any(noise <= 0):
        raise ValueError("noise_standard_deviation must be positive")
    jacobian = jacobian / noise[:, :, None]
    singular = torch.linalg.svdvals(jacobian)
    maximum = singular[:, 0]
    valid = singular > tolerance * maximum[:, None].clamp_min(tolerance)
    minimum = torch.where(
        valid, singular, torch.full_like(singular, float("inf"))
    ).min(dim=1).values
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    condition = torch.where(
        minimum > 0, maximum / minimum, torch.full_like(minimum, float("inf"))
    )
    fisher = jacobian.transpose(1, 2) @ jacobian
    return minimum, condition, fisher


def _nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)


def rank_spectrum_audit(
    power_repeats: torch.Tensor,
    codebook: MeasurementCodebook,
    *,
    steps: int = 400,
    seed: int = 21290,
    heldout_split: str = "validation",
    batch_size: int = 128,
) -> dict[str, object]:
    """Audit [Q,96,repeats] powers without training an angular model."""
    if power_repeats.ndim != 3 or power_repeats.shape[1] != 96:
        raise ValueError("power_repeats must have shape [Q,96,repeats]")
    if power_repeats.shape[2] < 2 or torch.any(power_repeats < 0):
        raise ValueError("at least two non-negative repeats are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = power_repeats.device
    commands = codebook.weights.to(device)
    split_indices = {name: codebook.indices(name).to(device) for name in codebook.splits}
    if heldout_split not in {"validation", "test", "validation_test"}:
        raise ValueError("heldout_split must be validation, test, or validation_test")
    mean_power = power_repeats.mean(dim=-1)
    repeat_standard_deviation = power_repeats.std(dim=-1, unbiased=True)
    records: list[dict[str, float]] = []
    q_hats: list[torch.Tensor] = []
    fishers: list[torch.Tensor] = []

    train = split_indices["train"]
    train_commands = commands[train]
    heldout = (
        torch.cat((split_indices["validation"], split_indices["test"]))
        if heldout_split == "validation_test"
        else split_indices[heldout_split]
    )
    for start in range(0, power_repeats.shape[0], batch_size):
        stop = min(start + batch_size, power_repeats.shape[0])
        batch_power = mean_power[start:stop]
        train_power = batch_power[:, train]
        graph_initial = _graph_synchronization_initialization_batch(
            batch_power, codebook, split="train"
        )
        rank1_factor = _fit_factor_batch(
            train_commands,
            train_power,
            rank=1,
            steps=steps,
            learning_rate=3e-2,
            seed=seed + start,
            initial_g=graph_initial,
        )
        rank2_factor = _fit_factor_batch(
            train_commands,
            train_power,
            rank=2,
            steps=steps,
            learning_rate=3e-2,
            seed=seed + 10_000 + start,
            initial_g=rank1_factor[:, 0],
        )
        target = batch_power[:, heldout]
        predicted_rank1 = _powers_from_factor_batch(rank1_factor, commands[heldout])
        predicted_rank2 = _powers_from_factor_batch(rank2_factor, commands[heldout])
        nmse1 = (
            (predicted_rank1 - target).square().sum(dim=1)
            / target.square().sum(dim=1).clamp_min(1e-12)
        )
        nmse2 = (
            (predicted_rank2 - target).square().sum(dim=1)
            / target.square().sum(dim=1).clamp_min(1e-12)
        )
        rank1_db = 10.0 * torch.log10(nmse1.clamp_min(1e-12))
        rank2_db = 10.0 * torch.log10(nmse2.clamp_min(1e-12))
        improvement = rank1_db - rank2_db
        batch_repeats = power_repeats[start:stop]
        mean_db = 10.0 * torch.log10(batch_power.clamp_min(1e-12))
        repeat_db_error = (
            10.0 * torch.log10(batch_repeats.clamp_min(1e-12))
            - mean_db[:, :, None]
        ).abs()
        repeat_p95_db = torch.quantile(repeat_db_error.flatten(1), 0.95, dim=1)

        direction_noise = repeat_standard_deviation[start:stop, train]
        numerical_floor = train_power.median(dim=1).values.clamp_min(1e-12) * 1e-6
        positive_median = []
        for row in direction_noise:
            positive = row[row > 0]
            positive_median.append(
                positive.median() if positive.numel() else row.new_zeros(())
            )
        numerical_floor = torch.maximum(
            numerical_floor, torch.stack(positive_median) * 0.1
        )
        direction_noise = torch.maximum(direction_noise, numerical_floor[:, None])
        minimum, condition, fisher = _jacobian_diagnostics_batch(
            rank1_factor[:, 0], train_commands, direction_noise
        )
        predicted_train = _powers_from_factor_batch(rank1_factor, train_commands)
        train_residual = (
            (predicted_train - train_power).square().sum(dim=1)
            / train_power.square().sum(dim=1).clamp_min(1e-12)
        )
        q_hats.append(rank_one_q_matrix(rank1_factor[:, 0]).detach().cpu())
        fishers.append(fisher.detach().cpu())
        for row in range(stop - start):
            records.append(
                {
                    "rank1_nmse_db": float(rank1_db[row].detach().cpu()),
                    "rank2_nmse_db": float(rank2_db[row].detach().cpu()),
                    "rank2_improvement_db": float(improvement[row].detach().cpu()),
                    "repeatability_p95_power_error_db": float(
                        repeat_p95_db[row].detach().cpu()
                    ),
                    "rank1_training_normalized_residual": float(
                        train_residual[row].detach().cpu()
                    ),
                    "minimum_jacobian_singular_value": float(
                        minimum[row].detach().cpu()
                    ),
                    "jacobian_condition": float(condition[row].detach().cpu()),
                }
            )

    improvement = torch.tensor([item["rank2_improvement_db"] for item in records])
    repeat_uncertainty = torch.tensor(
        [item["repeatability_p95_power_error_db"] for item in records]
    )
    exceeds_repeatability = improvement > repeat_uncertainty + 1e-6
    contract_failure = (improvement >= 0.5) & exceeds_repeatability
    failure_fraction = float(contract_failure.to(torch.float32).mean())
    accepted = (improvement <= 0.2 + 1e-6) & (~exceeds_repeatability)
    if failure_fraction >= 0.20:
        status = "contract_failure"
    elif float(accepted.to(torch.float32).mean()) >= 0.80:
        status = "rank1_accepted"
    else:
        status = "gray_zone"
    return {
        "schema": "v212-rank-spectrum-audit-1",
        "status": status,
        "heldout_split": heldout_split,
        "contract_failure_fraction": failure_fraction,
        "rank1_acceptance_fraction": float(accepted.to(torch.float32).mean()),
        "records": records,
        "q_hat": torch.cat(q_hats),
        "fisher": torch.cat(fishers),
    }
