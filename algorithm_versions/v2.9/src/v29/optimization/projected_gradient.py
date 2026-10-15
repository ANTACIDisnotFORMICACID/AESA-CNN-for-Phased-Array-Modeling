"""Deterministic multi-start projected optimization of the max-min objective."""

from __future__ import annotations

import torch

from ..contracts import QueryBatch
from torch import nn
from .problem import FourTargetProblem, FourTargetResult


def project_command_iq(
    command_iq: torch.Tensor,
    *,
    total_power: float = 1.0,
    amplitude_max: float = 1.0,
) -> torch.Tensor:
    """Project each [IQ,4,4] command onto fixed power and amplitude limits."""
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [B,2,4,4]")
    if total_power <= 0 or amplitude_max <= 0 or 16 * amplitude_max**2 < total_power:
        raise ValueError("invalid or infeasible command constraints")

    amplitude = torch.linalg.vector_norm(command_iq, dim=1)
    zero = amplitude.square().sum(dim=(1, 2), keepdim=True) <= torch.finfo(command_iq.dtype).eps
    amplitude = torch.where(zero, torch.ones_like(amplitude), amplitude)
    phase_iq = command_iq / torch.linalg.vector_norm(command_iq, dim=1, keepdim=True).clamp_min(1e-12)
    default_phase = torch.zeros_like(phase_iq)
    default_phase[:, 0] = 1.0
    phase_iq = torch.where(zero[:, None], default_phase, phase_iq)

    lower = torch.zeros(amplitude.shape[0], 1, 1, device=amplitude.device, dtype=amplitude.dtype)
    upper = torch.ones_like(lower)
    target = torch.as_tensor(total_power, device=amplitude.device, dtype=amplitude.dtype)
    limit = torch.as_tensor(amplitude_max, device=amplitude.device, dtype=amplitude.dtype)
    for _ in range(32):
        current_power = torch.minimum(upper * amplitude, limit).square().sum(dim=(1, 2), keepdim=True)
        upper = torch.where(current_power < target, upper * 2.0, upper)
    for _ in range(48):
        middle = (lower + upper) / 2.0
        current_power = torch.minimum(middle * amplitude, limit).square().sum(dim=(1, 2), keepdim=True)
        lower = torch.where(current_power < target, middle, lower)
        upper = torch.where(current_power >= target, middle, upper)
    projected_amplitude = torch.minimum(upper * amplitude, limit)
    return phase_iq * projected_amplitude[:, None]


def _repeat_query(query: QueryBatch, repeats: int) -> QueryBatch:
    range_m = query.range_m.expand(repeats, -1) if query.range_m is not None else None
    targets = query.target_xyz_m.expand(repeats, -1, -1) if query.target_xyz_m is not None else None
    return QueryBatch(
        unit_direction=query.unit_direction.expand(repeats, -1, -1),
        range_m=range_m,
        target_xyz_m=targets,
        coordinate_type=query.coordinate_type,
    )


def optimize_four_targets(
    model: nn.Module,
    problem: FourTargetProblem,
    *,
    steps: int = 256,
    learning_rate: float = 0.05,
    softmin_temperature_db: float = 0.25,
    random_restarts: int = 8,
    seed: int = 28001,
) -> FourTargetResult:
    """Maximize the weakest of four predicted received powers."""
    if steps <= 0 or random_restarts <= 0 or softmin_temperature_db <= 0:
        raise ValueError("steps, restarts and softmin temperature must be positive")
    parameter = next(model.parameters())
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    initial = torch.randn(
        random_restarts,
        2,
        4,
        4,
        generator=generator,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    initial = project_command_iq(
        initial,
        total_power=problem.total_command_power,
        amplitude_max=problem.per_element_amplitude_max,
    )
    command = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([command], lr=learning_rate)
    query = _repeat_query(problem.query, random_restarts)
    temperature = torch.as_tensor(softmin_temperature_db, device=parameter.device, dtype=parameter.dtype)
    history: list[float] = []

    training_state = model.training
    requires_grad = [value.requires_grad for value in model.parameters()]
    model.eval()
    for value in model.parameters():
        value.requires_grad_(False)
    try:
        for _ in range(steps):
            power = model(command, query).power.clamp_min(1e-20)
            power_db = 10.0 * torch.log10(power)
            soft_min = -temperature * torch.logsumexp(-power_db / temperature, dim=1)
            loss = -soft_min.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                command.copy_(
                    project_command_iq(
                        command,
                        total_power=problem.total_command_power,
                        amplitude_max=problem.per_element_amplitude_max,
                    )
                )
                true_min = model(command, query).power.min(dim=1).values.max()
                history.append(float(true_min.detach().cpu()))
        with torch.no_grad():
            final_power = model(command, query).power
            minima = final_power.min(dim=1).values
            best = int(torch.argmax(minima))
            return FourTargetResult(
                command_iq=command[best].detach().clone(),
                target_power=final_power[best].detach().clone(),
                min_target_power=float(minima[best].detach().cpu()),
                history=tuple(history),
            )
    finally:
        for value, state in zip(model.parameters(), requires_grad, strict=True):
            value.requires_grad_(state)
        model.train(training_state)
