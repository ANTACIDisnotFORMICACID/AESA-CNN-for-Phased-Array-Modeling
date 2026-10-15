"""O1 D-optimal optimization of the 32 supplemental constant-modulus commands."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .codebook import MeasurementCodebook


@dataclass(frozen=True)
class CodebookOptimizationResult:
    codebook: MeasurementCodebook
    validation_minimum_p05: float
    validation_median_minimum: float
    validation_mean_logdet: float
    validation_condition_median: float
    validation_condition_p95: float
    restart: int


def _batched_jacobian(g: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """Differentiable [Q,M,31] gauge-removed Jacobian."""
    if not torch.is_complex(g) or g.ndim != 2 or g.shape[-1] != 16:
        raise ValueError("g must be complex with shape [Q,16]")
    anchor = g.abs().argmax(dim=-1)
    anchor_value = g.gather(1, anchor[:, None])[:, 0]
    aligned = g * torch.exp(-1j * torch.angle(anchor_value))[:, None]
    field = torch.einsum("mn,qn->qm", commands, aligned)
    differential = field.conj()[..., None] * commands[None]
    full = torch.cat((2.0 * differential.real, -2.0 * differential.imag), dim=-1)
    columns = torch.arange(32, device=g.device)[None].expand(len(g), -1)
    keep = columns != (16 + anchor)[:, None]
    keep = keep[:, None, :].expand(-1, len(commands), -1)
    return full[keep].reshape(len(g), len(commands), 31)


def information_metrics(
    g: torch.Tensor,
    commands: torch.Tensor,
    noise_standard_deviation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    jacobian = _batched_jacobian(g, commands)
    if noise_standard_deviation is not None:
        noise = torch.as_tensor(
            noise_standard_deviation, dtype=jacobian.dtype, device=jacobian.device
        ).reshape(-1)
        if noise.shape != (len(commands),) or torch.any(noise <= 0):
            raise ValueError("noise_standard_deviation must be positive with shape [M]")
        jacobian = jacobian / noise[None, :, None]
    fisher = torch.einsum("qmi,qmj->qij", jacobian, jacobian)
    trace = fisher.diagonal(dim1=-2, dim2=-1).sum(-1)
    epsilon = 1e-6 * trace.clamp_min(1e-12) / 31.0
    identity = torch.eye(31, dtype=fisher.dtype, device=fisher.device)
    regularized = fisher + epsilon[:, None, None] * identity
    logdet = torch.linalg.slogdet(regularized).logabsdet
    singular = torch.linalg.svdvals(jacobian)
    minimum = singular[..., -1]
    condition = singular[..., 0] / minimum.clamp_min(1e-12)
    return logdet, minimum, condition


def optimize_supplemental_commands(
    baseline: MeasurementCodebook,
    train_g: torch.Tensor,
    validation_g: torch.Tensor,
    *,
    noise_standard_deviation: torch.Tensor | None = None,
    restarts: int = 5,
    steps: int = 2000,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    seed: int = 21310,
) -> CodebookOptimizationResult:
    if restarts <= 0 or steps <= 0 or learning_rate <= 0 or batch_size <= 0:
        raise ValueError("optimization settings must be positive")
    if train_g.shape[-1] != 16 or validation_g.shape[-1] != 16:
        raise ValueError("train_g and validation_g must end with 16 elements")
    device = train_g.device
    random_indices = [index for index, kind in enumerate(baseline.kinds) if kind == "random"]
    baseline_weights = baseline.weights.to(device)

    def assemble(supplemental: torch.Tensor) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        supplemental_index = 0
        for index in range(96):
            if index in random_indices:
                rows.append(supplemental[supplemental_index])
                supplemental_index += 1
            else:
                rows.append(baseline_weights[index])
        return torch.stack(rows)
    best: CodebookOptimizationResult | None = None

    for restart in range(restarts):
        generator = torch.Generator(device=device).manual_seed(seed + restart)
        phases = (2.0 * math.pi * torch.rand(32, 16, generator=generator, device=device)).requires_grad_()
        optimizer = torch.optim.Adam([phases], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            supplemental = torch.polar(torch.full_like(phases, 0.25), phases)
            commands = assemble(supplemental)
            if len(train_g) > batch_size:
                index = torch.randint(
                    len(train_g),
                    (batch_size,),
                    generator=generator,
                    device=device,
                )
                batch_g = train_g[index]
            else:
                batch_g = train_g
            logdet, _, _ = information_metrics(batch_g, commands, noise_standard_deviation)
            loss = -logdet.mean()
            loss.backward()
            optimizer.step()
        supplemental = torch.polar(torch.full_like(phases, 0.25), phases).detach()
        candidate = baseline.with_supplemental_commands(supplemental.cpu())
        logdet_chunks: list[torch.Tensor] = []
        minimum_chunks: list[torch.Tensor] = []
        condition_chunks: list[torch.Tensor] = []
        for start in range(0, len(validation_g), batch_size):
            logdet, minimum, condition = information_metrics(
                validation_g[start : start + batch_size],
                assemble(supplemental),
                noise_standard_deviation,
            )
            logdet_chunks.append(logdet.detach())
            minimum_chunks.append(minimum.detach())
            condition_chunks.append(condition.detach())
        logdet = torch.cat(logdet_chunks)
        minimum = torch.cat(minimum_chunks)
        condition = torch.cat(condition_chunks)
        result = CodebookOptimizationResult(
            codebook=candidate,
            validation_minimum_p05=float(torch.quantile(minimum, 0.05).cpu()),
            validation_median_minimum=float(minimum.median().cpu()),
            validation_mean_logdet=float(logdet.mean().cpu()),
            validation_condition_median=float(condition.median().cpu()),
            validation_condition_p95=float(torch.quantile(condition, 0.95).cpu()),
            restart=restart,
        )
        if best is None:
            best = result
        else:
            relative = (result.validation_minimum_p05 - best.validation_minimum_p05) / max(
                abs(best.validation_minimum_p05), 1e-12
            )
            if relative > 0.01 or (
                abs(relative) <= 0.01
                and result.validation_mean_logdet > best.validation_mean_logdet
            ):
                best = result
    if best is None:
        raise AssertionError("at least one restart must produce a codebook")
    return best
