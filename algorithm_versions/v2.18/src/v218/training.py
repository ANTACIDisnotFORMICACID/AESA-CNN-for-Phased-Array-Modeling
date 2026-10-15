"""Resumable V2.18 training selected only by validation amplitude NMSE."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from v210.contracts import QueryBatch
from v212.evaluation import stratified_separation_subset
from v212.training import amplitude_objective

from .canonical import PreparedData, atomic_json
from .experiment import (
    LOSS_CONTRACT,
    OPAUX_ARM,
    PHASE_AUX_ARMS,
    architecture_contract,
    build_model,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 240
    batch_size: int = 512
    learning_rate: float = 1e-3
    late_learning_rate: float = 3e-4
    learning_rate_drop_epoch: int = 181
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    residual_energy_weight: float = 1e-4
    separation_weight: float = 0.001
    separation_start_epoch: int = 61
    separation_interval_steps: int = 8
    validation_query_chunk: int = 1024


def _operator_complex(model: torch.nn.Module, direction: torch.Tensor) -> tuple[torch.Tensor, dict]:
    operator_iq, terms = model.operator(QueryBatch(direction[None]))
    operator = torch.complex(operator_iq[0, :, 0], operator_iq[0, :, 1]).reshape(len(direction), 16)
    return operator, terms


def complex_operator_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("complex operators must have matching shape [Q,16]")
    normalized = weights / weights.sum().clamp_min(1e-12)
    numerator = (normalized[:, None] * (prediction - target).abs().square()).sum()
    denominator = (normalized[:, None] * target.abs().square()).sum().clamp_min(1e-12)
    return numerator / denominator


def phase_aligned_operator_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Gauge-invariant operator NMSE with one optimal phase per direction."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("complex operators must have matching shape [Q,16]")
    if weights.shape != (len(prediction),):
        raise ValueError("solid-angle weights must have shape [Q]")
    inner = (prediction.conj() * target).sum(dim=1)
    phase = inner / inner.abs().clamp_min(1e-12)
    aligned = prediction * phase[:, None]
    normalized = weights / weights.sum().clamp_min(1e-12)
    numerator = (normalized[:, None] * (aligned - target).abs().square()).sum()
    denominator = (normalized[:, None] * target.abs().square()).sum().clamp_min(1e-12)
    return numerator / denominator


def _separation_fixture(data: PreparedData):
    direction = torch.from_numpy(data.train["direction"]).float()
    population_weights = torch.from_numpy(data.train["solid_angle_weights"]).double()
    index, weights = stratified_separation_subset(
        direction, per_cell=40, seed=21890, training_weights=population_weights
    )
    subset = direction[index]
    basis = data.transform.evaluate(subset.double()).float()
    weights = weights.float()
    gram = basis.T @ (weights[:, None] * basis)
    rhs = basis.T * weights[None]
    factor, info = torch.linalg.cholesky_ex(gram)
    analysis = (
        torch.cholesky_solve(rhs, factor)
        if int(info.max()) == 0 else torch.linalg.lstsq(gram, rhs).solution
    )
    return subset, weights, basis, analysis


def separation_loss(model, direction, weights, basis, analysis) -> torch.Tensor:
    residual_iq = model.residual(direction[None])[0]
    residual = torch.complex(residual_iq[:, 0], residual_iq[:, 1]).reshape(len(direction), 16)
    residual = model.envelope_scale.to(residual) * residual
    projection = basis.to(residual) @ (analysis.to(residual) @ residual)
    real_weights = weights.to(device=residual.device, dtype=residual.real.dtype)
    numerator = (real_weights[:, None] * projection.abs().square()).sum()
    denominator = (
        real_weights[:, None] * residual.abs().square()
    ).sum().detach().clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def evaluate_amplitude(model, direction, target, commands, *, chunk_size: int) -> dict[str, float]:
    model.eval()
    squared_error = torch.zeros((), dtype=torch.float64, device=direction.device)
    squared_target = torch.zeros_like(squared_error)
    absolute_db = torch.zeros_like(squared_error)
    count = 0
    for start in range(0, len(direction), chunk_size):
        stop = min(start + chunk_size, len(direction))
        prediction = model.forward_command_grid(commands, direction[start:stop]).T
        truth = target[start:stop]
        squared_error += (prediction.double() - truth.double()).square().sum()
        squared_target += truth.double().square().sum()
        absolute_db += (
            20 * torch.log10(prediction.clamp_min(1e-6))
            - 20 * torch.log10(truth.clamp_min(1e-6))
        ).abs().double().sum()
        count += truth.numel()
    nmse = squared_error / squared_target.clamp_min(1e-12)
    return {
        "linear_nmse": float(nmse),
        "linear_nmse_db": float(10 * torch.log10(nmse.clamp_min(1e-12))),
        "mae_db": float(absolute_db / max(count, 1)),
    }


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_run(
    arm: str,
    seed: int,
    data: PreparedData,
    run_dir: Path,
    *,
    device: str,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    config = config or TrainConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope_scale = float(data.manifest["envelope_scale"])
    model = build_model(
        arm, transform=data.transform, envelope_scale=envelope_scale, seed=seed
    ).to(device)
    parameter = next(model.parameters())
    dtype = parameter.dtype
    train_direction = torch.from_numpy(data.train["direction"]).to(device=device, dtype=dtype)
    train_target = torch.from_numpy(data.train["amplitude"]).to(device=device, dtype=dtype)
    train_operator = torch.from_numpy(data.train["g_ephi"]).to(device=device)
    train_weights = torch.from_numpy(data.train["solid_angle_weights"]).to(device=device, dtype=dtype)
    train_commands = torch.from_numpy(data.train["commands"]).to(device=device)
    val_direction = torch.from_numpy(data.validation["direction"]).to(device=device, dtype=dtype)
    val_target = torch.from_numpy(data.validation["amplitude"]).to(device=device, dtype=dtype)
    val_commands = torch.from_numpy(data.validation["commands"]).to(device=device)
    sep = tuple(value.to(device=device, dtype=dtype) for value in _separation_fixture(data))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    latest = run_dir / "latest.pt"
    start_epoch = 1
    global_step = 0
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_validation = math.inf
    best_state = None
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["arm"] != arm or int(state["seed"]) != seed:
            raise ValueError("resume checkpoint contract mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["generator_state"])
        start_epoch = int(state["completed_epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state["history"])
        best_epoch = int(state["best_epoch"])
        best_validation = float(state["best_validation"])
        best_state = state["best_state"]
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        learning_rate = (
            config.late_learning_rate if epoch >= config.learning_rate_drop_epoch
            else config.learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        permutation = torch.randperm(len(train_direction), generator=generator)
        total_loss = 0.0
        total_complex = 0.0
        train_squared_error = 0.0
        train_squared_target = 0.0
        seen = 0
        for start in range(0, len(permutation), config.batch_size):
            global_step += 1
            index = permutation[start:start + config.batch_size].to(device)
            direction = train_direction[index]
            target = train_target[index]
            operator_prediction = None
            if arm == OPAUX_ARM or arm in PHASE_AUX_ARMS:
                operator_prediction, terms = _operator_complex(model, direction)
                prediction = torch.einsum(
                    "qn,cn->cq", operator_prediction, train_commands.to(operator_prediction)
                ).abs()
            else:
                prediction, terms = model.forward_command_grid_with_terms(train_commands, direction)
            objective = amplitude_objective(prediction.T, target)
            regularization, _ = model.structured_regularization(
                terms,
                residual_energy_weight=config.residual_energy_weight,
                coupling_l2_weight=0.0,
                reciprocity_weight=0.0,
            )
            loss = objective.total + regularization
            complex_loss = torch.zeros_like(loss)
            if arm == OPAUX_ARM:
                complex_loss = complex_operator_nmse(
                    operator_prediction, train_operator[index], train_weights[index]
                )
                loss = loss + LOSS_CONTRACT["complex_operator_nmse"] * complex_loss
            elif arm in PHASE_AUX_ARMS:
                complex_loss = phase_aligned_operator_nmse(
                    operator_prediction, train_operator[index], train_weights[index]
                )
                loss = loss + LOSS_CONTRACT["complex_operator_nmse"] * complex_loss
            if epoch >= config.separation_start_epoch and global_step % config.separation_interval_steps == 0:
                loss = loss + (
                    config.separation_interval_steps * config.separation_weight
                    * separation_loss(model, *sep)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
            total_complex += float(complex_loss.detach()) * len(index)
            train_squared_error += float((prediction.T.detach() - target).double().square().sum())
            train_squared_target += float(target.double().square().sum())
            seen += len(index)
        validation = evaluate_amplitude(
            model, val_direction, val_target, val_commands,
            chunk_size=config.validation_query_chunk,
        )
        train_linear_nmse = train_squared_error / max(train_squared_target, 1e-12)
        train_linear_nmse_db = 10 * math.log10(max(train_linear_nmse, 1e-12))
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / seen,
            "train_linear_nmse": train_linear_nmse,
            "train_linear_nmse_db": train_linear_nmse_db,
            "train_complex_nmse": total_complex / seen,
            "learning_rate": learning_rate,
            "train_validation_amplitude_gap_db": validation["linear_nmse_db"] - train_linear_nmse_db,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if np.isfinite(validation["linear_nmse"]) and validation["linear_nmse"] < best_validation:
            best_validation = validation["linear_nmse"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        state = {
            "schema": "v218-resume-checkpoint-1",
            "arm": arm, "seed": seed, "completed_epoch": epoch,
            "global_step": global_step, "history": history,
            "best_epoch": best_epoch, "best_validation": best_validation,
            "best_state": best_state,
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "blind_test_status": "sealed_not_opened",
        }
        _atomic_torch_save(latest, state)
        if verbose:
            print(json.dumps({"arm": arm, "seed": seed, **row}), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint = {
        "schema": "v218-best-checkpoint-1",
        "arm": arm, "seed": seed,
        "architecture": architecture_contract(model, arm),
        "loss_contract": LOSS_CONTRACT,
        "data_fingerprints_sha256": data.manifest["source_fingerprints_sha256"],
        "envelope_scale": envelope_scale,
        "best_epoch": best_epoch,
        "best_validation_amplitude_linear_nmse": best_validation,
        "model_state": best_state,
        "blind_test_status": "sealed_not_opened",
    }
    _atomic_torch_save(run_dir / "checkpoint.pt", checkpoint)
    data.transform.save(run_dir / "filtration_transform.npz")
    result = {
        "schema": "v218-training-result-1",
        "arm": arm, "seed": seed, "epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_validation_amplitude_linear_nmse": best_validation,
        "best_validation_amplitude_linear_nmse_db": 10 * math.log10(max(best_validation, 1e-12)),
        "trained_from_scratch_or_exact_resume": True,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "blind_test_status": "sealed_not_opened",
    }
    atomic_json(run_dir / "result.json", result)
    atomic_json(run_dir / "history.json", {"history": history})
    return result
