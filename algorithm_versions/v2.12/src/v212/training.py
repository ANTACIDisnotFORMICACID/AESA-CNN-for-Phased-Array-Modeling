"""Amplitude-only G1/G2/G3 objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from v210.contracts import QueryBatch


@dataclass(frozen=True)
class AmplitudeObjective:
    total: torch.Tensor
    linear_nmse: torch.Tensor
    power_nmse: torch.Tensor
    db_huber: torch.Tensor
    operator_distillation: torch.Tensor


@dataclass(frozen=True)
class OperatorTrainConfig:
    epochs: int = 120
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    residual_energy_weight: float = 1e-4
    lambda_q: float = 0.0
    gradient_clip_norm: float = 5.0
    seed: int = 21212
    training_schedule: str = "joint"
    learning_rate_drop_epoch: int | None = None
    late_learning_rate: float = 3e-4
    ssh_phase_end_epoch: int = 60
    fourier_phase_end_epoch: int = 180
    lambda_separation: float = 0.0
    separation_start_epoch: int = 61
    separation_interval_steps: int = 8


@dataclass(frozen=True)
class OperatorTrainResult:
    history: list[dict[str, float | str]]
    best_epoch: int
    best_validation_nmse_db: float


def _training_phase(config: OperatorTrainConfig, epoch: int) -> str:
    if config.training_schedule == "joint":
        return "joint"
    if config.training_schedule != "ssh_fourier_joint":
        raise ValueError("training_schedule must be joint or ssh_fourier_joint")
    if epoch <= config.ssh_phase_end_epoch:
        return "ssh"
    if epoch <= config.fourier_phase_end_epoch:
        return "fourier"
    return "joint"


def _set_phase_trainability(model: torch.nn.Module, phase: str) -> None:
    residual = getattr(model, "residual", None)
    residual_ids = {id(parameter) for parameter in residual.parameters()} if residual is not None else set()
    for parameter in model.parameters():
        is_residual = id(parameter) in residual_ids
        parameter.requires_grad_(
            (phase == "joint") or (phase == "fourier" and is_residual) or (phase == "ssh" and not is_residual)
        )


def _fixed_separation_loss(
    model: torch.nn.Module,
    direction: torch.Tensor,
    weights: torch.Tensor,
    basis: torch.Tensor,
    analysis: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    residual_module = getattr(model, "residual", None)
    if residual_module is None:
        return torch.zeros((), dtype=direction.dtype, device=direction.device)
    residual_iq = residual_module(direction[None])[0]
    residual = torch.complex(residual_iq[:, 0], residual_iq[:, 1]).reshape(len(direction), -1)
    scale = getattr(model, "envelope_scale", torch.ones((), device=direction.device)).to(residual)
    residual = scale * residual
    coefficients = analysis.to(residual) @ residual
    projection = basis.to(residual) @ coefficients
    numerator = (weights[:, None] * projection.abs().square()).sum()
    denominator = (weights[:, None] * residual.abs().square()).sum().detach().clamp_min(epsilon)
    return numerator / denominator


def _nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)


def amplitude_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    predicted_q: torch.Tensor | None = None,
    target_q: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    lambda_q: float = 0.0,
    db_floor: float = 1e-6,
) -> AmplitudeObjective:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target amplitude shapes must match")
    if torch.any(target < 0) or lambda_q < 0:
        raise ValueError("amplitudes and lambda_q must be non-negative")
    linear = _nmse(prediction, target)
    power = _nmse(prediction.square(), target.square())
    prediction_db = 20.0 * torch.log10(prediction.clamp_min(db_floor))
    target_db = 20.0 * torch.log10(target.clamp_min(db_floor))
    db_huber = functional.smooth_l1_loss(prediction_db, target_db)
    if lambda_q:
        if predicted_q is None or target_q is None:
            raise ValueError("operator targets are required when lambda_q is non-zero")
        from .distillation import operator_distillation_loss

        q_loss = operator_distillation_loss(predicted_q, target_q, confidence)
    else:
        q_loss = torch.zeros_like(linear)
    total = linear + 0.1 * power + 0.03 * db_huber + lambda_q * q_loss
    return AmplitudeObjective(total, linear, power, db_huber, q_loss)


@torch.no_grad()
def _evaluate_grid(
    model: torch.nn.Module,
    direction: torch.Tensor,
    target: torch.Tensor,
    commands: torch.Tensor,
    db_floor: float,
) -> dict[str, float]:
    model.eval()
    prediction = model.forward_command_grid(commands, direction).T
    objective = amplitude_objective(prediction, target, db_floor=db_floor)
    return {
        "loss": float(objective.total),
        "linear_nmse": float(objective.linear_nmse),
        "linear_nmse_db": float(10.0 * torch.log10(objective.linear_nmse.clamp_min(1e-12))),
        "power_nmse": float(objective.power_nmse),
        "db_huber": float(objective.db_huber),
    }


def train_operator_model(
    model: torch.nn.Module,
    train_direction: torch.Tensor,
    train_target: torch.Tensor,
    train_commands: torch.Tensor,
    validation_direction: torch.Tensor,
    validation_target: torch.Tensor,
    validation_commands: torch.Tensor,
    *,
    operator_target: torch.Tensor | None = None,
    operator_confidence: torch.Tensor | None = None,
    separation_direction: torch.Tensor | None = None,
    separation_weights: torch.Tensor | None = None,
    separation_basis: torch.Tensor | None = None,
    db_floor: float = 1e-6,
    config: OperatorTrainConfig | None = None,
) -> OperatorTrainResult:
    """Train G1/G2/G3 from the A cell and select only on validation D."""
    config = config or OperatorTrainConfig()
    if config.lambda_separation < 0 or config.separation_interval_steps <= 0:
        raise ValueError("separation settings are invalid")
    if config.lambda_separation and any(
        value is None for value in (separation_direction, separation_weights, separation_basis)
    ):
        raise ValueError("fixed q_sep directions, weights and basis are required")
    if config.lambda_q and operator_target is None:
        raise ValueError("G3 requires operator_target from train commands only")
    if train_target.shape != (len(train_direction), len(train_commands)):
        raise ValueError("train_target must have shape [train_direction,train_command]")
    if validation_target.shape != (len(validation_direction), len(validation_commands)):
        raise ValueError("validation_target must have shape [validation_direction,validation_command]")
    if operator_target is not None and operator_target.shape != (len(train_direction), 16, 16):
        raise ValueError("operator_target must have shape [train_direction,16,16]")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    train_direction = train_direction.to(device=device, dtype=dtype)
    validation_direction = validation_direction.to(device=device, dtype=dtype)
    train_target = train_target.to(device=device, dtype=dtype)
    validation_target = validation_target.to(device=device, dtype=dtype)
    train_commands = train_commands.to(device=device)
    validation_commands = validation_commands.to(device=device)
    if operator_target is not None:
        operator_target = operator_target.to(device=device)
    if operator_confidence is not None:
        operator_confidence = operator_confidence.to(device=device, dtype=dtype)
    separation_analysis = None
    if config.lambda_separation:
        separation_direction = separation_direction.to(device=device, dtype=dtype)
        separation_weights = separation_weights.to(device=device, dtype=dtype)
        separation_weights = separation_weights / separation_weights.sum().clamp_min(1e-12)
        separation_basis = separation_basis.to(device=device, dtype=dtype)
        gram = separation_basis.T @ (separation_weights[:, None] * separation_basis)
        rhs = separation_basis.T * separation_weights[None]
        factor, info = torch.linalg.cholesky_ex(gram)
        separation_analysis = (
            torch.cholesky_solve(rhs, factor)
            if int(info.max()) == 0 else torch.linalg.lstsq(gram, rhs).solution
        )
    optimizer = torch.optim.AdamW(
        [item for item in model.parameters() if item.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = float("inf")
    history: list[dict[str, float | str]] = []
    global_step = 0
    for epoch in range(config.epochs):
        epoch_number = epoch + 1
        phase = _training_phase(config, epoch_number)
        _set_phase_trainability(model, phase)
        learning_rate = (
            config.late_learning_rate
            if config.learning_rate_drop_epoch is not None
            and epoch_number >= config.learning_rate_drop_epoch
            else config.learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        permutation = torch.randperm(len(train_direction), generator=generator)
        total_loss = 0.0
        seen = 0
        separation_total = 0.0
        separation_updates = 0
        for start in range(0, len(permutation), config.batch_size):
            global_step += 1
            index = permutation[start : start + config.batch_size].to(device)
            direction = train_direction[index]
            target = train_target[index]
            prediction, terms = model.forward_command_grid_with_terms(train_commands, direction)
            prediction = prediction.T
            predicted_q = None
            target_q = None
            confidence = None
            if config.lambda_q:
                predicted_q = model.q_matrix(QueryBatch(direction[None]))[0]
                target_q = operator_target[index]
                confidence = None if operator_confidence is None else operator_confidence[index]
            objective = amplitude_objective(
                prediction,
                target,
                predicted_q=predicted_q,
                target_q=target_q,
                confidence=confidence,
                lambda_q=config.lambda_q,
                db_floor=db_floor,
            )
            regularization, _ = model.structured_regularization(
                terms,
                residual_energy_weight=config.residual_energy_weight,
                coupling_l2_weight=0.0,
                reciprocity_weight=0.0,
            )
            loss = objective.total + regularization
            separation = torch.zeros_like(loss)
            if (
                config.lambda_separation
                and epoch_number >= config.separation_start_epoch
                and global_step % config.separation_interval_steps == 0
            ):
                separation = _fixed_separation_loss(
                    model,
                    separation_direction,
                    separation_weights,
                    separation_basis,
                    separation_analysis,
                )
                loss = loss + (
                    config.separation_interval_steps
                    * config.lambda_separation
                    * separation
                )
                separation_total += float(separation.detach())
                separation_updates += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
            seen += len(index)
        validation = _evaluate_grid(
            model,
            validation_direction,
            validation_target,
            validation_commands,
            db_floor,
        )
        row = {
            "epoch": float(epoch_number),
            "train_loss": total_loss / max(seen, 1),
            "learning_rate": float(learning_rate),
            "training_phase": phase,
            "separation_loss": separation_total / max(separation_updates, 1),
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        if validation["linear_nmse"] < best_validation:
            best_validation = validation["linear_nmse"]
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    return OperatorTrainResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_nmse_db=float(
            10.0 * torch.log10(torch.tensor(best_validation).clamp_min(1e-12))
        ),
    )
