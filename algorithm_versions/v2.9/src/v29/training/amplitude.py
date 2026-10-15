"""Amplitude-only datasets, losses, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..contracts import QueryBatch
from ..io import AmplitudeDataset
from ..models import AmplitudeOnlyForwardModel
from .datasets import SplitIndices


class AmplitudeObservationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Cartesian product of selected commands and selected spherical queries."""

    def __init__(
        self,
        dataset: AmplitudeDataset,
        command_indices: np.ndarray,
        query_indices: np.ndarray,
    ) -> None:
        self.direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
        self.command_iq = torch.as_tensor(dataset.command_iq, dtype=torch.float32)
        self.amplitude = torch.as_tensor(dataset.amplitude, dtype=torch.float32)
        self.command_indices = np.asarray(command_indices, dtype=np.int64)
        self.query_indices = np.asarray(query_indices, dtype=np.int64)
        if self.command_indices.ndim != 1 or self.query_indices.ndim != 1:
            raise ValueError("command and query indices must be one-dimensional")
        if self.command_indices.size == 0 or self.query_indices.size == 0:
            raise ValueError("amplitude observation split must not be empty")

    def __len__(self) -> int:
        return int(self.command_indices.size * self.query_indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        command_slot, query_slot = divmod(item, self.query_indices.size)
        command_index = int(self.command_indices[command_slot])
        query_index = int(self.query_indices[query_slot])
        return (
            self.command_iq[command_index],
            self.direction[query_index],
            self.amplitude[command_index, query_index],
        )


class AmplitudeDirectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One direction with every selected single-polarization amplitude command label."""

    def __init__(
        self,
        dataset: AmplitudeDataset,
        command_indices: np.ndarray,
        query_indices: np.ndarray,
    ) -> None:
        self.command_indices = np.asarray(command_indices, dtype=np.int64)
        self.query_indices = np.asarray(query_indices, dtype=np.int64)
        if self.command_indices.ndim != 1 or self.query_indices.ndim != 1:
            raise ValueError("command and query indices must be one-dimensional")
        if self.command_indices.size == 0 or self.query_indices.size == 0:
            raise ValueError("amplitude direction split must not be empty")
        self.direction = torch.as_tensor(
            dataset.direction[self.query_indices], dtype=torch.float32
        )
        self.command_iq = torch.as_tensor(
            dataset.command_iq[self.command_indices], dtype=torch.float32
        )
        selected = dataset.amplitude[np.ix_(self.command_indices, self.query_indices)]
        self.amplitude_by_direction = torch.as_tensor(selected.T.copy(), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.query_indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.direction[item], self.amplitude_by_direction[item]


def stratified_command_split(
    command_family: np.ndarray,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 28001,
) -> SplitIndices:
    """Hold out commands inside every code family, including unseen phase codes."""
    families = np.asarray(command_family)
    if families.ndim != 1:
        raise ValueError("command_family must be one-dimensional")
    generator = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for family in np.unique(families):
        indices = np.flatnonzero(families == family)
        shuffled = generator.permutation(indices)
        test_count = max(1, int(round(indices.size * test_fraction)))
        validation_count = max(1, int(round(indices.size * validation_fraction)))
        if test_count + validation_count >= indices.size:
            raise ValueError(f"command family {family!r} is too small for three-way splitting")
        test_parts.append(shuffled[:test_count])
        validation_parts.append(shuffled[test_count : test_count + validation_count])
        train_parts.append(shuffled[test_count + validation_count :])
    return SplitIndices(
        train=np.sort(np.concatenate(train_parts)),
        validation=np.sort(np.concatenate(validation_parts)),
        test=np.sort(np.concatenate(test_parts)),
    )


@dataclass(frozen=True)
class AmplitudeTrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    db_loss_weight: float = 0.03
    power_loss_weight: float = 0.10
    db_huber_delta: float = 1.0
    gradient_clip_norm: float = 5.0
    scheduler_patience: int = 6
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    residual_energy_weight: float = 1e-4
    coupling_l2_weight: float = 1e-4
    reciprocity_weight: float = 1e-3


@dataclass(frozen=True)
class AmplitudeLoss:
    total: torch.Tensor
    linear_nmse: torch.Tensor
    power_nmse: torch.Tensor
    amplitude_db_mae: torch.Tensor


@dataclass(frozen=True)
class AmplitudeTrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    initial_validation: dict[str, float] | None = None


def balanced_command_family_weights(command_family: np.ndarray) -> torch.Tensor:
    """Give every command family equal total weight while retaining every command."""
    families = np.asarray(command_family)
    if families.ndim != 1 or families.size == 0:
        raise ValueError("command_family must be a non-empty one-dimensional array")
    weights = np.zeros(families.size, dtype=np.float32)
    unique = np.unique(families)
    for family in unique:
        members = families == family
        weights[members] = 1.0 / float(members.sum())
    weights /= weights.sum()
    return torch.as_tensor(weights, dtype=torch.float32)


def _weighted_command_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if values.ndim == 0:
        if weights is not None:
            raise ValueError("command weights require a command-by-query loss tensor")
        return values
    if values.ndim != 1:
        raise ValueError("per-command values must be one-dimensional")
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape or torch.any(weights < 0) or not torch.any(weights > 0):
        raise ValueError("command weights must be non-negative and match the command axis")
    return torch.sum(values * (weights / weights.sum()))


def amplitude_only_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    noise_floor_amplitude: float,
    db_loss_weight: float = 0.03,
    power_loss_weight: float = 0.10,
    db_huber_delta: float = 1.0,
    command_weights: torch.Tensor | None = None,
) -> AmplitudeLoss:
    if prediction.shape != target.shape:
        raise ValueError("predicted and target amplitudes must have identical shapes")
    if torch.any(target < 0):
        raise ValueError("target amplitude must be non-negative")
    if prediction.ndim not in (1, 2):
        raise ValueError("amplitude loss expects [observation] or [command,query] tensors")
    if db_huber_delta <= 0:
        raise ValueError("db_huber_delta must be positive")
    if prediction.ndim == 1:
        scale = target.square().mean().clamp_min(1e-12)
        linear_by_command = (prediction - target).square().mean() / scale
        target_power = target.square()
        prediction_power = prediction.square()
        power_by_command = (
            (prediction_power - target_power).square().mean()
            / target_power.square().mean().clamp_min(1e-12)
        )
    else:
        scale = target.square().mean(dim=1).clamp_min(1e-12)
        linear_by_command = (prediction - target).square().mean(dim=1) / scale
        target_power = target.square()
        prediction_power = prediction.square()
        power_by_command = (
            (prediction_power - target_power).square().mean(dim=1)
            / target_power.square().mean(dim=1).clamp_min(1e-12)
        )
    linear_nmse = _weighted_command_mean(linear_by_command, command_weights)
    power_nmse = _weighted_command_mean(power_by_command, command_weights)
    numerical_floor = target.detach().max().clamp_min(1.0) * 1e-8
    floor = torch.maximum(
        numerical_floor,
        torch.as_tensor(noise_floor_amplitude, device=target.device, dtype=target.dtype),
    )
    valid = target >= floor
    if torch.any(valid):
        prediction_db = 20.0 * torch.log10(prediction.clamp_min(floor))
        target_db = 20.0 * torch.log10(target.clamp_min(floor))
        db_error = prediction_db - target_db
        if prediction.ndim == 1:
            db_mae_by_command = torch.abs(db_error[valid]).mean()
            db_objective_by_command = torch.nn.functional.huber_loss(
                db_error[valid],
                torch.zeros_like(db_error[valid]),
                delta=db_huber_delta,
            )
        else:
            valid_count = valid.sum(dim=1).clamp_min(1)
            db_mae_by_command = (torch.abs(db_error) * valid).sum(dim=1) / valid_count
            quadratic = 0.5 * db_error.square()
            linear = db_huber_delta * (torch.abs(db_error) - 0.5 * db_huber_delta)
            robust = torch.where(torch.abs(db_error) <= db_huber_delta, quadratic, linear)
            db_objective_by_command = (robust * valid).sum(dim=1) / valid_count
        db_mae = _weighted_command_mean(db_mae_by_command, command_weights)
        db_objective = _weighted_command_mean(db_objective_by_command, command_weights)
    else:
        db_mae = torch.zeros_like(linear_nmse)
        db_objective = torch.zeros_like(linear_nmse)
    return AmplitudeLoss(
        total=linear_nmse + power_loss_weight * power_nmse + db_loss_weight * db_objective,
        linear_nmse=linear_nmse,
        power_nmse=power_nmse,
        amplitude_db_mae=db_mae,
    )


def train_amplitude_model(
    model: AmplitudeOnlyForwardModel,
    batches: DataLoader,
    *,
    noise_floor_amplitude: float,
    config: AmplitudeTrainConfig | None = None,
) -> list[dict[str, float]]:
    config = config or AmplitudeTrainConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        total = 0.0
        linear = 0.0
        power = 0.0
        db = 0.0
        count = 0
        for command, direction, target in batches:
            parameter = next(model.parameters())
            command = command.to(device=parameter.device, dtype=parameter.dtype)
            direction = direction.to(device=parameter.device, dtype=parameter.dtype)
            target = target.to(device=parameter.device, dtype=parameter.dtype)
            prediction = model(command, QueryBatch(direction[:, None])).amplitude[:, 0]
            losses = amplitude_only_loss(
                prediction,
                target,
                noise_floor_amplitude=noise_floor_amplitude,
                db_loss_weight=config.db_loss_weight,
                power_loss_weight=config.power_loss_weight,
                db_huber_delta=config.db_huber_delta,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(losses.total.detach())
            linear += float(losses.linear_nmse.detach())
            power += float(losses.power_nmse.detach())
            db += float(losses.amplitude_db_mae.detach())
            count += 1
        if count == 0:
            raise ValueError("training loader yielded no amplitude observations")
        history.append(
            {
                "epoch": float(epoch),
                "loss": total / count,
                "linear_nmse": linear / count,
                "power_nmse": power / count,
                "amplitude_db_mae": db / count,
            }
        )
    return history


@torch.no_grad()
def evaluate_amplitude_grid_model(
    model: AmplitudeOnlyForwardModel,
    batches: DataLoader,
    command_iq: torch.Tensor,
    *,
    noise_floor_amplitude: float,
    command_weights: torch.Tensor | None = None,
    db_loss_weight: float = 0.0,
    power_loss_weight: float = 0.0,
    db_huber_delta: float = 1.0,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    parameter = next(model.parameters())
    commands = command_iq.to(device=parameter.device, dtype=parameter.dtype)
    for direction, target_by_command in batches:
        direction = direction.to(device=parameter.device, dtype=parameter.dtype)
        predictions.append(model.forward_command_grid(commands, direction).cpu())
        targets.append(target_by_command.T.contiguous().cpu())
    if not predictions:
        raise ValueError("evaluation loader yielded no amplitude directions")
    prediction = torch.cat(predictions, dim=1)
    target = torch.cat(targets, dim=1)
    weights = command_weights.cpu() if command_weights is not None else None
    losses = amplitude_only_loss(
        prediction,
        target,
        noise_floor_amplitude=noise_floor_amplitude,
        db_loss_weight=db_loss_weight,
        power_loss_weight=power_loss_weight,
        db_huber_delta=db_huber_delta,
        command_weights=weights,
    )
    absolute_error = torch.abs(prediction - target)
    global_linear_nmse = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-12)
    prediction_power = prediction.square()
    target_power = target.square()
    global_power_nmse = (
        (prediction_power - target_power).square().mean()
        / target_power.square().mean().clamp_min(1e-12)
    )
    prediction_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    correlation = (prediction_centered * target_centered).sum(dim=1) / (
        torch.linalg.vector_norm(prediction_centered, dim=1)
        * torch.linalg.vector_norm(target_centered, dim=1)
    ).clamp_min(1e-12)
    return {
        "loss": float(losses.total),
        "linear_nmse": float(losses.linear_nmse),
        "linear_nmse_db": float(10.0 * torch.log10(losses.linear_nmse.clamp_min(1e-20))),
        "global_linear_nmse": float(global_linear_nmse),
        "global_linear_nmse_db": float(10.0 * torch.log10(global_linear_nmse.clamp_min(1e-20))),
        "power_nmse": float(losses.power_nmse),
        "power_nmse_db": float(10.0 * torch.log10(losses.power_nmse.clamp_min(1e-20))),
        "global_power_nmse": float(global_power_nmse),
        "global_power_nmse_db": float(10.0 * torch.log10(global_power_nmse.clamp_min(1e-20))),
        "amplitude_db_mae": float(losses.amplitude_db_mae),
        "mean_command_amplitude_correlation": float(correlation.mean()),
        "absolute_error_p50": float(torch.quantile(absolute_error, 0.50)),
        "absolute_error_p95": float(torch.quantile(absolute_error, 0.95)),
    }


def train_grouped_amplitude_model(
    model: AmplitudeOnlyForwardModel,
    training_batches: DataLoader,
    training_command_iq: torch.Tensor,
    validation_batches: DataLoader,
    validation_command_iq: torch.Tensor,
    *,
    noise_floor_amplitude: float,
    training_command_weights: torch.Tensor | None = None,
    validation_command_weights: torch.Tensor | None = None,
    config: AmplitudeTrainConfig | None = None,
    select_initial_state: bool = False,
) -> AmplitudeTrainingResult:
    """Train one latent channel per direction against many known phase-coded amplitudes."""
    config = config or AmplitudeTrainConfig()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("at least one model parameter must remain trainable")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    parameter = next(model.parameters())
    training_commands = training_command_iq.to(device=parameter.device, dtype=parameter.dtype)
    training_weights = (
        training_command_weights.to(device=parameter.device, dtype=parameter.dtype)
        if training_command_weights is not None
        else None
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    initial_validation: dict[str, float] | None = None

    if select_initial_state:
        initial_validation = evaluate_amplitude_grid_model(
            model,
            validation_batches,
            validation_command_iq,
            noise_floor_amplitude=noise_floor_amplitude,
            command_weights=validation_command_weights,
            db_loss_weight=config.db_loss_weight,
            power_loss_weight=config.power_loss_weight,
            db_huber_delta=config.db_huber_delta,
        )
        best_validation_loss = initial_validation["loss"]
        best_epoch = 0
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        print(
            "epoch 000 "
            f"val_nmse_db={initial_validation['linear_nmse_db']:.3f} "
            f"val_db_mae={initial_validation['amplitude_db_mae']:.3f} "
            "source=initial_checkpoint"
        )

    for epoch in range(config.epochs):
        model.train()
        running = {
            "loss": 0.0,
            "data_loss": 0.0,
            "regularization": 0.0,
            "linear_nmse": 0.0,
            "power_nmse": 0.0,
            "amplitude_db_mae": 0.0,
        }
        query_count = 0
        for direction, target_by_command in training_batches:
            direction = direction.to(device=parameter.device, dtype=parameter.dtype)
            target = target_by_command.T.contiguous().to(device=parameter.device, dtype=parameter.dtype)
            prediction, terms = model.forward_command_grid_with_terms(
                training_commands,
                direction,
            )
            losses = amplitude_only_loss(
                prediction,
                target,
                noise_floor_amplitude=noise_floor_amplitude,
                db_loss_weight=config.db_loss_weight,
                power_loss_weight=config.power_loss_weight,
                db_huber_delta=config.db_huber_delta,
                command_weights=training_weights,
            )
            regularization, _ = model.structured_regularization(
                terms,
                residual_energy_weight=config.residual_energy_weight,
                coupling_l2_weight=config.coupling_l2_weight,
                reciprocity_weight=config.reciprocity_weight,
            )
            optimization_loss = losses.total + regularization
            optimizer.zero_grad(set_to_none=True)
            optimization_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip_norm)
            optimizer.step()
            batch_queries = int(direction.shape[0])
            running["loss"] += float(optimization_loss.detach()) * batch_queries
            running["data_loss"] += float(losses.total.detach()) * batch_queries
            running["regularization"] += float(regularization.detach()) * batch_queries
            running["linear_nmse"] += float(losses.linear_nmse.detach()) * batch_queries
            running["power_nmse"] += float(losses.power_nmse.detach()) * batch_queries
            running["amplitude_db_mae"] += float(losses.amplitude_db_mae.detach()) * batch_queries
            query_count += batch_queries
        if query_count == 0:
            raise ValueError("training loader yielded no amplitude directions")
        validation = evaluate_amplitude_grid_model(
            model,
            validation_batches,
            validation_command_iq,
            noise_floor_amplitude=noise_floor_amplitude,
            command_weights=validation_command_weights,
            db_loss_weight=config.db_loss_weight,
            power_loss_weight=config.power_loss_weight,
            db_huber_delta=config.db_huber_delta,
        )
        scheduler.step(validation["loss"])
        row = {
            "epoch": float(epoch + 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{name}": value / query_count for name, value in running.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d} "
            f"train_nmse={row['train_linear_nmse']:.5f} "
            f"val_nmse_db={row['validation_linear_nmse_db']:.3f} "
            f"val_db_mae={row['validation_amplitude_db_mae']:.3f} "
            f"lr={row['learning_rate']:.2e}"
        )
        if validation["loss"] < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation["loss"]
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    return AmplitudeTrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=len(history) < config.epochs,
        initial_validation=initial_validation,
    )


@torch.no_grad()
def evaluate_amplitude_model(
    model: AmplitudeOnlyForwardModel,
    batches: DataLoader,
    *,
    noise_floor_amplitude: float,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    for command, direction, target in batches:
        parameter = next(model.parameters())
        command = command.to(device=parameter.device, dtype=parameter.dtype)
        direction = direction.to(device=parameter.device, dtype=parameter.dtype)
        predictions.append(model(command, QueryBatch(direction[:, None])).amplitude[:, 0].cpu())
        targets.append(target.cpu())
    if not predictions:
        raise ValueError("evaluation loader yielded no amplitude observations")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    losses = amplitude_only_loss(
        prediction,
        target,
        noise_floor_amplitude=noise_floor_amplitude,
        db_loss_weight=0.0,
        power_loss_weight=0.0,
    )
    absolute_error = torch.abs(prediction - target)
    return {
        "linear_nmse": float(losses.linear_nmse),
        "linear_nmse_db": float(10.0 * torch.log10(losses.linear_nmse.clamp_min(1e-20))),
        "power_nmse": float(losses.power_nmse),
        "power_nmse_db": float(10.0 * torch.log10(losses.power_nmse.clamp_min(1e-20))),
        "amplitude_db_mae": float(losses.amplitude_db_mae),
        "absolute_error_p50": float(torch.quantile(absolute_error, 0.50)),
        "absolute_error_p95": float(torch.quantile(absolute_error, 0.95)),
    }
