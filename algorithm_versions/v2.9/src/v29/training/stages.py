"""Deterministic staged-training plan for the V2.9 physical hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import V29AmplitudeForwardModel


@dataclass(frozen=True)
class TrainingStage:
    name: str
    epochs: int
    learning_rate_scale: float


def default_stage_plan(total_epochs: int) -> tuple[TrainingStage, ...]:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if total_epochs < 4:
        return (TrainingStage("joint", total_epochs, 1.0),)
    fractions = (0.40, 0.20, 0.30, 0.10)
    targets = tuple(total_epochs * fraction for fraction in fractions)
    allocation = [1, 1, 1, 1]
    for _ in range(total_epochs - 4):
        index = max(range(4), key=lambda slot: targets[slot] - allocation[slot])
        allocation[index] += 1
    nominal, coupling, residual, joint = allocation
    stages = (
        TrainingStage("nominal", nominal, 1.0),
        TrainingStage("coupling", coupling, 0.5),
        TrainingStage("residual", residual, 0.5),
        TrainingStage("joint", joint, 0.2),
    )
    if sum(stage.epochs for stage in stages) != total_epochs:
        raise AssertionError("stage allocation must preserve total epochs")
    return stages


def configure_trainable_stage(
    model: V29AmplitudeForwardModel,
    stage_name: str,
) -> tuple[str, ...]:
    valid = {"nominal", "coupling", "residual", "joint"}
    if stage_name not in valid:
        raise ValueError(f"unknown V2.9 training stage: {stage_name}")
    enabled_modules = {
        "nominal": {"nominal_pattern"},
        "coupling": {"port_coupling"},
        "residual": {"angular_residual"},
        "joint": {"nominal_pattern", "port_coupling", "angular_residual"},
    }[stage_name]
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        module_name = name.split(".", 1)[0]
        parameter.requires_grad_(module_name in enabled_modules)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError(f"stage {stage_name!r} leaves no trainable parameters")
    return tuple(trainable)
