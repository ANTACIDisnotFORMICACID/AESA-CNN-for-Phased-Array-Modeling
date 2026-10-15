"""Leakage-aware datasets, losses and a minimal operator-training engine."""

from .datasets import OperatorDataset, SplitIndices, angular_block_split
from .engine import TrainConfig, train_operator_model
from .losses import OperatorLoss, operator_loss
from .metrics import evaluate_operator_model, operator_metrics
from .amplitude import (
    AmplitudeDirectionDataset,
    AmplitudeLoss,
    AmplitudeObservationDataset,
    AmplitudeTrainConfig,
    AmplitudeTrainingResult,
    amplitude_only_loss,
    balanced_command_family_weights,
    evaluate_amplitude_grid_model,
    evaluate_amplitude_model,
    stratified_command_split,
    train_grouped_amplitude_model,
    train_amplitude_model,
)

__all__ = [
    "OperatorDataset",
    "OperatorLoss",
    "SplitIndices",
    "TrainConfig",
    "angular_block_split",
    "operator_loss",
    "operator_metrics",
    "evaluate_operator_model",
    "train_operator_model",
    "AmplitudeLoss",
    "AmplitudeDirectionDataset",
    "AmplitudeObservationDataset",
    "AmplitudeTrainConfig",
    "AmplitudeTrainingResult",
    "amplitude_only_loss",
    "balanced_command_family_weights",
    "evaluate_amplitude_grid_model",
    "evaluate_amplitude_model",
    "stratified_command_split",
    "train_grouped_amplitude_model",
    "train_amplitude_model",
]
