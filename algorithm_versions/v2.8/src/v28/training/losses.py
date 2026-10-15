"""Losses defined on physical operators and their command-space consequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..complex_ops import contract_operator_iq


@dataclass(frozen=True)
class OperatorLoss:
    total: torch.Tensor
    normalized_operator_mse: torch.Tensor
    random_command_field_mse: torch.Tensor


def _normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square().sum(dim=tuple(range(2, prediction.ndim)))
    scale = target.square().sum(dim=tuple(range(2, target.ndim))).clamp_min(1e-12)
    return (error / scale).mean()


def operator_loss(
    prediction_iq: torch.Tensor,
    target_iq: torch.Tensor,
    *,
    random_commands: int = 4,
    field_weight: float = 0.25,
    generator: torch.Generator | None = None,
) -> OperatorLoss:
    """Fit basis fields and random superpositions without learned oracle labels."""
    if prediction_iq.shape != target_iq.shape:
        raise ValueError("prediction and target operator shapes must match")
    operator_nmse = _normalized_mse(prediction_iq, target_iq)
    if random_commands <= 0:
        field_nmse = torch.zeros_like(operator_nmse)
    else:
        batch = prediction_iq.shape[0]
        command = torch.randn(
            batch * random_commands,
            2,
            4,
            4,
            device=prediction_iq.device,
            dtype=prediction_iq.dtype,
            generator=generator,
        )
        norm = command.square().sum(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        command = command / norm
        predicted = prediction_iq.repeat_interleave(random_commands, dim=0)
        target = target_iq.repeat_interleave(random_commands, dim=0)
        predicted_field = contract_operator_iq(predicted, command)
        target_field = contract_operator_iq(target, command)
        field_nmse = _normalized_mse(predicted_field, target_field)
    return OperatorLoss(
        total=operator_nmse + field_weight * field_nmse,
        normalized_operator_mse=operator_nmse,
        random_command_field_mse=field_nmse,
    )

