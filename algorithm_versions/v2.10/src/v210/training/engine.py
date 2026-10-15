"""Small, explicit training loop for the forward operator model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..contracts import QueryBatch
from ..models import PhysicsStructuredForwardModel
from .losses import operator_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    random_commands_per_direction: int = 4
    field_loss_weight: float = 0.25
    gradient_clip_norm: float = 5.0


def train_operator_model(
    model: PhysicsStructuredForwardModel,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig | None = None,
) -> list[dict[str, float]]:
    """Train from measured/HFSS basis operators; no optimizer-oracle targets."""
    config = config or TrainConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        epoch_total = 0.0
        epoch_batches = 0
        for direction, target_operator in batches:
            parameter = next(model.parameters())
            direction = direction.to(device=parameter.device, dtype=parameter.dtype)[:, None, :]
            target_operator = target_operator.to(device=parameter.device, dtype=parameter.dtype)[:, None]
            query = QueryBatch(direction)
            prediction = model.operator(query).effective_iq
            losses = operator_loss(
                prediction,
                target_operator,
                random_commands=config.random_commands_per_direction,
                field_weight=config.field_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_total += float(losses.total.detach())
            epoch_batches += 1
        if epoch_batches == 0:
            raise ValueError("training iterator yielded no batches")
        history.append({"epoch": float(epoch), "loss": epoch_total / epoch_batches})
    return history

