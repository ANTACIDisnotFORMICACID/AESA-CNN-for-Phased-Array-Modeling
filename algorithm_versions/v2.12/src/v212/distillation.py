"""Gauge-invariant lifted-Q targets and G3 supervision."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class OperatorTarget:
    q_hat: torch.Tensor
    confidence: torch.Tensor
    inversion_residual: torch.Tensor
    minimum_jacobian_singular_value: torch.Tensor
    query_indices: torch.Tensor
    command_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not torch.is_complex(self.q_hat) or self.q_hat.ndim != 3:
            raise ValueError("q_hat must be complex with shape [Q,16,16]")
        if self.q_hat.shape[-2:] != (16, 16):
            raise ValueError("q_hat must have 16x16 matrices")
        count = self.q_hat.shape[0]
        for value in (
            self.confidence,
            self.inversion_residual,
            self.minimum_jacobian_singular_value,
        ):
            if value.shape != (count,):
                raise ValueError("OperatorTarget diagnostics must have shape [Q]")
        if self.query_indices.shape != (count,):
            raise ValueError("query_indices must have shape [Q]")
        if self.command_indices.ndim != 1 or self.command_indices.numel() == 0:
            raise ValueError("command_indices must be a non-empty vector")
        if len(torch.unique(self.query_indices)) != count:
            raise ValueError("query_indices must be unique")
        if len(torch.unique(self.command_indices)) != len(self.command_indices):
            raise ValueError("command_indices must be unique")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "v212-operator-target-1",
            "inversion_uses_train_commands_only": True,
        }
        np.savez(
            target,
            q_hat_real=self.q_hat.real.detach().cpu().numpy(),
            q_hat_imag=self.q_hat.imag.detach().cpu().numpy(),
            confidence=self.confidence.detach().cpu().numpy(),
            inversion_residual=self.inversion_residual.detach().cpu().numpy(),
            minimum_jacobian_singular_value=(
                self.minimum_jacobian_singular_value.detach().cpu().numpy()
            ),
            query_indices=self.query_indices.detach().cpu().numpy(),
            command_indices=self.command_indices.detach().cpu().numpy(),
            metadata=np.asarray(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OperatorTarget":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            if metadata.get("schema") != "v212-operator-target-1":
                raise ValueError("unsupported operator target schema")
            q_hat = torch.complex(
                torch.from_numpy(np.asarray(archive["q_hat_real"], dtype=np.float32).copy()),
                torch.from_numpy(np.asarray(archive["q_hat_imag"], dtype=np.float32).copy()),
            )
            return cls(
                q_hat=q_hat,
                confidence=torch.from_numpy(
                    np.asarray(archive["confidence"], dtype=np.float32).copy()
                ),
                inversion_residual=torch.from_numpy(
                    np.asarray(archive["inversion_residual"], dtype=np.float32).copy()
                ),
                minimum_jacobian_singular_value=torch.from_numpy(
                    np.asarray(
                        archive["minimum_jacobian_singular_value"], dtype=np.float32
                    ).copy()
                ),
                query_indices=torch.from_numpy(
                    np.asarray(archive["query_indices"], dtype=np.int64).copy()
                ),
                command_indices=torch.from_numpy(
                    np.asarray(archive["command_indices"], dtype=np.int64).copy()
                ),
            )


def confidence_from_fisher(
    fisher: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    minimum: float = 0.1,
    maximum: float = 10.0,
) -> torch.Tensor:
    if fisher.ndim != 3 or fisher.shape[-1] != fisher.shape[-2]:
        raise ValueError("fisher must have shape [Q,D,D]")
    identity = torch.eye(fisher.shape[-1], device=fisher.device, dtype=fisher.dtype)
    covariance = torch.linalg.pinv(fisher + epsilon * identity)
    raw = covariance.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(epsilon).reciprocal()
    normalized = raw / raw.median().clamp_min(epsilon)
    return normalized.clamp(minimum, maximum)


def operator_distillation_loss(
    prediction_q: torch.Tensor,
    target_q: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    if prediction_q.shape != target_q.shape or not torch.is_complex(prediction_q):
        raise ValueError("prediction_q and target_q must be matching complex tensors")
    if prediction_q.ndim != 3 or prediction_q.shape[-2:] != (16, 16):
        raise ValueError("operator tensors must have shape [Q,16,16]")
    error = (prediction_q - target_q).abs().square().sum(dim=(-2, -1))
    scale = target_q.abs().square().sum(dim=(-2, -1)).clamp_min(epsilon)
    normalized = error / scale
    if confidence is None:
        return normalized.mean()
    if confidence.shape != normalized.shape:
        raise ValueError("confidence must have shape [Q]")
    return (confidence * normalized).sum() / confidence.sum().clamp_min(epsilon)
