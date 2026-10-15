"""Verify candidate commands directly against canonical HFSS basis responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..complex_ops import contract_operator_iq, field_power, validate_command_iq
from ..io import BasisDataset


@dataclass(frozen=True)
class BasisVerification:
    grid_indices: np.ndarray
    angular_error_deg: np.ndarray
    field_iq: torch.Tensor
    target_power: torch.Tensor


def nearest_basis_indices(
    target_direction: np.ndarray,
    basis_direction: np.ndarray,
    *,
    max_angular_error_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_direction, dtype=np.float64)
    basis = np.asarray(basis_direction, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError("target and basis directions must have shape [Q,3]")
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    cosine = basis @ target.T
    indices = np.argmax(cosine, axis=0)
    best_cosine = cosine[indices, np.arange(target.shape[0])]
    error_deg = np.rad2deg(np.arccos(np.clip(best_cosine, -1.0, 1.0)))
    if np.any(error_deg > max_angular_error_deg):
        raise ValueError(
            "target direction is not represented closely enough by the FFD grid; "
            f"worst angular error={error_deg.max():.6g} deg, limit={max_angular_error_deg:.6g} deg"
        )
    return indices, error_deg


def verify_command_on_basis(
    command_iq: torch.Tensor,
    target_direction: np.ndarray,
    basis: BasisDataset,
    *,
    range_m: torch.Tensor | None = None,
    max_angular_error_deg: float = 0.51,
) -> BasisVerification:
    """Use FFD values directly; the learned model is not called."""
    validate_command_iq(command_iq)
    if command_iq.shape[0] != 1:
        raise ValueError("verification expects exactly one candidate command")
    indices, angular_error = nearest_basis_indices(
        target_direction,
        basis.direction,
        max_angular_error_deg=max_angular_error_deg,
    )
    selected = torch.as_tensor(
        basis.operator_iq[indices],
        device=command_iq.device,
        dtype=command_iq.dtype,
    )[None]
    field = contract_operator_iq(selected, command_iq)
    power = field_power(field, range_m)
    return BasisVerification(indices, angular_error, field, power)

