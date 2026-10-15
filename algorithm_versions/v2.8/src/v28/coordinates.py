"""Explicit conversions between HFSS spherical and algorithm Cartesian frames."""

from __future__ import annotations

import numpy as np
import torch

from .contracts import ArrayGeometry, CoordinateType, QueryBatch


def hfss_angles_to_direction(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """HFSS theta/phi to +X-boresight algorithm direction [..,3]."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    # Standard HFSS Cartesian order is X,Y,Z; theta is measured from +Z.
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )


def direction_to_hfss_angles(unit_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(unit_direction, dtype=np.float64)
    if direction.shape[-1] != 3:
        raise ValueError("unit_direction must end with xyz")
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    theta = np.rad2deg(np.arccos(np.clip(direction[..., 2], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(direction[..., 1], direction[..., 0]))
    return theta, phi


def spherical_basis(theta_deg: np.ndarray, phi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    e_theta = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        axis=-1,
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    return e_theta, e_phi


def query_from_direction(unit_direction: torch.Tensor, range_m: torch.Tensor | None = None) -> QueryBatch:
    unit_direction = unit_direction / torch.linalg.vector_norm(unit_direction, dim=-1, keepdim=True)
    return QueryBatch(unit_direction=unit_direction, range_m=range_m)


def query_from_cartesian_targets(
    target_xyz_m: torch.Tensor,
    geometry: ArrayGeometry,
    *,
    enforce_far_field: bool = True,
) -> QueryBatch:
    if target_xyz_m.ndim != 3 or target_xyz_m.shape[-1] != 3:
        raise ValueError("target_xyz_m must have shape [B,Q,3]")
    range_m = torch.linalg.vector_norm(target_xyz_m, dim=-1)
    if torch.any(range_m <= 0):
        raise ValueError("targets must not coincide with the array origin")
    if enforce_far_field and torch.any(range_m < geometry.fraunhofer_distance_m):
        minimum = float(range_m.min().detach().cpu())
        raise ValueError(
            f"far-field model received range {minimum:.4g} m below Fraunhofer "
            f"distance {geometry.fraunhofer_distance_m:.4g} m"
        )
    return QueryBatch(
        unit_direction=target_xyz_m / range_m[..., None],
        range_m=range_m,
        target_xyz_m=target_xyz_m,
        coordinate_type=CoordinateType.CARTESIAN_TARGET,
    )

