"""Geometry-only angular blocks and complementary-grid maximin sampling."""

from __future__ import annotations

import numpy as np


def load_amplitude_label_package(path) -> dict[str, np.ndarray]:
    """Load an amplitude-only label package and reject audit/complex-field files."""
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        if "not_for_training" in keys or any(key.startswith("g_") or "complex" in key for key in keys):
            raise ValueError("complex/audit packages are forbidden in the training loader")
        required = {"direction", "command_indices", "command_iq", "amplitude"}
        if not required.issubset(keys):
            raise ValueError(f"amplitude label package is missing {sorted(required - keys)}")
        result = {key: np.asarray(payload[key]) for key in payload.files}
    if result["direction"].ndim != 2 or result["direction"].shape[1] != 3:
        raise ValueError("direction must have shape [Q,3]")
    if result["amplitude"].shape != (len(result["command_indices"]), len(result["direction"])):
        raise ValueError("amplitude must have shape [C,Q]")
    if not np.isfinite(result["amplitude"]).all() or np.any(result["amplitude"] < 0):
        raise ValueError("amplitudes must be finite and non-negative")
    return result


def angular_block_ids(theta_deg: np.ndarray, phi_deg: np.ndarray, *, theta_origin_deg: float = 0.5) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if theta.shape != phi.shape or theta.ndim != 1:
        raise ValueError("theta and phi must be matching vectors")
    theta_bin = np.floor((theta - theta_origin_deg) / 10.0).astype(np.int64)
    phi_bin = np.floor(np.mod(phi + 180.0, 360.0) / 10.0).astype(np.int64)
    return np.stack((theta_bin, phi_bin), axis=1)


def block_set(blocks: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in row) for row in np.asarray(blocks)}


def nearest_angle(candidate: np.ndarray, reference: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    result = []
    for start in range(0, len(candidate), chunk_size):
        cosine = candidate[start:start + chunk_size] @ reference.T
        result.append(np.arccos(np.clip(cosine.max(axis=1), -1.0, 1.0)))
    return np.concatenate(result)


def select_per_block_maximin(
    candidate_direction: np.ndarray,
    candidate_blocks: np.ndarray,
    target_blocks: set[tuple[int, int]],
    training_direction: np.ndarray,
    *,
    per_block: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Select labels using geometry only; return selected and same-block probe indices."""
    direction = np.asarray(candidate_direction, dtype=np.float64)
    blocks = np.asarray(candidate_blocks, dtype=np.int64)
    if direction.shape[0] != blocks.shape[0] or direction.shape[1] != 3 or blocks.shape[1] != 2:
        raise ValueError("candidate arrays do not align")
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    base_distance = nearest_angle(direction, training_direction)
    selected: list[int] = []
    probe: list[int] = []
    for block in sorted(target_blocks):
        members = np.flatnonzero(np.all(blocks == np.asarray(block), axis=1))
        if len(members) < per_block + 1:
            raise ValueError(f"block {block} has insufficient candidates")
        remaining = members.copy()
        current_distance = base_distance[members].copy()
        chosen: list[int] = []
        for _ in range(per_block):
            member_lookup = {int(value): index for index, value in enumerate(members)}
            distance = np.asarray([current_distance[member_lookup[int(value)]] for value in remaining])
            local = int(np.argmax(distance))
            pick = int(remaining[local])
            chosen.append(pick)
            selected_angle = np.arccos(np.clip(direction[members] @ direction[pick], -1.0, 1.0))
            current_distance = np.minimum(current_distance, selected_angle)
            remaining = np.delete(remaining, local)
        selected.extend(chosen)
        probe.extend(int(value) for value in remaining)
    return np.asarray(selected, dtype=np.int64), np.asarray(probe, dtype=np.int64)


def deterministic_spatial_folds(theta_deg: np.ndarray, phi_deg: np.ndarray, count: int = 3) -> np.ndarray:
    if count < 2:
        raise ValueError("at least two folds are required")
    blocks = angular_block_ids(theta_deg, phi_deg)
    unique = sorted(block_set(blocks), key=lambda b: ((b[0] * 73856093) ^ (b[1] * 19349663), b))
    assignment = {block: index % count for index, block in enumerate(unique)}
    return np.asarray([assignment[tuple(int(v) for v in row)] for row in blocks], dtype=np.int64)
