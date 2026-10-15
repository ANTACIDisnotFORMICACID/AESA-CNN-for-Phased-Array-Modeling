from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


LIGHT_SPEED_M_S = 299_792_458.0


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


def nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum(weights[:, None] * np.abs(reference - prediction) ** 2)
    denominator = np.sum(weights[:, None] * np.abs(reference) ** 2)
    ratio = float(numerator / max(float(denominator), 1e-300))
    return float(10.0 * np.log10(max(ratio, 1e-300)))


def q_nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(reference), 2048):
        stop = min(start + 2048, len(reference))
        ref = reference[start:stop]
        pred = prediction[start:stop]
        q_ref = np.einsum("qi,qj->qij", np.conj(ref), ref)
        q_pred = np.einsum("qi,qj->qij", np.conj(pred), pred)
        local_weights = weights[start:stop, None, None]
        numerator += float(np.sum(local_weights * np.abs(q_ref - q_pred) ** 2))
        denominator += float(np.sum(local_weights * np.abs(q_ref) ** 2))
    ratio = numerator / max(denominator, 1e-300)
    return float(10.0 * np.log10(max(ratio, 1e-300)))


def load_train_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "canonical_indices", "direction", "g_ephi", "g_etheta", "envelope_ephi",
        "solid_angle_weights", "positions_m", "active_mask", "frequency_hz",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    if "blind" in path.name.lower() or "validation" in path.name.lower():
        raise ValueError(f"Only a train cache is permitted: {path}")
    if not all(np.isfinite(data[name]).all() for name in required if name != "active_mask"):
        raise ValueError(f"Non-finite values in {path}")
    return data


def weighted_residual_svd(
    residual: np.ndarray, weights: np.ndarray
) -> tuple[list[float], int]:
    weighted = residual * np.sqrt(weights[:, None])
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    energy = np.square(singular_values)
    fractions = energy / max(float(np.sum(energy)), 1e-300)
    cumulative = np.cumsum(fractions)
    rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return [float(value) for value in cumulative], rank_95


def analyse_array(array_id: str, data: dict[str, np.ndarray]) -> dict[str, object]:
    g = np.asarray(data["g_ephi"], dtype=np.complex128)
    envelope = np.asarray(data["envelope_ephi"], dtype=np.complex128)
    weights = np.asarray(data["solid_angle_weights"], dtype=np.float64)
    positions = np.asarray(data["positions_m"], dtype=np.float64)
    direction = np.asarray(data["direction"], dtype=np.float64)
    frequency = float(data["frequency_hz"])
    wavelength = LIGHT_SPEED_M_S / frequency
    k = 2.0 * math.pi / wavelength
    phase = np.exp(1j * k * np.einsum("qk,nk->qn", direction, positions))
    common = np.mean(envelope, axis=1, keepdims=True)
    m1 = np.repeat(common, envelope.shape[1], axis=1) * phase
    residual = envelope - common
    residual_fraction = float(
        np.sum(weights[:, None] * np.abs(residual) ** 2)
        / max(float(np.sum(weights[:, None] * np.abs(envelope) ** 2)), 1e-300)
    )
    cumulative, rank_95 = weighted_residual_svd(residual, weights)
    weighted_envelope = envelope * np.sqrt(weights[:, None])
    left, singular_values, right = np.linalg.svd(weighted_envelope, full_matrices=False)
    envelope_energy = np.square(singular_values)
    envelope_cumulative = np.cumsum(envelope_energy / np.sum(envelope_energy))
    rank1_weighted = singular_values[0] * left[:, :1] @ right[:1, :]
    rank1_envelope = rank1_weighted / np.sqrt(weights[:, None])
    rank1_g = rank1_envelope * phase
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    covariance = np.abs(residual.conj().T @ (weights[:, None] * residual))
    off_diagonal = ~np.eye(len(positions), dtype=bool)
    correlation = spearman_correlation(distance[off_diagonal], covariance[off_diagonal])
    ephi_power = float(np.sum(weights[:, None] * np.abs(g) ** 2))
    etheta_power = float(
        np.sum(weights[:, None] * np.abs(np.asarray(data["g_etheta"])) ** 2)
    )
    span = np.ptp(positions, axis=0)
    return {
        "array_id": array_id,
        "scope": "train_operator_cache_only",
        "train_queries": len(direction),
        "logical_shape": list(np.asarray(data["active_mask"]).shape),
        "elements": envelope.shape[1],
        "normalization_scale": float(data["normalization_scale"]),
        "max_D_over_lambda": float(np.max(span) / wavelength),
        "m1_common_eep_operator_nmse_db": nmse_db(g, m1, weights),
        "m1_common_eep_Q_nmse_db": q_nmse_db(g, m1, weights),
        "optimal_rank1_operator_nmse_db": nmse_db(g, rank1_g, weights),
        "optimal_rank1_Q_nmse_db": q_nmse_db(g, rank1_g, weights),
        "envelope_rank_95_percent": int(np.searchsorted(envelope_cumulative, 0.95) + 1),
        "envelope_rank_99_percent": int(np.searchsorted(envelope_cumulative, 0.99) + 1),
        "envelope_svd_cumulative_energy": [float(value) for value in envelope_cumulative],
        "element_nonuniformity_fraction": residual_fraction,
        "residual_rank_95_percent": rank_95,
        "residual_svd_cumulative_energy": cumulative,
        "residual_covariance_distance_spearman": float(correlation),
        "etheta_to_ephi_power_db": float(
            10.0 * np.log10(max(etheta_power / max(ephi_power, 1e-300), 1e-300))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict train-cache-only V2.17 pre-feasibility analysis."
    )
    parser.add_argument("--training-views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays: dict[str, dict[str, np.ndarray]] = {}
    array_ids = ("a1x4", "a1x8", "a2x4", "a4x4")
    for array_id in array_ids:
        cache = args.training_views / array_id / "train_operator_cache.npz"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        arrays[array_id] = load_train_cache(cache)
    per_array = {array_id: analyse_array(array_id, data) for array_id, data in arrays.items()}
    scales = [record["normalization_scale"] for record in per_array.values()]
    result = {
        "analysis_scope": "train_only_pre_feasibility",
        "arrays_included": list(array_ids),
        "arrays_excluded": [],
        "validation_cache_read": False,
        "blind_test_cache_read": False,
        "cnn_trained": False,
        "selection_authority": "none; descriptive evidence only",
        "model_scope": "one independently fitted structure per array; no parameter transfer evaluated",
        "per_array": per_array,
        "normalization_scale_relative_spread": float((max(scales) - min(scales)) / np.mean(scales)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
