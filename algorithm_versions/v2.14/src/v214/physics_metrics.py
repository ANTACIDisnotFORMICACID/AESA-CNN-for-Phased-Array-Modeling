"""Gauge-invariant and field-level metrics for V2.14 physics attribution."""

from __future__ import annotations

import numpy as np


def rank_one_operator(g: np.ndarray) -> np.ndarray:
    value = np.asarray(g)
    if value.ndim != 2 or value.shape[1] != 16 or not np.iscomplexobj(value):
        raise ValueError("g must be complex [Q,16]")
    return value.conj()[:, :, None] * value[:, None, :]


def weighted_nmse_db(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.shape[0] == 0:
        raise ValueError("prediction and target must be matching non-empty arrays")
    reduce_axes = tuple(range(1, target.ndim))
    error = np.sum(np.abs(prediction - target) ** 2, axis=reduce_axes)
    energy = np.sum(np.abs(target) ** 2, axis=reduce_axes)
    if weights is None:
        weight = np.full(len(target), 1.0 / len(target))
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight.shape != (len(target),) or np.any(weight < 0) or not np.isfinite(weight).all():
            raise ValueError("invalid weights")
        weight = weight / weight.sum()
    ratio = float(np.sum(weight * error) / max(np.sum(weight * energy), 1e-300))
    return 10.0 * np.log10(max(ratio, 1e-300))


def operator_metrics(prediction_g: np.ndarray, target_g: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    prediction_q = rank_one_operator(prediction_g)
    target_q = rank_one_operator(target_g)
    error = np.sum(np.abs(prediction_q - target_q) ** 2, axis=(1, 2))
    energy = np.maximum(np.sum(np.abs(target_q) ** 2, axis=(1, 2)), 1e-300)
    relative = error / energy
    return {
        "nmse_db": weighted_nmse_db(prediction_q, target_q, weights),
        "per_direction_relative_error_p50": float(np.quantile(relative, 0.50)),
        "per_direction_relative_error_p95": float(np.quantile(relative, 0.95)),
        "per_direction_relative_error_max": float(relative.max()),
    }


def amplitude_nmse_db(prediction_g: np.ndarray, target_g: np.ndarray, commands: np.ndarray) -> float:
    command = np.asarray(commands)
    prediction = np.abs(np.asarray(prediction_g) @ command.T)
    target = np.abs(np.asarray(target_g) @ command.T)
    return weighted_nmse_db(prediction, target)


def common_phase_aligned_nmse_db(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("complex fields must match [Q,N]")
    phase = np.angle(np.sum(prediction.conj() * target, axis=1))
    aligned = prediction * np.exp(1j * phase)[:, None]
    return weighted_nmse_db(aligned, target)


def transverse_diagnostic(direction: np.ndarray, h_cartesian: np.ndarray) -> dict[str, float]:
    direction = np.asarray(direction, dtype=np.float64)
    h = np.asarray(h_cartesian)
    if h.shape != (len(direction), 3, 16):
        raise ValueError("h_cartesian must be [Q,3,16]")
    longitudinal = np.einsum("qx,qxn->qn", direction, h)
    ratio = np.abs(longitudinal) ** 2 / np.maximum(np.sum(np.abs(h) ** 2, axis=1), 1e-300)
    return {"mean": float(ratio.mean()), "p95": float(np.quantile(ratio, 0.95)), "maximum": float(ratio.max())}


def polarization_diagnostic(g_ephi: np.ndarray, g_etheta: np.ndarray) -> dict[str, float]:
    ephi = np.sum(np.abs(g_ephi) ** 2, axis=1)
    etheta = np.sum(np.abs(g_etheta) ** 2, axis=1)
    ratio_db = 10.0 * np.log10(np.maximum(etheta, 1e-300) / np.maximum(ephi, 1e-300))
    return {
        "etheta_over_ephi_db_median": float(np.median(ratio_db)),
        "etheta_over_ephi_db_p05": float(np.quantile(ratio_db, 0.05)),
        "etheta_over_ephi_db_p95": float(np.quantile(ratio_db, 0.95)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("x and y must align")
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])

