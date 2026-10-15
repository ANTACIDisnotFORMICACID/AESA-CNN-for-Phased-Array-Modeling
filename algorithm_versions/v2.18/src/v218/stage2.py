"""V2.18 Stage-2 migration attribution without confirmatory-data access."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree

from v210.contracts import QueryBatch
from v216.experiment import load_model_from_run, load_stage1_model_from_run

from .audit import load_run
from .canonical import PreparedData, atomic_json, load_prepared
from .training import _operator_complex


OVERLAP_TOLERANCE = 1e-7
EQUIVALENCE_GATES = {
    "aggregate_nmse_db_max": -60.0,
    "worst_command_nmse_db_max": -50.0,
    "strong_relative_error_p99_max": 1e-3,
}


def _db(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-300))


def complex_commands(command_iq: np.ndarray) -> np.ndarray:
    value = np.asarray(command_iq, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (2, 16):
        raise ValueError("command IQ must have shape [C,2,16]")
    return (value[:, 0] + 1j * value[:, 1]).astype(np.complex64)


def direction_overlap(
    source_direction: np.ndarray,
    reference_direction: np.ndarray,
    *,
    tolerance: float = OVERLAP_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices, nearest reference indices, and Euclidean errors."""
    tree = cKDTree(np.asarray(reference_direction, dtype=np.float64))
    distance, index = tree.query(np.asarray(source_direction, dtype=np.float64), k=1)
    mask = distance <= tolerance
    return np.flatnonzero(mask), np.asarray(index[mask], dtype=np.int64), np.asarray(distance[mask])


def fit_positive_scale(canonical: np.ndarray, reference: np.ndarray) -> float:
    x = np.asarray(canonical, dtype=np.float64).ravel()
    y = np.asarray(reference, dtype=np.float64).ravel()
    scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-300))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("train-only normalization scale is not finite and positive")
    return scale


def equivalence_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    per_command = np.sum(error * error, axis=0) / np.maximum(
        np.sum(target * target, axis=0), 1e-300
    )
    aggregate = float(np.sum(error * error) / max(float(np.sum(target * target)), 1e-300))
    peak = np.max(target, axis=0, keepdims=True)
    strong = target >= peak * 1e-2
    relative = np.abs(error[strong]) / np.maximum(target[strong], 1e-300)
    return {
        "aggregate_nmse": aggregate,
        "aggregate_nmse_db": _db(aggregate),
        "per_command_nmse_db": [_db(value) for value in per_command],
        "worst_command_nmse_db": max(_db(value) for value in per_command),
        "strong_relative_error_p99": float(np.quantile(relative, 0.99)),
        "finite": bool(np.isfinite(prediction).all() and np.isfinite(target).all()),
    }


def _weighted_nmse(error: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    weight = np.asarray(weights, dtype=np.float64)[:, None]
    return float(np.sum(weight * np.square(error)) / max(float(np.sum(weight * np.square(target))), 1e-300))


def _subset_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0, "status": "empty"}
    error = prediction[mask] - target[mask]
    nmse = float(np.sum(error * error) / max(float(np.sum(np.square(target[mask]))), 1e-300))
    floor = max(float(np.max(target)) * 1e-6, 1e-12)
    db_error = 20 * np.log10(np.maximum(prediction[mask], floor)) - 20 * np.log10(
        np.maximum(target[mask], floor)
    )
    return {
        "count": count,
        "nmse": nmse,
        "nmse_db": _db(nmse),
        "mae_db": float(np.mean(np.abs(db_error))),
        "bias_db": float(np.mean(db_error)),
        "p95_abs_error_db": float(np.quantile(np.abs(db_error), 0.95)),
    }


def amplitude_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    direction_weights: np.ndarray | None = None,
    command_family: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("amplitude arrays must have matching shape [Q,C]")
    error = prediction - target
    grid_nmse = float(np.sum(error * error) / max(float(np.sum(target * target)), 1e-300))
    weights = np.ones(len(target), dtype=np.float64) if direction_weights is None else np.asarray(direction_weights, dtype=np.float64)
    weighted_nmse = _weighted_nmse(error, target, weights)
    peak = np.maximum(np.max(target, axis=0, keepdims=True), 1e-300)
    relative_db = 20 * np.log10(np.maximum(target / peak, 1e-300))
    bins = {
        "strong_-3_to_0_db": (relative_db >= -3),
        "mid_strong_-10_to_-3_db": (relative_db >= -10) & (relative_db < -3),
        "mid_weak_-20_to_-10_db": (relative_db >= -20) & (relative_db < -10),
        "weak_below_-20_db": relative_db < -20,
        "deep_null_below_-30_db": relative_db < -30,
    }
    families: dict[str, object] = {}
    family_macro = None
    directional_features: dict[str, object] = {}
    if command_family is not None:
        labels = np.asarray(command_family).astype(str)
        family_values = []
        for label in sorted(set(labels)):
            selected = labels == label
            value = _weighted_nmse(error[:, selected], target[:, selected], weights)
            families[label] = {"commands": int(selected.sum()), "solid_angle_nmse": value, "solid_angle_nmse_db": _db(value)}
            family_values.append(value)
        family_macro = float(np.mean(family_values))
        if direction is not None:
            vectors = np.asarray(direction, dtype=np.float64)
            scan = np.flatnonzero(labels == "scan")
            if len(scan):
                target_peak = np.argmax(target[:, scan], axis=0)
                prediction_peak = np.argmax(prediction[:, scan], axis=0)
                cosine = np.sum(vectors[target_peak] * vectors[prediction_peak], axis=1)
                angular = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                peak_amplitude_db = 20 * np.log10(
                    np.maximum(prediction[prediction_peak, scan], 1e-12)
                    / np.maximum(target[target_peak, scan], 1e-12)
                )
                directional_features["scan_peak"] = {
                    "commands": len(scan),
                    "mean_angular_error_deg": float(np.mean(angular)),
                    "p95_angular_error_deg": float(np.quantile(angular, 0.95)),
                    "mean_abs_peak_amplitude_error_db": float(np.mean(np.abs(peak_amplitude_db))),
                }
            hard_null = np.flatnonzero(labels == "hard_null")
            if len(hard_null):
                target_null = np.argmin(target[:, hard_null], axis=0)
                prediction_null = np.argmin(prediction[:, hard_null], axis=0)
                cosine = np.sum(vectors[target_null] * vectors[prediction_null], axis=1)
                angular = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                null_depth_db = 20 * np.log10(
                    np.maximum(prediction[prediction_null, hard_null], 1e-12)
                    / np.maximum(target[target_null, hard_null], 1e-12)
                )
                directional_features["hard_null"] = {
                    "commands": len(hard_null),
                    "mean_angular_error_deg": float(np.mean(angular)),
                    "p95_angular_error_deg": float(np.quantile(angular, 0.95)),
                    "mean_abs_null_depth_error_db": float(np.mean(np.abs(null_depth_db))),
                }
    return {
        "grid_micro_nmse": grid_nmse,
        "grid_micro_nmse_db": _db(grid_nmse),
        "solid_angle_micro_nmse": weighted_nmse,
        "solid_angle_micro_nmse_db": _db(weighted_nmse),
        "command_family_macro_nmse": family_macro,
        "command_family_macro_nmse_db": None if family_macro is None else _db(family_macro),
        "command_family": families,
        "directional_features": directional_features,
        "fixed_field_bins": {name: _subset_metrics(prediction, target, mask) for name, mask in bins.items()},
        "finite_nonnegative": bool(np.isfinite(prediction).all() and np.all(prediction >= 0)),
    }


def operator_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= np.sum(normalized)
    raw = np.sum(normalized[:, None] * np.abs(prediction - target) ** 2) / np.sum(
        normalized[:, None] * np.abs(target) ** 2
    )
    inner = np.sum(np.conj(prediction) * target, axis=1)
    phase = inner / np.maximum(np.abs(inner), 1e-300)
    aligned = prediction * phase[:, None]
    aligned_nmse = np.sum(normalized[:, None] * np.abs(aligned - target) ** 2) / np.sum(
        normalized[:, None] * np.abs(target) ** 2
    )
    prediction_norm = np.sum(np.abs(prediction) ** 2, axis=1)
    target_norm = np.sum(np.abs(target) ** 2, axis=1)
    q_error = prediction_norm**2 + target_norm**2 - 2 * np.abs(inner) ** 2
    q_nmse = np.sum(normalized * np.maximum(q_error, 0)) / np.sum(normalized * target_norm**2)
    return {
        "raw_complex_nmse_db_diagnostic_only": _db(raw),
        "phase_aligned_complex_nmse_db": _db(aligned_nmse),
        "q_nmse_db": _db(q_nmse),
    }


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    direction: np.ndarray,
    commands: np.ndarray,
    *,
    device: str,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    direction_tensor = torch.from_numpy(np.asarray(direction, dtype=np.float32)).to(device)
    command_tensor = torch.from_numpy(np.asarray(commands, dtype=np.complex64)).to(device)
    amplitude_parts: list[np.ndarray] = []
    operator_parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(direction_tensor), chunk):
        selected = direction_tensor[start:start + chunk]
        amplitude_parts.append(model.forward_command_grid(command_tensor, selected).T.cpu().numpy())
        operator, _ = _operator_complex(model, selected)
        operator_parts.append(operator.cpu().numpy())
    return np.concatenate(amplitude_parts), np.concatenate(operator_parts)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    arm: str
    seed: int
    run_dir: Path
    scale_to_canonical: float


def _model_specs(version_root: Path, scale: float) -> Iterable[ModelSpec]:
    v216 = version_root.parent / "v2.16" / "artifacts"
    v218 = version_root / "artifacts" / "training_4x4_20261014"
    for seed in (21601, 21602, 21603):
        yield ModelSpec("v216", "control_v212", seed, v216 / "training_20260914" / f"control_v212_seed{seed}", 1 / scale)
        yield ModelSpec("v216", "dcar_gated_pm", seed, v216 / "stage1_controlled_training_20260914" / f"dcar_gated_pm_seed{seed}", 1 / scale)
    for seed in (21801, 21802, 21803):
        yield ModelSpec("v218", "control_v212_canonical", seed, v218 / f"control_v212_canonical_seed{seed}", 1.0)
        yield ModelSpec("v218", "dcar_gated_pm_canonical", seed, v218 / f"dcar_gated_pm_canonical_seed{seed}", 1.0)


def _load_spec(spec: ModelSpec, data: PreparedData, device: str):
    if spec.family == "v218":
        return load_run(spec.run_dir, data, device)[0]
    if spec.arm == "control_v212":
        return load_model_from_run(spec.run_dir, device=device)[0]
    return load_stage1_model_from_run(spec.run_dir, device=device)[0]


def _panel_summary(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        key = f"{record['family']}:{record['arm']}"
        grouped.setdefault(key, []).append(float(record["amplitude"]["solid_angle_micro_nmse_db"]))
    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "per_seed_nmse_db": values,
            "mean_nmse_db": float(np.mean(values)),
            "std_nmse_db": float(np.std(values, ddof=1)),
            "range_nmse_db": float(np.ptp(values)),
            "worst_seed_nmse_db": float(max(values)),
        }
    return summary


def run_attribution(*, version_root: Path, output_dir: Path, device: str = "cpu") -> dict[str, object]:
    """Run the frozen P0-P3 attribution panels and write one recoverable receipt."""
    prepared = load_prepared(version_root / "artifacts" / "prepared_4x4_20261014")
    old_path = version_root.parent / "v2.16" / "artifacts" / "prepared_20260914" / "validation.npz"
    with np.load(old_path, allow_pickle=False) as archive:
        old = {name: np.asarray(archive[name]).copy() for name in archive.files}
    old_commands = complex_commands(old["command_iq"])
    old_direction = np.asarray(old["direction"], dtype=np.float32)

    old_train_i, canonical_train_i, train_distance = direction_overlap(old_direction, prepared.train["direction"])
    old_val_i, canonical_val_i, val_distance = direction_overlap(old_direction, prepared.validation["direction"])
    if len(np.intersect1d(old_train_i, old_val_i)):
        raise ValueError("canonical train and validation overlaps are not disjoint")
    canonical_train_amplitude = np.abs(prepared.train["g_ephi"][canonical_train_i] @ old_commands.T)
    scale = fit_positive_scale(canonical_train_amplitude, old["amplitude"][old_train_i])
    canonical_val_amplitude = np.abs(prepared.validation["g_ephi"][canonical_val_i] @ old_commands.T)
    equivalence = equivalence_metrics(scale * canonical_val_amplitude, old["amplitude"][old_val_i])
    equivalence_gates = {
        "train_overlap_exactly_3200": len(old_train_i) == 3200,
        "validation_overlap_exactly_800": len(old_val_i) == 800,
        "max_direction_error_le_1e-7": bool(max(np.max(train_distance), np.max(val_distance)) <= OVERLAP_TOLERANCE),
        "aggregate_nmse_db_le_-60": equivalence["aggregate_nmse_db"] <= EQUIVALENCE_GATES["aggregate_nmse_db_max"],
        "worst_command_nmse_db_le_-50": equivalence["worst_command_nmse_db"] <= EQUIVALENCE_GATES["worst_command_nmse_db_max"],
        "strong_relative_error_p99_lt_1e-3": equivalence["strong_relative_error_p99"] < EQUIVALENCE_GATES["strong_relative_error_p99_max"],
        "finite": equivalence["finite"],
    }
    equivalence_pass = all(equivalence_gates.values())
    if not equivalence_pass:
        receipt = {
            "schema": "v218-stage2-attribution-1",
            "status": "MIGRATION_ATTRIBUTION_FAIL",
            "failure_class": "DATA_ADAPTER_OR_REFERENCE_FAIL",
            "normalization_scale_from_canonical_to_v216": scale,
            "equivalence": equivalence,
            "equivalence_gates": equivalence_gates,
            "confirmation_status": "CONFIRMATION_NOT_OPENED",
        }
        atomic_json(output_dir / "MIGRATION_ATTRIBUTION.json", receipt)
        return receipt

    panel_records: dict[str, list[dict[str, object]]] = {name: [] for name in ("P0", "P1", "P2", "P3")}
    canonical_direction = prepared.validation["direction"]
    canonical_weights = prepared.validation["solid_angle_weights"]
    canonical_operator = prepared.validation["g_ephi"]
    canonical_old_target = np.abs(canonical_operator @ old_commands.T)
    for spec in _model_specs(version_root, scale):
        model = _load_spec(spec, prepared, device)
        if spec.family == "v216":
            panel_name = "P0"
            old_prediction_scale = 1.0
        else:
            panel_name = "P1"
            old_prediction_scale = scale
        old_prediction, _ = predict_model(model, old_direction, old_commands, device=device)
        panel_records[panel_name].append({
            "family": spec.family,
            "arm": spec.arm,
            "seed": spec.seed,
            "amplitude": amplitude_metrics(old_prediction * old_prediction_scale, old["amplitude"]),
        })
        prediction16, operator = predict_model(model, canonical_direction, old_commands, device=device)
        prediction16 *= spec.scale_to_canonical
        operator *= spec.scale_to_canonical
        panel_records["P2"].append({
            "family": spec.family,
            "arm": spec.arm,
            "seed": spec.seed,
            "amplitude": amplitude_metrics(prediction16, canonical_old_target, direction_weights=canonical_weights),
            "operator": operator_metrics(operator, canonical_operator, canonical_weights),
        })
        prediction512, _ = predict_model(model, canonical_direction, prepared.validation["commands"], device=device)
        prediction512 *= spec.scale_to_canonical
        panel_records["P3"].append({
            "family": spec.family,
            "arm": spec.arm,
            "seed": spec.seed,
            "amplitude": amplitude_metrics(
                prediction512,
                prepared.validation["amplitude"],
                direction_weights=canonical_weights,
                command_family=prepared.validation["command_family"],
                direction=canonical_direction,
            ),
            "operator": operator_metrics(operator, canonical_operator, canonical_weights),
        })
        del model

    summaries = {name: _panel_summary(records) for name, records in panel_records.items()}
    p3_control = summaries["P3"]["v218:control_v212_canonical"]
    p3_gated = summaries["P3"]["v218:dcar_gated_pm_canonical"]
    p2_control = summaries["P2"]["v218:control_v212_canonical"]
    p2_gated = summaries["P2"]["v218:dcar_gated_pm_canonical"]
    p2_advantage = p2_control["mean_nmse_db"] - p2_gated["mean_nmse_db"]
    p3_advantage = p3_control["mean_nmse_db"] - p3_gated["mean_nmse_db"]
    signals = {
        "command_shift_or_phase_gap": bool(p2_advantage - p3_advantage >= 0.30),
        "gated_optimization_instability": bool(
            p3_gated["range_nmse_db"] - p3_control["range_nmse_db"] >= 0.50
            or p3_gated["std_nmse_db"] - p3_control["std_nmse_db"] >= 0.30
        ),
    }
    attribution_pass = equivalence_pass and any(signals.values())
    receipt = {
        "schema": "v218-stage2-attribution-1",
        "status": "MIGRATION_ATTRIBUTION_PASS" if attribution_pass else "MIGRATION_ATTRIBUTION_FAIL",
        "failure_class": None if attribution_pass else "NO_PREREGISTERED_PHASE_OR_INSTABILITY_SIGNAL",
        "normalization_scale_from_canonical_to_v216": scale,
        "overlap": {
            "canonical_train": len(old_train_i),
            "canonical_validation": len(old_val_i),
            "unmatched_not_traced": len(old_direction) - len(old_train_i) - len(old_val_i),
            "maximum_direction_error": float(max(np.max(train_distance), np.max(val_distance))),
        },
        "equivalence": equivalence,
        "equivalence_gates": equivalence_gates,
        "attribution_signals": signals,
        "v218_gated_advantage_db": {"P2_old16": p2_advantage, "P3_frozen512": p3_advantage},
        "panel_summaries": summaries,
        "panels": panel_records,
        "phase_aux_training_authorized": attribution_pass,
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    }
    atomic_json(output_dir / "MIGRATION_ATTRIBUTION.json", receipt)
    return receipt
