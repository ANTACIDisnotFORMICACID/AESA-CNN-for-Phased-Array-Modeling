"""Resumable nominal 64x64 phase/continuous/INT8 oracle computation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from boundary_v27 import hpbw_aware_margin_db, tracking_grid
from v27_compat import V27_ROOT, enable_v26

enable_v26()
from hfss_physics_v26 import HfssPatternData  # noqa: E402
from oracle_v26 import OracleConfig, PhysicalOracle  # noqa: E402


THETA_GRID = np.linspace(60.0, 120.0, 64, dtype=np.float64)
PHI_GRID = np.linspace(-30.0, 30.0, 64, dtype=np.float64)
SHARD_ROOT = V27_ROOT / "output" / "oracle_shards"
MAP_ROOT = V27_ROOT / "output" / "maps"


def target_grid() -> np.ndarray:
    return np.asarray([(theta, phi) for theta in THETA_GRID for phi in PHI_GRID], dtype=np.float64)


def target_key(index: int) -> str:
    return f"target_{int(index):04d}.npz"


def _needs_recheck(result) -> bool:
    if not result.optimizer_success:
        return True
    metrics = (result.continuous_metrics, result.int8_metrics)
    return any(
        abs(metric.target_to_peak_db + 0.5) < 0.25
        or abs(metric.pointing_error_deg - 5.0) < 1.0
        or abs(metric.strong_peak_margin_db - 3.0) < 0.25
        for metric in metrics
    )


def _prefer(first, second):
    minimum = first.reference_target_power * 10.0 ** (-0.5 / 10.0)
    valid_first = first.continuous_metrics.target_power >= minimum * (1.0 - 1e-6)
    valid_second = second.continuous_metrics.target_power >= minimum * (1.0 - 1e-6)
    if valid_second and (not valid_first or second.continuous_metrics.strong_peak_margin_db > first.continuous_metrics.strong_peak_margin_db):
        return second
    return first


def _metric_vector(metric, hpbw_margin: float) -> np.ndarray:
    return np.asarray(
        [
            metric.target_power,
            metric.peak_power,
            metric.target_to_peak_db,
            metric.peak_theta_deg,
            metric.peak_phi_deg,
            metric.pointing_error_deg,
            metric.outside_peak_power,
            metric.strong_peak_margin_db,
            hpbw_margin,
            metric.etheta_power,
            metric.ephi_power,
            metric.dominant_polarization_ratio_db,
            float(metric.feasible),
        ],
        dtype=np.float64,
    )


METRIC_NAMES = np.asarray(
    [
        "target_power",
        "peak_power",
        "target_to_peak_db",
        "peak_theta_deg",
        "peak_phi_deg",
        "pointing_error_deg",
        "outside_peak_power",
        "strict_margin_db",
        "hpbw_aware_margin_db",
        "etheta_power",
        "ephi_power",
        "dominant_polarization_ratio_db",
        "strict_feasible",
    ]
)


def solve_one(
    index: int,
    force_recheck: bool = False,
    primary_seed: int = 27000,
    recheck_seed: int = 27197,
) -> dict[str, np.ndarray]:
    theta, phi = target_grid()[int(index)]
    data = HfssPatternData.load()
    primary = PhysicalOracle(data, OracleConfig(starts=32, max_steps=300, seed=int(primary_seed))).solve(float(theta), float(phi))
    rechecked = bool(force_recheck or _needs_recheck(primary))
    selected = primary
    if rechecked:
        confirmation = PhysicalOracle(data, OracleConfig(starts=64, max_steps=600, seed=int(recheck_seed))).solve(float(theta), float(phi))
        selected = _prefer(primary, confirmation)
    method_weights = (
        selected.phase_only_weights,
        selected.continuous_weights,
        selected.int8_weights,
    )
    method_metrics = (
        selected.phase_only_metrics,
        selected.continuous_metrics,
        selected.int8_metrics,
    )
    vectors = []
    hpbw_pixels = []
    for weights, metrics in zip(method_weights, method_metrics):
        margin, pixels = hpbw_aware_margin_db(data.power(weights), metrics.target_power)
        vectors.append(_metric_vector(metrics, margin))
        hpbw_pixels.append(pixels)
    return {
        "index": np.asarray(index, dtype=np.int32),
        "target_hfss_deg": np.asarray([theta, phi], dtype=np.float64),
        "method_names": np.asarray(["phase_only", "continuous_disk", "int8_disk"]),
        "metric_names": METRIC_NAMES,
        "metrics": np.stack(vectors).astype(np.float64),
        "weights": np.stack(method_weights).astype(np.complex64),
        "int8_codes": selected.int8_codes.astype(np.int8),
        "reference_target_power": np.asarray(selected.reference_target_power, dtype=np.float64),
        "optimizer_success": np.asarray(selected.optimizer_success, dtype=bool),
        "rechecked": np.asarray(rechecked, dtype=bool),
        "hpbw_connected_pixels": np.asarray(hpbw_pixels, dtype=np.int32),
        "starts_steps": np.asarray([64, 600] if rechecked else [32, 300], dtype=np.int32),
    }


def _valid_shard(path: Path, expected_index: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                int(archive["index"]) == int(expected_index)
                and archive["metrics"].shape == (3, len(METRIC_NAMES))
                and archive["weights"].shape == (3, 4)
                and np.isfinite(archive["metrics"]).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def _atomic_save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _worker(index: int) -> tuple[int, dict[str, np.ndarray], float]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    payload = solve_one(index)
    return index, payload, time.perf_counter() - started


def run_shards(workers: int = 1, limit: int | None = None) -> dict[str, Any]:
    indices = [index for index in range(4096) if not _valid_shard(SHARD_ROOT / target_key(index), index)]
    if limit is not None:
        indices = indices[: int(limit)]
    completed = 4096 - len([index for index in range(4096) if not _valid_shard(SHARD_ROOT / target_key(index), index)])
    durations: list[float] = []
    if workers <= 1:
        iterator = (_worker(index) for index in indices)
        for done, payload, duration in iterator:
            _atomic_save(SHARD_ROOT / target_key(done), payload)
            durations.append(duration)
            completed += 1
            if completed % 16 == 0:
                print(json.dumps({"completed": completed, "total": 4096, "last_seconds": duration}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_worker, index): index for index in indices}
            for future in as_completed(futures):
                done, payload, duration = future.result()
                _atomic_save(SHARD_ROOT / target_key(done), payload)
                durations.append(duration)
                completed += 1
                if completed % 16 == 0:
                    print(json.dumps({"completed": completed, "total": 4096, "last_seconds": duration}), flush=True)
    remaining = [index for index in range(4096) if not _valid_shard(SHARD_ROOT / target_key(index), index)]
    return {
        "completed": 4096 - len(remaining),
        "remaining": len(remaining),
        "mean_seconds_new": float(np.mean(durations)) if durations else 0.0,
    }


def merge_nominal_map(require_complete: bool = True) -> Path:
    missing = [index for index in range(4096) if not _valid_shard(SHARD_ROOT / target_key(index), index)]
    if missing and require_complete:
        raise RuntimeError(f"cannot merge: {len(missing)} oracle targets are missing; first={missing[:8]}")
    available = [index for index in range(4096) if index not in set(missing)]
    records = []
    for index in available:
        with np.load(SHARD_ROOT / target_key(index), allow_pickle=False) as archive:
            records.append({key: archive[key] for key in archive.files})
    if not records:
        raise RuntimeError("no valid oracle shards")
    metrics = np.stack([record["metrics"] for record in records])
    weights = np.stack([record["weights"] for record in records])
    result: dict[str, np.ndarray] = {
        "indices": np.asarray(available, dtype=np.int32),
        "targets_hfss_deg": np.stack([record["target_hfss_deg"] for record in records]),
        "method_names": records[0]["method_names"],
        "metric_names": records[0]["metric_names"],
        "metrics": metrics,
        "weights": weights,
        "int8_codes": np.stack([record["int8_codes"] for record in records]),
        "optimizer_success": np.asarray([record["optimizer_success"] for record in records], dtype=bool),
        "rechecked": np.asarray([record["rechecked"] for record in records], dtype=bool),
        "complete": np.asarray(len(records) == 4096, dtype=bool),
    }
    if len(records) == 4096:
        for method_index, method in enumerate(result["method_names"].tolist()):
            view_metrics = metrics[:, method_index].reshape(64, 64, -1)
            view_weights = weights[:, method_index].reshape(64, 64, 4)
            tracking = tracking_grid(
                view_metrics[:, :, 3], view_metrics[:, :, 4], view_weights
            )
            for name, value in tracking.items():
                result[f"{method}_{name}"] = value
    MAP_ROOT.mkdir(parents=True, exist_ok=True)
    destination = MAP_ROOT / "nominal_oracle_maps.npz"
    np.savez_compressed(destination, **result)
    return destination


def summarize_map(path: Path | None = None) -> dict[str, Any]:
    path = path or (MAP_ROOT / "nominal_oracle_maps.npz")
    with np.load(path, allow_pickle=False) as archive:
        names = archive["method_names"].tolist()
        metric_names = archive["metric_names"].tolist()
        metrics = archive["metrics"]
        complete = bool(archive["complete"])
        summary: dict[str, Any] = {
            "schema_version": 1,
            "complete": complete,
            "unique_targets": int(len(np.unique(archive["indices"]))),
            "missing_targets": int(4096 - len(archive["indices"])),
            "rechecked_targets": int(np.sum(archive["rechecked"])),
            "methods": {},
        }
        for mi, name in enumerate(names):
            metric = metrics[:, mi]
            get = lambda field: metric[:, metric_names.index(field)]
            gain = get("target_to_peak_db") >= -0.5
            pointing = gain & (get("pointing_error_deg") <= 5.0)
            strict = pointing & (get("strict_margin_db") >= 3.0)
            hpbw_diagnostic = pointing & (get("hpbw_aware_margin_db") >= 3.0)
            summary["methods"][name] = {
                "gain_reachable_count": int(np.sum(gain)),
                "pointing_reachable_count": int(np.sum(pointing)),
                "strict_unambiguous_count": int(np.sum(strict)),
                "hpbw_aware_diagnostic_count": int(np.sum(hpbw_diagnostic)),
                "target_to_peak_db_p05_median_p95": np.percentile(get("target_to_peak_db"), [5, 50, 95]).tolist(),
                "pointing_error_deg_p05_median_p95": np.percentile(get("pointing_error_deg"), [5, 50, 95]).tolist(),
                "strict_margin_db_p05_median_p95_max": np.percentile(get("strict_margin_db"), [5, 50, 95, 100]).tolist(),
            }
        output = MAP_ROOT / "nominal_oracle_summary.json"
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_shards(args.workers, args.limit), indent=2))
    if args.merge:
        merged = merge_nominal_map()
        print(json.dumps(summarize_map(merged), indent=2))
