"""1,024 state-specific continuous/INT8 references for representative audit states."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path

import numpy as np

from boundary_v27 import hpbw_aware_margin_db
from oracle_v27 import METRIC_NAMES, _metric_vector, _needs_recheck, _prefer
from states_v27 import all_audit_states
from v27_compat import V27_ROOT, enable_v26

enable_v26()
from dataset_v26 import state_pattern  # noqa: E402
from hfss_physics_v26 import HfssPatternData  # noqa: E402
from oracle_v26 import OracleConfig, PhysicalOracle  # noqa: E402


ROOT = V27_ROOT / "output" / "cache" / "state_oracle"
DESTINATION = V27_ROOT / "output" / "audit" / "state_continuous_reference_1024.npz"


def representative_state_indices() -> np.ndarray:
    states = all_audit_states()
    required_ids = {
        "baseline",
        "amp_rms_13pct",
        "phase_rms_22deg",
        "snr_25db",
        "failed_port_1",
        "failed_port_2",
        "failed_port_3",
        "failed_port_4",
    }
    selected = [state.state_index for state in states if state.state_id in required_ids]
    mixed = [state.state_index for state in states if state.state_class == "mixed"]
    selected.extend(mixed[index] for index in np.linspace(0, len(mixed) - 1, 8, dtype=int))
    value = np.asarray(selected, dtype=np.int32)
    if len(value) != 16 or len(np.unique(value)) != 16:
        raise RuntimeError("representative-state contract requires 16 unique states")
    return value


def representative_target_indices() -> tuple[np.ndarray, np.ndarray]:
    source = V27_ROOT / "output" / "maps" / "nominal_phase_only_map.npz"
    with np.load(source, allow_pickle=False) as archive:
        names = archive["metric_names"].tolist()
        metrics = archive["metrics"]
    gain = metrics[:, names.index("target_to_peak_db")]
    pointing = metrics[:, names.index("pointing_error_deg")]
    reachable = np.flatnonzero((gain >= -0.5) & (pointing <= 5.0))
    boundary_score = np.minimum(np.abs(gain + 0.5) / 0.25, np.abs(pointing - 5.0))
    boundary = np.argsort(boundary_score)
    outside_score = np.maximum(-(gain + 0.5), pointing - 5.0)
    outside = np.argsort(outside_score)[::-1]

    def take(values: np.ndarray, count: int, used: set[int]) -> list[int]:
        candidates = [int(value) for value in values if int(value) not in used]
        if len(candidates) < count:
            raise RuntimeError("insufficient targets for representative stratum")
        positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
        result = [candidates[index] for index in positions]
        used.update(result)
        return result

    used: set[int] = set()
    core = take(reachable, 22, used)
    edge = take(boundary, 21, used)
    impossible = take(outside, 21, used)
    indices = np.asarray(core + edge + impossible, dtype=np.int32)
    labels = np.asarray(["reachable_core"] * 22 + ["boundary"] * 21 + ["outside"] * 21)
    return indices, labels


def _solve_case(state_index: int, target_slot: int) -> tuple[int, int, dict[str, np.ndarray]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    state = all_audit_states()[int(state_index)]
    target_indices, labels = representative_target_indices()
    with np.load(V27_ROOT / "output" / "maps" / "nominal_phase_only_map.npz", allow_pickle=False) as archive:
        target = archive["targets_hfss_deg"][target_indices[int(target_slot)]]
    data = HfssPatternData.load()
    physical = state_pattern(data, state)
    primary = PhysicalOracle(physical, OracleConfig(starts=32, max_steps=300, seed=27200 + state_index)).solve(*map(float, target))
    rechecked = _needs_recheck(primary)
    selected = primary
    if rechecked:
        confirmation = PhysicalOracle(physical, OracleConfig(starts=64, max_steps=600, seed=28200 + state_index)).solve(*map(float, target))
        selected = _prefer(primary, confirmation)
    metrics = []
    for weights, value in (
        (selected.phase_only_weights, selected.phase_only_metrics),
        (selected.continuous_weights, selected.continuous_metrics),
        (selected.int8_weights, selected.int8_metrics),
    ):
        margin = hpbw_aware_margin_db(physical.power(weights), value.target_power)[0]
        metrics.append(_metric_vector(value, margin))
    return int(state_index), int(target_slot), {
        "target_hfss_deg": target.astype(np.float32),
        "stratum": labels[int(target_slot)],
        "metrics": np.stack(metrics).astype(np.float32),
        "weights": np.stack((selected.phase_only_weights, selected.continuous_weights, selected.int8_weights)).astype(np.complex64),
        "int8_codes": selected.int8_codes.astype(np.int8),
        "rechecked": np.asarray(rechecked, dtype=bool),
    }


def _path(state_index: int, slot: int) -> Path:
    return ROOT / f"state_{state_index:02d}_target_{slot:02d}.npz"


def _valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return archive["metrics"].shape == (3, len(METRIC_NAMES)) and np.isfinite(archive["metrics"]).all()
    except (OSError, KeyError, ValueError):
        return False


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def run(workers: int = 1, limit: int | None = None) -> dict:
    cases = [(int(state), slot) for state in representative_state_indices() for slot in range(64)]
    pending = [(state, slot) for state, slot in cases if not _valid(_path(state, slot))]
    if limit is not None:
        pending = pending[: int(limit)]
    completed = len(cases) - sum(not _valid(_path(state, slot)) for state, slot in cases)
    if workers <= 1:
        for state, slot in pending:
            _, _, payload = _solve_case(state, slot)
            _save(_path(state, slot), payload)
            completed += 1
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_solve_case, state, slot) for state, slot in pending]
            for future in as_completed(futures):
                state, slot, payload = future.result()
                _save(_path(state, slot), payload)
                completed += 1
                if completed % 16 == 0:
                    print(json.dumps({"state_oracle_completed": completed, "total": 1024}), flush=True)
    remaining = sum(not _valid(_path(state, slot)) for state, slot in cases)
    return {"completed": 1024 - remaining, "remaining": remaining}


def merge() -> Path:
    states = representative_state_indices()
    target_indices, labels = representative_target_indices()
    records = []
    for state in states:
        for slot in range(64):
            path = _path(int(state), slot)
            if not _valid(path):
                raise RuntimeError(f"missing state-oracle record {path.name}")
            with np.load(path, allow_pickle=False) as archive:
                records.append({key: archive[key] for key in archive.files})
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    state_vector = np.repeat(states, 64)
    target_vector = np.tile(target_indices, 16)
    stratum_vector = np.tile(labels, 16)
    metrics = np.stack([record["metrics"] for record in records])
    np.savez_compressed(
        DESTINATION,
        state_indices=state_vector,
        target_indices=target_vector,
        target_strata=stratum_vector,
        targets_hfss_deg=np.stack([record["target_hfss_deg"] for record in records]),
        method_names=np.asarray(["phase_only", "continuous_disk", "int8_disk"]),
        metric_names=METRIC_NAMES,
        metrics=metrics,
        weights=np.stack([record["weights"] for record in records]),
        int8_codes=np.stack([record["int8_codes"] for record in records]),
        rechecked=np.asarray([record["rechecked"] for record in records], dtype=bool),
    )
    names = METRIC_NAMES.tolist()
    rows = []
    for sample, (state_index, target_index, stratum) in enumerate(zip(state_vector, target_vector, stratum_vector)):
        for method_index, method in enumerate(("phase_only", "continuous_disk", "int8_disk")):
            value = metrics[sample, method_index]
            gain = value[names.index("target_to_peak_db")] >= -0.5
            pointing = gain and value[names.index("pointing_error_deg")] <= 5.0
            strict = pointing and value[names.index("strict_margin_db")] >= 3.0
            rows.append(
                {
                    "state_index": int(state_index),
                    "target_index": int(target_index),
                    "stratum": str(stratum),
                    "method": method,
                    **{name: float(value[index]) for index, name in enumerate(names)},
                    "gain_reachable": bool(gain),
                    "pointing_reachable": bool(pointing),
                    "strict_unambiguous": bool(strict),
                }
            )
    csv_path = DESTINATION.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    phase = metrics[:, 0]
    continuous = metrics[:, 1]
    phase_pointing = (phase[:, names.index("target_to_peak_db")] >= -0.5) & (phase[:, names.index("pointing_error_deg")] <= 5.0)
    continuous_pointing = (continuous[:, names.index("target_to_peak_db")] >= -0.5) & (continuous[:, names.index("pointing_error_deg")] <= 5.0)
    summary = {
        "schema_version": 1,
        "states": 16,
        "targets_per_state": 64,
        "cases": 1024,
        "phase_pointing_reachable_fraction": float(np.mean(phase_pointing)),
        "continuous_pointing_reachable_fraction": float(np.mean(continuous_pointing)),
        "continuous_margin_improvement_db_p05_median_p95": np.percentile(
            continuous[:, names.index("strict_margin_db")] - phase[:, names.index("strict_margin_db")], [5, 50, 95]
        ).tolist(),
        "continuous_strict_unambiguous_fraction": float(np.mean(
            continuous_pointing & (continuous[:, names.index("strict_margin_db")] >= 3.0)
        )),
        "rechecked": int(np.sum([record["rechecked"] for record in records])),
    }
    DESTINATION.with_name("state_continuous_reference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return DESTINATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.workers, args.limit)
    print(json.dumps(result, indent=2))
    if args.merge and result["remaining"] == 0:
        print(merge())
