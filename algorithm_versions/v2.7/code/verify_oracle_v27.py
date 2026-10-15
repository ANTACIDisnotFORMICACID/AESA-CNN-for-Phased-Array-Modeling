"""Independent 5 percent repeatability audit for the released nominal oracle map."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np

from oracle_v27 import MAP_ROOT, solve_one
from v27_compat import V27_ROOT


ROOT = V27_ROOT / "output" / "cache" / "oracle_repeatability"
DESTINATION = V27_ROOT / "output" / "reports" / "oracle_repeatability_5pct.json"


def verification_indices() -> np.ndarray:
    return np.sort(np.random.default_rng(27003).choice(4096, size=205, replace=False)).astype(np.int32)


def _path(index: int) -> Path:
    return ROOT / f"target_{index:04d}.npz"


def _worker(index: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # force_recheck guarantees the independent 64-start/600-step configuration is considered.
    return index, solve_one(index, force_recheck=True, primary_seed=37003, recheck_seed=37197)


def run(workers: int = 1, limit: int | None = None) -> dict:
    pending = [int(index) for index in verification_indices() if not _path(int(index)).is_file()]
    if limit is not None:
        pending = pending[: int(limit)]
    ROOT.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        iterator = (_worker(index) for index in pending)
        for index, payload in iterator:
            np.savez_compressed(_path(index), **payload)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_worker, index) for index in pending]
            for count, future in enumerate(as_completed(futures), start=1):
                index, payload = future.result()
                np.savez_compressed(_path(index), **payload)
                if count % 8 == 0:
                    print(json.dumps({"repeatability_new": count, "requested": len(pending)}), flush=True)
    return {"completed": sum(_path(int(index)).is_file() for index in verification_indices()), "total": 205}


def summarize() -> dict:
    source = MAP_ROOT / "nominal_oracle_maps.npz"
    with np.load(source, allow_pickle=False) as archive:
        names = archive["method_names"].tolist()
        metric_names = archive["metric_names"].tolist()
        released = archive["metrics"]
    method = names.index("continuous_disk")
    differences = []
    missing = []
    for index in verification_indices():
        path = _path(int(index))
        if not path.is_file():
            missing.append(int(index))
            continue
        with np.load(path, allow_pickle=False) as archive:
            repeat = archive["metrics"][method]
        original = released[int(index), method]
        differences.append(
            [
                abs(float(10.0 * np.log10((repeat[metric_names.index("target_power")] + 1e-30) / (original[metric_names.index("target_power")] + 1e-30)))),
                abs(float(repeat[metric_names.index("pointing_error_deg")] - original[metric_names.index("pointing_error_deg")])),
                abs(float(repeat[metric_names.index("strict_margin_db")] - original[metric_names.index("strict_margin_db")])),
            ]
        )
    values = np.asarray(differences, dtype=np.float64)
    maxima = np.max(values, axis=0).tolist() if len(values) else [float("inf")] * 3
    p95 = np.percentile(values, 95, axis=0).tolist() if len(values) else [float("inf")] * 3
    failures = [int(np.sum(values[:, index] > limit)) for index, limit in enumerate((0.02, 1.0, 0.05))] if len(values) else [205, 205, 205]
    payload = {
        "schema_version": 1,
        "requested": 205,
        "completed": len(values),
        "missing": missing,
        "maximum_absolute_difference": {
            "target_gain_db": maxima[0],
            "pointing_error_deg": maxima[1],
            "strict_margin_db": maxima[2],
        },
        "p95_absolute_difference": {
            "target_gain_db": p95[0],
            "pointing_error_deg": p95[1],
            "strict_margin_db": p95[2],
        },
        "failed_sample_count": {
            "target_gain_db": failures[0],
            "pointing_error_deg": failures[1],
            "strict_margin_db": failures[2],
        },
        "limits": {"target_gain_db": 0.02, "pointing_error_deg": 1.0, "strict_margin_db": 0.05},
        "passed": bool(not missing and maxima[0] <= 0.02 and maxima[1] <= 1.0 and maxima[2] <= 0.05),
        "interpretation": "target gain and strict margin are repeatable, but the raw global-peak location can jump between nearly equal lobes; this failure is retained rather than replaced by an HPBW-aware or tie-tolerant criterion",
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.workers, args.limit), indent=2))
    if run(1, 0)["completed"] == 205:
        print(json.dumps(summarize(), indent=2))
