"""Merge and validate resumable 64x64 oracle shards without accepting partial maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


V26_ROOT = Path(__file__).resolve().parents[1]


def merge(shard_directory: Path, output: Path, allow_incomplete: bool = False) -> dict:
    records = []
    sources = sorted(shard_directory.glob("oracle_*_of_*.jsonl"))
    for source in sources:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                item = json.loads(line)
                item["_source"] = f"{source.name}:{line_number}"
                records.append(item)
    by_target = {}
    duplicates = []
    for item in records:
        key = (round(float(item["target_theta_deg"]), 12), round(float(item["target_phi_deg"]), 12))
        if key in by_target:
            duplicates.append({"target": key, "first": by_target[key]["_source"], "second": item["_source"]})
        by_target[key] = item
    expected_theta = np.linspace(60.0, 120.0, 64)
    expected_phi = np.linspace(-30.0, 30.0, 64)
    expected = {(round(float(t), 12), round(float(p), 12)) for t in expected_theta for p in expected_phi}
    missing = sorted(expected - set(by_target))
    unexpected = sorted(set(by_target) - expected)
    complete = len(by_target) == 4096 and not duplicates and not missing and not unexpected
    if not complete and not allow_incomplete:
        raise RuntimeError(
            f"oracle map incomplete: unique={len(by_target)}, duplicates={len(duplicates)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
    summaries = {}
    for method in ("phase_only", "continuous", "int8"):
        metrics = [item[f"{method}_metrics"] for item in by_target.values()]
        if not metrics:
            summaries[method] = {"samples": 0}
            continue
        summaries[method] = {
            "samples": len(metrics),
            "feasible_count": int(sum(bool(row["feasible"]) for row in metrics)),
            "target_to_peak_db_p05_median_p95": np.percentile([row["target_to_peak_db"] for row in metrics], [5, 50, 95]).tolist(),
            "pointing_error_deg_p05_median_p95": np.percentile([row["pointing_error_deg"] for row in metrics], [5, 50, 95]).tolist(),
            "strong_peak_margin_db_p05_median_p95_max": np.percentile([row["strong_peak_margin_db"] for row in metrics], [5, 50, 95, 100]).tolist(),
        }
    payload = {
        "schema_version": 1,
        "complete": complete,
        "expected_targets": 4096,
        "unique_targets": len(by_target),
        "duplicates": duplicates[:32],
        "missing_count": len(missing),
        "missing_first": missing[:32],
        "unexpected": unexpected[:32],
        "summaries": summaries,
        "conclusion": (
            "fixed array physically limited; no amplitude/phase-feasible target"
            if complete and summaries["continuous"]["feasible_count"] == 0
            else "some amplitude/phase-feasible targets exist"
            if complete
            else "partial map; no whole-domain conclusion is permitted"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-directory", type=Path, default=V26_ROOT / "output" / "oracle_shards")
    parser.add_argument("--output", type=Path, default=V26_ROOT / "output" / "reports" / "oracle_map_summary.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(merge(args.shard_directory, args.output, args.allow_incomplete), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

