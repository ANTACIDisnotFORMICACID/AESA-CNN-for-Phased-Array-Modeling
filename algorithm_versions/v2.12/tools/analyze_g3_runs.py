"""Analyze the paired ten-seed G3 lifted-operator distillation experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v212.statistics import paired_comparison  # noqa: E402


FIELD_BINS = ("-3_to_0_db", "-10_to_-3_db", "-20_to_-10_db", "below_-20_db")


def _load(root: Path, pattern: str) -> dict[int, dict[str, object]]:
    runs: dict[int, dict[str, object]] = {}
    for path in root.glob(pattern):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[int(payload["seed"])] = payload
    return runs


def _nmse(payload: dict[str, object]) -> float:
    return float(payload["validation_metrics"]["nmse_db"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2-runs", required=True, type=Path)
    parser.add_argument("--g3-runs", required=True, type=Path)
    parser.add_argument("--lambda-q", default=0.01, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tag = str(args.lambda_q).replace(".", "p")
    g2 = _load(args.g2_runs, "G2_seed*/result.json")
    g3 = _load(args.g3_runs, f"G3_lambda{tag}_seed*/result.json")
    if len(g2) != 10 or len(g3) != 10:
        raise ValueError(f"paired G2/G3 analysis requires 10+10 runs, found {len(g2)}+{len(g3)}")
    seeds = sorted(set(g2).intersection(g3))
    if len(seeds) != 10:
        raise ValueError("G2 and G3 must use the same ten seeds")

    comparison = paired_comparison(
        np.asarray([_nmse(g2[seed]) for seed in seeds]),
        np.asarray([_nmse(g3[seed]) for seed in seeds]),
    )
    bins: dict[str, object] = {}
    for field_bin in FIELD_BINS:
        reference = [g2[s]["validation_metrics"]["field_strength_bins"][field_bin] for s in seeds]
        candidate = [g3[s]["validation_metrics"]["field_strength_bins"][field_bin] for s in seeds]
        bins[field_bin] = {
            "mean_linear_nmse_improvement_db": float(np.mean([
                float(a["linear_nmse_db"]) - float(b["linear_nmse_db"])
                for a, b in zip(reference, candidate)
            ])),
            "mean_mae_improvement_db": float(np.mean([
                float(a["mae_db"]) - float(b["mae_db"])
                for a, b in zip(reference, candidate)
            ])),
            "candidate_mean_bias_db": float(np.mean([float(row["bias_db"]) for row in candidate])),
            "mean_p95_improvement_db": float(np.mean([
                float(a["p95_abs_db"]) - float(b["p95_abs_db"])
                for a, b in zip(reference, candidate)
            ])),
        }

    stats = asdict(comparison)
    accepted = bool(stats["confidence_interval_95_db"][0] > 0 and stats["wins"] >= 8)
    payload = {
        "schema": "v212-g3-paired-statistics-1",
        "metric": "validation-D linear amplitude NMSE dB",
        "lambda_q": args.lambda_q,
        "operator_targets": "train_commands_only",
        "external_test_used": False,
        "seeds": seeds,
        "mean_validation_nmse_db": {
            "G2": float(np.mean([_nmse(g2[s]) for s in seeds])),
            "G3": float(np.mean([_nmse(g3[s]) for s in seeds])),
        },
        "G3_vs_G2": {**stats, "field_bins": bins},
        "diagnostic_acceptance": {
            "rule": "positive 95% CI and at least 8/10 paired wins",
            "accepted": accepted,
            "replaces_g2_main_model": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
