"""Analyze the ten-seed G1/G2 physical-reconstruction experiment."""

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

from v212.statistics import holm_adjust, paired_comparison  # noqa: E402


FIELD_BINS = ("-3_to_0_db", "-10_to_-3_db", "-20_to_-10_db", "below_-20_db")


def _load(root: Path, recipe: str) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for path in root.glob(f"{recipe}_seed*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[int(payload["seed"])] = payload
    return result


def _nmse(payload: dict[str, object]) -> float:
    return float(
        payload.get(
            "best_validation_nmse_db", payload.get("best_validation_linear_nmse_db")
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-runs", required=True, type=Path)
    parser.add_argument("--stage2-runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    collections = {
        "G0_E0": _load(args.stage1_runs, "E0"),
        "Stage1_E2-F5": _load(args.stage1_runs, "E2-F5"),
        "G1": _load(args.stage2_runs, "G1"),
        "G2": _load(args.stage2_runs, "G2"),
    }
    for name, runs in collections.items():
        if len(runs) != 10:
            raise ValueError(f"{name} requires exactly 10 completed seeds, found {len(runs)}")

    specifications = (
        ("G1_vs_G0_E0", "G0_E0", "G1"),
        ("G2_vs_G0_E0", "G0_E0", "G2"),
        ("G2_vs_G1", "G1", "G2"),
        ("G2_vs_Stage1_E2-F5", "Stage1_E2-F5", "G2"),
    )
    comparisons: dict[str, dict[str, object]] = {}
    raw_p: dict[str, float] = {}
    for key, reference_name, candidate_name in specifications:
        reference = collections[reference_name]
        candidate = collections[candidate_name]
        seeds = sorted(set(reference).intersection(candidate))
        if len(seeds) != 10:
            raise ValueError(f"{key} requires ten matching seeds")
        comparison = paired_comparison(
            np.asarray([_nmse(reference[seed]) for seed in seeds]),
            np.asarray([_nmse(candidate[seed]) for seed in seeds]),
        )
        field_bins: dict[str, object] = {}
        for field_bin in FIELD_BINS:
            reference_bin = [
                reference[seed]["validation_metrics"]["field_strength_bins"][field_bin]
                for seed in seeds
            ]
            candidate_bin = [
                candidate[seed]["validation_metrics"]["field_strength_bins"][field_bin]
                for seed in seeds
            ]
            field_bins[field_bin] = {
                "mean_linear_nmse_improvement_db": float(
                    np.mean(
                        [
                            float(left["linear_nmse_db"])
                            - float(right["linear_nmse_db"])
                            for left, right in zip(reference_bin, candidate_bin)
                        ]
                    )
                ),
                "mean_mae_improvement_db": float(
                    np.mean(
                        [
                            float(left["mae_db"]) - float(right["mae_db"])
                            for left, right in zip(reference_bin, candidate_bin)
                        ]
                    )
                ),
                "candidate_mean_bias_db": float(
                    np.mean([float(row["bias_db"]) for row in candidate_bin])
                ),
                "mean_p95_improvement_db": float(
                    np.mean(
                        [
                            float(left["p95_abs_db"])-float(right["p95_abs_db"])
                            for left, right in zip(reference_bin, candidate_bin)
                        ]
                    )
                ),
            }
        comparisons[key] = {**asdict(comparison), "seeds": seeds, "field_bins": field_bins}
        raw_p[key] = comparison.two_sided_sign_flip_p

    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        comparisons[key]["holm_adjusted_p"] = value

    g1, g2 = collections["G1"], collections["G2"]
    seeds = sorted(g1)
    mainlobe_delta = float(
        np.mean(
            [
                float(g2[seed]["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"])
                - float(g1[seed]["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"])
                for seed in seeds
            ]
        )
    )
    g2_vs_g1 = comparisons["G2_vs_G1"]
    payload = {
        "schema": "v212-stage2-paired-statistics-1",
        "metric": "validation-D linear amplitude NMSE dB",
        "external_test_used": False,
        "mean_validation_nmse_db": {
            name: float(np.mean([_nmse(row) for row in runs.values()]))
            for name, runs in collections.items()
        },
        "comparisons": comparisons,
        "g2_acceptance": {
            "ci_lower_positive_and_wins_at_least_8": bool(
                g2_vs_g1["confidence_interval_95_db"][0] > 0
                and g2_vs_g1["wins"] >= 8
            ),
            "mean_mainlobe_mae_change_vs_g1_db": mainlobe_delta,
            "mainlobe_non_degradation_limit_db": 0.05,
            "accepted": bool(
                g2_vs_g1["confidence_interval_95_db"][0] > 0
                and g2_vs_g1["wins"] >= 8
                and mainlobe_delta <= 0.05
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
