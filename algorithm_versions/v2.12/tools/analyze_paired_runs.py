"""Compute the pre-declared paired bootstrap, sign-flip and Holm statistics."""

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


def _load_runs(root: Path, recipe: str) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for path in root.glob(f"{recipe}_seed*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[int(payload["seed"])] = payload
    return result


def _nmse_db(payload: dict[str, object]) -> float:
    if "best_validation_linear_nmse_db" in payload:
        return float(payload["best_validation_linear_nmse_db"])
    return float(payload["best_validation_nmse_db"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="E0")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = _load_runs(args.runs_dir, args.baseline)
    comparisons: dict[str, dict[str, object]] = {}
    raw_p: dict[str, float] = {}
    acceptance: dict[str, object] = {}
    for candidate_name in args.candidates:
        candidate = _load_runs(args.runs_dir, candidate_name)
        references = [args.baseline]
        if candidate_name == "E2-F4":
            references.append("E0-M4")
        elif candidate_name == "E2-F5":
            references.append("E0-M5")
        candidate_checks: list[bool] = []
        for reference_name in references:
            reference = baseline if reference_name == args.baseline else _load_runs(
                args.runs_dir, reference_name
            )
            seeds = sorted(set(reference).intersection(candidate))
            if len(seeds) != 10:
                raise ValueError(
                    f"{candidate_name} vs {reference_name} requires exactly 10 paired seeds"
                )
            reference_values = np.asarray(
                [_nmse_db(reference[seed]) for seed in seeds]
            )
            candidate_values = np.asarray(
                [_nmse_db(candidate[seed]) for seed in seeds]
            )
            comparison = paired_comparison(reference_values, candidate_values)
            key = f"{candidate_name}_vs_{reference_name}"
            comparisons[key] = {**asdict(comparison), "seeds": seeds}
            raw_p[key] = comparison.two_sided_sign_flip_p
            candidate_checks.append(
                comparison.confidence_interval_95_db[0] > 0 and comparison.wins >= 8
            )
        if candidate_name.startswith("E2-"):
            seeds = sorted(set(baseline).intersection(candidate))
            baseline_main = np.asarray(
                [
                    baseline[seed]["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"]
                    for seed in seeds
                ]
            )
            candidate_main = np.asarray(
                [
                    candidate[seed]["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"]
                    for seed in seeds
                ]
            )
            mainlobe_delta = float(np.mean(candidate_main - baseline_main))
            acceptance[candidate_name] = {
                "ssh_value_established": bool(all(candidate_checks) and mainlobe_delta <= 0.05),
                "all_ci_lower_bounds_positive_and_wins_at_least_8": bool(all(candidate_checks)),
                "mean_mainlobe_mae_change_db": mainlobe_delta,
                "mainlobe_non_degradation_limit_db": 0.05,
            }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    order_selection: dict[str, object] | None = None
    if "E2-F4" in args.candidates and "E2-F5" in args.candidates:
        l4 = _load_runs(args.runs_dir, "E2-F4")
        l5 = _load_runs(args.runs_dir, "E2-F5")
        seeds = sorted(set(l4).intersection(l5))
        if len(seeds) != 10:
            raise ValueError("L4/L5 order selection requires exactly 10 paired seeds")
        l4_nmse = np.asarray([_nmse_db(l4[seed]) for seed in seeds])
        l5_nmse = np.asarray([_nmse_db(l5[seed]) for seed in seeds])
        improvement = float(np.mean(l4_nmse - l5_nmse))
        order_selection = {
            "mean_L5_improvement_over_L4_db": improvement,
            "threshold_db": 0.2,
            "selected_degree": 4 if improvement < 0.2 else 5,
            "rule": "select L4 when mean validation improvement from L5 is below 0.2 dB",
        }
    payload = {
        "schema": "v212-paired-statistics-1",
        "baseline": args.baseline,
        "metric": "best validation-D linear amplitude NMSE dB",
        "external_test_used": False,
        "comparisons": comparisons,
        "acceptance": acceptance,
        "order_selection": order_selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
