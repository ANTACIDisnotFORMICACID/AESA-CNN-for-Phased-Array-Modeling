"""Derive reproducible paired, field-bin and gate summaries from the frozen audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v216.experiment import ARMS, SEEDS  # noqa: E402
from v216.statistics import paired_three_seed_summary  # noqa: E402


BINS = ("-3_to_0_db", "-10_to_-3_db", "-20_to_-10_db", "below_-20_db")


def _plain_summary(values) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "sample_standard_deviation": float(value.std(ddof=1)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    if report["old_v215_external_test_used"] or report["new_confirmatory_test_defined_or_opened"]:
        raise ValueError("external-test seal failed")
    by_key = {(row["arm"], int(row["seed"])): row for row in report["runs"]}

    def improvement(reference: str, candidate: str, metric: str) -> list[float]:
        return [
            by_key[(reference, seed)][metric]["nmse_db"]
            - by_key[(candidate, seed)][metric]["nmse_db"]
            for seed in SEEDS
        ]

    comparisons = {
        "axial_gated_vs_control_validation": paired_three_seed_summary(
            improvement("control_v212", "axial_gated", "validation_metrics")
        ),
        "axial_gated_vs_control_coverage": paired_three_seed_summary(
            improvement("control_v212", "axial_gated", "coverage_probe_metrics")
        ),
        "axial_static_vs_control_validation": paired_three_seed_summary(
            improvement("control_v212", "axial_static", "validation_metrics")
        ),
        "axial_gated_vs_static_validation": paired_three_seed_summary(
            improvement("axial_static", "axial_gated", "validation_metrics")
        ),
        "axial_gated_vs_iso_validation": paired_three_seed_summary(
            improvement("iso_control", "axial_gated", "validation_metrics")
        ),
        "iso_vs_control_validation": paired_three_seed_summary(
            improvement("control_v212", "iso_control", "validation_metrics")
        ),
        "axial_gated_vs_control_complex_operator": paired_three_seed_summary(
            improvement("control_v212", "axial_gated", "operator_metrics")
        ),
    }
    field_bins = {}
    for metric_name in ("validation_metrics", "coverage_probe_metrics"):
        field_bins[metric_name] = {}
        for bin_name in BINS:
            values = [
                by_key[("control_v212", seed)][metric_name]["field_strength_bins"][bin_name]["linear_nmse_db"]
                - by_key[("axial_gated", seed)][metric_name]["field_strength_bins"][bin_name]["linear_nmse_db"]
                for seed in SEEDS
            ]
            field_bins[metric_name][bin_name] = paired_three_seed_summary(values)

    arm_metrics = {}
    for arm in ARMS:
        selected = [by_key[(arm, seed)] for seed in SEEDS]
        arm_metrics[arm] = {
            "validation_nmse_db": _plain_summary(
                [row["validation_metrics"]["nmse_db"] for row in selected]
            ),
            "validation_mae_db": _plain_summary(
                [row["validation_metrics"]["mae_db"] for row in selected]
            ),
            "coverage_nmse_db": _plain_summary(
                [row["coverage_probe_metrics"]["nmse_db"] for row in selected]
            ),
            "coverage_mae_db": _plain_summary(
                [row["coverage_probe_metrics"]["mae_db"] for row in selected]
            ),
            "complex_operator_nmse_db": _plain_summary(
                [row["operator_metrics"]["nmse_db"] for row in selected]
            ),
            "latency_microseconds_per_direction": _plain_summary(
                [row["latency"]["microseconds_per_direction"] for row in selected]
            ),
            "parameter_count": int(selected[0]["parameter_count"]),
        }

    raw_gate_correlations = {}
    gated = [by_key[("axial_gated", seed)] for seed in SEEDS]
    for axis in ("row_z", "column_y"):
        raw_gate_correlations[axis] = {}
        for name in ("pearson_with_qx", "pearson_with_qy", "pearson_with_qz"):
            raw_gate_correlations[axis][name] = _plain_summary(
                [row["validation_gate_diagnostics"][axis][name] for row in gated]
            )

    training_seconds = []
    best_epochs = []
    for arm in ARMS:
        for seed in SEEDS:
            result = json.loads(
                (args.training_dir / f"{arm}_seed{seed}" / "result.json").read_text(encoding="utf-8")
            )
            training_seconds.append(float(result["training_seconds"]))
            best_epochs.append(int(result["best_epoch"]))
    output = {
        "schema": "v216-stage0-statistical-analysis-1",
        "status": report["status"],
        "paired_improvement_convention": "positive means the second named model is better",
        "paired_comparisons": comparisons,
        "field_strength_bin_improvements_gated_vs_control": field_bins,
        "arm_metrics": arm_metrics,
        "raw_gate_correlations": raw_gate_correlations,
        "gate_identifiability_warning": (
            "A gate sign/scale can trade against its convolution weights; raw gate values are descriptive, "
            "not a gauge-invariant physical attribution."
        ),
        "training": {
            "runs": len(training_seconds),
            "epochs_per_run": 240,
            "total_seconds": float(sum(training_seconds)),
            "best_epoch_minimum": min(best_epochs),
            "best_epoch_maximum": max(best_epochs),
        },
        "interpretation_boundary": (
            "The single-spatial-layer 29-map arms differ from V2.12 in both spatial depth and removal "
            "of edge/corner maps; this audit cannot uniquely assign the loss to axial factorization."
        ),
        "old_v215_external_test_used": False,
        "new_confirmatory_test_defined_or_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
