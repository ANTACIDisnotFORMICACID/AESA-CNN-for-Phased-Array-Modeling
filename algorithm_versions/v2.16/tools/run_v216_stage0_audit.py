"""Evaluate V2.16 validation and coverage mechanisms without any external test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.14", ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v210.contracts import QueryBatch  # noqa: E402
from v212.evaluation import amplitude_metrics  # noqa: E402
from v214.angular_sampling import load_amplitude_label_package  # noqa: E402
from v214.physics_metrics import operator_metrics  # noqa: E402
from v216.experiment import (  # noqa: E402
    ARMS,
    PRIMARY_CANDIDATE,
    SEEDS,
    complex_commands,
    load_amplitude_package,
    load_model_from_run,
)


def _summary(values) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "sample_standard_deviation": float(value.std(ddof=1)) if len(value) > 1 else 0.0,
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "positive_count": int(np.sum(value > 0)),
    }


@torch.no_grad()
def _predict(model, commands: torch.Tensor, direction: torch.Tensor, chunk_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    result = []
    command = commands.to(device)
    for start in range(0, len(direction), chunk_size):
        chunk = direction[start:start + chunk_size].to(device)
        result.append(model.forward_command_grid(command, chunk).T.cpu())
    return torch.cat(result)


@torch.no_grad()
def _operator(model, direction: torch.Tensor, chunk_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(direction), chunk_size):
        chunk = direction[start:start + chunk_size].to(device)
        value, _ = model.operator(QueryBatch(chunk[None]))
        result.append(torch.complex(value[0, :, 0], value[0, :, 1]).reshape(-1, 16).cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def _latency(model, commands: torch.Tensor, direction: torch.Tensor, repeats: int) -> dict[str, float]:
    device = next(model.parameters()).device
    command = commands.to(device)
    query = direction[:512].to(device)
    for _ in range(3):
        model.forward_command_grid(command, query)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.forward_command_grid(command, query)
        samples.append(time.perf_counter() - started)
    median = float(np.median(samples))
    return {
        "batch_directions": 512,
        "commands": int(len(commands)),
        "median_seconds": median,
        "microseconds_per_direction": 1e6 * median / 512.0,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def _gate_diagnostics(model, direction: torch.Tensor) -> dict[str, object] | None:
    residual = model.residual
    if not hasattr(residual, "gate_values") or residual.mode != "axial_gated":
        return None
    device = next(model.parameters()).device
    gate = residual.gate_values(direction.to(device)[None])[0].cpu().numpy()
    q = direction.numpy()
    names = ("row_z", "column_y")
    result = {}
    for index, name in enumerate(names):
        value = gate[:, index]
        result[name] = {
            "mean": float(value.mean()),
            "standard_deviation": float(value.std()),
            "p05": float(np.quantile(value, 0.05)),
            "p50": float(np.quantile(value, 0.50)),
            "p95": float(np.quantile(value, 0.95)),
            "absolute_saturation_fraction_gt_0p95": float(np.mean(np.abs(value) > 0.95)),
            "pearson_with_qx": _safe_correlation(value, q[:, 0]),
            "pearson_with_qy": _safe_correlation(value, q[:, 1]),
            "pearson_with_qz": _safe_correlation(value, q[:, 2]),
            "spearman_with_qy": _safe_correlation(_rank(value), _rank(q[:, 1])),
            "spearman_with_qz": _safe_correlation(_rank(value), _rank(q[:, 2])),
        }
    return result


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(ARMS)
    validation = [report["arm_statistics"][arm]["validation_nmse_db"]["mean"] for arm in labels]
    probe = [report["arm_statistics"][arm]["coverage_probe_nmse_db"]["mean"] for arm in labels]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(labels, validation, color="#4e79a7")
    axes[0].set_ylabel("validation amplitude NMSE (dB)")
    axes[1].bar(labels, probe, color="#59a14f")
    axes[1].set_ylabel("coverage-probe amplitude NMSE (dB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.16 DCAR Stage-0 (all external tests sealed)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--physics-data", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--latency-repeats", type=int, default=7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.prepared_dir / "PREPARATION.json").read_text(encoding="utf-8"))
    train_config = json.loads((args.training_dir / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    sealed_manifest_flags = (
        "old_v215_test_indices_saved",
        "old_v215_test_directions_saved",
        "old_v215_test_commands_saved",
        "old_v215_test_amplitudes_saved",
        "old_v215_test_used_for_preparation_decisions",
        "new_confirmatory_test_defined",
    )
    if any(manifest[name] for name in sealed_manifest_flags):
        raise ValueError("prepared manifest violated the V2.16 external-test seal")
    if train_config["old_v215_test_used"] or train_config["new_confirmatory_test_defined_or_opened"]:
        raise ValueError("training configuration violated the external-test seal")
    validation = load_amplitude_package(args.prepared_dir / "validation.npz", expected_commands=16)
    validation_direction = torch.as_tensor(validation["direction"], dtype=torch.float32)
    validation_target = torch.as_tensor(validation["amplitude"], dtype=torch.float32)
    validation_commands = complex_commands(validation["command_iq"])
    probe = load_amplitude_label_package(args.label_dir / "coverage_probe.npz")
    probe_direction = torch.as_tensor(probe["direction"], dtype=torch.float32)
    probe_target = torch.as_tensor(probe["amplitude"].T.copy(), dtype=torch.float32)
    probe_commands = complex_commands(probe["command_iq"])
    if not np.array_equal(probe["command_indices"], validation["command_indices"]):
        raise ValueError("coverage probe command split differs from validation")
    with np.load(args.physics_data, allow_pickle=False) as physics:
        physics_direction = np.asarray(physics["direction_validation"], dtype=np.float32)
        physics_target = np.asarray(physics["g_ephi_validation"], dtype=np.complex128)
    if not np.allclose(physics_direction, validation["direction"], atol=1e-7, rtol=0.0):
        raise ValueError("physics audit and validation directions differ")

    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = args.training_dir / f"{arm}_seed{seed}"
            training_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            model, checkpoint = load_model_from_run(run_dir, args.device)
            if (
                training_result["old_v215_test_status"] != "not_used"
                or checkpoint["coverage_probe_used_for_selection"] is not False
                or checkpoint["new_confirmatory_test_opened"] is not False
            ):
                raise ValueError(f"run seal failed: {run_dir}")
            validation_prediction = _predict(model, validation_commands, validation_direction, args.chunk_size)
            probe_prediction = _predict(model, probe_commands, probe_direction, args.chunk_size)
            operator_prediction = _operator(model, validation_direction, args.chunk_size)
            finite = bool(
                torch.isfinite(validation_prediction).all()
                and torch.isfinite(probe_prediction).all()
                and torch.all(validation_prediction >= 0)
                and torch.all(probe_prediction >= 0)
                and np.isfinite(operator_prediction).all()
            )
            payload = {
                "arm": arm,
                "seed": seed,
                "parameter_count": int(checkpoint["parameter_count"]),
                "best_epoch": int(training_result["best_epoch"]),
                "validation_metrics": amplitude_metrics(validation_prediction, validation_target),
                "coverage_probe_metrics": amplitude_metrics(probe_prediction, probe_target),
                "operator_metrics": operator_metrics(operator_prediction, physics_target),
                "validation_gate_diagnostics": _gate_diagnostics(model, validation_direction),
                "coverage_gate_diagnostics": _gate_diagnostics(model, probe_direction),
                "latency": _latency(model, validation_commands, validation_direction, args.latency_repeats),
                "all_predictions_finite_nonnegative": finite,
                "checkpoint_selected_by_validation_only": True,
                "old_v215_test_used": False,
                "new_confirmatory_test_status": "not_defined_or_opened",
            }
            runs.append(payload)
            (args.output_dir / f"{arm}_seed{seed}_audit.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "validation_nmse_db": payload["validation_metrics"]["nmse_db"],
                "probe_nmse_db": payload["coverage_probe_metrics"]["nmse_db"],
            }), flush=True)

    by_key = {(row["arm"], row["seed"]): row for row in runs}
    arm_statistics = {}
    for arm in ARMS:
        selected = [by_key[(arm, seed)] for seed in SEEDS]
        arm_statistics[arm] = {
            "validation_nmse_db": _summary([row["validation_metrics"]["nmse_db"] for row in selected]),
            "coverage_probe_nmse_db": _summary([row["coverage_probe_metrics"]["nmse_db"] for row in selected]),
            "operator_nmse_db": _summary([row["operator_metrics"]["nmse_db"] for row in selected]),
            "latency_microseconds_per_direction": _summary([
                row["latency"]["microseconds_per_direction"] for row in selected
            ]),
            "parameter_count": selected[0]["parameter_count"],
        }

    def improvements(reference: str, candidate: str, metric: str) -> list[float]:
        return [
            by_key[(reference, seed)][metric]["nmse_db"]
            - by_key[(candidate, seed)][metric]["nmse_db"]
            for seed in SEEDS
        ]

    gated_vs_control_val = improvements("control_v212", "axial_gated", "validation_metrics")
    gated_vs_control_probe = improvements("control_v212", "axial_gated", "coverage_probe_metrics")
    static_vs_control_val = improvements("control_v212", "axial_static", "validation_metrics")
    gated_vs_static_val = improvements("axial_static", "axial_gated", "validation_metrics")
    gated_vs_iso_val = improvements("iso_control", "axial_gated", "validation_metrics")
    parameter_difference = (
        arm_statistics["axial_gated"]["parameter_count"]
        / arm_statistics["iso_control"]["parameter_count"] - 1.0
    )
    finite = all(row["all_predictions_finite_nonnegative"] for row in runs)
    checks = {
        "gated_vs_control_validation_mean_at_least_0p30": np.mean(gated_vs_control_val) >= 0.30,
        "gated_vs_control_validation_3_of_3_positive": all(x > 0 for x in gated_vs_control_val),
        "gated_vs_control_probe_mean_at_least_0p30": np.mean(gated_vs_control_probe) >= 0.30,
        "gated_vs_control_probe_3_of_3_positive": all(x > 0 for x in gated_vs_control_probe),
        "static_vs_control_validation_mean_at_least_0p10": np.mean(static_vs_control_val) >= 0.10,
        "static_vs_control_validation_at_least_2_of_3_positive": sum(x > 0 for x in static_vs_control_val) >= 2,
        "gated_vs_static_validation_mean_at_least_0p10": np.mean(gated_vs_static_val) >= 0.10,
        "gated_vs_static_validation_at_least_2_of_3_positive": sum(x > 0 for x in gated_vs_static_val) >= 2,
        "gated_vs_iso_validation_mean_at_least_0p10": np.mean(gated_vs_iso_val) >= 0.10,
        "gated_vs_iso_validation_at_least_2_of_3_positive": sum(x > 0 for x in gated_vs_iso_val) >= 2,
        "gated_vs_iso_parameter_difference_within_0p5_percent": abs(parameter_difference) <= 0.005,
        "all_predictions_finite_nonnegative": finite,
        "all_external_tests_unopened": all(
            not row["old_v215_test_used"]
            and row["new_confirmatory_test_status"] == "not_defined_or_opened"
            for row in runs
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    report = {
        "schema": "v216-stage0-audit-summary-1",
        "status": "v216_stage0_pass_eligible_for_fresh_confirmation" if passed else "v216_stage0_stop",
        "evidence_role": "post-v215 exploratory mechanism audit, not confirmatory qualification",
        "primary_candidate": PRIMARY_CANDIDATE,
        "old_v215_external_test_used": False,
        "new_confirmatory_test_defined_or_opened": False,
        "arm_statistics": arm_statistics,
        "paired_improvements_db": {
            "axial_gated_vs_control_validation": _summary(gated_vs_control_val),
            "axial_gated_vs_control_probe": _summary(gated_vs_control_probe),
            "axial_static_vs_control_validation": _summary(static_vs_control_val),
            "axial_gated_vs_axial_static_validation": _summary(gated_vs_static_val),
            "axial_gated_vs_iso_control_validation": _summary(gated_vs_iso_val),
        },
        "parameter_difference_gated_vs_iso_fraction": float(parameter_difference),
        "mechanism_gate": {**checks, "passed": passed},
        "runs": runs,
        "claim_boundary": "Validation and coverage exploration only; no old/new external test, hardware, OTA or FPGA claim.",
    }
    (args.output_dir / "STAGE0_AUDIT_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _make_plot(report, args.output_dir / "STAGE0_AUDIT_SUMMARY.png")
    print(json.dumps({
        "status": report["status"],
        "paired_improvements_db": report["paired_improvements_db"],
        "mechanism_gate": report["mechanism_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
