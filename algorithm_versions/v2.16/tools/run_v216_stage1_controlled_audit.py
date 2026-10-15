"""Audit the controlled V2.16 DCAR arms without opening any external test."""

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
    SEEDS, STAGE1_ARMS, STAGE1_PRIMARY_CANDIDATE, complex_commands,
    load_amplitude_package, load_model_from_run, load_stage1_model_from_run,
)
from v216.statistics import paired_three_seed_summary  # noqa: E402


@torch.no_grad()
def _predict(model, commands: torch.Tensor, direction: torch.Tensor, chunk: int) -> torch.Tensor:
    device = next(model.parameters()).device
    command = commands.to(device)
    values = []
    for start in range(0, len(direction), chunk):
        values.append(model.forward_command_grid(command, direction[start:start + chunk].to(device)).T.cpu())
    return torch.cat(values)


@torch.no_grad()
def _operator(model, direction: torch.Tensor, chunk: int) -> np.ndarray:
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(direction), chunk):
        query = direction[start:start + chunk].to(device)
        iq, _ = model.operator(QueryBatch(query[None]))
        values.append(torch.complex(iq[0, :, 0], iq[0, :, 1]).reshape(-1, 16).cpu().numpy())
    return np.concatenate(values)


@torch.no_grad()
def _latency(model, commands: torch.Tensor, direction: torch.Tensor, repeats: int) -> float:
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
    return float(1e6 * np.median(samples) / len(query))


def _summary(values) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "sample_standard_deviation": float(value.std(ddof=1)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


@torch.no_grad()
def _gate_diagnostics(model, direction: torch.Tensor) -> dict[str, object] | None:
    residual = model.residual
    if not hasattr(residual, "gate_values") or residual.mode != "gated":
        return None
    device = next(model.parameters()).device
    gate = residual.gate_values(direction.to(device)[None])[0].cpu().numpy()
    result = {}
    for block in range(gate.shape[1]):
        result[f"block_{block + 1}"] = {}
        for axis, index in (("row_z", 0), ("column_y", 1)):
            value = gate[:, block, index]
            result[f"block_{block + 1}"][axis] = {
                "mean": float(value.mean()),
                "standard_deviation": float(value.std()),
                "p05": float(np.quantile(value, 0.05)),
                "p50": float(np.quantile(value, 0.50)),
                "p95": float(np.quantile(value, 0.95)),
                "absolute_saturation_fraction_gt_0p95": float(np.mean(np.abs(value) > 0.95)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--physics-data", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--latency-repeats", type=int, default=7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.prepared_dir / "PREPARATION.json").read_text(encoding="utf-8"))
    stage1_config = json.loads((args.training_dir / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    if any(manifest[name] for name in (
        "old_v215_test_indices_saved", "old_v215_test_directions_saved",
        "old_v215_test_commands_saved", "old_v215_test_amplitudes_saved",
        "old_v215_test_used_for_preparation_decisions", "new_confirmatory_test_defined",
    )):
        raise ValueError("prepared package violates the external-test seal")
    if stage1_config["old_v215_test_used"] or stage1_config["coverage_probe_used"]:
        raise ValueError("training configuration violates the audit seal")

    validation = load_amplitude_package(args.prepared_dir / "validation.npz", expected_commands=16)
    q_val = torch.as_tensor(validation["direction"], dtype=torch.float32)
    y_val = torch.as_tensor(validation["amplitude"], dtype=torch.float32)
    w_val = complex_commands(validation["command_iq"])
    probe = load_amplitude_label_package(args.label_dir / "coverage_probe.npz")
    q_probe = torch.as_tensor(probe["direction"], dtype=torch.float32)
    y_probe = torch.as_tensor(probe["amplitude"].T.copy(), dtype=torch.float32)
    w_probe = complex_commands(probe["command_iq"])
    if not np.array_equal(probe["command_indices"], validation["command_indices"]):
        raise ValueError("coverage commands differ from validation")
    with np.load(args.physics_data, allow_pickle=False) as physics:
        q_physics = np.asarray(physics["direction_validation"], dtype=np.float32)
        g_physics = np.asarray(physics["g_ephi_validation"], dtype=np.complex128)
    if not np.allclose(q_physics, validation["direction"], atol=1e-7, rtol=0):
        raise ValueError("physics directions differ from validation")

    runs = []
    for arm in STAGE1_ARMS:
        for seed in SEEDS:
            run_dir = (
                args.control_dir / f"control_v212_seed{seed}"
                if arm == "control_v212" else args.training_dir / f"{arm}_seed{seed}"
            )
            model, checkpoint = (
                load_model_from_run(run_dir, args.device)
                if arm == "control_v212" else load_stage1_model_from_run(run_dir, args.device)
            )
            if (
                checkpoint["coverage_probe_used_for_selection"]
                or checkpoint["old_v215_test_used"]
                or checkpoint["new_confirmatory_test_opened"]
            ):
                raise ValueError(f"checkpoint violates the test seal: {run_dir}")
            val_prediction = _predict(model, w_val, q_val, args.chunk_size)
            probe_prediction = _predict(model, w_probe, q_probe, args.chunk_size)
            operator_prediction = _operator(model, q_val, args.chunk_size)
            finite = bool(
                torch.isfinite(val_prediction).all() and torch.isfinite(probe_prediction).all()
                and torch.all(val_prediction >= 0) and torch.all(probe_prediction >= 0)
                and np.isfinite(operator_prediction).all()
            )
            payload = {
                "arm": arm, "seed": seed,
                "parameter_count": int(checkpoint["parameter_count"]),
                "validation_metrics": amplitude_metrics(val_prediction, y_val),
                "coverage_probe_metrics": amplitude_metrics(probe_prediction, y_probe),
                "operator_metrics": operator_metrics(operator_prediction, g_physics),
                "validation_gate_diagnostics": _gate_diagnostics(model, q_val),
                "latency_microseconds_per_direction": _latency(
                    model, w_val, q_val, args.latency_repeats
                ),
                "all_predictions_finite_nonnegative": finite,
                "coverage_used_for_selection": False,
                "old_v215_test_used": False,
                "new_confirmatory_test_status": "not_defined_or_opened",
            }
            runs.append(payload)
            (args.output_dir / f"{arm}_seed{seed}_audit.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({
                "arm": arm, "seed": seed,
                "validation_nmse_db": payload["validation_metrics"]["nmse_db"],
                "coverage_nmse_db": payload["coverage_probe_metrics"]["nmse_db"],
            }), flush=True)

    by_key = {(row["arm"], row["seed"]): row for row in runs}

    def improvement(reference: str, candidate: str, metric: str) -> list[float]:
        return [
            by_key[(reference, seed)][metric]["nmse_db"]
            - by_key[(candidate, seed)][metric]["nmse_db"] for seed in SEEDS
        ]

    comparisons = {
        "gated_pm_vs_control_validation": paired_three_seed_summary(
            improvement("control_v212", "dcar_gated_pm", "validation_metrics")
        ),
        "gated_pm_vs_control_coverage": paired_three_seed_summary(
            improvement("control_v212", "dcar_gated_pm", "coverage_probe_metrics")
        ),
        "static_pm_vs_control_validation": paired_three_seed_summary(
            improvement("control_v212", "dcar_static_pm", "validation_metrics")
        ),
        "gated_pm_vs_static_pm_validation": paired_three_seed_summary(
            improvement("dcar_static_pm", "dcar_gated_pm", "validation_metrics")
        ),
        "gated_native_vs_static_native_validation": paired_three_seed_summary(
            improvement("dcar_static_native", "dcar_gated_native", "validation_metrics")
        ),
        "gated_pm_vs_gated_native_validation": paired_three_seed_summary(
            improvement("dcar_gated_native", "dcar_gated_pm", "validation_metrics")
        ),
        "gated_pm_vs_control_operator": paired_three_seed_summary(
            improvement("control_v212", "dcar_gated_pm", "operator_metrics")
        ),
    }
    arm_statistics = {}
    for arm in STAGE1_ARMS:
        selected = [by_key[(arm, seed)] for seed in SEEDS]
        arm_statistics[arm] = {
            "validation_nmse_db": _summary([x["validation_metrics"]["nmse_db"] for x in selected]),
            "validation_mae_db": _summary([x["validation_metrics"]["mae_db"] for x in selected]),
            "coverage_nmse_db": _summary([x["coverage_probe_metrics"]["nmse_db"] for x in selected]),
            "coverage_mae_db": _summary([x["coverage_probe_metrics"]["mae_db"] for x in selected]),
            "operator_nmse_db": _summary([x["operator_metrics"]["nmse_db"] for x in selected]),
            "latency_microseconds_per_direction": _summary(
                [x["latency_microseconds_per_direction"] for x in selected]
            ),
            "parameter_count": selected[0]["parameter_count"],
        }
    primary_parameters = arm_statistics[STAGE1_PRIMARY_CANDIDATE]["parameter_count"]
    control_parameters = arm_statistics["control_v212"]["parameter_count"]
    parameter_difference = primary_parameters / control_parameters - 1.0
    gate = {
        "gated_pm_vs_control_validation_mean_at_least_0p30": comparisons["gated_pm_vs_control_validation"]["mean_db"] >= 0.30,
        "gated_pm_vs_control_validation_3_of_3_positive": comparisons["gated_pm_vs_control_validation"]["positive_count"] == 3,
        "gated_pm_vs_control_coverage_mean_at_least_0p30": comparisons["gated_pm_vs_control_coverage"]["mean_db"] >= 0.30,
        "gated_pm_vs_control_coverage_3_of_3_positive": comparisons["gated_pm_vs_control_coverage"]["positive_count"] == 3,
        "static_pm_vs_control_validation_mean_at_least_minus_0p10": comparisons["static_pm_vs_control_validation"]["mean_db"] >= -0.10,
        "static_pm_vs_control_validation_at_least_2_of_3_nonnegative": comparisons["static_pm_vs_control_validation"]["positive_count"] >= 2,
        "gated_pm_vs_static_pm_mean_at_least_0p10": comparisons["gated_pm_vs_static_pm_validation"]["mean_db"] >= 0.10,
        "gated_pm_vs_static_pm_at_least_2_of_3_positive": comparisons["gated_pm_vs_static_pm_validation"]["positive_count"] >= 2,
        "gated_native_vs_static_native_mean_at_least_0p10": comparisons["gated_native_vs_static_native_validation"]["mean_db"] >= 0.10,
        "gated_native_vs_static_native_at_least_2_of_3_positive": comparisons["gated_native_vs_static_native_validation"]["positive_count"] >= 2,
        "primary_parameter_difference_within_0p5_percent": abs(parameter_difference) <= 0.005,
        "all_predictions_finite_nonnegative": all(x["all_predictions_finite_nonnegative"] for x in runs),
        "all_external_tests_unopened": all(
            not x["old_v215_test_used"] and x["new_confirmatory_test_status"] == "not_defined_or_opened"
            for x in runs
        ),
    }
    gate = {name: bool(value) for name, value in gate.items()}
    passed = all(gate.values())
    report = {
        "schema": "v216-stage1-controlled-audit-1",
        "status": "v216_stage1_pass_eligible_for_fresh_confirmation" if passed else "v216_stage1_stop",
        "evidence_role": "post-Stage-0 exploratory controlled DCAR audit",
        "primary_candidate": STAGE1_PRIMARY_CANDIDATE,
        "arm_statistics": arm_statistics,
        "paired_comparisons": comparisons,
        "primary_parameter_difference_fraction": float(parameter_difference),
        "mechanism_gate": {**gate, "passed": passed},
        "runs": runs,
        "old_v215_external_test_used": False,
        "new_confirmatory_test_defined_or_opened": False,
        "claim_boundary": "Validation and coverage exploration only; no external-test, hardware, OTA or FPGA claim.",
    }
    (args.output_dir / "STAGE1_CONTROLLED_AUDIT_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import matplotlib.pyplot as plt
    labels = list(STAGE1_ARMS)
    validation_means = [arm_statistics[name]["validation_nmse_db"]["mean"] for name in labels]
    coverage_means = [arm_statistics[name]["coverage_nmse_db"]["mean"] for name in labels]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].bar(labels, validation_means, color="#4e79a7")
    axes[1].bar(labels, coverage_means, color="#59a14f")
    axes[0].set_ylabel("validation amplitude NMSE (dB)")
    axes[1].set_ylabel("coverage amplitude NMSE (dB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=22)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.16 Stage-1 controlled DCAR (all external tests sealed)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "STAGE1_CONTROLLED_AUDIT_SUMMARY.png", dpi=180)
    plt.close(figure)
    print(json.dumps({
        "status": report["status"], "paired_comparisons": comparisons,
        "mechanism_gate": report["mechanism_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
