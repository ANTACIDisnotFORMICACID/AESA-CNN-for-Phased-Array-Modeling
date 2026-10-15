"""Audit all trained V2.15 forwards without opening the external test split."""

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
from v214.physics_ladder import element_classes  # noqa: E402
from v214.physics_metrics import operator_metrics  # noqa: E402
from v215.experiment import ARMS, PRIMARY_CANDIDATE, SEEDS, complex_commands, load_amplitude_package, load_model_from_run  # noqa: E402


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
        result.append(model.forward_command_grid(command, direction[start:start + chunk_size].to(device)).T.cpu())
    return torch.cat(result)


@torch.no_grad()
def _operator(model, direction: torch.Tensor, chunk_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(direction), chunk_size):
        value, _ = model.operator(QueryBatch(direction[start:start + chunk_size].to(device)[None]))
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
        start = time.perf_counter()
        model.forward_command_grid(command, query)
        samples.append(time.perf_counter() - start)
    return {
        "batch_directions": 512,
        "commands": int(len(commands)),
        "median_seconds": float(np.median(samples)),
        "microseconds_per_direction": float(1e6 * np.median(samples) / 512.0),
    }


def _aligned_residual(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    phase = np.angle(np.sum(prediction.conj() * target, axis=1))
    return target - prediction * np.exp(1j * phase)[:, None]


def _residual_diagnostics(prediction: np.ndarray, target: np.ndarray, s_matrix: np.ndarray) -> dict[str, object]:
    residual = _aligned_residual(prediction, target)
    covariance = np.abs(residual.conj().T @ residual)
    mask = ~np.eye(16, dtype=bool)
    row = np.arange(16) // 4
    col = np.arange(16) % 4
    adjacency = ((np.abs(row[:, None] - row[None, :]) + np.abs(col[:, None] - col[None, :])) == 1).astype(float)
    s_abs = np.abs(s_matrix).copy()
    np.fill_diagonal(s_abs, 0.0)
    classes = element_classes()
    return {
        "operator": operator_metrics(prediction, target),
        "aligned_residual_covariance_adjacency_correlation": float(np.corrcoef(covariance[mask], adjacency[mask])[0, 1]),
        "aligned_residual_covariance_s_correlation": float(np.corrcoef(covariance[mask], s_abs[mask])[0, 1]),
        "class_residual_energy": {
            name: float(np.mean(np.abs(residual[:, classes == class_id]) ** 2))
            for class_id, name in enumerate(("corner", "edge", "interior"))
        },
    }


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(ARMS)
    validation = [report["arm_statistics"][arm]["validation_nmse_db"]["mean"] for arm in labels]
    probe = [report["arm_statistics"][arm]["coverage_probe_nmse_db"]["mean"] for arm in labels]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(labels, validation, color="#4e79a7")
    axes[0].set_ylabel("validation amplitude NMSE (dB)")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, probe, color="#59a14f")
    axes[1].set_ylabel("coverage-probe amplitude NMSE (dB)")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.15 complete forward validation (test sealed)")
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

    train_manifest = json.loads((args.prepared_dir / "PREPARATION.json").read_text(encoding="utf-8"))
    train_config = json.loads((args.training_dir / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    if train_config.get("test_split_opened") is not False or train_config.get("coverage_probe_used") is not False:
        raise ValueError("training configuration violated the seal")
    validation = load_amplitude_package(args.prepared_dir / "validation.npz", expected_commands=16)
    validation_direction = torch.as_tensor(validation["direction"], dtype=torch.float32)
    validation_target = torch.as_tensor(validation["amplitude"], dtype=torch.float32)
    validation_commands = complex_commands(validation["command_iq"])
    probe = load_amplitude_label_package(args.label_dir / "coverage_probe.npz")
    probe_direction = torch.as_tensor(probe["direction"], dtype=torch.float32)
    probe_target = torch.as_tensor(probe["amplitude"].T.copy(), dtype=torch.float32)
    probe_commands = complex_commands(probe["command_iq"])
    if not torch.equal(torch.as_tensor(probe["command_indices"]), torch.as_tensor(validation["command_indices"])):
        raise ValueError("coverage probe command split differs from validation")
    with np.load(args.physics_data, allow_pickle=False) as physics:
        s_matrix = np.asarray(physics["s_parameters_47ghz"], dtype=np.complex128)
        physics_direction = np.asarray(physics["direction_validation"], dtype=np.float32)
        physics_target = np.asarray(physics["g_ephi_validation"], dtype=np.complex128)
    if not np.allclose(physics_direction, validation["direction"], atol=1e-7, rtol=0.0):
        raise ValueError("physics audit and validation directions differ")

    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = args.training_dir / f"{arm}_seed{seed}"
            training_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            model, checkpoint = load_model_from_run(run_dir, s_matrix, args.device)
            if training_result.get("test_status") != "sealed" or checkpoint.get("coverage_probe_used_for_selection") is not False:
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
                "best_epoch": training_result["best_epoch"],
                "validation_metrics": amplitude_metrics(validation_prediction, validation_target),
                "coverage_probe_metrics": amplitude_metrics(probe_prediction, probe_target),
                "physics_diagnostics": _residual_diagnostics(operator_prediction, physics_target, s_matrix),
                "latency": _latency(model, validation_commands, validation_direction, args.latency_repeats),
                "all_predictions_finite_nonnegative": finite,
                "checkpoint_selected_by_validation_only": True,
                "test_status": "sealed",
            }
            runs.append(payload)
            run_audit = args.output_dir / f"{arm}_seed{seed}_forward.json"
            run_audit.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"arm": arm, "seed": seed, "validation_nmse_db": payload["validation_metrics"]["nmse_db"], "probe_nmse_db": payload["coverage_probe_metrics"]["nmse_db"]}), flush=True)

    by_key = {(row["arm"], row["seed"]): row for row in runs}
    arm_statistics = {}
    for arm in ARMS:
        selected = [by_key[(arm, seed)] for seed in SEEDS]
        arm_statistics[arm] = {
            "validation_nmse_db": _summary([row["validation_metrics"]["nmse_db"] for row in selected]),
            "coverage_probe_nmse_db": _summary([row["coverage_probe_metrics"]["nmse_db"] for row in selected]),
            "operator_nmse_db": _summary([row["physics_diagnostics"]["operator"]["nmse_db"] for row in selected]),
            "latency_microseconds_per_direction": _summary([row["latency"]["microseconds_per_direction"] for row in selected]),
            "parameter_count": selected[0]["parameter_count"],
        }
    def improvements(reference: str, candidate: str, metric: str) -> list[float]:
        return [
            by_key[(reference, seed)][metric]["nmse_db"] - by_key[(candidate, seed)][metric]["nmse_db"]
            for seed in SEEDS
        ]
    candidate_validation = improvements("control", PRIMARY_CANDIDATE, "validation_metrics")
    candidate_probe = improvements("control", PRIMARY_CANDIDATE, "coverage_probe_metrics")
    true_vs_shuffled = improvements("graph_s_shuffled", PRIMARY_CANDIDATE, "validation_metrics")
    parameter_difference = arm_statistics[PRIMARY_CANDIDATE]["parameter_count"] / arm_statistics["control"]["parameter_count"] - 1.0
    finite = all(row["all_predictions_finite_nonnegative"] for row in runs)
    structure_gate = {
        "candidate_validation_mean_at_least_0p30": bool(np.mean(candidate_validation) >= 0.30),
        "candidate_validation_3_of_3_positive": bool(all(value > 0 for value in candidate_validation)),
        "true_s_vs_shuffled_mean_at_least_0p10": bool(np.mean(true_vs_shuffled) >= 0.10),
        "true_s_vs_shuffled_at_least_2_of_3_positive": bool(sum(value > 0 for value in true_vs_shuffled) >= 2),
        "candidate_probe_mean_at_least_0p30": bool(np.mean(candidate_probe) >= 0.30),
        "candidate_probe_3_of_3_positive": bool(all(value > 0 for value in candidate_probe)),
        "parameter_difference_within_5_percent": bool(abs(parameter_difference) <= 0.05),
        "all_predictions_finite_nonnegative": finite,
    }
    test_open_checks = {
        "predeclared_candidate_is_graph_s_true": PRIMARY_CANDIDATE == "graph_s_true",
        "all_12_runs_complete": len(runs) == 12,
        "all_checkpoints_selected_by_validation_only": all(row["checkpoint_selected_by_validation_only"] for row in runs),
        "prepared_packages_contain_no_test": not any(train_manifest[key] for key in (
            "test_indices_saved", "test_directions_saved", "test_commands_saved", "test_amplitudes_saved"
        )),
        "training_did_not_use_coverage_probe": train_config["coverage_probe_used"] is False,
        "all_predictions_finite_nonnegative": finite,
        "all_training_and_audit_test_sealed": all(row["test_status"] == "sealed" for row in runs),
    }
    report = {
        "schema": "v215-forward-validation-summary-1",
        "status": "forward_validation_complete",
        "predeclared_primary_candidate": PRIMARY_CANDIDATE,
        "test_used": False,
        "arm_statistics": arm_statistics,
        "paired_candidate_vs_control_validation_improvement_db": _summary(candidate_validation),
        "paired_candidate_vs_control_probe_improvement_db": _summary(candidate_probe),
        "paired_true_s_vs_shuffled_validation_improvement_db": _summary(true_vs_shuffled),
        "parameter_difference_fraction": float(parameter_difference),
        "structure_gate": {**structure_gate, "passed": all(structure_gate.values())},
        "test_open_gate": {**test_open_checks, "passed": all(test_open_checks.values())},
        "runs": runs,
    }
    (args.output_dir / "FORWARD_VALIDATION_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _make_plot(report, args.output_dir / "FORWARD_VALIDATION_SUMMARY.png")
    print(json.dumps({
        "status": report["status"],
        "candidate_validation": report["paired_candidate_vs_control_validation_improvement_db"],
        "candidate_probe": report["paired_candidate_vs_control_probe_improvement_db"],
        "true_s_vs_shuffled": report["paired_true_s_vs_shuffled_validation_improvement_db"],
        "structure_gate": report["structure_gate"],
        "test_open_gate": report["test_open_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
