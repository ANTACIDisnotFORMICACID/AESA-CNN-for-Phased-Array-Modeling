"""Open the frozen external test once for the preregistered V2.15 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v210.io import assert_amplitude_only_training_contract, assert_split_matches_dataset, load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.evaluation import amplitude_metrics  # noqa: E402
from v215.experiment import PRIMARY_CANDIDATE, SEEDS, load_model_from_run  # noqa: E402


def _summary(values) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    half_width = 4.3026527297 * float(value.std(ddof=1)) / np.sqrt(len(value)) if len(value) > 1 else 0.0
    return {
        "mean": float(value.mean()),
        "sample_standard_deviation": float(value.std(ddof=1)) if len(value) > 1 else 0.0,
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "positive_count": int(np.sum(value > 0)),
        "student_t_95_percent_ci": [float(value.mean() - half_width), float(value.mean() + half_width)],
    }


@torch.no_grad()
def _predict(model, commands, direction, chunk_size):
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(direction), chunk_size):
        result.append(model.forward_command_grid(
            commands.to(device), direction[start:start + chunk_size].to(device)
        ).T.cpu())
    return torch.cat(result)


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = [str(row["seed"]) for row in report["paired_results"]]
    improvement = [row["candidate_test_improvement_db"] for row in report["paired_results"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(seeds, improvement, color="#4e79a7")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.3, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("paired seed")
    axis.set_ylabel("test NMSE improvement (dB)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.15 one-time external test")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--physics-data", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--forward-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "FINAL_TEST_SUMMARY.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
        return

    summary_bytes = args.forward_summary.read_bytes()
    forward = json.loads(summary_bytes.decode("utf-8"))
    if forward.get("predeclared_primary_candidate") != PRIMARY_CANDIDATE:
        raise ValueError("forward summary candidate differs from preregistration")
    if forward.get("test_used") is not False or forward["test_open_gate"].get("passed") is not True:
        raise ValueError("protocol gate does not authorize the one-time test")
    opening = {
        "schema": "v215-one-time-test-opening-1",
        "predeclared_candidate": PRIMARY_CANDIDATE,
        "evaluated_arms": ["control", PRIMARY_CANDIDATE],
        "seeds": list(SEEDS),
        "forward_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "selection_and_checkpoints_frozen_before_test_read": True,
        "test_will_not_be_used_for_further_v215_tuning": True,
    }
    (args.output_dir / "TEST_OPENING_RECORD.json").write_text(
        json.dumps(opening, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # This is the only point in the V2.15 pipeline that loads and indexes q_test/w_test.
    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    amplitude = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    q_test = torch.as_tensor(split.query_test, dtype=torch.long)
    w_test = codebook.indices("test")
    test_direction = direction[q_test]
    test_target = amplitude[q_test][:, w_test]
    test_commands = codebook.weights[w_test]
    with np.load(args.physics_data, allow_pickle=False) as physics:
        s_matrix = np.asarray(physics["s_parameters_47ghz"], dtype=np.complex128)

    paired = []
    for seed in SEEDS:
        seed_result = {}
        for arm in ("control", PRIMARY_CANDIDATE):
            run_dir = args.training_dir / f"{arm}_seed{seed}"
            model, checkpoint = load_model_from_run(run_dir, s_matrix, args.device)
            if checkpoint.get("checkpoint_selection") != "validation_linear_nmse_only":
                raise ValueError("checkpoint was not selected only by validation")
            prediction = _predict(model, test_commands, test_direction, args.chunk_size)
            if not torch.isfinite(prediction).all() or torch.any(prediction < 0):
                raise ValueError("test prediction is invalid")
            seed_result[arm] = amplitude_metrics(prediction, test_target)
        improvement = seed_result["control"]["nmse_db"] - seed_result[PRIMARY_CANDIDATE]["nmse_db"]
        paired.append({
            "seed": seed,
            "candidate_test_improvement_db": improvement,
            "control": seed_result["control"],
            PRIMARY_CANDIDATE: seed_result[PRIMARY_CANDIDATE],
        })
        print(json.dumps({"seed": seed, "test_improvement_db": improvement}), flush=True)

    improvements = [row["candidate_test_improvement_db"] for row in paired]
    validation_gate = forward["structure_gate"]
    qualification = {
        "test_mean_improvement_at_least_0p30": bool(np.mean(improvements) >= 0.30),
        "test_3_of_3_positive": bool(all(value > 0 for value in improvements)),
        "validation_candidate_mean_at_least_0p30": bool(validation_gate["candidate_validation_mean_at_least_0p30"]),
        "validation_candidate_3_of_3_positive": bool(validation_gate["candidate_validation_3_of_3_positive"]),
        "no_test_leakage": True,
    }
    report = {
        "schema": "v215-final-test-summary-1",
        "status": "v215_candidate_qualified" if all(qualification.values()) else "v215_candidate_not_qualified",
        "test_opened": True,
        "test_opened_once_after_freeze": True,
        "test_directions": int(len(q_test)),
        "test_commands": int(len(w_test)),
        "evaluated_arms": ["control", PRIMARY_CANDIDATE],
        "paired_candidate_test_improvement_db": _summary(improvements),
        "qualification": qualification,
        "paired_results": paired,
        "claim_boundary": "HFSS Ephi amplitude held-out test only; no hardware or OTA qualification.",
    }
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_plot(report, args.output_dir / "FINAL_TEST_SUMMARY.png")
    print(json.dumps({
        "status": report["status"],
        "paired_candidate_test_improvement_db": report["paired_candidate_test_improvement_db"],
        "qualification": qualification,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
