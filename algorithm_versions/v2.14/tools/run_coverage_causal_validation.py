"""Run the preregistered nested train-only coverage-exchange validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V212_ROOT = ROOT.parent / "v2.12"
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(V212_ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import (  # noqa: E402
    assert_amplitude_only_training_contract,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    load_amplitude_split,
)
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.evaluation import amplitude_metrics, nearest_training_angle  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import operator_from_checkpoint  # noqa: E402
from v212.training import amplitude_objective  # noqa: E402


def _fine_angular_cell(direction: torch.Tensor) -> torch.Tensor:
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta = torch.rad2deg(torch.acos(direction[:, 2].clamp(-1.0, 1.0)))
    phi = torch.rad2deg(torch.atan2(direction[:, 1], direction[:, 0]))
    theta_bin = torch.floor(theta / 10.0).long().clamp(0, 17)
    phi_bin = torch.floor((phi + 90.0) / 10.0).long().clamp(0, 17)
    return theta_bin * 18 + phi_bin


def _maximin_per_cell(direction: torch.Tensor, per_cell: int) -> torch.Tensor:
    """Geometry-only deterministic maximin points inside each occupied 10-degree cell."""
    cell = _fine_angular_cell(direction)
    selected = []
    for cell_id in torch.unique(cell, sorted=True).tolist():
        members = torch.nonzero(cell == cell_id, as_tuple=False).flatten()
        take = min(per_cell, len(members) - 1)
        if take <= 0:
            continue
        sample = direction[members]
        sample = sample / sample.norm(dim=1, keepdim=True).clamp_min(1e-12)
        center = sample.mean(dim=0)
        center = center / center.norm().clamp_min(1e-12)
        chosen = [int(torch.argmax(sample @ center))]
        while len(chosen) < take:
            maximum_cosine = (sample @ sample[chosen].T).max(dim=1).values
            maximum_cosine[torch.as_tensor(chosen)] = 2.0
            chosen.append(int(torch.argmin(maximum_cosine)))
        selected.append(members[torch.as_tensor(chosen)])
    if not selected:
        raise ValueError("no augmentation points selected")
    return torch.cat(selected)


def _load_gap_model(run: Path, full_direction: torch.Tensor, device: str):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result.get("test_status") != "sealed" or checkpoint.get("test_split_opened") is not False:
        raise ValueError(f"source gap checkpoint is not sealed: {run}")
    q_keep = torch.as_tensor(checkpoint["train_query_indices"], dtype=torch.long)
    q_gap = torch.as_tensor(checkpoint["validation_query_indices"], dtype=torch.long)
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=full_direction[q_keep],
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(device)
    model.eval()
    return model, checkpoint, q_keep, q_gap


@torch.no_grad()
def _predict(model, commands, direction, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(direction), chunk_size):
        chunks.append(model.forward_command_grid(commands, direction[start:start + chunk_size]).T.cpu())
    return torch.cat(chunks)


def _fine_tune(
    model,
    direction,
    target,
    commands,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    db_floor: float,
):
    device = next(model.parameters()).device
    direction = direction.to(device)
    target = target.to(device)
    commands = commands.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 214300)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(direction), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size].to(device)
            prediction, terms = model.forward_command_grid_with_terms(commands, direction[index])
            objective = amplitude_objective(prediction.T, target[index], db_floor=db_floor)
            loss = objective.total + 1e-4 * terms["fourier_residual_iq"].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1)})
    model.eval()
    return history


def _summary(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_count": int(np.sum(array > 0)),
    }


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = [str(row["seed"]) for row in report["paired_results"]]
    nested = [row["nested_gap_improvement_db"] for row in report["paired_results"]]
    outer = [row["outer_validation_improvement_db"] for row in report["paired_results"]]
    distance = report["coverage_geometry"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    x = np.arange(len(seeds))
    axes[0].bar(x - 0.18, nested, 0.36, label="nested gap")
    axes[0].bar(x + 0.18, outer, 0.36, label="outer validation")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, seeds)
    axes[0].set_ylabel("paired NMSE improvement (dB)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(
        ["control", "coverage exchange"],
        [distance["control_nearest_angle_deg"]["median"], distance["coverage_nearest_angle_deg"]["median"]],
        color=["#bab0ac", "#59a14f"],
    )
    axes[1].set_ylabel("nested-gap nearest train angle median (deg)")
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.14 nested train-only coverage causal validation")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--gap-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21201, 21202, 21203])
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--db-floor", type=float, default=1e-8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "v214-coverage-causal-config-1",
        "dataset": str(args.dataset.resolve()), "split_dir": str(args.split_dir.resolve()),
        "codebook": str(args.codebook.resolve()), "gap_root": str(args.gap_root.resolve()),
        "seeds": args.seeds, "per_cell": args.per_cell, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "db_floor": args.db_floor, "external_test_used": False,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("output directory RUN_CONFIG does not match requested experiment")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    amplitude = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    w_train = codebook.indices("train")
    w_validation = codebook.indices("validation")
    q_outer_validation = torch.as_tensor(split.query_validation, dtype=torch.long)

    reference_run = args.gap_root / f"G2-J-LR240_sep0p001_seed{args.seeds[0]}_gap"
    reference_checkpoint = torch.load(reference_run / "checkpoint.pt", map_location="cpu", weights_only=True)
    q_keep = torch.as_tensor(reference_checkpoint["train_query_indices"], dtype=torch.long)
    q_gap = torch.as_tensor(reference_checkpoint["validation_query_indices"], dtype=torch.long)
    augmentation_local = _maximin_per_cell(direction[q_gap], args.per_cell)
    q_augmentation = q_gap[augmentation_local]
    evaluation_mask = torch.ones(len(q_gap), dtype=torch.bool)
    evaluation_mask[augmentation_local] = False
    q_nested_evaluation = q_gap[evaluation_mask]
    removal_generator = torch.Generator().manual_seed(21450)
    removal_local = torch.randperm(len(q_keep), generator=removal_generator)[:len(q_augmentation)]
    keep_mask = torch.ones(len(q_keep), dtype=torch.bool)
    keep_mask[removal_local] = False
    q_coverage_train = torch.cat((q_keep[keep_mask], q_augmentation))
    if len(q_coverage_train) != len(q_keep):
        raise AssertionError("coverage exchange must preserve training direction count")

    control_distance = torch.rad2deg(nearest_training_angle(direction[q_nested_evaluation], direction[q_keep]))
    coverage_distance = torch.rad2deg(nearest_training_angle(direction[q_nested_evaluation], direction[q_coverage_train]))
    distance_summary = lambda value: {
        "minimum": float(value.min()), "median": float(value.median()),
        "p95": float(torch.quantile(value, 0.95)), "maximum": float(value.max()),
    }
    paired_results = []
    for seed in args.seeds:
        run = args.gap_root / f"G2-J-LR240_sep0p001_seed{seed}_gap"
        _, _, seed_keep, seed_gap = _load_gap_model(run, direction, args.device)
        if not torch.equal(seed_keep, q_keep) or not torch.equal(seed_gap, q_gap):
            raise ValueError("gap split differs across paired seeds")
        arm_results = {}
        for arm, q_training in (("control", q_keep), ("coverage_exchange", q_coverage_train)):
            run_dir = args.output_dir / f"{arm}_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("external_test_used") is not False:
                    raise ValueError("saved run is not sealed")
                arm_results[arm] = saved
                continue
            model, checkpoint, _, _ = _load_gap_model(run, direction, args.device)
            history = _fine_tune(
                model, direction[q_training], amplitude[q_training][:, w_train], codebook.weights[w_train],
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, db_floor=args.db_floor,
            )
            nested_prediction = _predict(
                model, codebook.weights[w_validation].to(args.device),
                direction[q_nested_evaluation].to(args.device), args.chunk_size,
            )
            outer_prediction = _predict(
                model, codebook.weights[w_validation].to(args.device),
                direction[q_outer_validation].to(args.device), args.chunk_size,
            )
            payload = {
                "schema": "v214-coverage-causal-run-1", "arm": arm, "seed": seed,
                "training_direction_count": int(len(q_training)),
                "nested_evaluation_count": int(len(q_nested_evaluation)),
                "augmentation_count": int(len(q_augmentation)),
                "nested_gap_metrics": amplitude_metrics(
                    nested_prediction, amplitude[q_nested_evaluation][:, w_validation]
                ),
                "outer_validation_metrics": amplitude_metrics(
                    outer_prediction, amplitude[q_outer_validation][:, w_validation]
                ),
                "external_test_used": False,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.save({
                "schema": "v214-coverage-causal-checkpoint-1",
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "source_checkpoint": str(run / "checkpoint.pt"), "arm": arm, "seed": seed,
                "external_test_used": False,
            }, run_dir / "checkpoint.pt")
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            arm_results[arm] = payload
            print(json.dumps(payload, ensure_ascii=False))
        nested_improvement = (
            arm_results["control"]["nested_gap_metrics"]["nmse_db"]
            - arm_results["coverage_exchange"]["nested_gap_metrics"]["nmse_db"]
        )
        outer_improvement = (
            arm_results["control"]["outer_validation_metrics"]["nmse_db"]
            - arm_results["coverage_exchange"]["outer_validation_metrics"]["nmse_db"]
        )
        paired_results.append({
            "seed": seed, "nested_gap_improvement_db": nested_improvement,
            "outer_validation_improvement_db": outer_improvement,
            "control": arm_results["control"], "coverage_exchange": arm_results["coverage_exchange"],
        })

    nested_values = [row["nested_gap_improvement_db"] for row in paired_results]
    outer_values = [row["outer_validation_improvement_db"] for row in paired_results]
    equal_budget = all(
        row["control"]["training_direction_count"]
        == row["coverage_exchange"]["training_direction_count"]
        for row in paired_results
    )
    finite_metrics = bool(np.isfinite(nested_values + outer_values).all())
    passed = bool(
        np.mean(nested_values) >= 0.5
        and all(value > 0 for value in nested_values)
        and np.mean(outer_values) >= -0.1
        and min(outer_values) >= -0.5
        and equal_budget
        and finite_metrics
    )
    report = {
        "schema": "v214-coverage-causal-validation-1",
        "status": "coverage_hypothesis_pass" if passed else "coverage_hypothesis_fail",
        "external_test_used": False,
        "nested_train_only": True,
        "budget": {
            "control_training_directions": int(len(q_keep)),
            "coverage_training_directions": int(len(q_coverage_train)),
            "augmentation_directions": int(len(q_augmentation)),
            "nested_evaluation_directions": int(len(q_nested_evaluation)),
            "fine_10deg_cells_augmented": int(len(torch.unique(_fine_angular_cell(direction[q_augmentation])))),
        },
        "coverage_geometry": {
            "control_nearest_angle_deg": distance_summary(control_distance),
            "coverage_nearest_angle_deg": distance_summary(coverage_distance),
        },
        "paired_nested_gap_improvement_db": _summary(nested_values),
        "paired_outer_validation_improvement_db": _summary(outer_values),
        "acceptance": {
            "nested_mean_at_least_0p5": bool(np.mean(nested_values) >= 0.5),
            "nested_3_of_3_positive": bool(all(value > 0 for value in nested_values)),
            "outer_mean_noninferior_minus0p1": bool(np.mean(outer_values) >= -0.1),
            "outer_each_noncatastrophic_minus0p5": bool(min(outer_values) >= -0.5),
            "equal_training_budget": equal_budget,
            "all_primary_metrics_finite": finite_metrics,
            "external_test_sealed": True,
        },
        "paired_results": paired_results,
        "claim_boundary": "Supports or rejects targeted labeled angular coverage within nested query_train only; no model qualification or hardware claim.",
    }
    summary_path = args.output_dir / "COVERAGE_CAUSAL_SUMMARY.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_plot(report, args.output_dir / "COVERAGE_CAUSAL_SUMMARY.png")
    print(json.dumps({key: report[key] for key in (
        "status", "budget", "coverage_geometry", "paired_nested_gap_improvement_db",
        "paired_outer_validation_improvement_db", "acceptance"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
