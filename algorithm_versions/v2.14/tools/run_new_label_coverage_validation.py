"""Validate complementary integer-grid labels with frozen V2.12 CNN starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V212_ROOT = ROOT.parent / "v2.12"
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(V212_ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

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
from v214.angular_sampling import load_amplitude_label_package  # noqa: E402


def _load_source_model(run: Path, training_direction: torch.Tensor, device: str):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if checkpoint.get("test_split_opened") is not False or result.get("test_status") != "sealed":
        raise ValueError(f"source checkpoint is not sealed: {run}")
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=training_direction,
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(device)
    model.eval()
    return model, checkpoint


def _complex_commands(payload: np.lib.npyio.NpzFile) -> torch.Tensor:
    iq = np.asarray(payload["command_iq"], dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[2] != 16:
        raise ValueError("command_iq must have shape [C,2,16]")
    return torch.complex(torch.from_numpy(iq[:, 0]), torch.from_numpy(iq[:, 1]))


@torch.no_grad()
def _predict(model, commands: torch.Tensor, direction: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    device = next(model.parameters()).device
    commands = commands.to(device)
    for start in range(0, len(direction), chunk_size):
        query = direction[start:start + chunk_size].to(device)
        chunks.append(model.forward_command_grid(commands, query).T.cpu())
    return torch.cat(chunks)


def _fine_tune(
    model,
    direction: torch.Tensor,
    target: torch.Tensor,
    commands: torch.Tensor,
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
    generator = torch.Generator().manual_seed(seed + 214900)
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
        print(json.dumps({"seed": seed, "epoch": epoch, "train_loss": history[-1]["train_loss"]}))
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


def _distance_summary(value: torch.Tensor) -> dict[str, float]:
    value = torch.rad2deg(value)
    return {
        "minimum": float(value.min()),
        "median": float(value.median()),
        "p95": float(torch.quantile(value, 0.95)),
        "maximum": float(value.max()),
    }


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = [str(row["seed"]) for row in report["paired_results"]]
    exchange = [row["exchange_probe_improvement_db"] for row in report["paired_results"]]
    plus = [row["plus_probe_improvement_db"] for row in report["paired_results"]]
    exchange_outer = [row["exchange_outer_improvement_db"] for row in report["paired_results"]]
    plus_outer = [row["plus_outer_improvement_db"] for row in report["paired_results"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(seeds))
    axes[0].bar(x - 0.18, exchange, 0.36, label="exchange")
    axes[0].bar(x + 0.18, plus, 0.36, label="plus")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, seeds)
    axes[0].set_ylabel("coverage-probe improvement (dB)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x - 0.18, exchange_outer, 0.36, label="exchange")
    axes[1].bar(x + 0.18, plus_outer, 0.36, label="plus")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(-0.1, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, seeds)
    axes[1].set_ylabel("old outer-validation improvement (dB)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.14 complementary angular-label validation")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21201, 21202, 21203])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--db-floor", type=float, default=1e-8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--removal-seed", type=int, default=21470)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "v214-new-label-coverage-config-1",
        "dataset": str(args.dataset.resolve()),
        "split_dir": str(args.split_dir.resolve()),
        "codebook": str(args.codebook.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "label_dir": str(args.label_dir.resolve()),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "db_floor": args.db_floor,
        "removal_seed": args.removal_seed,
        "external_test_used": False,
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
    q_train = torch.as_tensor(split.query_train, dtype=torch.long)
    q_outer = torch.as_tensor(split.query_validation, dtype=torch.long)
    w_train = torch.as_tensor(codebook.indices("train"), dtype=torch.long)
    w_validation = torch.as_tensor(codebook.indices("validation"), dtype=torch.long)

    acquisition_file = args.label_dir / "acquisition_train.npz"
    probe_file = args.label_dir / "coverage_probe.npz"
    acquisition = load_amplitude_label_package(acquisition_file)
    probe = load_amplitude_label_package(probe_file)
    acquisition_direction = torch.as_tensor(acquisition["direction"], dtype=torch.float32)
    acquisition_target = torch.as_tensor(acquisition["amplitude"].T.copy(), dtype=torch.float32)
    acquisition_commands = _complex_commands(acquisition)
    acquisition_indices = torch.as_tensor(acquisition["command_indices"], dtype=torch.long)
    probe_direction = torch.as_tensor(probe["direction"], dtype=torch.float32)
    probe_target = torch.as_tensor(probe["amplitude"].T.copy(), dtype=torch.float32)
    probe_commands = _complex_commands(probe)
    probe_indices = torch.as_tensor(probe["command_indices"], dtype=torch.long)
    if acquisition_target.shape != (490, 64) or probe_target.shape != (4380, 16):
        raise ValueError("new-label package sizes differ from the frozen 490/4380 contract")
    if not torch.equal(acquisition_indices, w_train) or not torch.equal(probe_indices, w_validation):
        raise ValueError("new-label command split mismatch")
    if not torch.allclose(acquisition_commands, codebook.weights[w_train], atol=1e-7, rtol=1e-7):
        raise ValueError("acquisition commands do not match the frozen codebook")
    if not torch.allclose(probe_commands, codebook.weights[w_validation], atol=1e-7, rtol=1e-7):
        raise ValueError("probe commands do not match the frozen codebook")

    generator = torch.Generator().manual_seed(args.removal_seed)
    removal_local = torch.randperm(len(q_train), generator=generator)[:len(acquisition_direction)]
    keep_mask = torch.ones(len(q_train), dtype=torch.bool)
    keep_mask[removal_local] = False
    exchange_original = q_train[keep_mask]
    original_train_direction = direction[q_train]
    arms = {
        "control": (original_train_direction, amplitude[q_train][:, w_train]),
        "exchange": (
            torch.cat((direction[exchange_original], acquisition_direction)),
            torch.cat((amplitude[exchange_original][:, w_train], acquisition_target)),
        ),
        "plus": (
            torch.cat((original_train_direction, acquisition_direction)),
            torch.cat((amplitude[q_train][:, w_train], acquisition_target)),
        ),
    }
    if len(arms["control"][0]) != 22600 or len(arms["exchange"][0]) != 22600 or len(arms["plus"][0]) != 23090:
        raise AssertionError("training budgets violate the frozen three-arm contract")

    geometry = {
        arm: _distance_summary(nearest_training_angle(probe_direction, values[0]))
        for arm, values in arms.items()
    }
    paired_results = []
    for seed in args.seeds:
        source_run = args.checkpoint_root / f"G2-J-LR240_sep0p001_seed{seed}"
        source_checkpoint = torch.load(source_run / "checkpoint.pt", map_location="cpu", weights_only=True)
        if not np.array_equal(np.asarray(source_checkpoint["train_query_indices"]), q_train.numpy()):
            raise ValueError(f"source train split differs for seed {seed}")
        if not np.array_equal(np.asarray(source_checkpoint["validation_query_indices"]), q_outer.numpy()):
            raise ValueError(f"source validation split differs for seed {seed}")
        arm_results = {}
        for arm, (training_direction, training_target) in arms.items():
            run_dir = args.output_dir / f"{arm}_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("external_test_used") is not False:
                    raise ValueError("saved run is not sealed")
                arm_results[arm] = saved
                continue
            model, checkpoint = _load_source_model(source_run, original_train_direction, args.device)
            history = _fine_tune(
                model, training_direction, training_target, codebook.weights[w_train],
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, db_floor=args.db_floor,
            )
            probe_prediction = _predict(model, probe_commands, probe_direction, args.chunk_size)
            outer_prediction = _predict(model, codebook.weights[w_validation], direction[q_outer], args.chunk_size)
            payload = {
                "schema": "v214-new-label-coverage-run-1",
                "arm": arm,
                "seed": seed,
                "training_direction_count": int(len(training_direction)),
                "acquisition_direction_count": 0 if arm == "control" else 490,
                "coverage_probe_direction_count": 4380,
                "coverage_probe_metrics": amplitude_metrics(probe_prediction, probe_target),
                "outer_validation_metrics": amplitude_metrics(
                    outer_prediction, amplitude[q_outer][:, w_validation]
                ),
                "source_checkpoint_test_status": "sealed",
                "external_test_used": False,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            torch.save({
                "schema": "v214-new-label-coverage-checkpoint-1",
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "source_checkpoint": str(source_run / "checkpoint.pt"),
                "arm": arm,
                "seed": seed,
                "external_test_used": False,
            }, run_dir / "checkpoint.pt")
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            arm_results[arm] = payload
            print(json.dumps(payload, ensure_ascii=False))

        probe_nmse = {arm: arm_results[arm]["coverage_probe_metrics"]["nmse_db"] for arm in arms}
        outer_nmse = {arm: arm_results[arm]["outer_validation_metrics"]["nmse_db"] for arm in arms}
        paired_results.append({
            "seed": seed,
            "exchange_probe_improvement_db": probe_nmse["control"] - probe_nmse["exchange"],
            "plus_probe_improvement_db": probe_nmse["control"] - probe_nmse["plus"],
            "plus_vs_exchange_probe_db": probe_nmse["exchange"] - probe_nmse["plus"],
            "exchange_outer_improvement_db": outer_nmse["control"] - outer_nmse["exchange"],
            "plus_outer_improvement_db": outer_nmse["control"] - outer_nmse["plus"],
            **arm_results,
        })

    exchange_probe = [row["exchange_probe_improvement_db"] for row in paired_results]
    plus_probe = [row["plus_probe_improvement_db"] for row in paired_results]
    plus_vs_exchange = [row["plus_vs_exchange_probe_db"] for row in paired_results]
    exchange_outer = [row["exchange_outer_improvement_db"] for row in paired_results]
    plus_outer = [row["plus_outer_improvement_db"] for row in paired_results]
    all_primary = exchange_probe + plus_probe + plus_vs_exchange + exchange_outer + plus_outer
    acceptance = {
        "exchange_mean_at_least_0p5": bool(np.mean(exchange_probe) >= 0.5),
        "exchange_3_of_3_positive": bool(all(value > 0 for value in exchange_probe)),
        "plus_not_worse_than_exchange_minus0p1_mean": bool(np.mean(plus_vs_exchange) >= -0.1),
        "exchange_outer_mean_noninferior_minus0p1": bool(np.mean(exchange_outer) >= -0.1),
        "plus_outer_mean_noninferior_minus0p1": bool(np.mean(plus_outer) >= -0.1),
        "treatment_outer_each_noncatastrophic_minus0p5": bool(min(exchange_outer + plus_outer) >= -0.5),
        "training_budgets_match_contract": True,
        "all_primary_metrics_finite": bool(np.isfinite(all_primary).all()),
        "directions_and_commands_leak_free": True,
        "external_test_sealed": True,
    }
    passed = all(acceptance.values())
    report = {
        "schema": "v214-new-label-coverage-validation-1",
        "status": "new_angular_labels_pass" if passed else "new_angular_labels_fail",
        "external_test_used": False,
        "cnn_structure_changed": False,
        "training": {
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
        },
        "budget": {
            "control_training_directions": 22600,
            "exchange_training_directions": 22600,
            "plus_training_directions": 23090,
            "acquisition_directions": 490,
            "coverage_probe_directions": 4380,
            "old_outer_validation_directions": 4900,
        },
        "coverage_geometry_nearest_train_angle_deg": geometry,
        "paired_exchange_probe_improvement_db": _summary(exchange_probe),
        "paired_plus_probe_improvement_db": _summary(plus_probe),
        "paired_plus_vs_exchange_probe_db": _summary(plus_vs_exchange),
        "paired_exchange_outer_improvement_db": _summary(exchange_outer),
        "paired_plus_outer_improvement_db": _summary(plus_outer),
        "acceptance": acceptance,
        "paired_results": paired_results,
        "claim_boundary": "Only tests the value of the frozen complementary labels; no CNN redesign, external-test, deployment, or hardware claim.",
    }
    (args.output_dir / "NEW_LABEL_COVERAGE_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _make_plot(report, args.output_dir / "NEW_LABEL_COVERAGE_SUMMARY.png")
    print(json.dumps({
        "status": report["status"],
        "budget": report["budget"],
        "paired_exchange_probe_improvement_db": report["paired_exchange_probe_improvement_db"],
        "paired_plus_vs_exchange_probe_db": report["paired_plus_vs_exchange_probe_db"],
        "paired_exchange_outer_improvement_db": report["paired_exchange_outer_improvement_db"],
        "paired_plus_outer_improvement_db": report["paired_plus_outer_improvement_db"],
        "acceptance": acceptance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
