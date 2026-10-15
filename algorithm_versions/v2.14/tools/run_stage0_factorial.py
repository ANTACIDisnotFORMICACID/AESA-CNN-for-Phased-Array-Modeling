"""Run the preregistered V2.14 2x2 paired, validation-only Stage-0 screen."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V212_ROOT = ROOT.parent / "v2.12"
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
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
from v212.evaluation import amplitude_metrics  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.training import amplitude_objective  # noqa: E402
from v212.representation import tangent_frame  # noqa: E402
from v214.operator import CrossBranchFusionOperator  # noqa: E402


CELLS = ((False, False), (True, False), (False, True), (True, True))


def _candidate_run(soft: Path, formal: Path, candidate_fix: Path, seed: int) -> Path:
    root = soft if seed <= 21203 else (candidate_fix if seed >= 21209 else formal)
    return root / f"G2-J-LR240_sep0p001_seed{seed}"


def _load_model(run: Path, training_direction: torch.Tensor, cross_access: bool, device: str):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("test_split_opened") is not False:
        raise ValueError(f"external test state is not sealed: {run}")
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    torch.manual_seed(int(checkpoint["seed"]) + 214000)
    model = CrossBranchFusionOperator(
        ssh_transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
        envelope_scale=float(checkpoint["envelope_scale"]),
        ssh_seed=int(checkpoint["seed"]),
        fourier_input_mode=str(checkpoint.get("fourier_input_mode", "raw")),
        training_direction=training_direction,
        residual_enabled=True,
        cross_branch_access=cross_access,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if unexpected or set(missing) != {
        "fusion.geometry_maps", "fusion.stem.weight", "fusion.stem.bias",
        "fusion.head.weight", "fusion.head.bias",
    }:
        raise ValueError(f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, checkpoint


def _shift(direction: torch.Tensor, angle_deg: float, step: int) -> torch.Tensor:
    first, second = tangent_frame(direction)
    tangent = first if (step // 2) % 2 == 0 else second
    sign = 1.0 if step % 2 == 0 else -1.0
    angle = torch.as_tensor(math.radians(angle_deg), dtype=direction.dtype, device=direction.device)
    return torch.cos(angle) * direction + sign * torch.sin(angle) * tangent


@torch.no_grad()
def _predict(model, commands, direction, chunk_size: int) -> torch.Tensor:
    pieces = []
    for start in range(0, len(direction), chunk_size):
        pieces.append(model.forward_command_grid(commands, direction[start:start + chunk_size]).T.cpu())
    return torch.cat(pieces)


@torch.no_grad()
def _envelope_sensitivity(model, commands, direction, angle_deg: float, chunk_size: int) -> float:
    baseline = _predict(model, commands, direction, chunk_size)
    shifted_predictions = []
    first, second = tangent_frame(direction)
    angle = torch.as_tensor(math.radians(angle_deg), dtype=direction.dtype, device=direction.device)
    for tangent in (first, -first, second, -second):
        shifted = torch.cos(angle) * direction + torch.sin(angle) * tangent
        pieces = []
        for start in range(0, len(direction), chunk_size):
            pieces.append(model.forward_factorized_grid(
                commands,
                shifted[start:start + chunk_size],
                direction[start:start + chunk_size],
            ).T.cpu())
        shifted_predictions.append(torch.cat(pieces))
    stack = torch.stack(shifted_predictions)
    ratio = (stack - baseline[None]).square().sum() / baseline.square().sum().clamp_min(1e-30) / 4.0
    return float(10.0 * torch.log10(ratio.clamp_min(1e-30)))


@torch.no_grad()
def _fusion_energy_fraction(model, direction, chunk_size: int) -> float:
    numerator = denominator = 0.0
    from v210.contracts import QueryBatch
    for start in range(0, len(direction), chunk_size):
        envelope, terms = model.envelope(QueryBatch(direction[start:start + chunk_size][None]))
        numerator += float(terms["fusion_delta_iq"].square().sum())
        denominator += float(envelope.abs().square().sum())
    return numerator / max(denominator, 1e-30)


def _train_cell(
    model,
    train_direction,
    train_target,
    train_commands,
    validation_direction,
    validation_target,
    validation_commands,
    *,
    jitter: bool,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    jitter_angle_deg: float,
    jitter_weight: float,
    jitter_interval: int,
    chunk_size: int,
    db_floor: float,
):
    device = next(model.parameters()).device
    train_direction = train_direction.to(device)
    train_target = train_target.to(device)
    train_commands = train_commands.to(device)
    validation_direction_device = validation_direction.to(device)
    validation_commands_device = validation_commands.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 214100)
    best_nmse = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_direction), generator=generator)
        total_loss = total_jitter = 0.0
        seen = jitter_updates = 0
        for start in range(0, len(permutation), batch_size):
            global_step += 1
            index = permutation[start:start + batch_size].to(device)
            direction = train_direction[index]
            target = train_target[index]
            prediction, terms = model.forward_command_grid_with_terms(train_commands, direction)
            objective = amplitude_objective(prediction.T, target, db_floor=db_floor)
            residual_regularization = 1e-4 * terms["fourier_residual_iq"].square().mean()
            loss = objective.total + residual_regularization
            jitter_loss = torch.zeros((), dtype=loss.dtype, device=device)
            if jitter and global_step % jitter_interval == 0:
                shifted = _shift(direction, jitter_angle_deg, global_step)
                jitter_prediction = model.forward_factorized_grid(
                    train_commands, shifted, direction
                ).T
                jitter_loss = (jitter_prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)
                loss = loss + jitter_interval * jitter_weight * jitter_loss
                total_jitter += float(jitter_loss.detach())
                jitter_updates += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
            seen += len(index)
        model.eval()
        validation_prediction = _predict(
            model, validation_commands_device, validation_direction_device, chunk_size
        )
        validation_nmse = float(
            (validation_prediction - validation_target).square().sum()
            / validation_target.square().sum().clamp_min(1e-12)
        )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "jitter_linear_nmse": total_jitter / max(jitter_updates, 1),
            "validation_nmse_db": float(10.0 * math.log10(max(validation_nmse, 1e-30))),
        })
        if validation_nmse < best_nmse:
            best_nmse = validation_nmse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


def _summary(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_count": int(np.sum(array > 0)),
    }


def _make_plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    cells = ["C00", "C10", "C01", "C11"]
    nmse = [report["cell_summary"][cell]["validation_nmse_db"]["mean"] for cell in cells]
    improvement = [report["cell_summary"][cell]["paired_validation_improvement_vs_C00_db"]["mean"] for cell in cells]
    sensitivity = [report["cell_summary"][cell]["paired_sensitivity_improvement_vs_C00_db"]["mean"] for cell in cells]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].bar(cells, nmse, color="#4c78a8")
    axes[0].set_ylim(min(nmse) - 0.05, max(nmse) + 0.05)
    axes[0].set_ylabel("validation NMSE (dB)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(cells, improvement, color="#59a14f")
    axes[1].axhline(0.15, color="black", linestyle="--", linewidth=1, label="screen threshold")
    axes[1].set_ylabel("improvement vs C00 (dB)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[2].bar(cells, sensitivity, color="#e15759")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1, label="screen threshold")
    axes[2].set_ylabel("0.5 deg envelope robustness gain (dB)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.14 Stage-0 parameter-matched 2x2 screen")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--soft-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--candidate-fix-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21201, 21202, 21203])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--jitter-angle-deg", type=float, default=0.25)
    parser.add_argument("--jitter-weight", type=float, default=0.05)
    parser.add_argument("--jitter-interval", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--db-floor", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "v214-stage0-run-config-1",
        "dataset": str(args.dataset.resolve()),
        "split_dir": str(args.split_dir.resolve()),
        "codebook": str(args.codebook.resolve()),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "jitter_angle_deg": args.jitter_angle_deg,
        "jitter_weight": args.jitter_weight,
        "jitter_interval": args.jitter_interval,
        "db_floor": args.db_floor,
        "external_test_used": False,
    }
    run_config_path = args.output_dir / "RUN_CONFIG.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError("output directory RUN_CONFIG does not match requested experiment")
    else:
        run_config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    amplitude = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    q_train = torch.as_tensor(split.query_train, dtype=torch.long)
    q_validation = torch.as_tensor(split.query_validation, dtype=torch.long)
    w_train = codebook.indices("train")
    w_validation = codebook.indices("validation")
    results = []
    for seed in args.seeds:
        source_run = _candidate_run(args.soft_root, args.formal_root, args.candidate_fix_root, seed)
        for cross_access, jitter in CELLS:
            cell = f"C{int(cross_access)}{int(jitter)}"
            run_dir = args.output_dir / f"{cell}_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    saved.get("cell") == cell
                    and saved.get("seed") == seed
                    and saved.get("external_test_used") is False
                ):
                    results.append(saved)
                    print(json.dumps({"reused_completed_run": str(result_path)}, ensure_ascii=False))
                    continue
            model, checkpoint = _load_model(source_run, direction[q_train], cross_access, args.device)
            initial_parameter_count = sum(parameter.numel() for parameter in model.parameters())
            history, best_epoch = _train_cell(
                model,
                direction[q_train], amplitude[q_train][:, w_train], codebook.weights[w_train],
                direction[q_validation], amplitude[q_validation][:, w_validation], codebook.weights[w_validation],
                jitter=jitter, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, jitter_angle_deg=args.jitter_angle_deg,
                jitter_weight=args.jitter_weight, jitter_interval=args.jitter_interval,
                chunk_size=args.chunk_size, db_floor=args.db_floor,
            )
            model.eval()
            prediction = _predict(
                model, codebook.weights[w_validation].to(args.device),
                direction[q_validation].to(args.device), args.chunk_size,
            )
            metrics = amplitude_metrics(prediction, amplitude[q_validation][:, w_validation])
            sensitivity = _envelope_sensitivity(
                model, codebook.weights[w_validation].to(args.device),
                direction[q_validation].to(args.device), 0.5, args.chunk_size,
            )
            fusion_energy = _fusion_energy_fraction(
                model, direction[q_validation].to(args.device), args.chunk_size
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "v214-stage0-checkpoint-1",
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "source_checkpoint": str(source_run / "checkpoint.pt"),
                "cross_branch_access": cross_access,
                "jitter_regularization": jitter,
                "seed": seed,
                "external_test_used": False,
            }, run_dir / "checkpoint.pt")
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            row = {
                "cell": cell,
                "seed": seed,
                "cross_branch_access": cross_access,
                "jitter_regularization": jitter,
                "parameter_count": initial_parameter_count,
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "envelope_self_change_0p5deg_nmse_db": sensitivity,
                "fusion_delta_energy_fraction": fusion_energy,
                "external_test_used": False,
            }
            (run_dir / "result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    lookup = {(row["seed"], row["cell"]): row for row in results}
    effects = {"access_nmse_db": [], "jitter_nmse_db": [], "interaction_nmse_db": []}
    comparisons = []
    for seed in args.seeds:
        value = {cell: lookup[(seed, cell)]["validation_metrics"]["nmse_db"] for cell in ("C00", "C10", "C01", "C11")}
        access = 0.5 * ((value["C00"] - value["C10"]) + (value["C01"] - value["C11"]))
        jitter = 0.5 * ((value["C00"] - value["C01"]) + (value["C10"] - value["C11"]))
        interaction = (value["C10"] - value["C11"]) - (value["C00"] - value["C01"])
        effects["access_nmse_db"].append(access)
        effects["jitter_nmse_db"].append(jitter)
        effects["interaction_nmse_db"].append(interaction)
        comparisons.append({"seed": seed, "cell_nmse_db": value, "access_effect_db": access, "jitter_effect_db": jitter, "interaction_db": interaction})

    cell_summary = {}
    c00 = {seed: lookup[(seed, "C00")] for seed in args.seeds}
    expand_cells = []
    for cell in ("C00", "C10", "C01", "C11"):
        nmse = [lookup[(seed, cell)]["validation_metrics"]["nmse_db"] for seed in args.seeds]
        sensitivity = [lookup[(seed, cell)]["envelope_self_change_0p5deg_nmse_db"] for seed in args.seeds]
        improvement = [c00[seed]["validation_metrics"]["nmse_db"] - lookup[(seed, cell)]["validation_metrics"]["nmse_db"] for seed in args.seeds]
        sensitivity_improvement = [c00[seed]["envelope_self_change_0p5deg_nmse_db"] - lookup[(seed, cell)]["envelope_self_change_0p5deg_nmse_db"] for seed in args.seeds]
        qualifies = bool(
            cell != "C00"
            and (np.mean(improvement) >= 0.15 or np.mean(sensitivity_improvement) >= 0.5)
            and np.mean(improvement) >= -0.10
            and min(improvement) >= -0.50
        )
        if qualifies:
            expand_cells.append(cell)
        cell_summary[cell] = {
            "validation_nmse_db": _summary(nmse),
            "envelope_self_change_0p5deg_nmse_db": _summary(sensitivity),
            "paired_validation_improvement_vs_C00_db": _summary(improvement),
            "paired_sensitivity_improvement_vs_C00_db": _summary(sensitivity_improvement),
            "passes_stage0_expansion_rule": qualifies,
        }
    report = {
        "schema": "v214-stage0-factorial-screen-1",
        "status": "screen_complete_expand" if expand_cells else "screen_complete_stop",
        "external_test_used": False,
        "source_model": "frozen V2.12-S G2-J-LR240 sep=0.001 checkpoints",
        "screen_only_not_model_qualification": True,
        "configuration": {
            "seeds": args.seeds, "epochs": args.epochs, "learning_rate": args.learning_rate,
            "jitter_angle_deg": args.jitter_angle_deg, "jitter_weight": args.jitter_weight,
            "jitter_interval": args.jitter_interval,
            "db_floor": args.db_floor,
        },
        "parameter_counts": sorted(set(row["parameter_count"] for row in results)),
        "cell_summary": cell_summary,
        "factorial_effects_positive_is_better": {name: _summary(values) for name, values in effects.items()},
        "expand_cells": expand_cells,
        "paired_comparisons": comparisons,
        "runs": results,
    }
    (args.output_dir / "STAGE0_FACTORIAL_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _make_plot(report, args.output_dir / "STAGE0_FACTORIAL_SUMMARY.png")
    print(json.dumps({key: report[key] for key in ("status", "parameter_counts", "cell_summary", "factorial_effects_positive_is_better", "expand_cells")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
