"""Train G1, G2 or diagnostic G3; select on validation D without opening test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import (  # noqa: E402
    assert_amplitude_only_training_contract,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    load_amplitude_split,
)
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.distillation import OperatorTarget  # noqa: E402
from v212.encoding import MatchedFourierBasis  # noqa: E402
from v212.operator import CoherentRank1Operator  # noqa: E402
from v212.evaluation import (  # noqa: E402
    amplitude_metrics,
    spatial_gap_partition,
    stratified_separation_subset,
)
from v212.training import OperatorTrainConfig, train_operator_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--recipe",
        choices=(
            "G1", "G2", "G3", "G1-LR240", "G1-M5-LR240", "G2-J-LR240",
            "G2-I-LR240", "G2-T", "G2-TI", "SSH-only",
        ),
        required=True,
    )
    parser.add_argument("--ssh-degree", choices=(4, 5), type=int, default=4)
    parser.add_argument("--operator-target", type=Path)
    parser.add_argument("--lambda-q", type=float, choices=(0.01, 0.1, 1.0), default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lambda-separation", type=float, choices=(0.0, 0.001, 0.01, 0.05), default=0.0)
    parser.add_argument("--spatial-gap", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.recipe == "G3" and args.operator_target is None:
        raise ValueError("G3 requires --operator-target")
    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    codebook = MeasurementCodebook.load(args.codebook)
    dataset_commands = torch.complex(
        torch.as_tensor(dataset.command_iq[:, 0].reshape(96, 16)),
        torch.as_tensor(dataset.command_iq[:, 1].reshape(96, 16)),
    )
    torch.testing.assert_close(dataset_commands, codebook.weights, atol=1e-6, rtol=1e-6)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    amplitude = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    q_train = torch.as_tensor(split.query_train, dtype=torch.long)
    q_validation = torch.as_tensor(split.query_validation, dtype=torch.long)
    direction_protocol = "frozen_validation_D"
    if args.spatial_gap:
        keep_local, gap_local = spatial_gap_partition(direction[q_train])
        q_validation = q_train[gap_local]
        q_train = q_train[keep_local]
        direction_protocol = "q_train_fold0_spatial_gap"
    w_train = codebook.indices("train")
    w_validation = codebook.indices("validation")
    single_indices = torch.tensor(
        [
            index
            for index in codebook.splits["train"]
            if codebook.kinds[index] == "single"
        ],
        dtype=torch.long,
    )
    if len(single_indices) != 16:
        raise ValueError("training codebook must contain all 16 single-element commands")
    envelope_scale = float(amplitude[q_train][:, single_indices].median())
    if not envelope_scale > 0:
        raise ValueError("training-only envelope scale must be positive")
    ssh_recipes = {"G2", "G3", "G2-J-LR240", "G2-I-LR240", "G2-T", "G2-TI", "SSH-only"}
    matched_recipe = args.recipe == "G1-M5-LR240"
    transform = None
    if args.recipe in ssh_recipes or matched_recipe:
        transform = FiltrationSSHTransform.fit(direction[q_train].double(), args.ssh_degree)
    matched_basis = None
    if matched_recipe:
        matched_basis = MatchedFourierBasis(
            direction[q_train], transform.ranks, extended_bands=6, seed=args.seed
        )
    fourier_input_mode = (
        "filtration_residualized" if args.recipe in {"G2-I-LR240", "G2-TI"} else "raw"
    )
    residual_enabled = args.recipe != "SSH-only"
    torch.manual_seed(args.seed)
    model = CoherentRank1Operator(
        ssh_transform=transform if args.recipe in ssh_recipes else None,
        ssh_seed=args.seed,
        envelope_scale=envelope_scale,
        fourier_input_mode=fourier_input_mode,
        training_direction=direction[q_train] if fourier_input_mode != "raw" else None,
        matched_fourier_basis=matched_basis,
        residual_enabled=residual_enabled,
    ).to(args.device)
    target_q = confidence = None
    if args.operator_target is not None:
        target = OperatorTarget.load(args.operator_target)
        if not torch.equal(target.query_indices, q_train):
            raise ValueError("G3 target query indices must exactly match query_train")
        if not torch.equal(target.command_indices, w_train):
            raise ValueError("G3 inversion target must use exactly the 64 train commands")
        target_q = target.q_hat
        confidence = target.confidence
    new_schedule_recipe = args.recipe not in {"G1", "G2", "G3"}
    staged = args.recipe in {"G2-T", "G2-TI"}
    separation_recipes = {"G2-J-LR240", "G2-TI"}
    if args.lambda_separation and args.recipe not in separation_recipes:
        raise ValueError(
            "soft separation sweep requires a joint SSH/Fourier model "
            f"({sorted(separation_recipes)})"
        )
    config = OperatorTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_q=args.lambda_q if args.recipe == "G3" else 0.0,
        seed=args.seed,
        training_schedule="ssh_fourier_joint" if staged else "joint",
        learning_rate_drop_epoch=181 if new_schedule_recipe else None,
        lambda_separation=args.lambda_separation,
    )
    q_sep_direction = q_sep_weights = q_sep_basis = q_sep_global = None
    if args.lambda_separation:
        q_sep_local, q_sep_weights = stratified_separation_subset(
            direction[q_train],
            per_cell=40,
            seed=21290,
            require_all_cells=not args.spatial_gap,
        )
        q_sep_direction = direction[q_train][q_sep_local]
        q_sep_basis = transform.evaluate(q_sep_direction.double()).float()
        q_sep_global = q_train[q_sep_local]
    result = train_operator_model(
        model,
        direction[q_train],
        amplitude[q_train][:, w_train],
        codebook.weights[w_train],
        direction[q_validation],
        amplitude[q_validation][:, w_validation],
        codebook.weights[w_validation],
        operator_target=target_q,
        operator_confidence=confidence,
        separation_direction=q_sep_direction,
        separation_weights=q_sep_weights,
        separation_basis=q_sep_basis,
        db_floor=max(dataset.manifest.noise_floor_amplitude, 1e-8),
        config=config,
    )
    model.eval()
    with torch.no_grad():
        validation_prediction = model.forward_command_grid(
            codebook.weights[w_validation].to(args.device),
            direction[q_validation].to(args.device),
        ).T.cpu()
    detailed_validation = amplitude_metrics(
        validation_prediction,
        amplitude[q_validation][:, w_validation],
    )
    run_name = (
        f"G3_lambda{str(args.lambda_q).replace('.', 'p')}_seed{args.seed}"
        if args.recipe == "G3"
        else f"{args.recipe}_sep{str(args.lambda_separation).replace('.', 'p')}_seed{args.seed}"
        if args.lambda_separation else f"{args.recipe}_seed{args.seed}"
    )
    if args.spatial_gap:
        run_name += "_gap"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if transform is not None:
        transform.save(run_dir / "filtration_transform.npz")
    torch.save(
        {
            "schema_version": "v212-rank1-checkpoint-1",
            "recipe": args.recipe,
            "ssh_degree": args.ssh_degree if transform is not None else None,
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "train_config": asdict(config),
            "seed": args.seed,
            "test_split_opened": False,
            "envelope_scale": envelope_scale,
            "envelope_scale_source": "median_train_direction_single_command_amplitude",
            "fourier_input_mode": fourier_input_mode,
            "training_schedule": config.training_schedule,
            "learning_rate_phases": (
                [{"epochs": [1, 180], "lr": 1e-3}, {"epochs": [181, args.epochs], "lr": 3e-4}]
                if new_schedule_recipe else [{"epochs": [1, args.epochs], "lr": 1e-3}]
            ),
            "separation": {
                "lambda": args.lambda_separation,
                "start_epoch": config.separation_start_epoch,
                "interval_steps": config.separation_interval_steps,
                "query_indices": None if q_sep_global is None else q_sep_global,
                "source": "fixed_stratified_q_train_only",
            },
            "learnable_parameter_count": sum(value.numel() for value in model.parameters()),
            "direction_protocol": direction_protocol,
            "train_query_indices": q_train,
            "validation_query_indices": q_validation,
            "input_projection": (
                None if fourier_input_mode == "raw" else {
                    "gram_condition": model.residual.encoder.gram_condition,
                    "identity_error": model.residual.encoder.identity_error,
                    "orthogonality_residual": model.residual.encoder.orthogonality_residual,
                    "used_fast_path": model.residual.encoder.used_fast_path,
                }
            ),
            "observation": {
                **asdict(model.observation),
                "kind": model.observation.kind.value,
            },
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "history.json").write_text(
        json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "schema": "v212-rank1-training-result-1",
        "recipe": args.recipe,
        "seed": args.seed,
        "lambda_q": config.lambda_q,
        "lambda_separation": config.lambda_separation,
        "fourier_input_mode": fourier_input_mode,
        "training_schedule": config.training_schedule,
        "learnable_parameter_count": sum(value.numel() for value in model.parameters()),
        "direction_protocol": direction_protocol,
        "envelope_scale": envelope_scale,
        "best_epoch": result.best_epoch,
        "best_validation_nmse_db": result.best_validation_nmse_db,
        "validation_metrics": detailed_validation,
        "test_status": "sealed",
    }
    (run_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
