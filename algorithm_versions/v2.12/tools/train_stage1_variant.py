"""Train one V2.12 Stage-1 encoding variant without opening the test split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import (  # noqa: E402
    AMPLITUDE_ONLY_TRAINING_POLICY,
    assert_amplitude_only_training_contract,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    load_amplitude_split,
)
from v210.training import (  # noqa: E402
    AmplitudeDirectionDataset,
    AmplitudeTrainConfig,
    balanced_command_family_weights,
    configure_trainable_stage,
    train_grouped_amplitude_model,
)
from v212.stage1 import Stage1EncodingConfig, build_stage1_model  # noqa: E402
from v212.evaluation import amplitude_metrics  # noqa: E402


RECIPES = ("E0", "E0-M4", "E0-M5", "E-S4", "E1-R4", "E2-F4", "E2-F5", "E3-I")


def degree_for(recipe: str, e3_degree: int) -> int:
    if recipe in {"E0-M5", "E2-F5"}:
        return 5
    if recipe == "E3-I":
        return e3_degree
    return 4


def base_model_config(dataset) -> dict[str, object]:
    return {
        "architecture_version": "v210_nominal_coupling_residual_1",
        "basis_type": "ideal_phase",
        "source_type": dataset.manifest.source_type,
        "direction_fourier_bands": 4,
        "nominal_width": 24,
        "nominal_depth": 2,
        "nominal_max_log_amplitude": 3.0,
        "enable_coupling": True,
        "coupling_low_rank": 1,
        "enable_residual": True,
        "residual_width": 12,
        "residual_depth": 1,
        "residual_maximum_magnitude": 2.0,
        "reference_distance_m": dataset.manifest.reference_distance_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--e3-degree", type=int, choices=(4, 5), default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--queries-per-epoch", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_dir = args.output_dir / f"{args.recipe}_seed{args.seed}"
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"already complete: {result_path}")
        return

    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    train_data = AmplitudeDirectionDataset(dataset, split.command_train, split.query_train)
    validation_data = AmplitudeDirectionDataset(
        dataset, split.command_validation, split.query_validation
    )
    query_count = min(args.queries_per_epoch, len(train_data))
    sampler = RandomSampler(
        train_data,
        replacement=False,
        num_samples=query_count,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    validation_loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    encoding = Stage1EncodingConfig(
        recipe=args.recipe,
        ssh_degree=degree_for(args.recipe, args.e3_degree),
        seed=args.seed,
    )
    base_config = base_model_config(dataset)
    model, transform = build_stage1_model(
        base_config,
        encoding,
        train_data.direction,
    )
    model.to(args.device)
    configure_trainable_stage(model, "joint")
    train_weights = balanced_command_family_weights(dataset.command_family[split.command_train])
    validation_weights = balanced_command_family_weights(
        dataset.command_family[split.command_validation]
    )
    config = AmplitudeTrainConfig(
        epochs=args.epochs,
        learning_rate=1e-3,
        weight_decay=1e-5,
        db_loss_weight=0.03,
        power_loss_weight=0.10,
        early_stopping_patience=args.epochs + 1,
        residual_energy_weight=1e-4,
        coupling_l2_weight=1e-4,
        reciprocity_weight=1e-3,
    )
    started = time.perf_counter()
    result = train_grouped_amplitude_model(
        model,
        train_loader,
        train_data.command_iq,
        validation_loader,
        validation_data.command_iq,
        noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
        training_command_weights=train_weights,
        validation_command_weights=validation_weights,
        config=config,
        select_initial_state=True,
    )
    elapsed = time.perf_counter() - started
    if result.best_epoch == 0:
        best_metrics = result.initial_validation
    else:
        best_metrics = result.history[result.best_epoch - 1]
        best_metrics = {
            name.removeprefix("validation_"): value
            for name, value in best_metrics.items()
            if name.startswith("validation_")
        }
    if best_metrics is None:
        raise RuntimeError("best validation metrics are unavailable")
    model.eval()
    with torch.no_grad():
        validation_direction = torch.as_tensor(
            dataset.direction[split.query_validation],
            dtype=torch.float32,
            device=args.device,
        )
        validation_prediction = model.forward_command_grid(
            validation_data.command_iq.to(args.device), validation_direction
        ).T.cpu()
    validation_target = torch.as_tensor(
        dataset.amplitude[np.ix_(split.command_validation, split.query_validation)].T.copy(),
        dtype=torch.float32,
    )
    detailed_validation = amplitude_metrics(validation_prediction, validation_target)
    run_dir.mkdir(parents=True, exist_ok=True)
    if transform is not None:
        transform.save(run_dir / "filtration_transform.npz")
    checkpoint = {
        "schema_version": "v212-stage1-checkpoint-1",
        "model_kind": "v210_amplitude_with_v212_direction_encoding",
        "recipe": args.recipe,
        "base_model_config": base_config,
        "encoding_config": asdict(encoding),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "seed": args.seed,
        "train_config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_policy": AMPLITUDE_ONLY_TRAINING_POLICY,
        "input_contract": {
            "observable": "single_polarization_scalar_amplitude",
            "measured_phase_used": False,
            "complex_field_supervision_used": False,
            "test_split_opened": False,
        },
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "history.json").write_text(
        json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "schema": "v212-stage1-result-1",
        "recipe": args.recipe,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_linear_nmse_db": detailed_validation["nmse_db"],
        "best_validation_amplitude_db_mae": best_metrics["amplitude_db_mae"],
        "validation_metrics": detailed_validation,
        "parameter_count": checkpoint["parameter_count"],
        "elapsed_seconds": elapsed,
        "test_status": "sealed",
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
