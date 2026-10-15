"""Train V2.10 jointly, or use staged training as a controlled ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from ..io import (
    AMPLITUDE_ONLY_TRAINING_POLICY,
    assert_amplitude_only_training_contract,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    load_amplitude_split,
)
from ..models import model_from_config
from ..training import (
    AmplitudeDirectionDataset,
    AmplitudeTrainConfig,
    TrainingStage,
    balanced_command_family_weights,
    configure_trainable_stage,
    default_stage_plan,
    train_grouped_amplitude_model,
)


CHECKPOINT_SCHEMA = "v2.9-amplitude-operator-1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stage-plan", choices=("default", "joint"), default="joint")
    parser.add_argument("--batch-size", type=int, default=1024, help="directions per batch")
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--db-loss-weight", type=float, default=0.03)
    parser.add_argument("--power-loss-weight", type=float, default=0.10)
    parser.add_argument("--residual-energy-weight", type=float, default=1e-4)
    parser.add_argument("--coupling-l2-weight", type=float, default=1e-4)
    parser.add_argument("--reciprocity-weight", type=float, default=1e-3)
    parser.add_argument("--nominal-width", type=int, default=24)
    parser.add_argument("--nominal-depth", type=int, default=2)
    parser.add_argument("--coupling-low-rank", type=int, default=1)
    parser.add_argument("--residual-width", type=int, default=12)
    parser.add_argument("--residual-depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=28001)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument(
        "--queries-per-epoch",
        type=int,
        default=8192,
        help="sampled training directions per epoch; 0 uses all",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    if args.seed != split.seed:
        raise ValueError("training seed must match the precomputed split seed")

    torch.manual_seed(args.seed)
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else args.seed
    training_data = AmplitudeDirectionDataset(dataset, split.command_train, split.query_train)
    validation_data = AmplitudeDirectionDataset(
        dataset,
        split.command_validation,
        split.query_validation,
    )
    if args.queries_per_epoch < 0:
        raise ValueError("queries-per-epoch must be non-negative")
    if args.queries_per_epoch:
        query_count = min(args.queries_per_epoch, len(training_data))
        sampler = RandomSampler(
            training_data,
            replacement=False,
            num_samples=query_count,
            generator=torch.Generator().manual_seed(sampling_seed),
        )
    else:
        query_count = len(training_data)
        sampler = None
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=torch.Generator().manual_seed(sampling_seed),
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

    model_config: dict[str, object] = {
        "architecture_version": "v210_nominal_coupling_residual_1",
        "basis_type": "ideal_phase",
        "source_type": dataset.manifest.source_type,
        "direction_fourier_bands": 4,
        "nominal_width": args.nominal_width,
        "nominal_depth": args.nominal_depth,
        "nominal_max_log_amplitude": 3.0,
        "enable_coupling": True,
        "coupling_low_rank": args.coupling_low_rank,
        "enable_residual": True,
        "residual_width": args.residual_width,
        "residual_depth": args.residual_depth,
        "residual_maximum_magnitude": 2.0,
        "device_model": "identity",
        "reference_distance_m": dataset.manifest.reference_distance_m,
    }
    model = model_from_config(model_config).to(args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    training_weights = balanced_command_family_weights(
        dataset.command_family[split.command_train]
    )
    validation_weights = balanced_command_family_weights(
        dataset.command_family[split.command_validation]
    )
    if args.stage_plan == "default":
        stages = default_stage_plan(args.epochs)
    else:
        stages = (TrainingStage("joint", args.epochs, 1.0),)

    base_config = AmplitudeTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        db_loss_weight=args.db_loss_weight,
        power_loss_weight=args.power_loss_weight,
        early_stopping_patience=args.epochs + 1,
        residual_energy_weight=args.residual_energy_weight,
        coupling_l2_weight=args.coupling_l2_weight,
        reciprocity_weight=args.reciprocity_weight,
    )
    full_history: list[dict[str, float | str]] = []
    stage_results: list[dict[str, object]] = []
    global_epoch = 0
    for stage in stages:
        trainable_names = configure_trainable_stage(model, stage.name)
        stage_config = replace(
            base_config,
            epochs=stage.epochs,
            learning_rate=base_config.learning_rate * stage.learning_rate_scale,
            early_stopping_patience=stage.epochs + 1,
        )
        print(
            f"stage={stage.name} epochs={stage.epochs} "
            f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        result = train_grouped_amplitude_model(
            model,
            training_loader,
            training_data.command_iq,
            validation_loader,
            validation_data.command_iq,
            noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
            training_command_weights=training_weights,
            validation_command_weights=validation_weights,
            config=stage_config,
            select_initial_state=True,
        )
        for row in result.history:
            global_epoch += 1
            full_history.append({"stage": stage.name, "global_epoch": global_epoch, **row})
        stage_results.append(
            {
                "name": stage.name,
                "requested_epochs": stage.epochs,
                "executed_epochs": len(result.history),
                "selected_local_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "trainable_parameter_names": list(trainable_names),
                "initial_validation": result.initial_validation,
            }
        )

    configure_trainable_stage(model, "joint")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "dataset_schema_version": dataset.manifest.schema_version,
            "model_kind": "v210_physics_layered_amplitude_operator",
            "input_contract": {
                "command_iq": ["batch", "iq", "row", "col"],
                "unit_direction": ["batch", "query", "xyz"],
                "receiver_observation": "single_polarization_scalar_amplitude",
                "complex_convention": "transmit_h_transpose_w",
                "measured_phase_used": False,
                "s_parameters_used": False,
                "complex_field_supervision_used": False,
                "training_label": "single_polarization_scalar_amplitude",
                "physical_domain": "strict_front_hemisphere_q_x_gt_0",
            },
            "training_policy": AMPLITUDE_ONLY_TRAINING_POLICY,
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "dataset_fingerprint": split.dataset_fingerprint,
            "split_schema_version": split.schema_version,
            "queries_per_epoch": query_count,
            "requested_total_epochs": args.epochs,
            "executed_total_epochs": global_epoch,
            "stage_plan": [asdict(stage) for stage in stages],
            "stage_results": stage_results,
            "history": full_history,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "train_config": asdict(base_config),
            "parameter_count": parameter_count,
            "query_split_indices": {
                "train": torch.as_tensor(split.query_train, dtype=torch.int64),
                "validation": torch.as_tensor(split.query_validation, dtype=torch.int64),
                "test": torch.as_tensor(split.query_test, dtype=torch.int64),
            },
            "command_split_indices": {
                "train": torch.as_tensor(split.command_train, dtype=torch.int64),
                "validation": torch.as_tensor(split.command_validation, dtype=torch.int64),
                "test": torch.as_tensor(split.command_test, dtype=torch.int64),
            },
            "dataset_manifest": asdict(dataset.manifest),
        },
        args.checkpoint,
    )
    print(
        f"saved V2.10 checkpoint after {global_epoch} executed epochs "
        f"to {args.checkpoint.resolve()}"
    )


if __name__ == "__main__":
    main()

