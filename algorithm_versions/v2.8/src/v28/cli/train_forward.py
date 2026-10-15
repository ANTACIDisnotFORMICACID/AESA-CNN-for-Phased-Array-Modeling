"""Train the official v2.8 forward model using amplitude-only observations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from ..io import assert_split_matches_dataset, load_amplitude_dataset, load_amplitude_split
from ..models import AmplitudeOnlyForwardModel, AmplitudeOperatorResidualCNN
from ..training import (
    AmplitudeDirectionDataset,
    AmplitudeTrainConfig,
    balanced_command_family_weights,
    train_grouped_amplitude_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="resume/fine-tune a compatible amplitude-only checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024, help="directions per training batch")
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--db-loss-weight", type=float, default=0.03)
    parser.add_argument("--power-loss-weight", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--element-pattern-width", type=int, default=20)
    parser.add_argument("--disable-element-pattern", action="store_true")
    parser.add_argument("--freeze-element-gain", action="store_true")
    parser.add_argument("--freeze-local-coupling", action="store_true")
    parser.add_argument("--seed", type=int, default=28001)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="independent query-sampling seed; defaults to the split seed",
    )
    parser.add_argument(
        "--queries-per-epoch",
        type=int,
        default=8192,
        help="random training directions per epoch; every selected command supervises each direction; 0 uses all",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_from_config(config: dict[str, object]) -> AmplitudeOnlyForwardModel:
    return AmplitudeOnlyForwardModel(
        correction=AmplitudeOperatorResidualCNN(
            fourier_bands=int(config["direction_fourier_bands"]),
            directional_width=int(config["directional_width"]),
            edge_width=int(config["edge_width"]),
            element_pattern_width=int(config.get("element_pattern_width", 20)),
            multihop_width=int(config["multihop_width"]),
            enable_element_pattern=bool(config.get("enable_element_pattern", False)),
            enable_multihop=bool(config["enable_multihop"]),
        ),
        reference_distance_m=float(config["reference_distance_m"]),
    )


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    if args.seed != split.seed:
        raise ValueError("training seed must match the precomputed split seed")

    torch.manual_seed(args.seed)
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else args.seed
    training_data = AmplitudeDirectionDataset(dataset, split.command_train, split.query_train)
    validation_data = AmplitudeDirectionDataset(
        dataset, split.command_validation, split.query_validation
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
    initial_checkpoint = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        if initial_checkpoint.get("schema_version") != "v2.8-amplitude-1":
            raise ValueError("initial checkpoint is not an amplitude-only v2.8 model")
        contract = initial_checkpoint.get("input_contract", {})
        if (
            contract.get("measured_phase_used") is not False
            or contract.get("s_parameters_used") is not False
        ):
            raise ValueError("initial checkpoint violates the amplitude-only contract")
        if initial_checkpoint.get("dataset_fingerprint") != split.dataset_fingerprint:
            raise ValueError("initial checkpoint was trained from a different amplitude dataset")
        if initial_checkpoint.get("split_schema_version") != split.schema_version:
            raise ValueError("initial checkpoint uses a different data split schema")
        if initial_checkpoint.get("seed") != split.seed:
            raise ValueError("initial checkpoint uses a different split seed")
        for part in ("train", "validation", "test"):
            expected_query = torch.as_tensor(
                getattr(split, f"query_{part}"), dtype=torch.int64
            )
            expected_command = torch.as_tensor(
                getattr(split, f"command_{part}"), dtype=torch.int64
            )
            if not torch.equal(initial_checkpoint["query_split_indices"][part], expected_query):
                raise ValueError(f"initial checkpoint query {part} indices do not match")
            if not torch.equal(
                initial_checkpoint["command_split_indices"][part], expected_command
            ):
                raise ValueError(f"initial checkpoint command {part} indices do not match")
        model_config = dict(initial_checkpoint["model_config"])
        model = _model_from_config(model_config)
        model.load_state_dict(initial_checkpoint["model_state"], strict=True)
    else:
        enable_element_pattern = not args.disable_element_pattern
        model_config = {
            "architecture_version": "single_pol_pattern_residual_2",
            "direction_fourier_bands": 4,
            "directional_width": 24,
            "edge_width": 16,
            "element_pattern_width": args.element_pattern_width,
            "multihop_width": 16,
            "enable_element_pattern": enable_element_pattern,
            "enable_multihop": False,
            "reference_distance_m": dataset.manifest.reference_distance_m,
        }
        model = _model_from_config(model_config)
    model = model.to(args.device)

    frozen_parameter_names: list[str] = []
    freeze_prefixes = []
    if args.freeze_element_gain:
        freeze_prefixes.append("correction.element_gain.")
    if args.freeze_local_coupling:
        freeze_prefixes.append("correction.local_coupling.")
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_prefixes):
            parameter.requires_grad_(False)
            frozen_parameter_names.append(name)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count == 0:
        raise ValueError("freezing configuration leaves no trainable parameters")
    config = AmplitudeTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        db_loss_weight=args.db_loss_weight,
        power_loss_weight=args.power_loss_weight,
        scheduler_patience=args.scheduler_patience,
        minimum_learning_rate=args.minimum_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
    )
    training_weights = balanced_command_family_weights(
        dataset.command_family[split.command_train]
    )
    validation_weights = balanced_command_family_weights(
        dataset.command_family[split.command_validation]
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
        config=config,
        select_initial_state=initial_checkpoint is not None,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "v2.8-amplitude-1",
            "model_kind": "amplitude_only_latent_complex_operator",
            "input_contract": {
                "command_iq": ["batch", "iq", "row", "col"],
                "unit_direction": ["batch", "query", "xyz"],
                "receiver_observation": "single_polarization_scalar_amplitude",
                "measured_phase_used": False,
                "s_parameters_used": False,
            },
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "dataset_fingerprint": split.dataset_fingerprint,
            "split_schema_version": split.schema_version,
            "queries_per_epoch": query_count,
            "effective_observations_per_epoch": int(query_count * split.command_train.size),
            "training_protocol": {
                "direction_grouped_commands": True,
                "command_family_balanced_loss": True,
                "validation_selected_checkpoint": True,
                "test_used_for_selection": False,
                "fine_tuned_from_checkpoint": initial_checkpoint is not None,
                "frozen_parameter_names": frozen_parameter_names,
            },
            "model_state": model.state_dict(),
            "model_config": model_config,
            "train_config": asdict(config),
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
            "initial_checkpoint": (
                {
                    "path": str(args.initial_checkpoint.resolve()),
                    "sha256": _sha256(args.initial_checkpoint),
                    "parent_best_epoch": initial_checkpoint.get("best_epoch"),
                }
                if initial_checkpoint is not None
                else None
            ),
            "initial_validation": result.initial_validation,
            "history": result.history,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "stopped_early": result.stopped_early,
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
        f"saved amplitude-only checkpoint from validation-best epoch "
        f"{result.best_epoch} to {args.checkpoint.resolve()}"
    )


if __name__ == "__main__":
    main()
