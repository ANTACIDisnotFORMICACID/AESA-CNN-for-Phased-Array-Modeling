"""Train one V2.11 direction-encoding variant under the frozen V2.10 protocol."""

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
    evaluate_amplitude_grid_model,
    train_grouped_amplitude_model,
)
from v211.training_bridge import build_v211_model  # noqa: E402


VARIANTS: dict[str, dict[str, object]] = {
    "fourier_v210": {"kind": "fourier", "fourier_bands": 4, "output_channels": 27},
    "ssh_l4": {"kind": "ssh", "ssh_degree": 4, "output_channels": 27},
    "ssh_l5": {"kind": "ssh", "ssh_degree": 5, "output_channels": 27},
    "fourier_plus_ssh_l4": {
        "kind": "fourier_plus_ssh",
        "fourier_bands": 4,
        "ssh_degree": 4,
        "output_channels": 27,
    },
}


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _base_model_config(dataset) -> dict[str, object]:
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
        "device_model": "identity",
        "reference_distance_m": dataset.manifest.reference_distance_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--ssh-statistics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--queries-per-epoch", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=28101)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.queries_per_epoch <= 0:
        raise ValueError("epochs and queries-per-epoch must be positive")
    run_dir = args.output_dir.resolve() / f"{args.variant}_seed{args.seed}"
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"already complete: {result_path}", flush=True)
        return

    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed

    training_data = AmplitudeDirectionDataset(dataset, split.command_train, split.query_train)
    validation_data = AmplitudeDirectionDataset(dataset, split.command_validation, split.query_validation)
    test_data = AmplitudeDirectionDataset(dataset, split.command_test, split.query_test)
    query_count = min(args.queries_per_epoch, len(training_data))
    sampler = RandomSampler(
        training_data,
        replacement=False,
        num_samples=query_count,
        generator=torch.Generator().manual_seed(sampling_seed),
    )
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        sampler=sampler,
        generator=torch.Generator().manual_seed(sampling_seed),
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

    base_config = _base_model_config(dataset)
    encoding_config = {**VARIANTS[args.variant], "normalization_epsilon": 1e-6}
    statistics_path = args.ssh_statistics if encoding_config["kind"] != "fourier" else None
    model = build_v211_model(
        base_config,
        encoding_config,
        statistics_path=statistics_path,
    ).to(args.device)
    configure_trainable_stage(model, "joint")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    training_weights = balanced_command_family_weights(dataset.command_family[split.command_train])
    validation_weights = balanced_command_family_weights(dataset.command_family[split.command_validation])
    test_weights = balanced_command_family_weights(dataset.command_family[split.command_test])
    train_config = AmplitudeTrainConfig(
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
    print(
        f"START variant={args.variant} seed={args.seed} epochs={args.epochs} "
        f"parameters={parameter_count} device={args.device}",
        flush=True,
    )
    started = time.perf_counter()
    result = train_grouped_amplitude_model(
        model,
        training_loader,
        training_data.command_iq,
        validation_loader,
        validation_data.command_iq,
        noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
        training_command_weights=training_weights,
        validation_command_weights=validation_weights,
        config=train_config,
        select_initial_state=True,
    )
    elapsed = time.perf_counter() - started
    test_metrics = evaluate_amplitude_grid_model(
        model,
        test_loader,
        test_data.command_iq,
        noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
        command_weights=test_weights,
        db_loss_weight=0.0,
        power_loss_weight=0.0,
    )
    validation_best = {
        key.removeprefix("validation_"): value
        for key, value in result.history[result.best_epoch - 1].items()
        if key.startswith("validation_")
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "v211-encoding-ablation-checkpoint-1",
        "model_kind": "v210_physics_layers_with_v211_direction_encoder",
        "base_model_config": base_config,
        "encoding_config": encoding_config,
        "model_state": _cpu_state(model),
        "parameter_count": parameter_count,
        "seed": args.seed,
        "sampling_seed": sampling_seed,
        "queries_per_epoch": query_count,
        "train_config": asdict(train_config),
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
        "input_contract": {
            "observable": "single-polarization scalar amplitude",
            "physical_domain": "strict q_x > 0",
            "measured_phase_used": False,
            "complex_field_supervision_used": False,
            "s_parameters_used": False,
            "ssh_statistics_fit": "training query split only",
        },
        "training_policy": AMPLITUDE_ONLY_TRAINING_POLICY,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "history.json").write_text(
        json.dumps(result.history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload = {
        "variant": args.variant,
        "encoding_config": encoding_config,
        "seed": args.seed,
        "sampling_seed": sampling_seed,
        "requested_epochs": args.epochs,
        "executed_epochs": len(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "validation": validation_best,
        "test_reused_benchmark": test_metrics,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "test_blind_status": "not blind: V2.10 previously opened this split",
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"COMPLETE {args.variant} seed={args.seed} seconds={elapsed:.1f}", flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
