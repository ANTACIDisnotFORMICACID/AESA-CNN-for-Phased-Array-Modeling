"""Train every preregistered V2.15 CNN arm from scratch without test access."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.evaluation import amplitude_metrics, stratified_separation_subset  # noqa: E402
from v212.training import OperatorTrainConfig, train_operator_model  # noqa: E402
from v215.experiment import ARMS, SEEDS, build_model, complex_commands, load_amplitude_package  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--physics-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--late-learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-separation", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.prepared_dir / "PREPARATION.json").read_text(encoding="utf-8"))
    if any(manifest[key] for key in (
        "test_indices_saved", "test_directions_saved", "test_commands_saved", "test_amplitudes_saved"
    )):
        raise ValueError("prepared data is not test-sealed")
    train = load_amplitude_package(args.prepared_dir / "train_exchange.npz", expected_commands=64)
    validation = load_amplitude_package(args.prepared_dir / "validation.npz", expected_commands=16)
    train_direction = torch.as_tensor(train["direction"], dtype=torch.float32)
    train_target = torch.as_tensor(train["amplitude"], dtype=torch.float32)
    train_commands = complex_commands(train["command_iq"])
    validation_direction = torch.as_tensor(validation["direction"], dtype=torch.float32)
    validation_target = torch.as_tensor(validation["amplitude"], dtype=torch.float32)
    validation_commands = complex_commands(validation["command_iq"])
    with np.load(args.physics_data, allow_pickle=False) as physics:
        s_matrix = np.asarray(physics["s_parameters_47ghz"], dtype=np.complex128)

    config_payload = {
        "schema": "v215-complete-training-config-1",
        "prepared_dir": str(args.prepared_dir.resolve()),
        "physics_data": str(args.physics_data.resolve()),
        "arms": list(args.arms),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "late_learning_rate": args.late_learning_rate,
        "learning_rate_drop_epoch": 181,
        "lambda_separation": args.lambda_separation,
        "ssh_degree": 4,
        "checkpoint_selection": "validation_linear_nmse_only",
        "coverage_probe_used": False,
        "test_split_opened": False,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
        raise ValueError("existing RUN_CONFIG differs from the requested complete experiment")
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    single_local = np.flatnonzero(np.isin(train["command_indices"], np.arange(16)))
    if len(single_local) != 16:
        raise ValueError("training package must contain the 16 single-element commands")
    envelope_scale = float(np.median(train["amplitude"][:, single_local]))
    if not envelope_scale > 0:
        raise ValueError("training-only envelope scale must be positive")
    transform = FiltrationSSHTransform.fit(train_direction.double(), degree=4)
    q_sep_local, q_sep_weights = stratified_separation_subset(
        train_direction, per_cell=40, seed=21290, require_all_cells=True
    )
    q_sep_direction = train_direction[q_sep_local]
    q_sep_basis = transform.evaluate(q_sep_direction.double()).float()

    for arm in args.arms:
        for seed in args.seeds:
            run_dir = args.output_dir / f"{arm}_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("test_status") != "sealed":
                    raise ValueError(f"existing result is not sealed: {run_dir}")
                print(json.dumps({"resume": "skip_complete", "arm": arm, "seed": seed}), flush=True)
                continue
            print(json.dumps({"start": "full_training", "arm": arm, "seed": seed}), flush=True)
            model = build_model(
                arm,
                transform=transform,
                s_matrix=s_matrix,
                envelope_scale=envelope_scale,
                seed=seed,
            ).to(args.device)
            parameter_count = sum(value.numel() for value in model.parameters())
            model.eval()
            with torch.no_grad():
                zero_start_residual = float(
                    model.residual(train_direction[:32].to(args.device)[None]).abs().max().cpu()
                )
            train_config = OperatorTrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                late_learning_rate=args.late_learning_rate,
                learning_rate_drop_epoch=181,
                lambda_separation=args.lambda_separation,
                seed=seed,
                training_schedule="joint",
            )
            started = time.perf_counter()
            result = train_operator_model(
                model,
                train_direction,
                train_target,
                train_commands,
                validation_direction,
                validation_target,
                validation_commands,
                separation_direction=q_sep_direction,
                separation_weights=q_sep_weights,
                separation_basis=q_sep_basis,
                db_floor=1e-8,
                config=train_config,
            )
            duration = time.perf_counter() - started
            model.eval()
            with torch.no_grad():
                prediction = model.forward_command_grid(
                    validation_commands.to(args.device), validation_direction.to(args.device)
                ).T.cpu()
            metrics = amplitude_metrics(prediction, validation_target)
            if not np.isfinite(metrics["nmse_db"]):
                raise ValueError("training produced non-finite validation metrics")
            run_dir.mkdir(parents=True, exist_ok=True)
            transform.save(run_dir / "filtration_transform.npz")
            architecture = (
                {
                    "family": "v212-control",
                    "residual_width": 24,
                    "residual_depth": 2,
                    "analytic_phase_preserved": True,
                    "zero_start_output_head": True,
                }
                if arm == "control" else model.architecture_metadata()
            )
            torch.save(
                {
                    "schema": "v215-complete-checkpoint-1",
                    "arm": arm,
                    "seed": seed,
                    "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "train_config": asdict(train_config),
                    "envelope_scale": envelope_scale,
                    "architecture": architecture,
                    "parameter_count": parameter_count,
                    "train_directions": len(train_direction),
                    "train_commands": len(train_commands),
                    "validation_directions": len(validation_direction),
                    "validation_commands": len(validation_commands),
                    "checkpoint_selection": "validation_linear_nmse_only",
                    "training_data_has_test_fields": False,
                    "coverage_probe_used_for_selection": False,
                    "test_split_opened": False,
                },
                run_dir / "checkpoint.pt",
            )
            (run_dir / "history.json").write_text(
                json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            payload = {
                "schema": "v215-complete-training-result-1",
                "arm": arm,
                "seed": seed,
                "epochs": args.epochs,
                "trained_from_scratch": True,
                "parameter_count": parameter_count,
                "parameter_difference_vs_control_checked_in_summary": True,
                "zero_start_residual_max_abs": zero_start_residual,
                "best_epoch": result.best_epoch,
                "best_validation_nmse_db": result.best_validation_nmse_db,
                "validation_metrics": metrics,
                "training_seconds": duration,
                "coverage_probe_status": "not_opened_by_training",
                "test_status": "sealed",
            }
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            del model
            gc.collect()


if __name__ == "__main__":
    main()
