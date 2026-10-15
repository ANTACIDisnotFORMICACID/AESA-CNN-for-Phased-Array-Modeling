"""Train the preregistered depth/context-controlled V2.16 DCAR arms."""

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
from v216.experiment import (  # noqa: E402
    SEEDS,
    STAGE1_TRAIN_ARMS,
    build_stage1_model,
    complex_commands,
    load_amplitude_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", choices=STAGE1_TRAIN_ARMS, default=list(STAGE1_TRAIN_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.prepared_dir / "PREPARATION.json").read_text(encoding="utf-8"))
    forbidden = (
        "old_v215_test_indices_saved", "old_v215_test_directions_saved",
        "old_v215_test_commands_saved", "old_v215_test_amplitudes_saved",
        "old_v215_test_used_for_preparation_decisions", "new_confirmatory_test_defined",
    )
    if any(manifest[name] for name in forbidden):
        raise ValueError("prepared package violates the external-test seal")
    control_config = json.loads((args.control_dir / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    frozen_protocol = {
        "epochs": 240, "batch_size": 512, "learning_rate": 1e-3,
        "late_learning_rate": 3e-4, "learning_rate_drop_epoch": 181,
        "lambda_separation": 0.001,
    }
    if any(control_config[key] != value for key, value in frozen_protocol.items()):
        raise ValueError("Stage-0 control protocol differs from the Stage-1 freeze")
    if control_config["old_v215_test_used"] or control_config["coverage_probe_used"]:
        raise ValueError("control checkpoints do not satisfy the Stage-1 seal")

    train = load_amplitude_package(args.prepared_dir / "train_exchange.npz", expected_commands=64)
    validation = load_amplitude_package(args.prepared_dir / "validation.npz", expected_commands=16)
    train_direction = torch.as_tensor(train["direction"], dtype=torch.float32)
    train_target = torch.as_tensor(train["amplitude"], dtype=torch.float32)
    train_commands = complex_commands(train["command_iq"])
    validation_direction = torch.as_tensor(validation["direction"], dtype=torch.float32)
    validation_target = torch.as_tensor(validation["amplitude"], dtype=torch.float32)
    validation_commands = complex_commands(validation["command_iq"])
    single_local = np.flatnonzero(np.isin(train["command_indices"], np.arange(16)))
    if len(single_local) != 16:
        raise ValueError("training package must contain the 16 single-element commands")
    envelope_scale = float(np.median(train["amplitude"][:, single_local]))
    transform = FiltrationSSHTransform.fit(train_direction.double(), degree=4)
    control_transform = FiltrationSSHTransform.load(
        args.control_dir / "control_v212_seed21601" / "filtration_transform.npz"
    )
    np.testing.assert_allclose(transform.transform.numpy(), control_transform.transform.numpy(), atol=0, rtol=0)
    control_checkpoint = torch.load(
        args.control_dir / "control_v212_seed21601" / "checkpoint.pt",
        map_location="cpu", weights_only=True,
    )
    if envelope_scale != float(control_checkpoint["envelope_scale"]):
        raise ValueError("training-only envelope scale differs from the control")

    q_sep_local, q_sep_weights = stratified_separation_subset(
        train_direction, per_cell=40, seed=21290, require_all_cells=True
    )
    q_sep_direction = train_direction[q_sep_local]
    q_sep_basis = transform.evaluate(q_sep_direction.double()).float()
    run_config = {
        "schema": "v216-stage1-controlled-training-config-1",
        "arms": list(args.arms), "seeds": list(args.seeds), **frozen_protocol,
        "prepared_dir": str(args.prepared_dir.resolve()),
        "control_dir": str(args.control_dir.resolve()),
        "control_checkpoint_reused_not_retrained": True,
        "ssh_degree": 4,
        "checkpoint_selection": "validation_linear_nmse_only",
        "coverage_probe_used": False,
        "old_v215_test_used": False,
        "new_confirmatory_test_defined_or_opened": False,
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != run_config:
        raise ValueError("existing Stage-1 RUN_CONFIG differs from the frozen experiment")
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    train_config = OperatorTrainConfig(
        epochs=240, batch_size=512, learning_rate=1e-3,
        late_learning_rate=3e-4, learning_rate_drop_epoch=181,
        lambda_separation=0.001, seed=0, training_schedule="joint",
    )
    for arm in args.arms:
        for seed in args.seeds:
            run_dir = args.output_dir / f"{arm}_seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("old_v215_test_status") != "not_used":
                    raise ValueError(f"existing result violates the seal: {run_dir}")
                print(json.dumps({"resume": "skip_complete", "arm": arm, "seed": seed}), flush=True)
                continue
            print(json.dumps({"start": "full_training", "arm": arm, "seed": seed}), flush=True)
            model = build_stage1_model(
                arm, transform=transform, envelope_scale=envelope_scale, seed=seed
            ).to(args.device)
            parameter_count = sum(value.numel() for value in model.parameters())
            with torch.no_grad():
                zero_start = float(model.residual(train_direction[:32].to(args.device)[None]).abs().max().cpu())
            per_run_config = OperatorTrainConfig(**{**asdict(train_config), "seed": seed})
            started = time.perf_counter()
            result = train_operator_model(
                model, train_direction, train_target, train_commands,
                validation_direction, validation_target, validation_commands,
                separation_direction=q_sep_direction, separation_weights=q_sep_weights,
                separation_basis=q_sep_basis, db_floor=1e-8, config=per_run_config,
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
            torch.save({
                "schema": "v216-stage1-controlled-checkpoint-1",
                "arm": arm, "seed": seed,
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "train_config": asdict(per_run_config),
                "envelope_scale": envelope_scale,
                "architecture": model.architecture_metadata(),
                "parameter_count": parameter_count,
                "checkpoint_selection": "validation_linear_nmse_only",
                "coverage_probe_used_for_selection": False,
                "old_v215_test_used": False,
                "new_confirmatory_test_opened": False,
            }, run_dir / "checkpoint.pt")
            (run_dir / "history.json").write_text(
                json.dumps(result.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            payload = {
                "schema": "v216-stage1-controlled-training-result-1",
                "arm": arm, "seed": seed, "epochs": 240, "trained_from_scratch": True,
                "parameter_count": parameter_count, "zero_start_residual_max_abs": zero_start,
                "best_epoch": result.best_epoch,
                "best_validation_nmse_db": result.best_validation_nmse_db,
                "validation_metrics": metrics, "training_seconds": duration,
                "coverage_probe_status": "not_opened_by_training",
                "old_v215_test_status": "not_used",
                "new_confirmatory_test_status": "not_defined_or_opened",
            }
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            del model
            gc.collect()


if __name__ == "__main__":
    main()
