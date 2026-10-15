"""V2.4 refinement: test low-rate, head-only adaptation without retraining all layers."""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from beamforming_pre_fpga import ArrayGeometryConfig, TopologyAdaptiveBeamformingCNN
from experiment_common import SOURCE, TARGET, make_samples
from run_v24_sparse_end_to_end_study import (
    TRAIN_SEEDS,
    aggregate_rows,
    d_optimal_fixed_angles,
    evaluate_sparse_model,
    seed_everything,
    train_sparse_controller,
    with_physics_sparse_calibration,
)
from sparse_beamforming_v24 import ALGORITHM_VERSION


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
RESULT_DIR = VERSION_DIR / "output" / "controller_training"
METHODS = {"single_board_pretrained_zero_shot", "single_board_head_only"}


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_pretrained(
    config: ArrayGeometryConfig,
    topology_index: int,
    source_train,
    fixed_angles: torch.Tensor,
    seed: int,
) -> TopologyAdaptiveBeamformingCNN:
    measurement_count = int(fixed_angles.shape[0]) + 1
    seed_everything(seed + topology_index * 1000 + measurement_count)
    model = TopologyAdaptiveBeamformingCNN(
        config, width=16, num_blocks=3, global_context="none", output_bits=None
    )
    train_sparse_controller(
        config,
        model,
        source_train,
        fixed_angles,
        seed=seed + measurement_count * 10,
        mode="coherent_iq",
        steps=300,
    )
    return model


def main() -> None:
    payload_path = RESULT_DIR / "metrics.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    per_seed = [row for row in payload["per_seed"] if row.get("method") not in METHODS]
    training_log = [
        row for row in read_csv(RESULT_DIR / "training_log.csv")
        if row.get("method") not in METHODS
    ]
    selected: Dict[str, object] = {}

    for topology_index, config in enumerate(
        (ArrayGeometryConfig.theoretical_4x4(), ArrayGeometryConfig.hardware_1x4())
    ):
        recommended = 32 if config.x_num > 1 else 8
        fixed_angles = d_optimal_fixed_angles(config, recommended - 1)
        source_train = make_samples(
            768, SOURCE, config, seed=7100 + topology_index, targets_per_profile=8
        )
        board_raw = make_samples(
            256, TARGET, config, seed=7275 + topology_index, targets_per_profile=256
        )
        board = with_physics_sparse_calibration(
            config, board_raw, seed=7475 + topology_index
        )
        board_train = board.take(torch.arange(0, 128))
        board_validation = board.take(torch.arange(128, 192))
        board_test = board.take(torch.arange(192, 256))
        best_score = -math.inf
        best_model = None
        best_seed = None
        best_update = None

        for seed in TRAIN_SEEDS:
            pretrained = source_pretrained(
                config, topology_index, source_train, fixed_angles, seed
            )
            zero_metric = evaluate_sparse_model(
                config,
                pretrained,
                board_test,
                name="single_board_pretrained_zero_shot",
                seed=seed,
            )
            per_seed.append(
                {
                    **zero_metric,
                    "method": "single_board_pretrained_zero_shot",
                    "measurement_count": 0,
                    "train_samples": 0,
                    "train_profiles": 1,
                    "label_mode": "none",
                }
            )
            zero_validation = evaluate_sparse_model(
                config,
                pretrained,
                board_validation,
                name="single_board_pretrained_zero_shot_validation",
                seed=seed,
            )
            zero_score = float(zero_validation["mean_oracle_fraction"])
            if zero_score > best_score:
                best_score = zero_score
                best_model = copy.deepcopy(pretrained)
                best_seed = seed
                best_update = "zero_shot"

            for train_size in (8, 16, 32, 64, 128):
                model = copy.deepcopy(pretrained)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                for parameter in model.head.parameters():
                    parameter.requires_grad_(True)
                subset = board_train.take(torch.arange(train_size))
                log = train_sparse_controller(
                    config,
                    model,
                    subset,
                    fixed_angles,
                    seed=seed + 9000 + train_size,
                    mode="coherent_iq",
                    steps=100,
                    lr=3.0e-4,
                    coefficient_mode="calibrated_nominal",
                )
                for row in log:
                    training_log.append(
                        {
                            "topology": config.topology_name,
                            "method": "single_board_head_only",
                            "measurement_count": recommended,
                            "train_samples": train_size,
                            "train_profiles": 1,
                            "trainable_parameters": sum(
                                p.numel() for p in model.parameters() if p.requires_grad
                            ),
                            "seed": seed,
                            **row,
                        }
                    )
                metric = evaluate_sparse_model(
                    config,
                    model,
                    board_test,
                    name="single_board_head_only",
                    seed=seed,
                )
                per_seed.append(
                    {
                        **metric,
                        "method": "single_board_head_only",
                        "measurement_count": recommended,
                        "train_samples": train_size,
                        "train_profiles": 1,
                        "label_mode": "coherent_iq",
                    }
                )
                if train_size == 128:
                    validation = evaluate_sparse_model(
                        config,
                        model,
                        board_validation,
                        name="single_board_head_only_validation",
                        seed=seed,
                    )
                    score = float(validation["mean_oracle_fraction"])
                    if score > best_score:
                        best_score = score
                        best_model = copy.deepcopy(model)
                        best_seed = seed
                        best_update = "head_only_P128"

        if best_model is None or best_seed is None or best_update is None:
            raise RuntimeError("refinement selection failed")
        checkpoint_name = f"v24_{config.topology_name}_low_update_controller.pt"
        torch.save(
            {
                "schema_version": 2,
                "algorithm_version": ALGORITHM_VERSION,
                "array_config": asdict(config),
                "model_config": {
                    "width": 16,
                    "num_blocks": 3,
                    "global_context": "none",
                    "state_dim": 4,
                    "output_bits": None,
                },
                "training_regime": "single_board_low_update",
                "selected_update": best_update,
                "selection_split": "single_board_validation",
                "selected_seed": best_seed,
                "total_measurement_count": recommended,
                "dynamic_target_measurement_count": 1,
                "fixed_measurement_angles_rad": fixed_angles,
                "measurement_mode": "coherent_iq",
                "state_dict": best_model.state_dict(),
            },
            RESULT_DIR / checkpoint_name,
        )
        payload["checkpoints"][config.topology_name]["single_board_low_update"] = (
            checkpoint_name
        )
        selected[config.topology_name] = {
            "checkpoint": checkpoint_name,
            "selected_seed": best_seed,
            "selected_update": best_update,
            "validation_mean_oracle_fraction": best_score,
        }
        print(
            json.dumps(
                {"stage": "refinement", "topology": config.topology_name, **selected[config.topology_name]}
            ),
            flush=True,
        )

    payload["per_seed"] = per_seed
    payload["aggregate"] = aggregate_rows(
        [row for row in per_seed if int(row["seed"]) >= 0]
    )
    payload["refinement"] = {
        "purpose": "avoid catastrophic forgetting during single-board adaptation",
        "head_only_trainable_parameters": 34,
        "learning_rate": 3.0e-4,
        "steps": 100,
        "selection": selected,
    }
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(RESULT_DIR / "per_seed.csv", per_seed)
    write_csv(RESULT_DIR / "aggregate.csv", payload["aggregate"])
    write_csv(RESULT_DIR / "training_log.csv", training_log)


if __name__ == "__main__":
    main()
