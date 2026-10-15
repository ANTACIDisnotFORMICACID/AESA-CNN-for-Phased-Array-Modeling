"""Re-evaluate V2.4 integer output and create paired statistical comparisons.

This is intentionally separate from the 7-minute training runner so a quantizer-only
algorithm correction never causes stochastic retraining or overwrites training logs.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    profile_topology_model,
)
from experiment_common import TARGET, make_samples, model_forward
from physics_initialized_frontend import PhysicsInitializedApertureEstimator
from run_v24_sparse_end_to_end_study import (
    QuantizedOutputWrapper,
    evaluate_sparse_model,
    with_physics_sparse_calibration,
)
from sparse_beamforming_v24 import ComplexDiskInt8Quantizer


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
RESULT_DIR = VERSION_DIR / "output" / "controller_training"


def legacy_direction_map_macs(num_elements: int) -> int:
    """原 64×64 方向图 CNN 的 Conv/Linear 乘法 MAC（统一不计 bias）。"""

    encoder = 4 * num_elements * (
        64 * 1 * 3 + 128 * 64 * 3 + 64 * 128 * 3
    )
    fully_connected = 512 * 128 + 4096 * 512
    decoder = 4 * (
        64 * 8 * 8 * 32 * 16
        + 32 * 16 * 16 * 16 * 16
        + 16 * 32 * 32 * 8 * 16
    )
    output_head = 4 * (8 * 64 * 64 * 1 * 9)
    return encoder + fully_connected + decoder + output_head


def release_complexity_rows() -> List[Dict[str, object]]:
    scope = "Conv/Linear multiplication MACs; bias and nonlinear ops excluded"
    legacy_parameters = 2_352_770
    rows: List[Dict[str, object]] = []
    for topology, elements in (("4x4", 16), ("1x4", 4)):
        rows.append(
            {
                "model": "legacy_direction_map_cnn",
                "topology": topology,
                "input_elements": elements,
                "parameters": legacy_parameters,
                "macs_per_sample": legacy_direction_map_macs(elements),
                "fp32_parameter_bytes": legacy_parameters * 4,
                "int8_parameter_bytes_approx": legacy_parameters,
                "scope": scope,
            }
        )
    controller_profiles: Dict[str, Dict[str, int]] = {}
    for config in (
        ArrayGeometryConfig.theoretical_4x4(),
        ArrayGeometryConfig.hardware_1x4(),
    ):
        topology = config.topology_name
        controller = profile_topology_model(TopologyAdaptiveBeamformingCNN(config))
        controller_profiles[topology] = controller
        rows.append(
            {
                "model": "v24_topology_cnn",
                "topology": topology,
                "input_elements": config.num_elements,
                "parameters": controller["total_parameters"],
                "macs_per_sample": controller["macs_per_sample"],
                "fp32_parameter_bytes": controller["fp32_parameter_bytes"],
                "int8_parameter_bytes_approx": controller[
                    "int8_parameter_bytes_approx"
                ],
                "scope": scope,
            }
        )
    for config, measurements in (
        (ArrayGeometryConfig.theoretical_4x4(), 32),
        (ArrayGeometryConfig.hardware_1x4(), 12),
    ):
        topology = config.topology_name
        pseudoinverse = torch.zeros(
            config.num_elements, measurements, dtype=torch.complex64
        )
        frontend = PhysicsInitializedApertureEstimator(
            pseudoinverse, adapter_rank=4
        ).profile()
        controller = controller_profiles[topology]
        parameters = frontend["total_parameters"] + controller["total_parameters"]
        macs = frontend["macs_per_sample"] + controller["macs_per_sample"]
        rows.append(
            {
                "model": "v23_physics_frontend_plus_v24_cnn",
                "topology": topology,
                "input_elements": config.num_elements,
                "parameters": parameters,
                "macs_per_sample": macs,
                "fp32_parameter_bytes": parameters * 4,
                "int8_parameter_bytes_approx": parameters,
                "scope": scope,
            }
        )
    return rows


def release_measurement_budget_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    protocol = payload["protocol"]
    rows: List[Dict[str, object]] = []

    def append(topology: str, scenario: str, q: int, p: int) -> None:
        far = int(protocol["recommended_measurements"][topology])
        near = int(protocol["near_field_calibration_points"][topology])
        far_repeats = int(protocol["measurement_repeats_per_command"])
        near_repeats = int(protocol["near_field_calibration_repeats"])
        rows.append(
            {
                "topology": topology,
                "scenario": scenario,
                "Q_profiles": q,
                "P_commands_per_profile": p,
                "M_far": far,
                "R_far": far_repeats,
                "M_near": near,
                "R_near": near_repeats,
                "total_far_complex_observations": q * p * far * far_repeats,
                "total_near_complex_observations": q * near * near_repeats,
            }
        )

    for topology in ("4x4", "1x4"):
        append(
            topology,
            "source_sparse_training",
            int(protocol["source_train_profiles"]),
            int(protocol["directions_per_profile"]),
        )
    for topology in ("4x4", "1x4"):
        for commands in protocol["single_board_data_curve_commands_P"]:
            append(
                topology,
                f"single_board_P{commands}",
                int(protocol["single_board_data_curve_profiles_Q"]),
                int(commands),
            )
    return rows


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


def paired_comparisons(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for topology in ("4x4", "1x4"):
        for train_samples in (8, 16, 32, 64, 128):
            scratch = {
                int(row["seed"]): row
                for row in rows
                if row["topology"] == topology
                and int(row["train_samples"]) == train_samples
                and row["method"] == "single_board_scratch"
            }
            for candidate_method, comparison in (
                ("single_board_pretrained", "full_finetune_vs_scratch"),
                ("single_board_head_only", "head_only_vs_scratch"),
                ("single_board_pretrained_zero_shot", "zero_shot_vs_scratch"),
            ):
                candidate = {
                    int(row["seed"]): row
                    for row in rows
                    if row["topology"] == topology
                    and row["method"] == candidate_method
                    and (
                        int(row["train_samples"]) == train_samples
                        or candidate_method == "single_board_pretrained_zero_shot"
                    )
                }
                for metric, orientation in (
                    ("mean_oracle_fraction", "candidate_minus_scratch"),
                    ("p05_oracle_fraction", "candidate_minus_scratch"),
                    ("mean_gain_gap_db", "candidate_minus_scratch"),
                    ("pattern_nmse_db", "scratch_minus_candidate_lower_is_better"),
                ):
                    differences = []
                    for seed in (24001, 24002, 24003, 24004, 24005):
                        candidate_value = float(candidate[seed][metric])
                        scratch_value = float(scratch[seed][metric])
                        differences.append(
                            scratch_value - candidate_value
                            if metric == "pattern_nmse_db"
                            else candidate_value - scratch_value
                        )
                    mean = statistics.fmean(differences)
                    std = statistics.stdev(differences)
                    ci = 2.776 * std / math.sqrt(len(differences))
                    output.append(
                        {
                            "topology": topology,
                            "comparison": comparison,
                            "train_samples": train_samples,
                            "metric": metric,
                            "orientation": orientation,
                            "paired_seeds": len(differences),
                            "mean_difference": mean,
                            "ci95_low": mean - ci,
                            "ci95_high": mean + ci,
                            "supports_candidate_better": mean - ci > 0.0,
                        }
                    )
    return output


def load_board_test(
    config: ArrayGeometryConfig, topology_index: int
):
    raw = make_samples(
        256,
        TARGET,
        config,
        seed=7275 + topology_index,
        targets_per_profile=256,
    )
    calibrated = with_physics_sparse_calibration(
        config, raw, seed=7475 + topology_index
    )
    return calibrated.take(torch.arange(192, 256))


def main() -> None:
    torch.set_num_threads(1)
    metrics_path = RESULT_DIR / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    quantization: Dict[str, object] = {}
    checkpoint_validation: Dict[str, object] = {}

    for topology_index, config in enumerate(
        (ArrayGeometryConfig.theoretical_4x4(), ArrayGeometryConfig.hardware_1x4())
    ):
        topology = config.topology_name
        checkpoint_name = payload["checkpoints"][topology]["single_board_low_update"]
        checkpoint = torch.load(
            RESULT_DIR / checkpoint_name, map_location="cpu", weights_only=True
        )
        model = TopologyAdaptiveBeamformingCNN(
            config,
            width=int(checkpoint["model_config"]["width"]),
            num_blocks=int(checkpoint["model_config"]["num_blocks"]),
            global_context=str(checkpoint["model_config"]["global_context"]),
            state_dim=int(checkpoint["model_config"]["state_dim"]),
            output_bits=None,
        )
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"{topology} checkpoint failed strict load")
        test = load_board_test(config, topology_index)
        selected_seed = int(checkpoint["selected_seed"])
        fp32_metric = evaluate_sparse_model(
            config, model, test, name="best_single_board_sparse_fp32", seed=selected_seed
        )
        int8_model = QuantizedOutputWrapper(model)
        int8_metric = evaluate_sparse_model(
            config,
            int8_model,
            test,
            name="best_single_board_sparse_int8_output",
            seed=selected_seed,
        )
        with torch.no_grad():
            fp32_output = model_forward(model, test)["weights_iq"]
            integer = ComplexDiskInt8Quantizer()(fp32_output)
        quantization[topology] = {
            "fp32": fp32_metric,
            "int8_output": int8_metric,
            "mean_oracle_fraction_delta": int8_metric["mean_oracle_fraction"]
            - fp32_metric["mean_oracle_fraction"],
            "gain_gap_delta_db": int8_metric["mean_gain_gap_db"]
            - fp32_metric["mean_gain_gap_db"],
            "max_integer_radius_sq": int(integer["max_integer_radius_sq"]),
            "integer_radius_limit_sq": 127 * 127,
            "input_disk_projection_rate": float(integer["input_disk_projection_rate"]),
            "lattice_constraint_adjustment_rate": float(
                integer["lattice_constraint_adjustment_rate"]
            ),
            "scope": "exact nearest INT8 output-disk lattice reference; internal CNN remains FP32",
        }
        checkpoint_validation[topology] = {
            "checkpoint": checkpoint_name,
            "strict_load": True,
            "selected_seed": selected_seed,
        }
        golden = test.take(torch.arange(8))
        with torch.no_grad():
            golden_fp32 = model_forward(model, golden)["weights_iq"]
            golden_integer = ComplexDiskInt8Quantizer()(golden_fp32)
        np.savez_compressed(
            RESULT_DIR / f"golden_vectors_{topology}.npz",
            targets_rad=golden.targets.numpy(),
            initial_weights_iq=golden.initial.numpy(),
            calibration_iq=golden.calibration_input.numpy(),
            position_errors=golden.position_input.numpy(),
            state=golden.state_input.numpy(),
            output_fp32_iq=golden_fp32.numpy(),
            output_int8_iq=golden_integer["codes_iq"].numpy(),
        )

    comparisons = paired_comparisons(payload["per_seed"])
    payload["quantization"] = quantization
    payload["checkpoint_validation"] = checkpoint_validation
    payload["paired_comparisons"] = comparisons
    payload["postprocess"] = {
        "quantizer": "exact_nearest_integer_lattice_point_in_complex_unit_disk",
        "student_t_ci": "t(0.975,4)=2.776",
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "paired_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(RESULT_DIR / "paired_comparisons.csv", comparisons)
    write_csv(RESULT_DIR / "complexity.csv", release_complexity_rows())
    write_csv(
        RESULT_DIR / "measurement_budget.csv",
        release_measurement_budget_rows(payload),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "quantization": quantization,
                "paired_comparisons": len(comparisons),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
