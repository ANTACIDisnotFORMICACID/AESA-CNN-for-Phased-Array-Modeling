"""验证 V2.5 内嵌的 V2.4 控制器训练子包及全部冻结数值证据。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from beamforming_pre_fpga import ArrayGeometryConfig, TopologyAdaptiveBeamformingCNN
from export_v24_controller_bundle import check_bundle
from finalize_v24_results import (
    paired_comparisons,
    release_complexity_rows,
    release_measurement_budget_rows,
)
from run_v24_sparse_end_to_end_study import aggregate_rows
from sparse_beamforming_v24 import ComplexDiskInt8Quantizer, PreFpgaAlgorithmContract


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
INPUT_DIR = VERSION_DIR / "input" / "controller_training"
OUTPUT_DIR = VERSION_DIR / "output" / "controller_training"
PICTURE_DIR = OUTPUT_DIR / "pictures"

CONTROLLER_CODE_FILES = {
    "beamforming_common.py", "beamforming_pre_fpga.py", "experiment_common.py",
    "export_v24_controller_bundle.py", "finalize_v24_results.py",
    "physics_initialized_frontend.py", "run_v24_finetune_refinement.py",
    "run_v24_release_pipeline.py", "run_v24_sparse_end_to_end_study.py",
    "sparse_beamforming_v24.py", "test_beamforming_common.py",
    "test_beamforming_pre_fpga.py", "test_experiment_common.py",
    "validate_v24_controller_artifacts.py",
}
OUTPUT_FILES = {
    "KNOWN_LIMITATIONS.md", "aggregate.csv", "comparison_results.json",
    "complexity.csv", "contracts.json", "golden_vectors_1x4.npz",
    "golden_vectors_4x4.npz", "measurement_budget.csv",
    "measurement_design.csv", "metrics.json", "paired_comparisons.csv",
    "paired_comparisons.json", "per_seed.csv", "test_log.txt",
    "v24_1x4_calibrated_surrogate_controller.pt",
    "v24_1x4_low_update_controller.pt", "v24_1x4_oracle_simulation_controller.pt",
    "v24_1x4_single_board_adapted_controller.pt",
    "v24_4x4_calibrated_surrogate_controller.pt", "v24_4x4_low_update_controller.pt",
    "v24_4x4_oracle_simulation_controller.pt",
    "v24_4x4_single_board_adapted_controller.pt", "报告.md",
}
PICTURE_FILES = {
    "golden_case_1x4.png", "golden_case_4x4.png", "performance_comparison.png"
}
CSV_ROWS = {
    "aggregate.csv": 56, "complexity.csv": 6, "measurement_budget.csv": 12,
    "measurement_design.csv": 12, "paired_comparisons.csv": 120,
    "per_seed.csv": 282,
}


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"非有限数值：{path}={value}")


def load_finite_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"CSV 为空：{path.name}")
    for row_index, row in enumerate(rows, start=2):
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise AssertionError(
                    f"CSV 非有限数值：{path.name}:{row_index}:{key}={value}"
                )
    return rows


def _cell_matches(cell: str, expected: Any) -> bool:
    if expected is None:
        return cell == ""
    if isinstance(expected, bool):
        return cell.lower() == str(expected).lower()
    if isinstance(expected, int):
        return int(cell) == expected
    if isinstance(expected, float):
        return math.isclose(float(cell), expected, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(expected, (dict, list)):
        return ast.literal_eval(cell) == expected
    return cell == str(expected)


def assert_csv_mirrors(
    actual: list[dict[str, str]], expected: list[dict[str, Any]], name: str
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{name} 行数与权威数据不一致")
    for index, (actual_row, expected_row) in enumerate(zip(actual, expected), start=2):
        for key, cell in actual_row.items():
            if key not in expected_row and cell == "":
                continue
            if not _cell_matches(cell, expected_row.get(key)):
                raise AssertionError(
                    f"{name}:{index}:{key} 不一致：{cell!r} != {expected_row.get(key)!r}"
                )


def load_controller(path: Path) -> tuple[dict[str, Any], TopologyAdaptiveBeamformingCNN]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("algorithm_version") != "2.4.0":
        raise AssertionError(f"checkpoint 版本不匹配：{path.name}")
    config = ArrayGeometryConfig(**checkpoint["array_config"])
    model_config = checkpoint["model_config"]
    model = TopologyAdaptiveBeamformingCNN(
        config, width=model_config["width"], num_blocks=model_config["num_blocks"],
        global_context=model_config["global_context"], state_dim=model_config["state_dim"],
        output_bits=None,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("selection_split") == "target_test":
        raise AssertionError(f"{path.name} 使用了测试集选模")
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise AssertionError(f"{path.name} 含非有限参数")
    model.eval()
    return checkpoint, model


def validate_v24() -> dict[str, Any]:
    actual_code_files = {path.name for path in CODE_DIR.glob("*.py")}
    missing_code_files = CONTROLLER_CODE_FILES - actual_code_files
    if missing_code_files:
        raise AssertionError(
            f"V2.4 控制器子包缺少代码依赖：{sorted(missing_code_files)}"
        )
    if {path.name for path in INPUT_DIR.iterdir() if path.is_file()} != {
        "comparison_inputs.json"
    }:
        raise AssertionError("V2.4 input/ 必须精确包含 comparison_inputs.json")
    if {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} != OUTPUT_FILES:
        raise AssertionError("V2.4 output/ 顶层文件集合不匹配")
    if {path.name for path in PICTURE_DIR.iterdir() if path.is_file()} != PICTURE_FILES:
        raise AssertionError("V2.4 output/pictures 文件集合不匹配")
    check_bundle()

    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert_finite(metrics)
    if metrics["algorithm_version"] != "2.4.0" or len(metrics["train_seeds"]) != 5:
        raise AssertionError("V2.4 版本或训练种子数不匹配")
    selection = metrics["protocol"]["checkpoint_selection"]
    if selection.get("test_used_for_selection") is not False:
        raise AssertionError("V2.4 checkpoint 选择协议未明确排除测试集")
    if any(int(row["seeds"]) != 5 for row in metrics["aggregate"]):
        raise AssertionError("V2.4 aggregate 存在非5种子分组")
    if any(float(row["amplitude_violation_rate"]) != 0.0 for row in metrics["per_seed"]):
        raise AssertionError("V2.4 出现复幅度约束违规")

    csv_data = {name: load_finite_csv(OUTPUT_DIR / name) for name in CSV_ROWS}
    actual_counts = {name: len(rows) for name, rows in csv_data.items()}
    if actual_counts != CSV_ROWS:
        raise AssertionError(f"V2.4 CSV 行数不匹配：{actual_counts}")
    assert_csv_mirrors(csv_data["aggregate.csv"], metrics["aggregate"], "aggregate.csv")
    assert_csv_mirrors(
        csv_data["measurement_design.csv"], metrics["measurement_design"],
        "measurement_design.csv",
    )
    assert_csv_mirrors(csv_data["per_seed.csv"], metrics["per_seed"], "per_seed.csv")
    assert_csv_mirrors(
        csv_data["paired_comparisons.csv"], metrics["paired_comparisons"],
        "paired_comparisons.csv",
    )
    assert_csv_mirrors(
        csv_data["complexity.csv"], release_complexity_rows(), "complexity.csv"
    )
    assert_csv_mirrors(
        csv_data["measurement_budget.csv"], release_measurement_budget_rows(metrics),
        "measurement_budget.csv",
    )
    recomputed_aggregate = aggregate_rows(
        [row for row in metrics["per_seed"] if int(row["seed"]) >= 0]
    )
    if recomputed_aggregate != metrics["aggregate"]:
        raise AssertionError("V2.4 aggregate 不能由 per_seed 确定性重算")
    if paired_comparisons(metrics["per_seed"]) != metrics["paired_comparisons"]:
        raise AssertionError("V2.4 paired comparisons 不能由 per_seed 重算")

    contracts = json.loads((OUTPUT_DIR / "contracts.json").read_text(encoding="utf-8"))
    paired_json = json.loads(
        (OUTPUT_DIR / "paired_comparisons.json").read_text(encoding="utf-8")
    )
    if contracts != metrics["contracts"] or paired_json != metrics["paired_comparisons"]:
        raise AssertionError("V2.4 JSON 镜像与 metrics.json 不一致")

    loaded: list[str] = []
    golden_checked: list[str] = []
    for topology, recommended, config in (
        ("4x4", 32, ArrayGeometryConfig.theoretical_4x4()),
        ("1x4", 8, ArrayGeometryConfig.hardware_1x4()),
    ):
        for regime, checkpoint_name in metrics["checkpoints"][topology].items():
            checkpoint, _model = load_controller(OUTPUT_DIR / checkpoint_name)
            if checkpoint["array_config"] != asdict(config):
                raise AssertionError(f"{topology}/{regime} 阵列配置不匹配")
            loaded.append(checkpoint_name)
        expected_contract = PreFpgaAlgorithmContract.create(
            config, recommended, "coherent_iq"
        ).to_dict()
        if contracts[topology] != expected_contract:
            raise AssertionError(f"{topology} FPGA 前合同不能从代码重建")

        low_name = metrics["checkpoints"][topology]["single_board_low_update"]
        _checkpoint, model = load_controller(OUTPUT_DIR / low_name)
        with np.load(
            OUTPUT_DIR / f"golden_vectors_{topology}.npz", allow_pickle=False
        ) as data:
            with torch.no_grad():
                replay = model(
                    torch.from_numpy(data["initial_weights_iq"]).float(),
                    torch.from_numpy(data["targets_rad"]).float(),
                    calibration_iq=torch.from_numpy(data["calibration_iq"]).float(),
                    position_errors=torch.from_numpy(data["position_errors"]).float(),
                    state=torch.from_numpy(data["state"]).float(),
                    quantize_output=False,
                )["weights_iq"]
                integer = ComplexDiskInt8Quantizer()(replay)["codes_iq"]
            np.testing.assert_allclose(
                replay.numpy(), data["output_fp32_iq"], rtol=0.0, atol=3e-7
            )
            np.testing.assert_array_equal(integer.numpy(), data["output_int8_iq"])
            if not all(np.isfinite(data[name]).all() for name in data.files):
                raise AssertionError(f"{topology} golden vector 含非有限数")
            codes = data["output_int8_iq"].astype(np.int64)
            if int((codes[:, 0] ** 2 + codes[:, 1] ** 2).max()) > 127 * 127:
                raise AssertionError(f"{topology} golden vector 越过整数圆盘")
        golden_checked.append(topology)

        design = next(
            row for row in metrics["measurement_design"]
            if row["topology"] == topology and int(row["measurement_count"]) == recommended
        )
        if design["full_column_rank_rate"] != 1.0 or design["condition_number"]["p90"] > 2.0:
            raise AssertionError(f"{topology} 推荐测点设计未达标")
        for method in (
            "single_board_scratch", "single_board_pretrained", "single_board_head_only"
        ):
            groups = [
                row for row in metrics["aggregate"]
                if row["topology"] == topology and row["method"] == method
            ]
            if len(groups) != 5:
                raise AssertionError(f"{topology}/{method} 学习曲线不完整")
        quantization = metrics["quantization"][topology]
        if quantization["mean_oracle_fraction_delta"] < -0.005:
            raise AssertionError(f"{topology} INT8 oracle fraction 退化超过0.005")
        if quantization["gain_gap_delta_db"] < -0.1:
            raise AssertionError(f"{topology} INT8 增益退化超过0.1dB")
        if quantization["max_integer_radius_sq"] > 127 * 127:
            raise AssertionError(f"{topology} INT8 码字越过复圆盘")

    protocol = metrics["protocol"]
    if protocol["measurement_repeats_per_command"] != 1:
        raise AssertionError("V2.4 数据预算必须使用固定单次观测")
    if protocol["single_board_data_curve_profiles_Q"] != 1:
        raise AssertionError("V2.4 同板曲线必须固定 Q=1")
    test_log = (OUTPUT_DIR / "test_log.txt").read_text(encoding="utf-8")
    if "22 tests, PASS" not in test_log:
        raise AssertionError("V2.4 test_log 未记录独立分支 22/22 结果")
    return {
        "version": "v2.4", "package_layout": "pass",
        "scope": "v2.5/controller_training",
        "controller_code_dependencies": len(CONTROLLER_CODE_FILES),
        "input_json_files": 1,
        "output_files": len(OUTPUT_FILES), "picture_files": len(PICTURE_FILES),
        "csv_rows": actual_counts, "train_seed_count": 5,
        "checkpoints_strict_loaded": loaded,
        "golden_vectors_replayed": golden_checked,
        "comparison_bundle": "pass", "selection_protocol": "validation_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=("v2.4",))
    parser.parse_args()
    print(json.dumps(validate_v24(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
