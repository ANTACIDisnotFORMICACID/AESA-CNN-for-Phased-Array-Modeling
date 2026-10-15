"""Total release validator for the self-contained V2.5 project package.

The validator deliberately separates two evidence scopes:

* ``controller_training`` is the complete inherited V2.4 CNN training package;
* ``full_pattern`` is the V2.5 zero-parameter physics renderer and compatibility
  layer.

Passing this file proves reproducibility against the recorded simulation and
numerical contracts.  It does not claim FPGA deployment or OTA validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    profile_topology_model,
)
from full_pattern_v25 import (
    ALGORITHM_VERSION,
    FullPatternPhysics,
    PatternGridConfig,
    V25BeamformingSystem,
    legacy_forward_1x4_config,
    profile_full_pattern,
)
from validate_v24_controller_artifacts import load_controller, validate_v24


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
INPUT_DIR = VERSION_DIR / "input"
OUTPUT_DIR = VERSION_DIR / "output"
CONTROLLER_INPUT_DIR = INPUT_DIR / "controller_training"
FULL_INPUT_DIR = INPUT_DIR / "full_pattern"
CONTROLLER_OUTPUT_DIR = OUTPUT_DIR / "controller_training"
FULL_OUTPUT_DIR = OUTPUT_DIR / "full_pattern"
CONTROLLER_PICTURE_DIR = CONTROLLER_OUTPUT_DIR / "pictures"
FULL_PICTURE_DIR = FULL_OUTPUT_DIR / "pictures"

CODE_FILES = {
    "beamforming_common.py",
    "beamforming_pre_fpga.py",
    "experiment_common.py",
    "export_v24_controller_bundle.py",
    "export_v25_comparison_bundle.py",
    "finalize_v24_results.py",
    "full_pattern_v25.py",
    "physics_initialized_frontend.py",
    "run_v24_finetune_refinement.py",
    "run_v24_release_pipeline.py",
    "run_v24_sparse_end_to_end_study.py",
    "run_v25_full_pattern_study.py",
    "run_v25_release_pipeline.py",
    "sparse_beamforming_v24.py",
    "test_beamforming_common.py",
    "test_beamforming_pre_fpga.py",
    "test_experiment_common.py",
    "test_full_pattern_v25.py",
    "validate_v24_controller_artifacts.py",
    "validate_v25_artifacts.py",
}

CONTROLLER_INPUT_FILES = {"comparison_inputs.json"}
FULL_INPUT_FILES = {"comparison_inputs.json", "historical_replay_input.json"}

CONTROLLER_OUTPUT_FILES = {
    "KNOWN_LIMITATIONS.md",
    "aggregate.csv",
    "comparison_results.json",
    "complexity.csv",
    "contracts.json",
    "golden_vectors_1x4.npz",
    "golden_vectors_4x4.npz",
    "measurement_budget.csv",
    "measurement_design.csv",
    "metrics.json",
    "paired_comparisons.csv",
    "paired_comparisons.json",
    "per_seed.csv",
    "test_log.txt",
    "v24_1x4_calibrated_surrogate_controller.pt",
    "v24_1x4_low_update_controller.pt",
    "v24_1x4_oracle_simulation_controller.pt",
    "v24_1x4_single_board_adapted_controller.pt",
    "v24_4x4_calibrated_surrogate_controller.pt",
    "v24_4x4_low_update_controller.pt",
    "v24_4x4_oracle_simulation_controller.pt",
    "v24_4x4_single_board_adapted_controller.pt",
    "报告.md",
}
CONTROLLER_PICTURE_FILES = {
    "golden_case_1x4.png",
    "golden_case_4x4.png",
    "performance_comparison.png",
}

FULL_OUTPUT_FILES = {
    "KNOWN_LIMITATIONS.md",
    "V2.5组会讲解与答辩手册.md",
    "benchmark.csv",
    "comparison_results.json",
    "complexity.csv",
    "contract_1x4.json",
    "contract_4x4.json",
    "golden_full_pattern_1x4.npz",
    "golden_full_pattern_4x4.npz",
    "intermediate_values.json",
    "legacy_full_pattern_1x4.json",
    "metrics.json",
    "numerical_cases.csv",
    "test_log.txt",
    "报告.md",
}
FULL_PICTURE_FILES = {
    "golden_full_pattern_1x4.png",
    "golden_full_pattern_4x4.png",
    "performance_comparison.png",
}

NUMERICAL_GENERATED_FILES = [
    "benchmark.csv",
    "complexity.csv",
    "contract_4x4.json",
    "contract_1x4.json",
    "golden_full_pattern_4x4.npz",
    "golden_full_pattern_1x4.npz",
    "intermediate_values.json",
    "legacy_full_pattern_1x4.json",
    "metrics.json",
    "numerical_cases.csv",
]

RELEASE_FILES = [
    "KNOWN_LIMITATIONS.md",
    "V2.5组会讲解与答辩手册.md",
    "benchmark.csv",
    "comparison_results.json",
    "complexity.csv",
    "contract_1x4.json",
    "contract_4x4.json",
    "golden_full_pattern_1x4.npz",
    "golden_full_pattern_4x4.npz",
    "intermediate_values.json",
    "legacy_full_pattern_1x4.json",
    "metrics.json",
    "numerical_cases.csv",
    "test_log.txt",
    "报告.md",
]

RELEASE_PICTURE_FILES = [
    "golden_full_pattern_1x4.png",
    "golden_full_pattern_4x4.png",
    "performance_comparison.png",
]

EXPECTED_ACCEPTANCE_GATES = {
    "all_fp64_max_abs_le_1e-10",
    "all_fp32_max_abs_le_5e-5",
    "legacy_scalar_formula_parity_le_1e-10",
    "untouched_historical_json_field_parity_le_1e-10",
    "v24_controller_bit_exact",
}

HISTORICAL_SOURCE_COMMIT = "bf352ac73ba2e2024081b98591e05409eac89317"
HISTORICAL_SOURCE_BLOB = "6d11801fcc420937c381e475b7bfbe8345a2ddbf"
HISTORICAL_SOURCE_PATH = "output_json/output_20260425_202604_01be8f59.json"


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise AssertionError(f"non-finite value at {path}: {value}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root must be an object: {path}")
    assert_finite(value, path.name)
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"CSV is empty: {path.name}")
    for row_number, row in enumerate(rows, start=2):
        for key, cell in row.items():
            if cell in (None, ""):
                continue
            try:
                number = float(cell)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise AssertionError(
                    f"non-finite CSV value: {path.name}:{row_number}:{key}={cell}"
                )
    return rows


def _cell_matches(cell: str | None, expected: Any) -> bool:
    if expected is None:
        return cell in (None, "")
    if cell in (None, ""):
        return False
    if isinstance(expected, bool):
        return cell.lower() == str(expected).lower()
    if isinstance(expected, int):
        return int(cell) == expected
    if isinstance(expected, float):
        return math.isclose(float(cell), expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
    return cell == str(expected)


def assert_csv_mirror(
    actual: list[dict[str, str]], expected: list[dict[str, Any]], name: str
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{name} row count differs from metrics.json")
    for row_number, (actual_row, expected_row) in enumerate(
        zip(actual, expected), start=2
    ):
        for key, expected_value in expected_row.items():
            if key not in actual_row or not _cell_matches(
                actual_row[key], expected_value
            ):
                raise AssertionError(
                    f"{name}:{row_number}:{key} differs from metrics.json"
                )
        for key, cell in actual_row.items():
            if key not in expected_row and cell not in (None, ""):
                raise AssertionError(
                    f"{name}:{row_number}:{key} unexpectedly contains {cell!r}"
                )


def _assert_exact_files(directory: Path, expected: set[str], label: str) -> None:
    if not directory.is_dir():
        raise AssertionError(f"missing directory: {label}={directory}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise AssertionError(
            f"{label} inventory mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def validate_layout() -> dict[str, int]:
    version_dirs = {
        path.name
        for path in VERSION_DIR.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    if version_dirs != {"code", "input", "output"}:
        raise AssertionError(f"V2.5 top-level directory mismatch: {version_dirs}")
    input_dirs = {path.name for path in INPUT_DIR.iterdir() if path.is_dir()}
    output_dirs = {path.name for path in OUTPUT_DIR.iterdir() if path.is_dir()}
    if input_dirs != {"controller_training", "full_pattern"}:
        raise AssertionError(f"V2.5 input scopes mismatch: {input_dirs}")
    if output_dirs != {"controller_training", "full_pattern"}:
        raise AssertionError(f"V2.5 output scopes mismatch: {output_dirs}")
    if any(path.is_file() for path in INPUT_DIR.iterdir()):
        raise AssertionError("V2.5 input root must not contain loose files")
    if any(path.is_file() for path in OUTPUT_DIR.iterdir()):
        raise AssertionError("V2.5 output root must not contain loose files")

    _assert_exact_files(CODE_DIR, CODE_FILES, "code")
    _assert_exact_files(
        CONTROLLER_INPUT_DIR, CONTROLLER_INPUT_FILES, "input/controller_training"
    )
    _assert_exact_files(FULL_INPUT_DIR, FULL_INPUT_FILES, "input/full_pattern")
    _assert_exact_files(
        CONTROLLER_OUTPUT_DIR,
        CONTROLLER_OUTPUT_FILES,
        "output/controller_training",
    )
    _assert_exact_files(
        CONTROLLER_PICTURE_DIR,
        CONTROLLER_PICTURE_FILES,
        "output/controller_training/pictures",
    )
    _assert_exact_files(FULL_OUTPUT_DIR, FULL_OUTPUT_FILES, "output/full_pattern")
    _assert_exact_files(
        FULL_PICTURE_DIR, FULL_PICTURE_FILES, "output/full_pattern/pictures"
    )
    return {
        "code": len(CODE_FILES),
        "controller_input": len(CONTROLLER_INPUT_FILES),
        "full_pattern_input": len(FULL_INPUT_FILES),
        "controller_output": len(CONTROLLER_OUTPUT_FILES),
        "controller_pictures": len(CONTROLLER_PICTURE_FILES),
        "full_pattern_output": len(FULL_OUTPUT_FILES),
        "full_pattern_pictures": len(FULL_PICTURE_FILES),
    }


def validate_metrics_header(metrics: dict[str, Any]) -> None:
    if ALGORITHM_VERSION != "2.5.0" or metrics.get("algorithm_version") != ALGORITHM_VERSION:
        raise AssertionError("V2.5 algorithm version mismatch")
    if metrics.get("seed") != 20260823:
        raise AssertionError("V2.5 frozen study seed mismatch")
    grid = metrics.get("grid", {})
    if grid != {
        "theta_deg": [60.0, 120.0, 64],
        "phi_deg": [-30.0, 30.0, 64],
        "matrix_semantics": "[phi_row, theta_column]",
    }:
        raise AssertionError("V2.5 angular-grid contract mismatch")
    gates = metrics.get("acceptance_gates", {})
    if set(gates) != EXPECTED_ACCEPTANCE_GATES or not all(
        value is True for value in gates.values()
    ):
        raise AssertionError(f"V2.5 acceptance gates mismatch: {gates}")
    if "result_files" in metrics:
        raise AssertionError("obsolete ambiguous metrics.result_files is forbidden")
    if metrics.get("numerical_generated_files") != NUMERICAL_GENERATED_FILES:
        raise AssertionError(
            "metrics.numerical_generated_files must list exactly the ten files "
            "written by the numerical study"
        )
    if metrics.get("release_files") != RELEASE_FILES:
        raise AssertionError("metrics.release_files must list the exact 15-file release")
    if metrics.get("release_picture_files") != RELEASE_PICTURE_FILES:
        raise AssertionError(
            "metrics.release_picture_files must list the exact three release pictures"
        )
    if set(RELEASE_FILES) != FULL_OUTPUT_FILES:
        raise AssertionError("validator release manifest and layout constants disagree")
    if set(RELEASE_PICTURE_FILES) != FULL_PICTURE_FILES:
        raise AssertionError("validator picture manifest and layout constants disagree")
    for name in NUMERICAL_GENERATED_FILES:
        if not (FULL_OUTPUT_DIR / name).is_file():
            raise AssertionError(f"runner result is missing: {name}")


def validate_numerical_cases(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = load_csv(FULL_OUTPUT_DIR / "numerical_cases.csv")
    if len(rows) != 64:
        raise AssertionError(f"expected 64 numerical cases, got {len(rows)}")
    identities = {
        (
            row["topology"],
            int(row["case_index"]),
            row["mode"],
            row["dtype"],
        )
        for row in rows
    }
    expected_identities = {
        (topology, case_index, mode, dtype)
        for topology in ("4x4", "1x4")
        for case_index in range(8)
        for mode in ("ideal", "position_amp_phase_coupling")
        for dtype in ("float64", "float32")
    }
    if identities != expected_identities or len(identities) != len(rows):
        raise AssertionError("numerical_cases.csv has missing or duplicate cases")

    maxima: dict[str, float] = {"float64": 0.0, "float32": 0.0}
    for topology in ("4x4", "1x4"):
        for mode in ("ideal", "position_amp_phase_coupling"):
            for dtype in ("float64", "float32"):
                subset = [
                    row
                    for row in rows
                    if row["topology"] == topology
                    and row["mode"] == mode
                    and row["dtype"] == dtype
                ]
                if len(subset) != 8:
                    raise AssertionError(f"bad numerical group: {topology}/{mode}/{dtype}")
                recomputed = {
                    "case_count": 8,
                    "worst_max_abs_error": max(
                        float(row["max_abs_error"]) for row in subset
                    ),
                    "worst_relative_l2": max(
                        float(row["relative_l2"]) for row in subset
                    ),
                    "worst_nmse_db": max(float(row["nmse_db"]) for row in subset),
                    "mean_rmse": sum(float(row["rmse"]) for row in subset) / 8.0,
                }
                summary = metrics["accuracy_summary"][topology][mode][dtype]
                for key, expected in recomputed.items():
                    actual = summary[key]
                    if isinstance(expected, int):
                        matches = actual == expected
                    else:
                        matches = math.isclose(
                            float(actual), expected, rel_tol=1.0e-13, abs_tol=1.0e-18
                        )
                    if not matches:
                        raise AssertionError(
                            f"accuracy summary mismatch: {topology}/{mode}/{dtype}/{key}"
                        )
                maxima[dtype] = max(maxima[dtype], recomputed["worst_max_abs_error"])
    if maxima["float64"] > 1.0e-10 or maxima["float32"] > 5.0e-5:
        raise AssertionError(f"full-pattern numerical tolerance failed: {maxima}")
    return {"count": len(rows), "worst_max_abs_error": maxima}


def validate_contracts() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for topology, config in (
        ("4x4", ArrayGeometryConfig.theoretical_4x4()),
        ("1x4", ArrayGeometryConfig.hardware_1x4()),
    ):
        contract = load_json(FULL_OUTPUT_DIR / f"contract_{topology}.json")
        renderer = FullPatternPhysics(config, PatternGridConfig.legacy_64())
        expected = renderer.export_contract()
        if contract != expected:
            raise AssertionError(f"{topology} contract cannot be rebuilt from code")
        if renderer.trainable_parameter_count != 0 or renderer.state_dict():
            raise AssertionError(f"{topology} renderer is not a zero-state observer")
        if contract["pattern_output_shape"] != ["B", 2, 64, 64]:
            raise AssertionError(f"{topology} full-pattern output shape mismatch")
        result[topology] = contract
    return result


def validate_complexity(
    metrics: dict[str, Any], controllers: dict[str, torch.nn.Module]
) -> dict[str, Any]:
    rows = load_csv(FULL_OUTPUT_DIR / "complexity.csv")
    if len(rows) != 2:
        raise AssertionError("complexity.csv must contain exactly two rows")
    assert_csv_mirror(rows, metrics["complexity"], "complexity.csv")
    summaries: dict[str, Any] = {}
    for row in rows:
        topology = row["topology"]
        if topology not in controllers:
            raise AssertionError(f"unexpected complexity topology: {topology}")
        controller_profile = profile_topology_model(controllers[topology])
        config = controllers[topology].config
        renderer_profile = profile_full_pattern(
            config, controller_macs=controller_profile["macs_per_sample"]
        )
        expected_fields = {
            "controller_parameters": controller_profile["total_parameters"],
            "full_pattern_parameters": renderer_profile["trainable_parameters"],
            "total_trainable_parameters": controller_profile["total_parameters"],
            "controller_macs": controller_profile["macs_per_sample"],
            "renderer_real_macs": renderer_profile["renderer_real_macs"],
            "optional_coupling_real_macs": renderer_profile[
                "optional_coupling_real_macs"
            ],
            "optional_element_response_real_macs": renderer_profile[
                "optional_element_response_real_macs"
            ],
            "nonideal_contraction_real_macs": renderer_profile[
                "nonideal_contraction_real_macs"
            ],
            "end_to_end_real_macs": renderer_profile["end_to_end_real_macs"],
            "end_to_end_nonideal_contraction_real_macs": renderer_profile[
                "end_to_end_nonideal_contraction_real_macs"
            ],
            "cached_steering_bytes_fp32": renderer_profile[
                "cached_steering_bytes"
            ],
        }
        for key, expected in expected_fields.items():
            if int(row[key]) != int(expected):
                raise AssertionError(f"complexity mismatch: {topology}/{key}")
        if int(row["legacy_forward_cnn_parameters"]) != 2_352_770:
            raise AssertionError("legacy CNN parameter baseline changed")
        if int(row["legacy_forward_cnn_input_elements"]) != config.num_elements:
            raise AssertionError(f"legacy input width mismatch: {topology}")
        legacy_macs = int(row["legacy_forward_cnn_macs"])
        if not math.isclose(
            float(row["legacy_over_renderer_macs_ratio"]),
            legacy_macs / renderer_profile["renderer_real_macs"],
            rel_tol=1.0e-12,
        ):
            raise AssertionError(f"legacy/renderer ratio mismatch: {topology}")
        if not math.isclose(
            float(row["legacy_over_end_to_end_macs_ratio"]),
            legacy_macs / renderer_profile["end_to_end_real_macs"],
            rel_tol=1.0e-12,
        ):
            raise AssertionError(f"legacy/end-to-end ratio mismatch: {topology}")
        summaries[topology] = {
            "controller_parameters": controller_profile["total_parameters"],
            "controller_macs": controller_profile["macs_per_sample"],
            "renderer_real_macs": renderer_profile["renderer_real_macs"],
        }
    return summaries


def validate_benchmark(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = load_csv(FULL_OUTPUT_DIR / "benchmark.csv")
    if len(rows) != 6:
        raise AssertionError("benchmark.csv must contain 2 topologies x 3 operations")
    assert_csv_mirror(rows, metrics["benchmark"], "benchmark.csv")
    expected = {
        (topology, operation)
        for topology in ("4x4", "1x4")
        for operation in (
            "controller_only",
            "full_pattern_only_cached_ideal",
            "controller_plus_full_pattern",
        )
    }
    if {(row["topology"], row["operation"]) for row in rows} != expected:
        raise AssertionError("benchmark operation inventory mismatch")
    for row in rows:
        if int(row["warmup"]) != 25 or int(row["repeats"]) != 200:
            raise AssertionError("benchmark protocol changed")
        minimum = float(row["min_ms"])
        median = float(row["median_ms"])
        mean = float(row["mean_ms"])
        p95 = float(row["p95_ms"])
        maximum = float(row["max_ms"])
        if not (0.0 < minimum <= median <= p95 <= maximum):
            raise AssertionError(f"invalid benchmark order: {row}")
        if not minimum <= mean <= maximum:
            raise AssertionError(f"invalid benchmark mean: {row}")
    return {"rows": len(rows), "environment": metrics["environment"]}


def _torch_from(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array))


def validate_golden_vectors() -> dict[str, Any]:
    expected_keys = {
        "weights_iq",
        "position_errors",
        "element_response_iq",
        "coupling_real",
        "coupling_imag",
        "field_iq",
        "reference_iq",
        "theta_axis_rad",
        "phi_axis_rad",
    }
    summaries: dict[str, Any] = {}
    for topology, config, weight_shape in (
        ("4x4", ArrayGeometryConfig.theoretical_4x4(), (1, 2, 4, 4)),
        ("1x4", ArrayGeometryConfig.hardware_1x4(), (1, 2, 1, 4)),
    ):
        path = FULL_OUTPUT_DIR / f"golden_full_pattern_{topology}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise AssertionError(f"{topology} golden NPZ key inventory mismatch")
            if any(archive[key].dtype != np.float64 for key in archive.files):
                raise AssertionError(f"{topology} golden NPZ must be float64")
            if not all(np.isfinite(archive[key]).all() for key in archive.files):
                raise AssertionError(f"{topology} golden NPZ contains non-finite values")
            if archive["weights_iq"].shape != weight_shape:
                raise AssertionError(f"{topology} golden weight shape mismatch")
            if archive["position_errors"].shape != (
                1,
                3,
                config.x_num,
                config.y_num,
            ):
                raise AssertionError(f"{topology} golden position shape mismatch")
            if archive["field_iq"].shape != (1, 2, 64, 64):
                raise AssertionError(f"{topology} golden field shape mismatch")
            if archive["reference_iq"].shape != (1, 2, 64, 64):
                raise AssertionError(f"{topology} golden reference shape mismatch")

            renderer = FullPatternPhysics(config, dtype=torch.float64)
            coupling = torch.complex(
                _torch_from(archive["coupling_real"]),
                _torch_from(archive["coupling_imag"]),
            )
            with torch.no_grad():
                result = renderer(
                    _torch_from(archive["weights_iq"]),
                    position_errors=_torch_from(archive["position_errors"]),
                    element_response_iq=_torch_from(archive["element_response_iq"]),
                    coupling_iq=coupling,
                )
            np.testing.assert_allclose(
                result["field_iq"].numpy(),
                archive["field_iq"],
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                result["theta_axis_rad"].numpy(),
                archive["theta_axis_rad"],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                result["phi_axis_rad"].numpy(),
                archive["phi_axis_rad"],
                rtol=0.0,
                atol=0.0,
            )
            max_error = float(
                np.max(np.abs(archive["field_iq"] - archive["reference_iq"]))
            )
            if max_error > 1.0e-10:
                raise AssertionError(f"{topology} golden/reference mismatch: {max_error}")
            summaries[topology] = {
                "field_shape": list(archive["field_iq"].shape),
                "renderer_replay": "pass",
                "max_abs_reference_error": max_error,
            }
    return summaries


def validate_intermediate_values() -> dict[str, Any]:
    intermediate = load_json(FULL_OUTPUT_DIR / "intermediate_values.json")
    if set(intermediate) != {"4x4", "1x4"}:
        raise AssertionError("intermediate_values.json topology inventory mismatch")
    checked: dict[str, int] = {}
    for topology in ("4x4", "1x4"):
        item = intermediate[topology]
        path = FULL_OUTPUT_DIR / f"golden_full_pattern_{topology}.npz"
        config = (
            ArrayGeometryConfig.theoretical_4x4()
            if topology == "4x4"
            else ArrayGeometryConfig.hardware_1x4()
        )
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_allclose(item["weights_iq"], archive["weights_iq"][0])
            np.testing.assert_allclose(
                item["position_errors"], archive["position_errors"][0]
            )
            np.testing.assert_allclose(
                item["element_response_iq"], archive["element_response_iq"][0]
            )
            np.testing.assert_allclose(
                item["coupling_real"], archive["coupling_real"][0]
            )
            np.testing.assert_allclose(
                item["coupling_imag"], archive["coupling_imag"][0]
            )
            positions = [
                [0.0, config.x_spacing * xi, config.y_spacing * yi]
                for xi in range(config.x_num)
                for yi in range(config.y_num)
            ]
            np.testing.assert_allclose(item["array_positions"], positions)
            points = item["selected_grid_points"]
            if len(points) != 6:
                raise AssertionError(f"{topology} selected-point count mismatch")
            for point in points:
                phi_index = int(point["phi_index"])
                theta_index = int(point["theta_index"])
                field_i = float(archive["field_iq"][0, 0, phi_index, theta_index])
                field_q = float(archive["field_iq"][0, 1, phi_index, theta_index])
                reference_i = float(
                    archive["reference_iq"][0, 0, phi_index, theta_index]
                )
                reference_q = float(
                    archive["reference_iq"][0, 1, phi_index, theta_index]
                )
                expected = {
                    "phi_deg": math.degrees(float(archive["phi_axis_rad"][phi_index])),
                    "theta_deg": math.degrees(
                        float(archive["theta_axis_rad"][theta_index])
                    ),
                    "field_i": field_i,
                    "field_q": field_q,
                    "reference_i": reference_i,
                    "reference_q": reference_q,
                    "magnitude": math.hypot(field_i, field_q),
                    "magnitude_db": 20.0 * math.log10(math.hypot(field_i, field_q)),
                    "phase_rad": math.atan2(field_q, field_i) % (2.0 * math.pi),
                }
                for key, expected_value in expected.items():
                    if not math.isclose(
                        float(point[key]),
                        expected_value,
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-13,
                    ):
                        raise AssertionError(
                            f"intermediate point mismatch: {topology}/{phi_index}/{theta_index}/{key}"
                        )
            if item.get("axis_contract") != {
                "rows": "phi",
                "columns": "theta",
                "shape": [64, 64],
            }:
                raise AssertionError(f"{topology} intermediate axis contract mismatch")
        checked[topology] = len(points)
    return checked


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _complex_field_from_iq(field_iq: torch.Tensor) -> np.ndarray:
    return field_iq[0, 0].numpy() + 1j * field_iq[0, 1].numpy()


def _replay_legacy_payload(
    payload: dict[str, Any], config: ArrayGeometryConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude = torch.tensor(payload["applied_amp_errors"], dtype=torch.float64).reshape(
        1, config.x_num, config.y_num
    )
    phase = torch.tensor(
        payload["applied_phase_errors_rad"], dtype=torch.float64
    ).reshape(1, config.x_num, config.y_num)
    renderer = FullPatternPhysics(config, dtype=torch.float64)
    with torch.no_grad():
        result = renderer.from_amplitude_phase(amplitude, phase)
    field = _complex_field_from_iq(result["field_iq"])
    magnitude_db = result["magnitude_db"][0].numpy()
    phase_rad = result["phase_rad"][0].numpy()
    return field, magnitude_db, phase_rad


def validate_historical_replay(metrics: dict[str, Any]) -> dict[str, Any]:
    path = FULL_INPUT_DIR / "historical_replay_input.json"
    raw = path.read_bytes()
    # Git stores this text blob with LF while a Windows checkout may materialize
    # CRLF.  Hash the canonical Git text representation, but reject any other
    # stray carriage returns so normalization cannot hide a content change.
    canonical_raw = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical_raw or _git_blob_sha1(canonical_raw) != HISTORICAL_SOURCE_BLOB:
        raise AssertionError("historical input is not the immutable Git blob")
    payload = load_json(path)
    if payload.get("data_shape") != [64, 64]:
        raise AssertionError("historical input grid shape mismatch")
    if len(payload.get("applied_amp_errors", [])) != 4 or len(
        payload.get("applied_phase_errors_rad", [])
    ) != 4:
        raise AssertionError("historical input does not contain four element weights")
    recorded_magnitude_db = np.asarray(payload["mag_db_rows"], dtype=np.float64)
    recorded_phase = np.asarray(payload["phase_rows"], dtype=np.float64)
    if recorded_magnitude_db.shape != (64, 64) or recorded_phase.shape != (64, 64):
        raise AssertionError("historical input matrices are not 64x64")
    if not np.isfinite(recorded_magnitude_db).all() or not np.isfinite(
        recorded_phase
    ).all():
        raise AssertionError("historical input contains non-finite pattern values")

    field, magnitude_db, phase_rad = _replay_legacy_payload(
        payload, legacy_forward_1x4_config()
    )
    recorded_field = np.power(10.0, recorded_magnitude_db / 20.0) * np.exp(
        1j * recorded_phase
    )
    max_complex = float(np.max(np.abs(field - recorded_field)))
    # Avoid NumPy's platform BLAS here: the validator also loads PyTorch and
    # mixing independently bundled OpenMP runtimes is unsafe on Windows.
    error_energy = float(np.sum(np.abs(field - recorded_field) ** 2))
    reference_energy = float(np.sum(np.abs(recorded_field) ** 2))
    if reference_energy <= 0.0:
        raise AssertionError("historical reference field has zero energy")
    relative_l2 = math.sqrt(error_energy / reference_energy)
    magnitude_error = float(np.max(np.abs(magnitude_db - recorded_magnitude_db)))
    phase_phasor_error = float(
        np.max(np.abs(np.exp(1j * phase_rad) - np.exp(1j * recorded_phase)))
    )
    replay = metrics["untouched_historical_json_replay"]
    expected_metadata = {
        "git_path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "source_blob": HISTORICAL_SOURCE_BLOB,
        "topology": "1x4",
        "wavelength": 6.83,
        "shape": [64, 64],
    }
    for key, expected in expected_metadata.items():
        if replay.get(key) != expected:
            raise AssertionError(f"historical replay metadata mismatch: {key}")
    recomputed = {
        "complex_field_max_abs_error": max_complex,
        "complex_field_relative_l2": relative_l2,
        "magnitude_db_max_abs_error": magnitude_error,
        "phase_unit_phasor_max_abs_error": phase_phasor_error,
    }
    for key, expected in recomputed.items():
        if not math.isclose(
            float(replay[key]), expected, rel_tol=1.0e-12, abs_tol=1.0e-16
        ):
            raise AssertionError(f"historical replay metric mismatch: {key}")
    if max_complex > 1.0e-10:
        raise AssertionError(f"historical pointwise replay failed: {max_complex}")
    return {"git_blob_sha1": HISTORICAL_SOURCE_BLOB, **recomputed}


def validate_legacy_output_schema() -> dict[str, Any]:
    payload = load_json(FULL_OUTPUT_DIR / "legacy_full_pattern_1x4.json")
    required = {
        "header",
        "targets",
        "data_file",
        "data_shape",
        "dtype",
        "mag_db_rows",
        "phase_rows",
        "run_id",
        "notes",
        "applied_amp_errors",
        "applied_phase_errors_rad",
    }
    if set(payload) != required:
        raise AssertionError("legacy output schema key inventory mismatch")
    if payload["data_shape"] != [64, 64] or not payload["run_id"]:
        raise AssertionError("legacy output shape/run_id mismatch")
    field, magnitude_db, phase_rad = _replay_legacy_payload(
        payload, ArrayGeometryConfig.hardware_1x4()
    )
    recorded_magnitude = np.asarray(payload["mag_db_rows"], dtype=np.float64)
    recorded_phase = np.asarray(payload["phase_rows"], dtype=np.float64)
    np.testing.assert_allclose(magnitude_db, recorded_magnitude, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.exp(1j * phase_rad), np.exp(1j * recorded_phase), rtol=0.0, atol=1.0e-12
    )
    if not np.isfinite(field).all():
        raise AssertionError("legacy output replay is non-finite")
    return {"schema": "pass", "shape": [64, 64]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_bytes(data: bytes) -> bytes:
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _sha256_canonical_text(path: Path) -> str:
    return hashlib.sha256(_canonical_text_bytes(path.read_bytes())).hexdigest()


def validate_source_artifact_hashes() -> dict[str, Any]:
    comparison_input = load_json(FULL_INPUT_DIR / "comparison_inputs.json")
    expected_contract = {
        "algorithm": "SHA-256",
        "encoding": "UTF-8 without BOM",
        "line_endings": "LF",
        "scope": "canonical decoded text bytes",
    }
    if comparison_input.get("source_artifact_hash_contract") != expected_contract:
        raise AssertionError("full-pattern source hash contract mismatch")

    source_paths = {
        name: FULL_OUTPUT_DIR / name
        for name in (
            "metrics.json",
            "complexity.csv",
            "benchmark.csv",
            "numerical_cases.csv",
        )
    }
    expected_hashes = {
        name: _sha256_canonical_text(path) for name, path in source_paths.items()
    }
    if comparison_input.get("source_artifacts") != expected_hashes:
        raise AssertionError("full-pattern canonical source hashes mismatch")

    # This invariant catches the Windows-CRLF/Linux-LF portability regression
    # even when the validator itself is run on only one operating system.
    lf = b"first line\nsecond line\n"
    crlf = b"first line\r\nsecond line\r\n"
    if _canonical_text_bytes(lf) != _canonical_text_bytes(crlf):
        raise AssertionError("line-ending canonicalization is not portable")
    return {"contract": expected_contract, "files": expected_hashes}


def _assert_nested_bit_exact(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(expected, actual):
            raise AssertionError(f"wrapper tensor is not bit-exact: {path}")
    elif isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise AssertionError(f"wrapper mapping keys differ: {path}")
        for key in expected:
            _assert_nested_bit_exact(expected[key], actual[key], f"{path}.{key}")
    elif expected != actual:
        raise AssertionError(f"wrapper value differs: {path}")


def _controller_provenance_item(metrics: dict[str, Any], topology: str) -> dict[str, Any]:
    checkpoints = metrics["controller_provenance"].get("checkpoints")
    if isinstance(checkpoints, dict):
        item = checkpoints.get(topology)
    elif isinstance(checkpoints, list):
        matches = [row for row in checkpoints if row.get("topology") == topology]
        item = matches[0] if len(matches) == 1 else None
    else:
        item = None
    if not isinstance(item, dict):
        raise AssertionError(f"missing controller provenance: {topology}")
    return item


def validate_low_update_wrappers(
    metrics: dict[str, Any],
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    provenance = metrics["controller_provenance"]
    if provenance.get("origin_version") != "v2.4":
        raise AssertionError("controller origin version mismatch")
    if provenance.get("integration_version") != "v2.5":
        raise AssertionError("controller integration version mismatch")
    controllers: dict[str, torch.nn.Module] = {}
    summaries: dict[str, Any] = {}
    for topology, config in (
        ("4x4", ArrayGeometryConfig.theoretical_4x4()),
        ("1x4", ArrayGeometryConfig.hardware_1x4()),
    ):
        checkpoint_name = f"v24_{topology}_low_update_controller.pt"
        checkpoint_path = CONTROLLER_OUTPUT_DIR / checkpoint_name
        checkpoint, controller = load_controller(checkpoint_path)
        if checkpoint["array_config"] != asdict(config):
            raise AssertionError(f"low-update geometry mismatch: {topology}")
        regression = metrics["v24_controller_regression"][topology]
        if regression.get("checkpoint") != checkpoint_name:
            raise AssertionError(f"regression checkpoint mismatch: {topology}")
        if regression.get("algorithm_version") != "2.4.0":
            raise AssertionError(f"regression controller version mismatch: {topology}")
        if regression.get("selected_update") != "zero_shot":
            raise AssertionError(f"regression update mismatch: {topology}")
        if regression.get("selection_split") != "single_board_validation":
            raise AssertionError(f"regression selection split mismatch: {topology}")
        if regression.get("strict_load") is not True:
            raise AssertionError(f"regression strict-load flag mismatch: {topology}")
        if float(regression.get("max_abs_difference", math.inf)) != 0.0:
            raise AssertionError(f"regression is not bit-exact: {topology}")
        if regression.get("renderer_state_dict_keys") != []:
            raise AssertionError(f"renderer state leaked into checkpoint: {topology}")
        if regression.get("conclusion") != "bit_exact_unchanged":
            raise AssertionError(f"regression conclusion mismatch: {topology}")
        if not all(torch.isfinite(tensor).all() for tensor in checkpoint["state_dict"].values()):
            raise AssertionError(f"non-finite low-update checkpoint: {topology}")

        provenance_item = _controller_provenance_item(metrics, topology)
        expected_package_path = (
            f"output/controller_training/{checkpoint_name}"
        )
        if provenance_item.get("package_path") != expected_package_path:
            raise AssertionError(f"controller package path mismatch: {topology}")
        if provenance_item.get("sha256") != _sha256(checkpoint_path):
            raise AssertionError(f"controller SHA-256 mismatch: {topology}")

        with np.load(
            CONTROLLER_OUTPUT_DIR / f"golden_vectors_{topology}.npz",
            allow_pickle=False,
        ) as golden:
            arguments = {
                "initial_weights_iq": _torch_from(golden["initial_weights_iq"]).float(),
                "target_angles_rad": _torch_from(golden["targets_rad"]).float(),
                "calibration_iq": _torch_from(golden["calibration_iq"]).float(),
                "position_errors": _torch_from(golden["position_errors"]).float(),
                "state": _torch_from(golden["state"]).float(),
                "quantize_output": False,
            }
            renderer = FullPatternPhysics(config)
            if renderer.state_dict():
                raise AssertionError(f"renderer persistent state is not empty: {topology}")
            wrapper = V25BeamformingSystem(controller, renderer)
            with torch.no_grad():
                direct = controller(
                    arguments.pop("initial_weights_iq"),
                    arguments.pop("target_angles_rad"),
                    **arguments,
                )
                # Recreate the positional inputs because they were popped above.
                wrapped = wrapper(
                    _torch_from(golden["initial_weights_iq"]).float(),
                    _torch_from(golden["targets_rad"]).float(),
                    calibration_iq=_torch_from(golden["calibration_iq"]).float(),
                    position_errors=_torch_from(golden["position_errors"]).float(),
                    state=_torch_from(golden["state"]).float(),
                    quantize_output=False,
                    render_pattern=True,
                )
            controller_part = {
                key: value for key, value in wrapped.items() if key != "full_pattern"
            }
            _assert_nested_bit_exact(direct, controller_part, topology)
            pattern = wrapped["full_pattern"]
            if tuple(pattern["field_iq"].shape) != (
                len(golden["targets_rad"]),
                2,
                64,
                64,
            ):
                raise AssertionError(f"wrapped full-pattern shape mismatch: {topology}")
            if not all(
                torch.isfinite(value).all()
                for value in pattern.values()
                if isinstance(value, torch.Tensor)
            ):
                raise AssertionError(f"wrapped full pattern is non-finite: {topology}")
        controllers[topology] = controller
        summaries[topology] = {
            "checkpoint": checkpoint_name,
            "sha256": provenance_item["sha256"],
            "controller_output": "bit_exact",
            "renderer_state_dict_keys": [],
        }
    return controllers, summaries


def validate_formula_parity(metrics: dict[str, Any]) -> None:
    parity = metrics["legacy_visualization_formula_parity"]
    for topology in ("4x4", "1x4"):
        item = parity[topology]
        if item["shape"] != [64, 64]:
            raise AssertionError(f"formula-parity shape mismatch: {topology}")
        if item["row_axis"] != "phi" or item["column_axis"] != "theta":
            raise AssertionError(f"formula-parity axes mismatch: {topology}")
        if item["max_abs_complex_error_vs_original_visualization_formula"] > 1.0e-10:
            raise AssertionError(f"legacy formula parity failed: {topology}")


def validate_boundaries(metrics: dict[str, Any]) -> dict[str, str]:
    boundary = str(metrics.get("interpretation_boundary", ""))
    if "不等于" not in boundary or "真实天线" not in boundary:
        raise AssertionError("metrics does not state the real-antenna boundary")
    known = (FULL_OUTPUT_DIR / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
    report = (FULL_OUTPUT_DIR / "报告.md").read_text(encoding="utf-8")
    manual = (FULL_OUTPUT_DIR / "V2.5组会讲解与答辩手册.md").read_text(
        encoding="utf-8"
    )
    required_known = ("不是暗室、VNA 或 OTA 实测", "FPGA", "实物上校准验证")
    required_report = (
        "FPGA 综合前算法版本",
        "没有生成 HLS、RTL、DPU 模型或 bitstream",
    )
    required_manual = ("不宣称已完成 FPGA", "板级实测")
    for text, phrases, name in (
        (known, required_known, "KNOWN_LIMITATIONS.md"),
        (report, required_report, "报告.md"),
        (manual, required_manual, "V2.5组会讲解与答辩手册.md"),
    ):
        missing = [phrase for phrase in phrases if phrase not in text]
        if missing:
            raise AssertionError(f"{name} misses interpretation boundary: {missing}")
    for name in ("报告.md", "V2.5组会讲解与答辩手册.md", "test_log.txt"):
        if (FULL_OUTPUT_DIR / name).stat().st_size < 200:
            raise AssertionError(f"document is unexpectedly small: {name}")
    test_log = (FULL_OUTPUT_DIR / "test_log.txt").read_text(encoding="utf-8")
    required_test_evidence = (
        "Result: 35 tests, PASS",
        "V2.4 controller chain: 22/22",
        "V2.5 full-pattern tests: 13/13",
    )
    missing_test_evidence = [
        phrase for phrase in required_test_evidence if phrase not in test_log
    ]
    if missing_test_evidence:
        raise AssertionError(
            "V2.5 complete-package test evidence is incomplete: "
            f"{missing_test_evidence}"
        )
    obsolete_counts = [phrase for phrase in ("17 tests, PASS", "40/40") if phrase in test_log]
    if obsolete_counts:
        raise AssertionError(f"obsolete V2.5 test-count claims remain: {obsolete_counts}")
    return {"fpga": "not_claimed", "ota": "not_claimed"}


def _check_export_bundle() -> dict[str, Any]:
    # The exporter intentionally imports Matplotlib while this validator must
    # import PyTorch.  Some Windows builds bundle incompatible copies of Intel
    # OpenMP, so executing both runtimes in one process can abort before Python
    # raises an exception.  Run the exporter's ``--check`` entry point in an
    # isolated process: that entry point calls ``check_release_bundle()``,
    # rebuilds expected JSON values and PNG canvases in memory, and performs no
    # writes. Snapshotting the five files here independently enforces that
    # read-only contract rather than trusting only its status message.
    artifacts = [
        FULL_INPUT_DIR / "comparison_inputs.json",
        FULL_OUTPUT_DIR / "comparison_results.json",
        *(FULL_PICTURE_DIR / name for name in RELEASE_PICTURE_FILES),
    ]

    def snapshot() -> dict[str, tuple[int, int, str]]:
        return {
            str(path.relative_to(VERSION_DIR)): (
                path.stat().st_mtime_ns,
                path.stat().st_size,
                _sha256(path),
            )
            for path in artifacts
        }

    before = snapshot()
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(CODE_DIR / "export_v25_comparison_bundle.py"),
            "--check",
        ],
        cwd=CODE_DIR,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    after = snapshot()
    if before != after:
        raise AssertionError("V2.5 comparison --check modified release artifacts")
    if completed.returncode != 0:
        raise AssertionError(
            "V2.5 comparison in-memory check failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError("V2.5 comparison checker emitted invalid JSON") from exc
    if not isinstance(result, dict):
        raise AssertionError("V2.5 comparison checker must return a JSON object")
    expected = {
        "version": "v2.5",
        "comparison_bundle": "pass",
        "input_cases": 2,
        "output_cases": 2,
        "pictures": [
            "golden_full_pattern_4x4.png",
            "golden_full_pattern_1x4.png",
            "performance_comparison.png",
        ],
        "check_mode_writes": 0,
    }
    if result != expected:
        raise AssertionError(f"unexpected V2.5 comparison check result: {result}")
    return result


def validate_png(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"invalid PNG signature: {path.name}")
    if data[12:16] != b"IHDR":
        raise AssertionError(f"missing PNG IHDR: {path.name}")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != expected_size:
        raise AssertionError(f"unexpected PNG size: {path.name}={(width, height)}")
    if not data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
        raise AssertionError(f"truncated PNG: {path.name}")
    if len(data) < 20_000:
        raise AssertionError(f"suspiciously small PNG: {path.name}")
    return {"width": width, "height": height, "bytes": len(data)}


def validate_v25() -> dict[str, Any]:
    layout = validate_layout()

    # Reuse the entire controller-package validator, including all eight strict
    # checkpoint loads, deterministic aggregates, INT8 golden replays, and its
    # own comparison-bundle in-memory check.
    controller_validation = validate_v24()
    if controller_validation.get("package_layout") != "pass":
        raise AssertionError("embedded V2.4 controller validation did not pass")

    metrics = load_json(FULL_OUTPUT_DIR / "metrics.json")
    validate_metrics_header(metrics)
    numerical = validate_numerical_cases(metrics)
    contracts = validate_contracts()
    controllers, wrapper_regression = validate_low_update_wrappers(metrics)
    complexity = validate_complexity(metrics, controllers)
    benchmark = validate_benchmark(metrics)
    golden = validate_golden_vectors()
    intermediate = validate_intermediate_values()
    validate_formula_parity(metrics)
    historical = validate_historical_replay(metrics)
    legacy_output = validate_legacy_output_schema()
    source_artifact_hashes = validate_source_artifact_hashes()
    comparison_bundle = _check_export_bundle()
    boundaries = validate_boundaries(metrics)

    pictures = {
        "golden_full_pattern_4x4.png": validate_png(
            FULL_PICTURE_DIR / "golden_full_pattern_4x4.png", (1728, 536)
        ),
        "golden_full_pattern_1x4.png": validate_png(
            FULL_PICTURE_DIR / "golden_full_pattern_1x4.png", (1728, 536)
        ),
        "performance_comparison.png": validate_png(
            FULL_PICTURE_DIR / "performance_comparison.png", (2080, 584)
        ),
    }

    return {
        "version": "v2.5",
        "status": "pass",
        "layout_files": layout,
        "controller_training": controller_validation,
        "full_pattern": {
            "algorithm_version": metrics["algorithm_version"],
            "numerical_generated_files": len(NUMERICAL_GENERATED_FILES),
            "release_files": len(RELEASE_FILES),
            "release_picture_files": len(RELEASE_PICTURE_FILES),
            "numerical_cases": numerical,
            "contracts": {
                topology: contract["pattern_output_shape"]
                for topology, contract in contracts.items()
            },
            "complexity": complexity,
            "benchmark": benchmark,
            "golden_vectors": golden,
            "intermediate_points": intermediate,
            "historical_replay": historical,
            "legacy_output": legacy_output,
            "source_artifact_hashes": source_artifact_hashes,
            "low_update_wrappers": wrapper_regression,
            "comparison_bundle": comparison_bundle,
            "pictures": pictures,
            "acceptance_gates": "pass",
        },
        "boundaries": boundaries,
        "conclusion": "simulation_and_pre_fpga_validation_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2.5", choices=("v2.5",))
    parser.parse_args()
    print(json.dumps(validate_v25(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
