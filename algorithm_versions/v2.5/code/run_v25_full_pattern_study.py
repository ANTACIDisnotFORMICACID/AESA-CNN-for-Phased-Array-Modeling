"""复现 V2.5 完整复方向图层并生成可审计的数值、复杂度和接口记录。"""

from __future__ import annotations

import csv
import cmath
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    analytic_far_field_weights,
    element_response_from_amp_phase,
    profile_topology_model,
)
from full_pattern_v25 import (
    ALGORITHM_VERSION,
    FullPatternPhysics,
    PatternGridConfig,
    V25BeamformingSystem,
    profile_full_pattern,
    weights_iq_from_amplitude_phase,
)


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
INPUT_DIR = VERSION_DIR / "input" / "full_pattern"
OUTPUT_DIR = VERSION_DIR / "output" / "full_pattern"
CONTROLLER_OUTPUT_DIR = VERSION_DIR / "output" / "controller_training"
BASE_SEED = 20260823
LEGACY = {
    "name": "original_learned_forward_cnn",
    "parameters": 2_352_770,
    "macs_per_sample_n4_historical_path": 29_552_128,
    "macs_per_sample_n16_analysis_example": 31_932_928,
    "standardized_complex_loss": 0.00282475,
    "physical_complex_nmse_db": -28.592,
    "mean_sample_relative_l2_percent": 3.416,
    "magnitude_correlation": 0.99854,
    "cpu_batch1_median_ms_historical": 3.99,
    "scope_note": "历史验证集上的学习代理结果；不是浮点公式一致性误差。",
}
HISTORICAL_SAMPLE = "output_json/output_20260425_202604_01be8f59.json"
HISTORICAL_INPUT = INPUT_DIR / "historical_replay_input.json"
HISTORICAL_COMMIT = "bf352ac73ba2e2024081b98591e05409eac89317"
HISTORICAL_BLOB = "6d11801fcc420937c381e475b7bfbe8345a2ddbf"


def explicit_array_positions(
    config: ArrayGeometryConfig, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Study-local coordinate construction, independent of the production helper."""

    return torch.tensor(
        [
            (0.0, config.x_spacing * xi, config.y_spacing * yi)
            for xi in range(config.x_num)
            for yi in range(config.y_num)
        ],
        dtype=dtype,
    )


def legacy_forward_cnn_macs(elements: int) -> int:
    if elements not in (4, 16):
        raise ValueError("legacy comparison is defined only for N=4 or N=16")
    return 29_552_128 if elements == 4 else 31_932_928


def direct_complex_reference(
    config: ArrayGeometryConfig,
    grid: PatternGridConfig,
    weights_iq: torch.Tensor,
    *,
    position_errors: Optional[torch.Tensor] = None,
    element_response_iq: Optional[torch.Tensor] = None,
    coupling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """独立复数公式真值，不调用 V2 的 steering/field 实现。"""

    batch = weights_iq.shape[0]
    theta = torch.deg2rad(
        torch.linspace(
            grid.theta_min_deg,
            grid.theta_max_deg,
            grid.theta_points,
            dtype=weights_iq.dtype,
            device=weights_iq.device,
        )
    )
    phi = torch.deg2rad(
        torch.linspace(
            grid.phi_min_deg,
            grid.phi_max_deg,
            grid.phi_points,
            dtype=weights_iq.dtype,
            device=weights_iq.device,
        )
    )
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="xy")
    directions = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ),
        dim=-1,
    )
    positions = explicit_array_positions(config, dtype=weights_iq.dtype).unsqueeze(0).expand(
        batch, -1, -1
    )
    if position_errors is not None:
        positions = positions + position_errors.flatten(2).transpose(1, 2)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "ptc,bnc->bptn", directions, positions
    )
    propagation = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    weights = torch.complex(weights_iq[:, 0], weights_iq[:, 1]).flatten(1)
    if coupling is not None:
        weights = torch.einsum("bnm,bm->bn", coupling, weights)
    if element_response_iq is not None:
        response = torch.complex(
            element_response_iq[:, 0], element_response_iq[:, 1]
        ).flatten(1)
        weights = response * weights
    field = torch.einsum("bptn,bn->bpt", propagation, weights)
    return torch.stack((field.real, field.imag), dim=1)


def make_case(
    config: ArrayGeometryConfig, seed: int
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (1, config.x_num, config.y_num)
    amplitude = 0.25 + 0.75 * torch.rand(shape, generator=generator, dtype=torch.float64)
    phase = (2.0 * math.pi) * torch.rand(shape, generator=generator, dtype=torch.float64) - math.pi
    weights = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=1)

    position = (
        0.02
        * config.wavelength
        * torch.randn(1, 3, config.x_num, config.y_num, generator=generator, dtype=torch.float64)
    )
    response_amplitude = (
        1.0
        + 0.08
        * torch.randn(1, config.x_num, config.y_num, generator=generator, dtype=torch.float64)
    ).clamp_min(0.65)
    response_phase = 0.10 * torch.randn(
        1, config.x_num, config.y_num, generator=generator, dtype=torch.float64
    )
    response = element_response_from_amp_phase(response_amplitude, response_phase)

    elements = config.num_elements
    eye = torch.eye(elements, dtype=torch.complex128).unsqueeze(0)
    off_diagonal = 1.0 - torch.eye(elements, dtype=torch.float64)
    coupling_noise = torch.complex(
        torch.randn(1, elements, elements, generator=generator, dtype=torch.float64),
        torch.randn(1, elements, elements, generator=generator, dtype=torch.float64),
    )
    coupling = eye + (0.02 / math.sqrt(elements)) * coupling_noise * off_diagonal
    return {
        "weights_iq": weights,
        "position_errors": position,
        "element_response_iq": response,
        "coupling": coupling,
    }


def cast_case(case: Mapping[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return {
        "weights_iq": case["weights_iq"].to(dtype),
        "position_errors": case["position_errors"].to(dtype),
        "element_response_iq": case["element_response_iq"].to(dtype),
        "coupling": case["coupling"].to(complex_dtype),
    }


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    actual64 = actual.to(torch.float64)
    reference64 = reference.to(torch.float64)
    error = actual64 - reference64
    squared_error = error.square().sum()
    squared_reference = reference64.square().sum()
    ratio = float(squared_error / squared_reference.clamp_min(1.0e-300))
    return {
        "max_abs_error": float(error.abs().amax()),
        "rmse": float(torch.sqrt(error.square().mean())),
        "relative_l2": float(torch.sqrt(squared_error / squared_reference.clamp_min(1.0e-300))),
        "nmse_db": 10.0 * math.log10(max(ratio, 1.0e-300)),
    }


def scalar_legacy_reference(
    config: ArrayGeometryConfig, grid: PatternGridConfig, weights_iq: torch.Tensor
) -> torch.Tensor:
    """逐标量对应原 visualization.py，且不依赖被测张量公式。"""

    theta_axis = [
        math.radians(value)
        for value in torch.linspace(
            grid.theta_min_deg, grid.theta_max_deg, grid.theta_points, dtype=torch.float64
        ).tolist()
    ]
    phi_axis = [
        math.radians(value)
        for value in torch.linspace(
            grid.phi_min_deg, grid.phi_max_deg, grid.phi_points, dtype=torch.float64
        ).tolist()
    ]
    positions = explicit_array_positions(config, dtype=torch.float64).tolist()
    weights = torch.complex(weights_iq[0, 0], weights_iq[0, 1]).flatten().tolist()
    wave_number = 2.0 * math.pi / config.wavelength
    fields = []
    for phi in phi_axis:
        for theta in theta_axis:
            direction = (
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            )
            value = 0.0j
            for weight, position in zip(weights, positions):
                projection = sum(component * coordinate for component, coordinate in zip(direction, position))
                value += weight * cmath.exp(1.0j * wave_number * projection)
            fields.append(value)
    return torch.tensor(fields, dtype=torch.complex128).reshape(
        grid.phi_points, grid.theta_points
    )


def replay_historical_json() -> Dict[str, object]:
    """Replay the immutable historical JSON bundled with this version package.

    The SHA-1 check uses Git's blob-object framing, so it proves that the packaged
    bytes are the exact historical object without requiring a Git executable or
    repository history at runtime.
    """

    raw = HISTORICAL_INPUT.read_bytes()

    def git_blob_sha1(payload: bytes) -> str:
        framed = f"blob {len(payload)}\0".encode("ascii") + payload
        return hashlib.sha1(framed).hexdigest()

    blob = git_blob_sha1(raw)
    normalized_line_endings = False
    if blob != HISTORICAL_BLOB and b"\r\n" in raw:
        # Git stores this text as LF even when core.autocrlf materializes CRLF in
        # a Windows working tree.  Validate the canonical blob bytes as Git does.
        blob = git_blob_sha1(raw.replace(b"\r\n", b"\n"))
        normalized_line_endings = True
    if blob != HISTORICAL_BLOB:
        raise AssertionError(
            f"packaged historical input changed: {blob} != {HISTORICAL_BLOB}"
        )
    payload = json.loads(raw.decode("utf-8-sig"))
    header = payload["header"]
    config = ArrayGeometryConfig(
        wavelength=float(header["wavlen"]),
        x_spacing=float(header["x_dis"]),
        y_spacing=float(header["y_dis"]),
        x_num=int(header["array_rows"]),
        y_num=int(header["array_cols"]),
    )
    grid = PatternGridConfig(
        theta_min_deg=float(header["theta_span_deg"][0]),
        theta_max_deg=float(header["theta_span_deg"][1]),
        theta_points=int(header["cols"]),
        phi_min_deg=float(header["phi_span_deg"][0]),
        phi_max_deg=float(header["phi_span_deg"][1]),
        phi_points=int(header["rows"]),
    )
    amplitude = torch.tensor(
        payload["applied_amp_errors"], dtype=torch.float64
    ).reshape(1, config.x_num, config.y_num)
    phase = torch.tensor(
        payload["applied_phase_errors_rad"], dtype=torch.float64
    ).reshape(1, config.x_num, config.y_num)
    weights = weights_iq_from_amplitude_phase(amplitude, phase, config)
    result = FullPatternPhysics(config, grid, dtype=torch.float64)(weights)
    expected_db = torch.tensor(payload["mag_db_rows"], dtype=torch.float64)
    expected_phase = torch.tensor(payload["phase_rows"], dtype=torch.float64)
    expected_magnitude = torch.pow(10.0, expected_db / 20.0)
    expected_field = torch.complex(
        expected_magnitude * torch.cos(expected_phase),
        expected_magnitude * torch.sin(expected_phase),
    )
    actual_field = torch.complex(result["field_iq"][0, 0], result["field_iq"][0, 1])
    field_error = actual_field - expected_field
    return {
        "git_path": HISTORICAL_SAMPLE,
        "package_input": "input/full_pattern/historical_replay_input.json",
        "source_commit": HISTORICAL_COMMIT,
        "source_blob": blob,
        "git_text_line_endings_normalized": normalized_line_endings,
        "topology": config.topology_name,
        "wavelength": config.wavelength,
        "shape": list(expected_db.shape),
        "complex_field_max_abs_error": float(field_error.abs().amax()),
        "complex_field_relative_l2": float(
            torch.linalg.vector_norm(field_error)
            / torch.linalg.vector_norm(expected_field).clamp_min(1.0e-300)
        ),
        "magnitude_db_max_abs_error": float(
            (result["magnitude_db"][0] - expected_db).abs().amax()
        ),
        "phase_unit_phasor_max_abs_error": float(
            (
                torch.exp(torch.complex(torch.zeros_like(result["phase_rad"][0]), result["phase_rad"][0]))
                - torch.exp(torch.complex(torch.zeros_like(expected_phase), expected_phase))
            )
            .abs()
            .amax()
        ),
    }


def benchmark(callable_object, warmup: int = 25, repeats: int = 200) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            callable_object()
        durations = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            callable_object()
            durations.append((time.perf_counter_ns() - start) / 1.0e6)
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.fmean(durations),
        "p95_ms": p95,
        "min_ms": min(durations),
        "max_ms": max(durations),
    }


def load_published_v24_controller(
    topology: str,
) -> tuple[ArrayGeometryConfig, TopologyAdaptiveBeamformingCNN, Dict[str, object]]:
    path = CONTROLLER_OUTPUT_DIR / f"v24_{topology}_low_update_controller.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ArrayGeometryConfig(**checkpoint["array_config"])
    controller = TopologyAdaptiveBeamformingCNN(
        config, **checkpoint["model_config"]
    )
    controller.load_state_dict(checkpoint["state_dict"], strict=True)
    controller.eval()
    metadata = {
        "checkpoint": path.name,
        "package_path": f"output/controller_training/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "algorithm_version": checkpoint["algorithm_version"],
        "training_regime": checkpoint["training_regime"],
        "selected_update": checkpoint["selected_update"],
        "selection_split": checkpoint["selection_split"],
        "selected_seed": checkpoint["selected_seed"],
        "strict_load": True,
    }
    return config, controller, metadata


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"没有可写入 {path.name} 的记录")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def selected_intermediates(
    config: ArrayGeometryConfig,
    grid: PatternGridConfig,
    case: Mapping[str, torch.Tensor],
    result: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
) -> Dict[str, object]:
    indices = ((0, 0), (0, 63), (31, 31), (32, 32), (63, 0), (63, 63))
    weights = case["weights_iq"]
    amplitude = torch.sqrt(weights[:, 0].square() + weights[:, 1].square())
    phase = torch.remainder(torch.atan2(weights[:, 1], weights[:, 0]), 2.0 * math.pi)
    points = []
    for phi_index, theta_index in indices:
        points.append(
            {
                "phi_index": phi_index,
                "theta_index": theta_index,
                "phi_deg": math.degrees(float(result["phi_axis_rad"][phi_index])),
                "theta_deg": math.degrees(float(result["theta_axis_rad"][theta_index])),
                "field_i": float(result["field_iq"][0, 0, phi_index, theta_index]),
                "field_q": float(result["field_iq"][0, 1, phi_index, theta_index]),
                "reference_i": float(reference[0, 0, phi_index, theta_index]),
                "reference_q": float(reference[0, 1, phi_index, theta_index]),
                "magnitude": float(result["magnitude"][0, phi_index, theta_index]),
                "magnitude_db": float(result["magnitude_db"][0, phi_index, theta_index]),
                "phase_rad": float(result["phase_rad"][0, phi_index, theta_index]),
            }
        )
    return {
        "topology": config.topology_name,
        "formula": "E=sum_n[(Cw)_n*g_n*exp(j*2pi/lambda*(r_n+delta_r_n) dot u)]",
        "array_positions": explicit_array_positions(config, dtype=torch.float64).tolist(),
        "weights_iq": weights[0].tolist(),
        "weights_amplitude_flat": amplitude[0].flatten().tolist(),
        "weights_phase_rad_flat": phase[0].flatten().tolist(),
        "position_errors": case["position_errors"][0].tolist(),
        "element_response_iq": case["element_response_iq"][0].tolist(),
        "coupling_real": case["coupling"][0].real.tolist(),
        "coupling_imag": case["coupling"][0].imag.tolist(),
        "selected_grid_points": points,
        "axis_contract": {
            "rows": "phi",
            "columns": "theta",
            "shape": [grid.phi_points, grid.theta_points],
        },
    }


def main() -> None:
    torch.manual_seed(BASE_SEED)
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = PatternGridConfig.legacy_64()
    configs = (
        ArrayGeometryConfig.theoretical_4x4(),
        ArrayGeometryConfig.hardware_1x4(),
    )

    numerical_rows = []
    legacy_parity = {}
    controller_regression = {}
    complexity_rows = []
    benchmark_rows = []
    contracts = {}
    intermediate = {}
    historical_replay = replay_historical_json()

    for config in configs:
        topology = config.topology_name
        canonical_case = make_case(config, BASE_SEED + config.num_elements)
        canonical_reference = direct_complex_reference(
            config,
            grid,
            canonical_case["weights_iq"],
            position_errors=canonical_case["position_errors"],
            element_response_iq=canonical_case["element_response_iq"],
            coupling=canonical_case["coupling"],
        )

        for case_index in range(8):
            source_case = make_case(config, BASE_SEED + 100 * config.num_elements + case_index)
            for mode in ("ideal", "position_amp_phase_coupling"):
                reference = direct_complex_reference(
                    config,
                    grid,
                    source_case["weights_iq"],
                    position_errors=(source_case["position_errors"] if mode != "ideal" else None),
                    element_response_iq=(
                        source_case["element_response_iq"] if mode != "ideal" else None
                    ),
                    coupling=(source_case["coupling"] if mode != "ideal" else None),
                )
                for dtype in (torch.float64, torch.float32):
                    cast = cast_case(source_case, dtype)
                    renderer = FullPatternPhysics(config, grid, dtype=dtype)
                    result = renderer(
                        cast["weights_iq"],
                        position_errors=(cast["position_errors"] if mode != "ideal" else None),
                        element_response_iq=(
                            cast["element_response_iq"] if mode != "ideal" else None
                        ),
                        coupling_iq=(cast["coupling"] if mode != "ideal" else None),
                    )["field_iq"]
                    row = {
                        "topology": topology,
                        "case_index": case_index,
                        "mode": mode,
                        "dtype": str(dtype).replace("torch.", ""),
                        **error_metrics(result, reference),
                    }
                    numerical_rows.append(row)

        ideal_weights = canonical_case["weights_iq"]
        ideal_renderer = FullPatternPhysics(config, grid, dtype=torch.float64)
        ideal_result = ideal_renderer(ideal_weights)
        scalar_field = scalar_legacy_reference(config, grid, ideal_weights)
        torch_field = torch.complex(
            ideal_result["field_iq"][0, 0], ideal_result["field_iq"][0, 1]
        )
        legacy_parity[topology] = {
            "max_abs_complex_error_vs_original_visualization_formula": float(
                (torch_field - scalar_field).abs().amax()
            ),
            "rmse_complex_vs_original_visualization_formula": float(
                torch.sqrt((torch_field - scalar_field).abs().square().mean())
            ),
            "shape": list(torch_field.shape),
            "row_axis": "phi",
            "column_axis": "theta",
        }

        nonideal_renderer = FullPatternPhysics(config, grid, dtype=torch.float64)
        nonideal_result = nonideal_renderer(
            canonical_case["weights_iq"],
            position_errors=canonical_case["position_errors"],
            element_response_iq=canonical_case["element_response_iq"],
            coupling_iq=canonical_case["coupling"],
        )
        intermediate[topology] = selected_intermediates(
            config, grid, canonical_case, nonideal_result, canonical_reference
        )
        np.savez_compressed(
            OUTPUT_DIR / f"golden_full_pattern_{topology}.npz",
            weights_iq=canonical_case["weights_iq"].numpy(),
            position_errors=canonical_case["position_errors"].numpy(),
            element_response_iq=canonical_case["element_response_iq"].numpy(),
            coupling_real=canonical_case["coupling"].real.numpy(),
            coupling_imag=canonical_case["coupling"].imag.numpy(),
            field_iq=nonideal_result["field_iq"].numpy(),
            reference_iq=canonical_reference.numpy(),
            theta_axis_rad=nonideal_result["theta_axis_rad"].numpy(),
            phi_axis_rad=nonideal_result["phi_axis_rad"].numpy(),
        )

        renderer32 = FullPatternPhysics(config, grid, dtype=torch.float32).eval()
        contract = renderer32.export_contract()
        contracts[topology] = contract
        with (OUTPUT_DIR / f"contract_{topology}.json").open("w", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)

        if topology == "1x4":
            payload = ideal_renderer.to_legacy_payload(
                ideal_result,
                weights_iq=ideal_weights,
                run_id="v2.5-reproducible-1x4-example",
                targets={"thetadir_deg": [], "phidir_deg": []},
                saved_at="2026-08-23 00:00:00",
            )
            with (OUTPUT_DIR / "legacy_full_pattern_1x4.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

        checkpoint_config, controller, checkpoint_metadata = load_published_v24_controller(
            topology
        )
        if checkpoint_config != config:
            raise AssertionError("published V2.4 checkpoint geometry mismatch")
        system = V25BeamformingSystem(controller, renderer32).eval()
        target = torch.tensor(
            [[math.radians(81.0), math.radians(-13.0)]], dtype=torch.float32
        )
        initial = analytic_far_field_weights(config, target)
        regression_generator = torch.Generator().manual_seed(
            BASE_SEED + 10_000 + config.num_elements
        )
        calibration = torch.zeros_like(initial)
        calibration[:, 0] = 1.0
        calibration = calibration + 0.01 * torch.randn(
            calibration.shape, generator=regression_generator
        )
        health = torch.ones(1, 1, config.x_num, config.y_num)
        position = 0.002 * config.wavelength * torch.randn(
            1,
            3,
            config.x_num,
            config.y_num,
            generator=regression_generator,
        )
        state = 0.05 * torch.randn(
            1, controller.state_dim, generator=regression_generator
        )
        controller_inputs = {
            "calibration_iq": calibration,
            "health_mask": health,
            "position_errors": position,
            "state": state,
            "quantize_output": False,
        }
        with torch.inference_mode():
            direct_controller = controller(initial, target, **controller_inputs)
            combined = system(
                initial,
                target,
                **controller_inputs,
                render_pattern=True,
            )
        regression_max = max(
            float((direct_controller[key] - combined[key]).abs().amax())
            for key in ("weights_iq", "prequant_weights_iq", "delta_iq")
        )
        controller_regression[topology] = {
            **checkpoint_metadata,
            "max_abs_difference": regression_max,
            "renderer_state_dict_keys": [
                key for key in system.state_dict() if key.startswith("renderer.")
            ],
            "conclusion": "bit_exact_unchanged" if regression_max == 0.0 else "changed",
        }

        controller_profile = profile_topology_model(controller)
        full_profile = profile_full_pattern(
            config, controller_macs=controller_profile["macs_per_sample"]
        )
        legacy_macs = legacy_forward_cnn_macs(config.num_elements)
        complexity_rows.append(
            {
                "topology": topology,
                "controller_parameters": controller_profile["total_parameters"],
                "full_pattern_parameters": full_profile["trainable_parameters"],
                "total_trainable_parameters": controller_profile["total_parameters"],
                "controller_macs": controller_profile["macs_per_sample"],
                "renderer_real_macs": full_profile["renderer_real_macs"],
                "optional_coupling_real_macs": full_profile[
                    "optional_coupling_real_macs"
                ],
                "optional_element_response_real_macs": full_profile[
                    "optional_element_response_real_macs"
                ],
                "nonideal_contraction_real_macs": full_profile[
                    "nonideal_contraction_real_macs"
                ],
                "end_to_end_real_macs": full_profile["end_to_end_real_macs"],
                "end_to_end_nonideal_contraction_real_macs": full_profile[
                    "end_to_end_nonideal_contraction_real_macs"
                ],
                "cached_steering_bytes_fp32": full_profile["cached_steering_bytes"],
                "legacy_forward_cnn_parameters": LEGACY["parameters"],
                "legacy_forward_cnn_input_elements": config.num_elements,
                "legacy_forward_cnn_macs": legacy_macs,
                "legacy_cnn_training_evidence_for_topology": topology == "1x4",
                "legacy_over_renderer_macs_ratio": legacy_macs
                / full_profile["renderer_real_macs"],
                "legacy_over_end_to_end_macs_ratio": legacy_macs
                / full_profile["end_to_end_real_macs"],
            }
        )

        benchmark_sources = {
            "controller_only": lambda: controller(initial, target, quantize_output=False),
            "full_pattern_only_cached_ideal": lambda: renderer32(initial),
            "controller_plus_full_pattern": lambda: system(
                initial, target, quantize_output=False, render_pattern=True
            ),
        }
        for operation, function in benchmark_sources.items():
            benchmark_rows.append(
                {"topology": topology, "operation": operation, **benchmark(function)}
            )

    write_csv(OUTPUT_DIR / "numerical_cases.csv", numerical_rows)
    write_csv(OUTPUT_DIR / "complexity.csv", complexity_rows)
    write_csv(OUTPUT_DIR / "benchmark.csv", benchmark_rows)
    with (OUTPUT_DIR / "intermediate_values.json").open("w", encoding="utf-8") as handle:
        json.dump(intermediate, handle, ensure_ascii=False, indent=2)

    accuracy_summary = {}
    for topology in ("4x4", "1x4"):
        accuracy_summary[topology] = {}
        for mode in ("ideal", "position_amp_phase_coupling"):
            accuracy_summary[topology][mode] = {}
            for dtype in ("float64", "float32"):
                subset = [
                    row
                    for row in numerical_rows
                    if row["topology"] == topology
                    and row["mode"] == mode
                    and row["dtype"] == dtype
                ]
                accuracy_summary[topology][mode][dtype] = {
                    "case_count": len(subset),
                    "worst_max_abs_error": max(row["max_abs_error"] for row in subset),
                    "worst_relative_l2": max(row["relative_l2"] for row in subset),
                    "worst_nmse_db": max(row["nmse_db"] for row in subset),
                    "mean_rmse": statistics.fmean(row["rmse"] for row in subset),
                }

    gates = {
        "all_fp64_max_abs_le_1e-10": max(
            row["max_abs_error"] for row in numerical_rows if row["dtype"] == "float64"
        )
        <= 1.0e-10,
        "all_fp32_max_abs_le_5e-5": max(
            row["max_abs_error"] for row in numerical_rows if row["dtype"] == "float32"
        )
        <= 5.0e-5,
        "legacy_scalar_formula_parity_le_1e-10": max(
            item["max_abs_complex_error_vs_original_visualization_formula"]
            for item in legacy_parity.values()
        )
        <= 1.0e-10,
        "untouched_historical_json_field_parity_le_1e-10": historical_replay[
            "complex_field_max_abs_error"
        ]
        <= 1.0e-10,
        "v24_controller_bit_exact": all(
            item["max_abs_difference"] == 0.0 for item in controller_regression.values()
        ),
    }
    if not all(gates.values()):
        raise AssertionError(f"V2.5 gate failed: {gates}")

    metrics = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "study": "V2.5 full complex pattern compatibility",
        "seed": BASE_SEED,
        "grid": {
            "theta_deg": [grid.theta_min_deg, grid.theta_max_deg, grid.theta_points],
            "phi_deg": [grid.phi_min_deg, grid.phi_max_deg, grid.phi_points],
            "matrix_semantics": "[phi_row, theta_column]",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "accuracy_summary": accuracy_summary,
        "legacy_visualization_formula_parity": legacy_parity,
        "untouched_historical_json_replay": historical_replay,
        "v24_controller_regression": controller_regression,
        "complexity": complexity_rows,
        "benchmark": benchmark_rows,
        "legacy_learned_cnn_historical_baseline": LEGACY,
        "acceptance_gates": gates,
        "controller_provenance": {
            "origin_version": "v2.4",
            "integration_version": "v2.5",
            "storage_scope": "output/controller_training",
            "checkpoints": {
                topology: {
                    "package_path": row["package_path"],
                    "sha256": row["sha256"],
                    "algorithm_version": row["algorithm_version"],
                    "selected_seed": row["selected_seed"],
                    "selection_split": row["selection_split"],
                }
                for topology, row in controller_regression.items()
            },
            "statement": (
                "V2.5 adds the full-pattern renderer and wrapper; the inherited "
                "V2.4 controllers remain algorithm_version 2.4.0 and are loaded "
                "strictly from this version package without numerical modification."
            ),
        },
        "numerical_generated_files": [
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
        ],
        "release_files": [
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
        ],
        "release_picture_files": [
            "golden_full_pattern_1x4.png",
            "golden_full_pattern_4x4.png",
            "performance_comparison.png",
        ],
        "interpretation_boundary": (
            "公式一致性只证明理想/显式非理想阵列模型计算正确；"
            "不等于未经实测校准即可精确代表真实天线。"
        ),
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(json.dumps({"gates": gates, "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
