"""21,248-case V2.5/V2.6/phase-reference audit on fixed HFSS physics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from boundary_v27 import boundary_reasons, classify_boundary, global_phase_aligned_weight_change, hpbw_aware_margin_db
from states_v27 import AuditState, all_audit_states, stratified_targets
from v25_adapter_v27 import FrozenV25Adapter
from v27_compat import V27_ROOT, enable_v26

enable_v26()
from calibration_v26 import build_d_optimal_design, simulate_measurements  # noqa: E402
from controller_v26 import BeamformingControllerV26  # noqa: E402
from dataset_v26 import state_pattern  # noqa: E402
from hfss_physics_v26 import HfssPatternData, angular_distance_deg, quantize_int8_disk  # noqa: E402
from oracle_v26 import OracleConfig, PhysicalOracle  # noqa: E402


AUDIT_ROOT = V27_ROOT / "output" / "audit"
CACHE_ROOT = V27_ROOT / "output" / "cache" / "audit_states"


def _v26_profile(state: AuditState):
    """Duck-typed profile accepted by frozen state_pattern()."""
    return state


def _batch_metrics(data: HfssPatternData, weights: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.asarray(weights, dtype=np.complex64)
    targets = np.asarray(targets, dtype=np.float64)
    etheta, ephi = data.fields(weights)
    power = (np.abs(etheta) ** 2 + np.abs(ephi) ** 2).astype(np.float32)
    flat = power.reshape(len(weights), -1)
    peak_index = np.argmax(flat, axis=1)
    peak_pi, peak_ti = np.unravel_index(peak_index, power.shape[1:])
    peak_power = flat[np.arange(len(weights)), peak_index]
    peak_theta = data.theta_deg[peak_ti]
    peak_phi = data.phi_deg[peak_pi]
    response = [data.response_at(float(theta), float(phi)) for theta, phi in targets]
    ht = np.stack([item[0] for item in response])
    hp = np.stack([item[1] for item in response])
    target_et = np.sum(ht * weights, axis=1)
    target_ep = np.sum(hp * weights, axis=1)
    target_power = (np.abs(target_et) ** 2 + np.abs(target_ep) ** 2).astype(np.float64)
    target_to_peak = 10.0 * np.log10((target_power + 1e-30) / (peak_power + 1e-30))
    pointing = angular_distance_deg(peak_theta, peak_phi, targets[:, 0], targets[:, 1])
    grid_theta, grid_phi = data.flat_theta_phi
    strict_margin = np.empty(len(weights), dtype=np.float64)
    hpbw_margin = np.empty(len(weights), dtype=np.float64)
    for index, (theta, phi) in enumerate(targets):
        outside = angular_distance_deg(grid_theta, grid_phi, theta, phi) > 8.0
        outside_peak = float(np.max(flat[index, outside]))
        strict_margin[index] = 10.0 * np.log10((target_power[index] + 1e-30) / (outside_peak + 1e-30))
        hpbw_margin[index] = hpbw_aware_margin_db(power[index], target_power[index])[0]
    polarization_ratio = 10.0 * np.log10(
        (np.maximum(np.abs(target_et) ** 2, np.abs(target_ep) ** 2) + 1e-30)
        / (np.minimum(np.abs(target_et) ** 2, np.abs(target_ep) ** 2) + 1e-30)
    )
    return {
        "target_power": target_power,
        "peak_power": peak_power.astype(np.float64),
        "target_to_peak_db": target_to_peak,
        "peak_theta_deg": peak_theta.astype(np.float64),
        "peak_phi_deg": peak_phi.astype(np.float64),
        "pointing_error_deg": np.asarray(pointing, dtype=np.float64),
        "strict_margin_db": strict_margin,
        "hpbw_aware_margin_db": hpbw_margin,
        "dominant_polarization_ratio_db": polarization_ratio.astype(np.float64),
    }


def _phase_commands(data: HfssPatternData, targets: np.ndarray) -> np.ndarray:
    oracle = PhysicalOracle(data, OracleConfig(starts=32, max_steps=300, seed=27050))
    return np.stack([oracle.target_power_phase_oracle(float(theta), float(phi)) for theta, phi in targets]).astype(np.complex64)


def _nearest_neighbor_diagnostics(targets: np.ndarray, metric: np.ndarray, commands: np.ndarray) -> np.ndarray:
    """Peak jump and phase-aligned command change to the closest sampled target."""
    count = len(targets)
    result = np.zeros((count, 2), dtype=np.float32)
    peak_theta = metric[:, METRIC_NAMES.index("peak_theta_deg")]
    peak_phi = metric[:, METRIC_NAMES.index("peak_phi_deg")]
    for index in range(count):
        target_distance = angular_distance_deg(
            targets[:, 0], targets[:, 1], targets[index, 0], targets[index, 1]
        )
        target_distance[index] = np.inf
        neighbor = int(np.argmin(target_distance))
        result[index, 0] = angular_distance_deg(
            peak_theta[index], peak_phi[index], peak_theta[neighbor], peak_phi[neighbor]
        )
        result[index, 1] = global_phase_aligned_weight_change(commands[index], commands[neighbor])
    return result


def _v26_commands(
    controller: BeamformingControllerV26,
    targets: np.ndarray,
    estimated_iq: np.ndarray,
    confidence: np.ndarray,
    health: np.ndarray,
) -> np.ndarray:
    # The frozen V2.6 head is zero, therefore deployment_input() is the exact zero-residual baseline.
    values = []
    for theta, phi in targets:
        _, initial = controller.deployment_input((float(theta), float(phi)), estimated_iq, confidence, health)
        values.append(initial * health)
    return np.asarray(values, dtype=np.complex64)


def _save_state_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def evaluate_state(
    state: AuditState,
    data: HfssPatternData,
    design,
    v25: FrozenV25Adapter,
) -> Path:
    destination = CACHE_ROOT / f"state_{state.state_index:02d}.npz"
    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as archive:
                if (
                    archive["targets_hfss_deg"].shape == (256, 2)
                    and archive["metrics"].shape == (6, 256, 9)
                    and archive["tracking_diagnostics"].shape == (6, 256, 2)
                ):
                    return destination
        except (OSError, KeyError, ValueError):
            pass
    health = state.health.astype(np.float32)
    measurement = simulate_measurements(
        design,
        state.calibration,
        snr_db=state.measurement_snr_db,
        global_phase_rad=np.deg2rad(state.global_phase_deg),
        seed=state.seed,
    )
    controller = BeamformingControllerV26(data, calibration_design=design)
    estimate = controller.calibrator.calibrate(measurement, health)
    estimated_iq = estimate.calibration_iq[0]
    estimated = estimated_iq[0, 0] + 1j * estimated_iq[1, 0]
    targets = stratified_targets(state.state_index)
    transformed = state_pattern(data, _v26_profile(state))
    phase = _phase_commands(transformed, targets)
    v26 = _v26_commands(controller, targets, estimated_iq, estimate.confidence[0], health)
    v25_commands = v25.commands(targets, estimated, health)
    command_methods = (phase, v26, v25_commands)
    method_names = []
    metric_values = []
    command_values = []
    code_values = []
    tracking_values = []
    for name, commands in zip(("phase_reference", "v2.6_zero_residual", "v2.5_low_update_15mm"), command_methods):
        codes = []
        integers = []
        for command in commands:
            code, restored = quantize_int8_disk(command)
            codes.append(code)
            integers.append(restored)
        integers = np.asarray(integers, dtype=np.complex64)
        effective_fp32 = commands * state.calibration[None, :] * health[None, :]
        effective_int8 = integers * state.calibration[None, :] * health[None, :]
        for suffix, effective, stored_command in (
            ("fp32", effective_fp32, commands),
            ("int8", effective_int8, integers),
        ):
            metric = _batch_metrics(data, effective, targets)
            metric_array = np.stack([metric[key] for key in METRIC_NAMES], axis=-1)
            metric_values.append(metric_array)
            method_names.append(f"{name}_{suffix}")
            command_values.append(stored_command)
            code_values.append(np.asarray(codes, dtype=np.int8))
            tracking_values.append(_nearest_neighbor_diagnostics(targets, metric_array, stored_command))
    calibration_holdout = controller.calibrator.holdout_nmse_db(
        data,
        estimated,
        state.calibration,
        np.asarray([[65.0, -25.0], [80.0, 10.0], [100.0, -5.0], [118.0, 25.0]]),
    )[0]
    payload = {
        "state_index": np.asarray(state.state_index, dtype=np.int32),
        "state_id": np.asarray(state.state_id),
        "state_class": np.asarray(state.state_class),
        "targets_hfss_deg": targets,
        "method_names": np.asarray(method_names),
        "metric_names": np.asarray(METRIC_NAMES),
        "metrics": np.asarray(metric_values, dtype=np.float32),
        "commands": np.asarray(command_values, dtype=np.complex64),
        "int8_codes": np.asarray(code_values, dtype=np.int8),
        "tracking_diagnostics": np.asarray(tracking_values, dtype=np.float32),
        "calibration_nmse_db": np.asarray(calibration_holdout, dtype=np.float32),
        "calibration_confidence": np.asarray(estimate.confidence[0, 0], dtype=np.float32),
        "calibration_condition_number": np.asarray(estimate.condition_number, dtype=np.float32),
    }
    _save_state_shard(destination, payload)
    return destination


METRIC_NAMES = (
    "target_power",
    "peak_power",
    "target_to_peak_db",
    "peak_theta_deg",
    "peak_phi_deg",
    "pointing_error_deg",
    "strict_margin_db",
    "hpbw_aware_margin_db",
    "dominant_polarization_ratio_db",
)


def _state_worker(state_index: int) -> tuple[int, str]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    state = all_audit_states()[int(state_index)]
    data = HfssPatternData.load()
    path = evaluate_state(state, data, build_d_optimal_design(data), FrozenV25Adapter())
    return int(state_index), str(path)


def run_audit(limit_states: int | None = None, workers: int = 1) -> dict[str, Any]:
    states = all_audit_states()
    if limit_states is not None:
        states = states[: int(limit_states)]
    paths = []
    if workers <= 1:
        data = HfssPatternData.load()
        design = build_d_optimal_design(data)
        v25 = FrozenV25Adapter()
        for offset, state in enumerate(states, start=1):
            paths.append(evaluate_state(state, data, design, v25))
            print(json.dumps({"audit_state": offset, "requested": len(states), "state_id": state.state_id}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_state_worker, state.state_index): state for state in states}
            for offset, future in enumerate(as_completed(futures), start=1):
                state_index, path = future.result()
                paths.append(Path(path))
                print(json.dumps({"audit_state": offset, "requested": len(states), "state_id": all_audit_states()[state_index].state_id}), flush=True)
    return {"completed_states": len(paths), "completed_cases": len(paths) * 256, "paths": [str(path) for path in paths]}


def merge_audit(require_complete: bool = True) -> tuple[Path, Path]:
    states = all_audit_states()
    missing = [state.state_index for state in states if not (CACHE_ROOT / f"state_{state.state_index:02d}.npz").is_file()]
    if missing and require_complete:
        raise RuntimeError(f"audit has {len(missing)} missing state shards: {missing[:8]}")
    rows = []
    arrays = []
    for state in states:
        path = CACHE_ROOT / f"state_{state.state_index:02d}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            methods = archive["method_names"].tolist()
            metric_names = archive["metric_names"].tolist()
            metrics = archive["metrics"]
            tracking = archive["tracking_diagnostics"]
            targets = archive["targets_hfss_deg"]
            phase_fp32_target = metrics[0, :, metric_names.index("target_power")]
            for method_index, method in enumerate(methods):
                values = metrics[method_index]
                target_power = values[:, metric_names.index("target_power")]
                for sample_index, (target, metric) in enumerate(zip(targets, values)):
                    flags = classify_boundary(
                        float(metric[metric_names.index("target_to_peak_db")]),
                        float(metric[metric_names.index("pointing_error_deg")]),
                        float(metric[metric_names.index("strict_margin_db")]),
                    )
                    reasons = list(boundary_reasons(flags))
                    if float(archive["calibration_confidence"]) < 0.5:
                        reasons.append("CALIBRATION_UNCERTAIN")
                    if method.startswith("v2.5_"):
                        reasons.append("GEOMETRY_MISMATCH_15MM_VS_6P8MM")
                    base_target_power = (
                        values[sample_index, metric_names.index("target_power")]
                        if method.endswith("_fp32")
                        else metrics[method_index - 1, sample_index, metric_names.index("target_power")]
                    )
                    int8_degradation = (
                        0.0
                        if method.endswith("_fp32")
                        else float(10.0 * np.log10((base_target_power + 1e-30) / (target_power[sample_index] + 1e-30)))
                    )
                    rows.append(
                        {
                            "state_index": state.state_index,
                            "state_id": state.state_id,
                            "state_class": state.state_class,
                            "amplitude_rms_fraction": state.amplitude_rms_fraction,
                            "phase_rms_deg": state.phase_rms_deg,
                            "measurement_snr_db": state.measurement_snr_db,
                            "failed_port": "" if state.failed_port is None else state.failed_port + 1,
                            "global_phase_deg": state.global_phase_deg,
                            "sample_index": sample_index,
                            "theta_deg": float(target[0]),
                            "phi_deg": float(target[1]),
                            "method": method,
                            **{name: float(metric[index]) for index, name in enumerate(metric_names)},
                            "target_power_db": float(10.0 * np.log10(metric[metric_names.index("target_power")] + 1e-30)),
                            "global_peak_power_db": float(10.0 * np.log10(metric[metric_names.index("peak_power")] + 1e-30)),
                            "target_power_regret_vs_phase_db": float(10.0 * np.log10((phase_fp32_target[sample_index] + 1e-30) / (target_power[sample_index] + 1e-30))),
                            "gain_reachable": flags.gain_reachable,
                            "pointing_reachable": flags.pointing_reachable,
                            "strict_unambiguous": flags.strict_unambiguous,
                            "int8_degradation_db": int8_degradation,
                            "nearest_sample_peak_jump_deg": float(tracking[method_index, sample_index, 0]),
                            "nearest_sample_weight_change": float(tracking[method_index, sample_index, 1]),
                            "reason": "|".join(dict.fromkeys(reasons)),
                            "calibration_nmse_db": float(archive["calibration_nmse_db"]),
                            "calibration_confidence": float(archive["calibration_confidence"]),
                        }
                    )
            arrays.append(
                {
                    "state_index": state.state_index,
                    "targets": targets,
                    "metrics": metrics,
                    "commands": archive["commands"],
                    "int8_codes": archive["int8_codes"],
                    "calibration_nmse_db": float(archive["calibration_nmse_db"]),
                    "calibration_confidence": float(archive["calibration_confidence"]),
                }
            )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = AUDIT_ROOT / "control_cases.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    npz_path = AUDIT_ROOT / "control_cases.npz"
    np.savez_compressed(
        npz_path,
        state_indices=np.asarray([item["state_index"] for item in arrays], dtype=np.int32),
        targets_hfss_deg=np.stack([item["targets"] for item in arrays]) if arrays else np.empty((0, 256, 2)),
        method_names=np.asarray(["phase_reference_fp32", "phase_reference_int8", "v2.6_zero_residual_fp32", "v2.6_zero_residual_int8", "v2.5_low_update_15mm_fp32", "v2.5_low_update_15mm_int8"]),
        metric_names=np.asarray(METRIC_NAMES),
        metrics=np.stack([item["metrics"] for item in arrays]) if arrays else np.empty((0, 6, 256, 9)),
        commands=np.stack([item["commands"] for item in arrays]) if arrays else np.empty((0, 6, 256, 4)),
        int8_codes=np.stack([item["int8_codes"] for item in arrays]) if arrays else np.empty((0, 6, 256, 4, 2), dtype=np.int8),
        calibration_nmse_db=np.asarray([item["calibration_nmse_db"] for item in arrays], dtype=np.float32),
        calibration_confidence=np.asarray([item["calibration_confidence"] for item in arrays], dtype=np.float32),
        complete=np.asarray(len(arrays) == 83),
    )
    return csv_path, npz_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_audit(args.limit_states, args.workers), ensure_ascii=False, indent=2))
    if args.merge:
        print([str(path) for path in merge_audit()])
