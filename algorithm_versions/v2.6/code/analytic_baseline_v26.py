"""384-sample calibrated analytic/zero-residual baseline on unseen states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration_v26 import build_d_optimal_design, simulate_measurements
from controller_v26 import BeamformingControllerV26
from dataset_v26 import generate_state_profiles, state_pattern, target_angles_for_state
from hfss_physics_v26 import HfssPatternData
from oracle_v26 import OracleConfig, PhysicalOracle


V26_ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    controller = BeamformingControllerV26(data, calibration_design=design)
    profiles = generate_state_profiles("test")
    healthy = [profile for profile in profiles if profile.failed_port is None][:3]
    failed = [profile for profile in profiles if profile.failed_port is not None][:3]
    selected = healthy + failed
    rows = []
    for profile in selected:
        rng = np.random.default_rng(profile.seed ^ 0xA26)
        measurement = simulate_measurements(
            design,
            profile.calibration,
            snr_db=profile.measurement_snr_db,
            global_phase_rad=float(rng.uniform(-np.pi, np.pi)),
            seed=profile.seed,
        )
        health = np.asarray(profile.health_mask, dtype=np.float32)
        calibration_iq, confidence = controller.calibrate(measurement, health)
        physical = state_pattern(data, profile)
        oracle = PhysicalOracle(physical, OracleConfig(starts=32, max_steps=300))
        for theta, phi in target_angles_for_state(profile):
            result = controller.beamform((float(theta), float(phi)), calibration_iq[0], confidence[0], health)
            actual = physical.metrics(result.weights_int8, float(theta), float(phi))
            oracle_weights = oracle.target_power_phase_oracle(float(theta), float(phi))
            reference = physical.metrics(oracle_weights, float(theta), float(phi))
            rows.append(
                {
                    "state_index": profile.state_index,
                    "failed_port": profile.failed_port,
                    "theta_deg": float(theta),
                    "phi_deg": float(phi),
                    "target_power_regret_db": 10.0 * np.log10((reference.target_power + 1e-30) / (actual.target_power + 1e-30)),
                    "target_to_peak_db": actual.target_to_peak_db,
                    "pointing_error_deg": actual.pointing_error_deg,
                    "strong_peak_margin_db": actual.strong_peak_margin_db,
                    "feasible": bool(actual.feasible),
                    "calibration_confidence": float(confidence[0, 0]),
                }
            )
    regret = [row["target_power_regret_db"] for row in rows]
    pointing = [row["pointing_error_deg"] for row in rows]
    margin = [row["strong_peak_margin_db"] for row in rows]
    payload = {
        "schema_version": 1,
        "scope": "zero-residual CNN equals calibrated analytic initial; this is not trained V2.6",
        "states": len(selected),
        "samples": len(rows),
        "healthy_states": len(healthy),
        "failed_port_states": len(failed),
        "all_finite": bool(np.isfinite(regret).all() and np.isfinite(pointing).all() and np.isfinite(margin).all()),
        "feasible_count": sum(row["feasible"] for row in rows),
        "target_power_regret_db_p05_median_p95": np.percentile(regret, [5, 50, 95]).tolist(),
        "pointing_error_deg_p05_median_p95": np.percentile(pointing, [5, 50, 95]).tolist(),
        "strong_peak_margin_db_p05_median_p95_max": np.percentile(margin, [5, 50, 95, 100]).tolist(),
        "rows": rows,
    }
    output = V26_ROOT / "output" / "reports" / "analytic_baseline_study.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))

