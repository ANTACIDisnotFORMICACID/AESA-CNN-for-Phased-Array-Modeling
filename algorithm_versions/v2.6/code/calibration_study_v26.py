"""Deterministic unseen-state calibration acceptance study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration_v26 import PhaseInvariantCalibrator, build_d_optimal_design, simulate_measurements
from dataset_v26 import generate_state_profiles
from hfss_physics_v26 import HfssPatternData


V26_ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    calibrator = PhaseInvariantCalibrator(design)
    holdout = np.asarray(
        [(65.0, -26.0), (72.0, -11.0), (81.0, 7.0), (94.0, 23.0), (103.0, -20.0), (110.0, -2.0), (117.0, 14.0), (119.0, 28.0)]
    )
    rows = []
    for profile in generate_state_profiles("test"):
        rng = np.random.default_rng(profile.seed ^ 0xC411B)
        measurement = simulate_measurements(
            design,
            profile.calibration,
            snr_db=profile.measurement_snr_db,
            global_phase_rad=float(rng.uniform(-np.pi, np.pi)),
            seed=profile.seed,
        )
        health = np.asarray(profile.health_mask)
        estimate = calibrator.calibrate(measurement, health)
        recovered = estimate.calibration_iq[0, 0, 0] + 1j * estimate.calibration_iq[0, 1, 0]
        nmse = float(calibrator.holdout_nmse_db(data, recovered, profile.calibration, holdout)[0])
        rows.append(
            {
                "state_index": profile.state_index,
                "snr_db": profile.measurement_snr_db,
                "failed_port": profile.failed_port,
                "global_phase_rad": float(np.angle(np.sum(measurement[0, 0] + 1j * measurement[0, 1]))),
                "fit_residual_nmse_db": float(estimate.residual_nmse_db[0]),
                "holdout_nmse_db": nmse,
                "confidence": float(estimate.confidence[0, 0]),
                "reliable": nmse <= -20.0 and design.condition_number <= 20.0,
            }
        )
    nmse_values = [row["holdout_nmse_db"] for row in rows]
    payload = {
        "schema_version": 1,
        "states": len(rows),
        "condition_number": design.condition_number,
        "condition_limit": 20.0,
        "holdout_nmse_limit_db": -20.0,
        "reliable_count": sum(row["reliable"] for row in rows),
        "all_reliable": all(row["reliable"] for row in rows),
        "holdout_nmse_db_p05_median_p95_worst": np.percentile(nmse_values, [5, 50, 95, 100]).tolist(),
        "rows": rows,
    }
    output = V26_ROOT / "output" / "reports" / "calibration_study.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

