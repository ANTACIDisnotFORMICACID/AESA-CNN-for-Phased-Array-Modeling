"""Fast complete 64x64 phase-only baseline map for the canonical HFSS array."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hfss_physics_v26 import HfssPatternData, unit_vector_hfss
from oracle_v26 import OracleConfig, PhysicalOracle


V26_ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    data = HfssPatternData.load()
    oracle = PhysicalOracle(data, OracleConfig(starts=32, max_steps=300))
    theta_axis = np.linspace(60.0, 120.0, 64)
    phi_axis = np.linspace(-30.0, 30.0, 64)
    targets = np.asarray([(theta, phi) for theta in theta_axis for phi in phi_axis], dtype=np.float64)
    weights = np.stack([oracle.target_power_phase_oracle(float(theta), float(phi)) for theta, phi in targets])
    grid_theta, grid_phi = data.flat_theta_phi
    grid_unit = unit_vector_hfss(grid_theta, grid_phi)
    target_unit = unit_vector_hfss(targets[:, 0], targets[:, 1])
    htheta = data.etheta.reshape(4, -1)
    hphi = data.ephi.reshape(4, -1)
    rows = []
    for start in range(0, len(targets), 32):
        stop = min(start + 32, len(targets))
        w = weights[start:stop]
        et = np.einsum("bn,nk->bk", w, htheta, optimize=True)
        ep = np.einsum("bn,nk->bk", w, hphi, optimize=True)
        power = np.abs(et) ** 2 + np.abs(ep) ** 2
        peak_index = np.argmax(power, axis=1)
        peak_power = power[np.arange(len(w)), peak_index]
        ht, hp = zip(*(data.response_at(float(t), float(p)) for t, p in targets[start:stop]))
        ht = np.asarray(ht)
        hp = np.asarray(hp)
        target_power = np.abs(np.sum(w * ht, axis=1)) ** 2 + np.abs(np.sum(w * hp, axis=1)) ** 2
        outside = target_unit[start:stop] @ grid_unit.T < np.cos(np.deg2rad(8.0))
        outside_peak = np.max(np.where(outside, power, -np.inf), axis=1)
        peak_unit = grid_unit[peak_index]
        pointing = np.rad2deg(np.arccos(np.clip(np.sum(peak_unit * target_unit[start:stop], axis=1), -1.0, 1.0)))
        rows.append(
            np.stack(
                (
                    10.0 * np.log10((target_power + 1e-30) / (peak_power + 1e-30)),
                    pointing,
                    10.0 * np.log10((target_power + 1e-30) / (outside_peak + 1e-30)),
                ),
                axis=1,
            )
        )
    metrics = np.concatenate(rows, axis=0)
    feasible = (metrics[:, 0] >= -0.5) & (metrics[:, 1] <= 5.0) & (metrics[:, 2] >= 3.0)
    output = V26_ROOT / "output" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "phase_only_map.npz",
        target_hfss_deg=targets.astype(np.float32),
        weights=weights.astype(np.complex64),
        target_to_peak_db=metrics[:, 0].astype(np.float32),
        pointing_error_deg=metrics[:, 1].astype(np.float32),
        strong_peak_margin_db=metrics[:, 2].astype(np.float32),
        feasible=feasible,
    )
    summary = {
        "schema_version": 1,
        "samples": 4096,
        "starts": 32,
        "feasible_count": int(np.sum(feasible)),
        "target_to_peak_db_p05_median_p95": np.percentile(metrics[:, 0], [5, 50, 95]).tolist(),
        "pointing_error_deg_p05_median_p95": np.percentile(metrics[:, 1], [5, 50, 95]).tolist(),
        "strong_peak_margin_db_p05_median_p95_max": np.percentile(metrics[:, 2], [5, 50, 95, 100]).tolist(),
        "conclusion": "phase-only has no feasible direction" if not np.any(feasible) else "phase-only has feasible directions",
    }
    (output / "phase_only_map_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

