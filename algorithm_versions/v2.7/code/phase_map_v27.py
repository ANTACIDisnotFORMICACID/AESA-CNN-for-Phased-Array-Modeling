"""Promote the frozen V2.6 4096-point phase map into the V2.7 metric schema."""

from __future__ import annotations

import json
import numpy as np

from boundary_v27 import tracking_grid
from evaluate_v27 import METRIC_NAMES, _batch_metrics
from oracle_v27 import MAP_ROOT, target_grid
from v27_compat import REPO_ROOT, V26_ROOT, enable_v26

enable_v26()
from hfss_physics_v26 import HfssPatternData  # noqa: E402


SOURCE = V26_ROOT / "output" / "reports" / "phase_only_map.npz"


def run() -> dict:
    with np.load(SOURCE, allow_pickle=False) as archive:
        targets = archive["target_hfss_deg"].astype(np.float64)
        weights = archive["weights"].astype(np.complex64)
    expected = target_grid()
    if targets.shape != (4096, 2) or not np.allclose(targets, expected, atol=2e-5, rtol=0):
        raise ValueError("frozen V2.6 phase map does not match the V2.7 64x64 target contract")
    data = HfssPatternData.load()
    rows = []
    for start in range(0, 4096, 64):
        result = _batch_metrics(data, weights[start:start + 64], targets[start:start + 64])
        rows.append(np.stack([result[key] for key in METRIC_NAMES], axis=-1))
    metrics = np.concatenate(rows).astype(np.float32)
    track = tracking_grid(
        metrics[:, METRIC_NAMES.index("peak_theta_deg")].reshape(64, 64),
        metrics[:, METRIC_NAMES.index("peak_phi_deg")].reshape(64, 64),
        weights.reshape(64, 64, 4),
    )
    MAP_ROOT.mkdir(parents=True, exist_ok=True)
    destination = MAP_ROOT / "nominal_phase_only_map.npz"
    np.savez_compressed(
        destination,
        targets_hfss_deg=targets,
        weights=weights,
        metric_names=np.asarray(METRIC_NAMES),
        metrics=metrics,
        **track,
    )
    gain = metrics[:, METRIC_NAMES.index("target_to_peak_db")] >= -0.5
    pointing = gain & (metrics[:, METRIC_NAMES.index("pointing_error_deg")] <= 5.0)
    strict = pointing & (metrics[:, METRIC_NAMES.index("strict_margin_db")] >= 3.0)
    summary = {
        "source": SOURCE.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "unique_targets": 4096,
        "gain_reachable": int(np.sum(gain)),
        "pointing_reachable": int(np.sum(pointing)),
        "strict_unambiguous": int(np.sum(strict)),
        "continuous_tracking_fraction": float(np.mean(track["continuous_tracking"])),
    }
    (MAP_ROOT / "nominal_phase_only_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
