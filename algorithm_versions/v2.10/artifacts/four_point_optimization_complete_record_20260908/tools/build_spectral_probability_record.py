"""Recompute finite-grid single-direction upper bounds and random-four guarantees."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION_ROOT = ROOT.parents[1]
DATA = VERSION_ROOT / "artifacts" / "front_hemisphere_20260906" / "datasets"
OUT = ROOT / "analysis"


def choose4(n: int) -> int:
    return n * (n - 1) * (n - 2) * (n - 3) // 24


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    global_index = 0
    for split, relative in (
        ("train", "dev/train.npz"),
        ("validation", "dev/validation.npz"),
        ("calibration", "reserved/calibration.npz"),
        ("holdout", "reserved/holdout.npz"),
    ):
        with np.load(DATA / relative, allow_pickle=False) as values:
            q = values["q"]
            h = values["h"]
            theta = values["theta"]
            phi = values["phi"]
            for source_row in np.flatnonzero(q[:, 0] > 0):
                key = tuple(float(item) for item in q[source_row])
                if key in seen:
                    raise RuntimeError(f"duplicate front direction: {key}")
                seen.add(key)
                singular_max = float(np.linalg.svd(h[source_row], compute_uv=False)[0])
                upper_linear = singular_max * singular_max / 60.0
                records.append({
                    "global_front_index": global_index,
                    "source_split": split,
                    "source_row": int(source_row),
                    "q_x": key[0],
                    "q_y": key[1],
                    "q_z": key[2],
                    "theta_deg": float(theta[source_row]),
                    "phi_deg": float(phi[source_row]),
                    "arbitrary_amplitude_unit_power_upper_linear": upper_linear,
                    "arbitrary_amplitude_unit_power_upper_db": 10.0 * math.log10(max(upper_linear, 1e-300)),
                })
                global_index += 1

    if len(records) != 32400:
        raise RuntimeError(f"expected 32400 unique front directions, got {len(records)}")
    with (OUT / "spectral_single_direction_bounds.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    upper_db = np.asarray([row["arbitrary_amplitude_unit_power_upper_db"] for row in records])
    total = choose4(len(records))
    probabilities = []
    for threshold in (0.0, 1.0, 3.0, 5.0):
        weak = int(np.sum(upper_db < threshold))
        guaranteed = total - choose4(len(records) - weak)
        probabilities.append({
            "threshold_db": threshold,
            "weak_direction_count": weak,
            "front_direction_count": len(records),
            "total_four_point_combinations": total,
            "guaranteed_low_four_point_combinations": guaranteed,
            "uniform_without_replacement_probability_lower_bound": guaranteed / total,
        })
    with (OUT / "random_four_probability_bounds.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(probabilities[0]))
        writer.writeheader()
        writer.writerows(probabilities)

    weakest = records[int(np.argmin(upper_db))]
    summary = {
        "front_direction_count": len(records),
        "unique_direction_count": len(seen),
        "spectral_upper_db": {
            "minimum": float(np.min(upper_db)),
            "p1": float(np.percentile(upper_db, 1)),
            "p5": float(np.percentile(upper_db, 5)),
            "median": float(np.median(upper_db)),
            "p95": float(np.percentile(upper_db, 95)),
            "maximum": float(np.max(upper_db)),
        },
        "weakest_direction": weakest,
        "probability_rows": probabilities,
        "proof_contract": (
            "For Q=H^H H/60 and every |w_n|=1/4, ||w||_2=1, hence "
            "w^H Q w <= lambda_max(Q). Any four-set containing a direction "
            "whose bound is below tau has optimized Gmin below tau."
        ),
    }
    (OUT / "spectral_probability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
