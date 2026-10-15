"""Verify that the package contains one consistent O3 method and no refinement data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION_ROOT = ROOT.parents[3]
FRONT = VERSION_ROOT / "artifacts" / "front_hemisphere_20260906"
METHOD = "H0_O3_32x320_Adam_phase_only"


def verify_split(split: str, expected: int, expected_per_group: int) -> dict[str, object]:
    suite = np.load(FRONT / f"{split}_front_suite.npz", allow_pickle=False)
    result = np.load(ROOT / "raw" / "o3_results" / f"{split}_o3_results.npz", allow_pickle=False)
    upper = np.load(ROOT / "raw" / "sdp_upper_bound" / f"{split}_sdp_upper_bound.npz", allow_pickle=False)
    assert len(result["weight"]) == expected
    assert str(result["method"]) == METHOD
    assert result["cnn_files_loaded"].size == 0
    groups, counts = np.unique(result["stratum"], return_counts=True)
    assert set(groups.tolist()) == {"clustered", "grazing", "moderate", "wide"}
    assert np.all(counts == expected_per_group)
    modulus_error = float(np.max(np.abs(np.abs(result["weight"]) - 0.25)))
    assert modulus_error < 1e-6
    field = np.einsum(
        "bkrn,bn->bkr",
        suite["h"].astype(np.complex128),
        result["weight"].astype(np.complex128),
    )
    gain = np.sum(np.abs(field) ** 2, axis=-1) / 60.0
    gain_error = float(np.max(np.abs(gain - result["target_gain"])))
    assert gain_error < 2e-5
    assert np.all(upper["constant_modulus_sdp_upper"] + 1e-10 >= upper["o3_minimum"])
    with (ROOT / "tables" / f"o3_{split}_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    assert {row["method"] for row in rows} == {METHOD}
    return {
        "split": split,
        "cases": expected,
        "cases_per_group": expected_per_group,
        "method": METHOD,
        "cnn_files_loaded": [],
        "maximum_modulus_error": modulus_error,
        "maximum_hfss_gain_recompute_error": gain_error,
        "status": "PASS",
    }


def main() -> None:
    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and any(word in path.name.lower() for word in ("targeted", "refine", "slsqp"))
    ]
    assert not forbidden, forbidden
    report = {
        "status": "CONSISTENT_O3_ONLY_PACKAGE_OK",
        "comparison_method_count": 1,
        "refinement_files": [],
        "splits": [verify_split("validation", 240, 60), verify_split("holdout", 600, 150)],
    }
    (ROOT / "PACKAGE_VERIFICATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
