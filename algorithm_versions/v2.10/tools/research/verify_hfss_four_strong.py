"""Verify the frozen exact-HFSS four-strong-point evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = BASE / "artifacts" / "front_hemisphere_20260906"
DEFAULT_EVIDENCE = BASE / "artifacts" / "hfss_four_strong_highsearch_20260906"
BASELINE_SCRIPT = BASE / "tools" / "research" / "hfss_four_strong_baseline.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_suite(source: Path, evidence: Path, suite: str) -> dict:
    protocol_path = evidence / f"{suite}_protocol.json"
    result_path = evidence / f"{suite}_results.json"
    arrays_path = evidence / f"{suite}_arrays.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}

    assert protocol == result["protocol"]
    assert protocol["purpose"].startswith("exact-HFSS")
    assert protocol["cnn_files_loaded"] == []
    assert protocol["reverse_directions_used"] == 0
    assert protocol["forward_grid_directions"] == 32400
    assert protocol["continuous_hemisphere_certificate"] is False
    assert len(arrays["q"]) == protocol["cases"]
    assert bool((arrays["q"][..., 0] > 0).all())

    source_hashes = protocol["source_hashes"]
    assert sha256(BASELINE_SCRIPT) == source_hashes["script"]
    for filename, expected in source_hashes.items():
        if filename == "script":
            continue
        assert sha256(source / filename) == expected

    spectral = arrays["spectral_solo_upper_bound"]
    floor = {
        "H0_target_maxmin": 0.0,
        "H1_ratio_only": None,
        "H2_preserve_0p5db": 0.5,
        "H3_preserve_1db": 1.0,
    }
    checked = []
    for method, allowed_loss in floor.items():
        prefix = f"{method}_"
        if f"{prefix}weight" not in arrays:
            continue
        weight = arrays[f"{prefix}weight"]
        rho = arrays[f"{prefix}minimum_rho"]
        utilization = arrays[f"{prefix}minimum_solo_utilization"]
        loss = arrays[f"{prefix}target_loss_db"]
        strict = arrays[f"{prefix}strict_four_strong"]
        assert float(np.max(np.abs(np.abs(weight) - 0.25))) < 1e-6
        assert bool((rho <= 1.0 + 1e-6).all())
        assert bool((utilization <= spectral + 2e-5).all())
        assert np.array_equal(strict, (rho >= 0.5) & (utilization >= 0.5))
        if allowed_loss is not None:
            assert float(loss.max()) <= allowed_loss + 2e-4
        reported = result["methods"][method]
        assert abs(float(strict.mean()) - reported["fraction_strict_four_strong"]) < 1e-7
        assert abs(
            float((rho >= 0.5).mean())
            - reported["fraction_spatial_strong"]["rho_ge_0_5"]
        ) < 1e-7
        checked.append(method)

    return {
        "suite": suite,
        "cases": int(protocol["cases"]),
        "methods": checked,
        "cnn_files_loaded": 0,
        "reverse_directions_used": 0,
        "maximum_phase_only_magnitude_error": float(
            max(
                np.max(np.abs(np.abs(arrays[f"{method}_weight"]) - 0.25))
                for method in checked
            )
        ),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    suites = [verify_suite(args.source.resolve(), args.evidence.resolve(), name) for name in ("validation", "holdout")]
    print(json.dumps({"status": "HFSS_FOUR_STRONG_VERIFICATION_OK", "suites": suites}, indent=2))


if __name__ == "__main__":
    main()
