"""Verify the complete four-target optimization record and write manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "evidence" / "hfss_o3_sdp_cross_validation"
VERSION_ROOT = ROOT.parents[1]
FRONT = VERSION_ROOT / "artifacts" / "front_hemisphere_20260906"
DATA = FRONT / "datasets"


def required(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    required(ROOT / "README.md")
    required(ROOT / "COMPLETE_TECHNICAL_REPORT.md")
    required(ROOT / "analysis" / "spectral_single_direction_bounds.csv")
    checks = {}
    split_records = []
    for split, expected_cases, expected_per_group in (
        ("validation", 240, 60),
        ("holdout", 600, 150),
    ):
        suite_path = FRONT / f"{split}_front_suite.npz"
        o3_path = CORE / "raw" / "o3_results" / f"{split}_o3_results.npz"
        sdp_path = CORE / "raw" / "sdp_upper_bound" / f"{split}_sdp_upper_bound.npz"
        with np.load(suite_path, allow_pickle=False) as suite, np.load(o3_path, allow_pickle=False) as o3, np.load(sdp_path, allow_pickle=False) as sdp:
            cases = len(suite["q"])
            groups, counts = np.unique(suite["stratum"].astype(str), return_counts=True)
            method_values = np.unique(o3["method"].astype(str))
            cnn_values = o3["cnn_files_loaded"]
            weight = o3["weight"]
            gain = np.sum(np.abs(np.einsum("bkrn,bn->bkr", suite["h"], weight)) ** 2, axis=-1) / 60.0
            gain_error = float(np.max(np.abs(gain - o3["target_gain"])))
            modulus_error = float(np.max(np.abs(np.abs(weight) - 0.25)))
            sdp_violation = float(np.max(sdp["o3_minimum"] - sdp["constant_modulus_sdp_upper"]))
            row = {
                "split": split,
                "cases": cases,
                "groups": dict(zip(groups.tolist(), counts.astype(int).tolist())),
                "method": method_values.tolist(),
                "cnn_files_loaded_size": int(cnn_values.size),
                "maximum_modulus_error": modulus_error,
                "maximum_hfss_gain_recompute_error": gain_error,
                "maximum_o3_minus_sdp_linear": sdp_violation,
                "passed": bool(
                    cases == expected_cases
                    and len(groups) == 4
                    and np.all(counts == expected_per_group)
                    and method_values.tolist() == ["H0_O3_32x320_Adam_phase_only"]
                    and cnn_values.size == 0
                    and modulus_error < 1e-6
                    and gain_error < 2e-5
                    and sdp_violation <= 2e-5
                ),
            }
            split_records.append(row)
    checks["hfss_o3_sdp"] = all(row["passed"] for row in split_records)

    ideal = json.loads(
        (ROOT / "evidence" / "ideal_array_control" / "verification.json").read_text(encoding="utf-8")
    )
    checks["ideal_array_control"] = bool(ideal["all_checks_passed"])

    spectral = json.loads(
        (ROOT / "analysis" / "spectral_probability_summary.json").read_text(encoding="utf-8")
    )
    checks["spectral_direction_count"] = (
        spectral["front_direction_count"] == 32400
        and spectral["unique_direction_count"] == 32400
    )

    full_counts = {}
    total_front = 0
    for split, relative in (
        ("train", "dev/train.npz"),
        ("validation", "dev/validation.npz"),
        ("calibration", "reserved/calibration.npz"),
        ("holdout", "reserved/holdout.npz"),
    ):
        with np.load(DATA / relative, allow_pickle=False) as values:
            front = int(np.sum(values["q"][:, 0] > 0))
            full_counts[split] = {"total": len(values["q"]), "front": front}
            total_front += front
    checks["full_dataset_front_count"] = total_front == 32400

    geometry = json.loads(
        (ROOT / "evidence" / "hfss_geometry_baseline" / "artifacts" / "build_4x4_report.json").read_text(encoding="utf-8")
    )
    checks["hfss_geometry_baseline"] = bool(
        geometry["passed"]
        and geometry["logical_shape"] == [4, 4]
        and len(geometry["ports"]) == 16
        and geometry["radiation_box"]["single_radiation_boundary"]
        and geometry["radiation_box"]["old_box9_absent"]
    )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "splits": split_records,
        "ideal_array_verification": ideal,
        "full_direction_counts": full_counts,
        "spectral_summary": spectral,
        "ground_connectivity_status": "NOT_PROVEN_BY_EXISTING_GEOMETRY_AUDIT",
        "known_failed_record": (
            "hfss_geometry_baseline/artifacts/solve_export_report.json records a failed "
            "Touchstone export and is retained as provenance, not counted as a passed check."
        ),
    }
    (ROOT / "PACKAGE_VERIFICATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, ensure_ascii=False))

    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "FILE_INDEX.csv":
            continue
        entries.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
        })
    with (ROOT / "FILE_INDEX.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)

    print(json.dumps({"status": report["status"], "files": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
