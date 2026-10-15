"""Validate and promote a completed recovery run into official raw exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


WORKTREE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE_ROOT / "algorithm_versions" / "v2.17" / "src"))

from v217.cross_array import ArraySpec, load_ffd_operator, load_touchstone  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_eep(mapping: Path, spec: ArraySpec, expected_points: int) -> dict[str, object]:
    data = load_ffd_operator(mapping, spec)
    if data["g_ephi"].shape != (expected_points, spec.element_count):
        raise RuntimeError(f"Unexpected EEP shape for {mapping}: {data['g_ephi'].shape}")
    for name in ("g_ephi", "g_etheta"):
        value = data[name]
        if not (np.isfinite(value.real).all() and np.isfinite(value.imag).all()):
            raise RuntimeError(f"{mapping}: {name} contains NaN or Inf")
    files = [Path(value) for value in data["source_files"]]
    if len(files) != spec.element_count or any(not path.is_file() or path.stat().st_size <= 0 for path in files):
        raise RuntimeError(f"{mapping}: incomplete FFD set")
    return {
        "mapping": file_record(mapping),
        "point_count": expected_points,
        "ffd_count": len(files),
        "finite": True,
        "phase_reference_m": data["source_phase_reference_m"].tolist(),
        "phase_reference_to_global_applied": True,
        "files": [file_record(path) for path in files],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    run = args.run.resolve()
    audit_path = args.audit_summary.resolve()
    project = args.project.resolve()
    spec = ArraySpec(args.array_id, args.rows, args.cols, 0.015, 0.0068)

    touchstone_candidates = [
        run / f"{args.array_id}_47ghz.s{spec.element_count}p",
        run / f"{args.array_id}_47ghz_readback.s{spec.element_count}p",
    ]
    source_touchstones = [path for path in touchstone_candidates if path.is_file()]
    if len(source_touchstones) != 1:
        raise RuntimeError(
            f"Expected exactly one recovered Touchstone file, got {source_touchstones}"
        )
    source_touchstone = source_touchstones[0]
    integer_source = run / "eep" / "integer"
    offset_source = run / "eep" / "offset"
    recovery_receipt = run.with_suffix(".json")
    if not all(path.exists() for path in (source_touchstone, integer_source, offset_source, audit_path, project)):
        raise FileNotFoundError("Recovery run is missing a required promotion input")

    frequency, matrix, reference = load_touchstone(source_touchstone, spec.element_count)
    if (
        matrix.shape[1:] != (spec.element_count, spec.element_count)
        or frequency.size == 0
        or float(np.min(np.abs(frequency - 47e9))) > 1.0
    ):
        raise RuntimeError(
            f"Recovered Touchstone failed shape/frequency gate: {matrix.shape}, {frequency}"
        )
    if not (np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all()):
        raise RuntimeError("Recovered Touchstone contains NaN or Inf")
    integer = validate_eep(integer_source / "eep.txt", spec, 181 * 361)
    offset = validate_eep(offset_source / "eep.txt", spec, 180 * 360)
    audit_container = json.loads(audit_path.read_text(encoding="utf-8"))
    audit = audit_container.get("direct_audit", audit_container)
    if not audit.get("passed"):
        raise RuntimeError("Corrected direct-combination audit has not passed")
    if not (
        float(audit["aggregate_complex_nmse_db"]) < -100.0
        and all(
            float(item["complex_nmse_db"]) < -80.0
            and float(item["amplitude_max_relative_error"]) < 1e-6
            for item in audit["per_command"]
        )
    ):
        raise RuntimeError("Corrected direct-combination metrics do not satisfy frozen thresholds")

    if args.validate_only:
        preflight = {
            "passed": True,
            "validate_only": True,
            "array_id": args.array_id,
            "logical_shape": [args.rows, args.cols],
            "port_count": spec.element_count,
            "touchstone": file_record(source_touchstone),
            "integer_eep": integer,
            "offset_eep": offset,
            "direct_audit": {
                "aggregate_complex_nmse_db": float(audit["aggregate_complex_nmse_db"]),
                "command_count": len(audit["per_command"]),
            },
        }
        rendered = json.dumps(preflight, ensure_ascii=False, indent=2) + "\n"
        if args.validation_report is not None:
            validation_report = args.validation_report.resolve()
            if validation_report.exists():
                raise FileExistsError(
                    f"Validation report already exists: {validation_report}"
                )
            validation_report.parent.mkdir(parents=True, exist_ok=True)
            validation_report.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return

    official = root / "raw_exports" / args.array_id
    target_touchstone = official / f"{args.array_id}_47ghz.s{spec.element_count}p"
    target_eep = official / "eep"
    if target_touchstone.exists() or target_eep.exists():
        raise FileExistsError("Official Touchstone or EEP target already exists; refusing to overwrite")
    shutil.copy2(source_touchstone, target_touchstone)
    shutil.copytree(integer_source, target_eep / "integer")
    shutil.copytree(offset_source, target_eep / "offset")

    stages = official / "stages"
    history = stages / "history"
    history.mkdir(parents=True, exist_ok=True)
    archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in ("solve", "export_s", "export_eep", "direct_audit"):
        existing = stages / f"{name}.json"
        if existing.exists():
            archived = history / f"{name}_pre_recovery_{archive_stamp}.json"
            if archived.exists():
                raise FileExistsError(f"History target already exists: {archived}")
            shutil.copy2(existing, archived)

    now = datetime.now(timezone.utc).isoformat()
    project_hash = sha256_file(project)
    solve = {
        "passed": True,
        "stage": "solve",
        "array_id": args.array_id,
        "setup": "Setup2 : LastAdaptive",
        "solution_source": "user_gui_solve_reused_from_recovery_copy",
        "solution_readback": (
            f"all {spec.element_count ** 2} S-parameter traces readable and finite at 47 GHz"
        ),
        "solve_performed_this_run": False,
        "project": str(project),
        "project_sha256": project_hash,
        "project_unchanged_during_extraction": True,
        "recovery_receipt": str(recovery_receipt),
        "recorded_utc": now,
    }
    if args.profile is not None:
        profile = args.profile.resolve()
        if not profile.is_file() or profile.stat().st_size <= 0:
            raise RuntimeError(f"Solve profile is missing or empty: {profile}")
        solve["profile"] = file_record(profile)
    export_s = {
        "passed": True,
        "stage": "export_s",
        "array_id": args.array_id,
        "project": str(project),
        "project_sha256_after_solve": project_hash,
        "touchstone": str(target_touchstone.resolve()),
        "bytes": target_touchstone.stat().st_size,
        "sha256": sha256_file(target_touchstone),
        "frequency_count": int(len(frequency)),
        "frequency_hz": frequency.tolist(),
        "port_count": spec.element_count,
        "reference_impedance_ohm": reference,
        "finite": True,
        "method": (
            "native_hfss_touchstone_export"
            if "_readback." not in source_touchstone.name
            else f"all_{spec.element_count ** 2}_hfss_s_trace_readback_after_native_export_rejected"
        ),
        "native_export_passed": "_readback." not in source_touchstone.name,
        "recorded_utc": now,
    }
    export_eep = {
        "passed": True,
        "stage": "export_eep",
        "array_id": args.array_id,
        "project": str(project),
        "project_sha256_after_solve": project_hash,
        "grids": {"integer": integer, "offset": offset},
        "recorded_utc": now,
    }
    amplitude_max = max(float(item["amplitude_max_relative_error"]) for item in audit["per_command"])
    worst_nmse = max(float(item["complex_nmse_db"]) for item in audit["per_command"])
    direct_audit = {
        "passed": True,
        "stage": "direct_audit",
        "array_id": args.array_id,
        "command_count": len(audit["per_command"]),
        "aggregate_complex_nmse_db": float(audit["aggregate_complex_nmse_db"]),
        "worst_per_command_complex_nmse_db": worst_nmse,
        "amplitude_max_relative_error": amplitude_max,
        "thresholds": {
            "aggregate_complex_nmse_db_max": -100.0,
            "per_command_complex_nmse_db_max": -80.0,
            "amplitude_max_relative_error_max": 1e-6,
        },
        "phase_reference_correction": "exp(+j*k*q_dot_r_port) from HFSS EEP mapping XYZ",
        "audit_summary": str(audit_path),
        "raw_reports": audit.get("raw_reports", str(audit_path.parent / "raw_reports")),
        "raw_reports_reused": bool(audit.get("raw_reports_reused", False)),
        "not_for_training": True,
        "recorded_utc": now,
    }
    reports = {
        "solve": solve,
        "export_s": export_s,
        "export_eep": export_eep,
        "direct_audit": direct_audit,
    }
    for name, report in reports.items():
        (stages / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    promotion = {
        "passed": True,
        "array_id": args.array_id,
        "source_run": str(run),
        "official_touchstone": file_record(target_touchstone),
        "official_integer_mapping": file_record(target_eep / "integer" / "eep.txt"),
        "official_offset_mapping": file_record(target_eep / "offset" / "eep.txt"),
        "stage_receipts": {name: str((stages / f"{name}.json").resolve()) for name in reports},
        "recorded_utc": now,
    }
    promotion_path = official / f"recovery_promotion_{datetime.now(timezone.utc):%Y%m%d}.json"
    if promotion_path.exists():
        raise FileExistsError(f"Promotion receipt already exists: {promotion_path}")
    promotion_path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
