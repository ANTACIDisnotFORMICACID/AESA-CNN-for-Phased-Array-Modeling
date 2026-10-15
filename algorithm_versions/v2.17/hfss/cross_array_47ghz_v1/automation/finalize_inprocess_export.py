"""Validate and audit raw files produced by export_existing_solution_in_gui.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


DATASET_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE_ROOT / "algorithm_versions" / "v2.17" / "src"))

from v217.cross_array import ArraySpec, load_ffd_operator, load_touchstone  # noqa: E402
from recover_existing_solution import parse_direct_report  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_touchstone(raw: dict, output: Path, spec: ArraySpec) -> dict:
    scales = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    unit = raw["frequency_unit"]
    if unit not in scales:
        raise RuntimeError(f"Unsupported frequency unit {unit!r}")
    frequencies = np.asarray(raw["frequency_values"], dtype=np.float64) * scales[unit]
    expected_expressions = [
        f"S({receive},{source})"
        for source in range(1, spec.element_count + 1)
        for receive in range(1, spec.element_count + 1)
    ]
    if raw["expressions"] != expected_expressions or raw["port_count"] != spec.element_count:
        raise RuntimeError("S-parameter expression list or port count is incomplete")
    columns = []
    for expression in expected_expressions:
        item = raw["values"][expression]
        real = np.asarray(item["real"], dtype=np.float64)
        imag = np.asarray(item["imag"], dtype=np.float64)
        if real.shape != frequencies.shape or imag.shape != frequencies.shape:
            raise RuntimeError(f"Incomplete S trace {expression}")
        columns.append(real + 1j * imag)
    matrix = np.stack(columns, axis=1)
    if not (np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all()):
        raise RuntimeError("S matrix contains NaN or Inf")
    lines = [
        "! Reconstructed from ordinary-GUI in-process readable S traces\n",
        "# Hz S RI R 50\n",
    ]
    for frequency, row in zip(frequencies, matrix):
        tokens = [f"{frequency:.15g}"]
        for value in row:
            tokens.extend((f"{value.real:.15g}", f"{value.imag:.15g}"))
        lines.append(" ".join(tokens) + "\n")
    expected_bytes = "".join(lines).encode("ascii")
    if output.exists():
        if output.read_bytes() != expected_bytes:
            raise RuntimeError(f"Existing reconstructed Touchstone differs from current readback: {output}")
    else:
        output.write_bytes(expected_bytes)
    return {"trace_count": len(expected_expressions), "sample_count": int(frequencies.size)}


def validate_eep(mapping: Path, spec: ArraySpec, expected_points: int) -> dict:
    operator = load_ffd_operator(mapping, spec)
    phi = operator["g_ephi"]
    theta = operator["g_etheta"]
    expected_shape = (expected_points, spec.element_count)
    if phi.shape != expected_shape or theta.shape != expected_shape:
        raise RuntimeError(f"EEP shape mismatch: {phi.shape}, {theta.shape}")
    if not all(np.isfinite(v.real).all() and np.isfinite(v.imag).all() for v in (phi, theta)):
        raise RuntimeError("EEP contains NaN or Inf")
    files = [Path(value) for value in operator["source_files"]]
    return {
        "mapping": str(mapping),
        "point_count": expected_points,
        "port_count": spec.element_count,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }


def audit_direct(run: Path, integer_mapping: Path, spec: ArraySpec, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = np.load(manifest_path.parent / "commands.npz", allow_pickle=False)["weights"]
    eep = load_ffd_operator(integer_mapping, spec)
    fields = []
    for item in manifest["commands"]:
        path = run / "direct_audit" / "raw_reports" / f"{item['command_id']}.csv"
        theta, phi, field = parse_direct_report(path)
        if theta.size != 181 or phi.size != 361 or field.size != 181 * 361:
            raise RuntimeError(f"Incomplete direct report {path.name}: {(theta.size, phi.size, field.size)}")
        fields.append(field)
    direct = np.stack(fields, axis=1)
    synthesis = eep["g_ephi"] @ weights.T
    error = direct - synthesis
    aggregate = float(np.vdot(error, error).real / np.vdot(direct, direct).real)
    per_command = []
    for index, item in enumerate(manifest["commands"]):
        denominator = float(np.vdot(direct[:, index], direct[:, index]).real)
        nmse = float(np.vdot(error[:, index], error[:, index]).real / denominator)
        amplitude = float(
            np.max(np.abs(np.abs(direct[:, index]) - np.abs(synthesis[:, index])))
            / max(float(np.max(np.abs(direct[:, index]))), 1e-300)
        )
        per_command.append({
            "command_id": item["command_id"],
            "complex_nmse_db": 10 * math.log10(max(nmse, 1e-300)),
            "amplitude_max_relative_error": amplitude,
        })
    aggregate_db = 10 * math.log10(max(aggregate, 1e-300))
    passed = bool(
        aggregate_db < -100.0
        and all(item["complex_nmse_db"] < -80.0 and item["amplitude_max_relative_error"] < 1e-6 for item in per_command)
    )
    return {"passed": passed, "aggregate_complex_nmse_db": aggregate_db, "per_command": per_command}


def load_direct_api_field(command_dir: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads((command_dir / "metadata.json").read_text(encoding="utf-8"))
    count = int(metadata["sample_count"])
    real = np.fromfile(command_dir / "real.float64le", dtype="<f8")
    imag = np.fromfile(command_dir / "imag.float64le", dtype="<f8")
    if real.size != count or imag.size != count or count != 181 * 361:
        raise RuntimeError(f"Incomplete direct API payload in {command_dir}")
    field = real + 1j * imag
    if not (np.isfinite(field.real).all() and np.isfinite(field.imag).all()):
        raise RuntimeError(f"Non-finite direct API payload in {command_dir}")
    return metadata, field


def audit_direct_api(run: Path, integer_mapping: Path, spec: ArraySpec, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = np.load(manifest_path.parent / "commands.npz", allow_pickle=False)["weights"]
    eep = load_ffd_operator(integer_mapping, spec)
    raw_fields = []
    metadata = []
    for item in manifest["commands"]:
        meta, field = load_direct_api_field(run / item["command_id"])
        metadata.append(meta)
        raw_fields.append(field)
    raw = np.stack(raw_fields, axis=1)
    synthesis = eep["g_ephi"] @ weights.T
    # AEDT may expose the two angular axes in either flattened order.  Choose
    # one grid ordering once for the whole audit using D00, then require all 16
    # commands to pass the fixed numerical thresholds.
    orderings = {
        "raw": raw,
        "theta_phi_transpose": raw.reshape(361, 181, 16).transpose(1, 0, 2).reshape(181 * 361, 16),
    }
    # The low-level AEDT API returns values in the data object's engineering
    # unit, while report CSV export labels/converts them to V.  The in-process
    # exporter predates unit capture, so reconcile only against AEDT's discrete
    # engineering prefixes; never fit an arbitrary continuous gain.
    unit_scales_to_v = (1.0, 1e-3, 1e-6, 1e-9, 1e3, 1e6, 1e9)
    reconciliation_nmse = {}
    for ordering_name, ordered in orderings.items():
        for scale in unit_scales_to_v:
            candidate0 = ordered[:, 0] * scale
            error0 = candidate0 - synthesis[:, 0]
            key = f"{ordering_name}|scale={scale:.0e}"
            reconciliation_nmse[key] = float(
                np.vdot(error0, error0).real / np.vdot(candidate0, candidate0).real
            )
    reconciliation = min(reconciliation_nmse, key=reconciliation_nmse.get)
    ordering, scale_text = reconciliation.split("|scale=")
    raw_to_v_scale = float(scale_text)
    direct = orderings[ordering] * raw_to_v_scale
    error = direct - synthesis
    aggregate = float(np.vdot(error, error).real / np.vdot(direct, direct).real)
    per_command = []
    for index, item in enumerate(manifest["commands"]):
        denominator = float(np.vdot(direct[:, index], direct[:, index]).real)
        nmse = float(np.vdot(error[:, index], error[:, index]).real / denominator)
        amplitude = float(
            np.max(np.abs(np.abs(direct[:, index]) - np.abs(synthesis[:, index])))
            / max(float(np.max(np.abs(direct[:, index]))), 1e-300)
        )
        per_command.append({
            "command_id": item["command_id"],
            "complex_nmse_db": 10 * math.log10(max(nmse, 1e-300)),
            "amplitude_max_relative_error": amplitude,
        })
    aggregate_db = 10 * math.log10(max(aggregate, 1e-300))
    passed = bool(
        aggregate_db < -100.0
        and all(item["complex_nmse_db"] < -80.0 and item["amplitude_max_relative_error"] < 1e-6 for item in per_command)
    )
    return {
        "passed": passed,
        "aggregate_complex_nmse_db": aggregate_db,
        "per_command": per_command,
        "raw_reports": str(run),
        "raw_reports_reused": False,
        "raw_format": "per-command little-endian float64 real/imag plus JSON angular metadata",
        "grid_ordering": ordering,
        "raw_to_v_scale": raw_to_v_scale,
        "unit_reconciliation_probe": "D00_discrete_AEDT_engineering_unit_and_grid_order_only",
        "unit_reconciliation_nmse": reconciliation_nmse,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--direct-api-run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    receipt = run / "GUI_EXPORT_RECEIPT.log"
    receipt_text = receipt.read_text(encoding="utf-8") if receipt.is_file() else ""
    if args.direct_api_run:
        required = ("S_PARAMETERS_EXPORTED", "EEP_EXPORTED integer", "EEP_EXPORTED offset")
        if any(marker not in receipt_text for marker in required):
            raise RuntimeError("GUI exporter did not complete the reusable S/EEP stages")
    elif "PASS raw content exported" not in receipt_text:
        raise RuntimeError("GUI exporter has no PASS receipt")
    spec = ArraySpec("a1x8", 1, 8, 0.015, 0.0068)
    touchstone = run / "a1x8_47ghz_readback.s8p"
    s_info = write_touchstone(json.loads((run / "s_parameters.json").read_text(encoding="utf-8")), touchstone, spec)
    parsed_frequency, parsed_s, parsed_reference = load_touchstone(touchstone, spec.element_count)
    if (
        parsed_s.shape != (s_info["sample_count"], 8, 8)
        or not np.any(np.isclose(parsed_frequency, 47e9))
        or not np.isclose(parsed_reference, 50.0)
    ):
        raise RuntimeError("Reconstructed S8P parse gate failed")
    integer_mapping = run / "eep" / "integer" / "eep.txt"
    eep = {
        "integer": validate_eep(integer_mapping, spec, 65341),
        "offset": validate_eep(run / "eep" / "offset" / "eep.txt", spec, 64800),
    }
    if args.direct_api_run:
        direct_api_run = args.direct_api_run.resolve()
        api_receipt = direct_api_run / "DIRECT_API_RECEIPT.log"
        if not api_receipt.is_file() or "PASS all direct fields exported" not in api_receipt.read_text(encoding="utf-8"):
            raise RuntimeError("Direct API exporter has no PASS receipt")
        direct = audit_direct_api(
            direct_api_run, integer_mapping, spec, DATASET_ROOT / "direct_audit" / "a1x8" / "manifest.json"
        )
    else:
        direct = audit_direct(run, integer_mapping, spec, DATASET_ROOT / "direct_audit" / "a1x8" / "manifest.json")
    report = {
        "passed": direct["passed"],
        "stage": "ordinary_gui_inprocess_full_gate",
        "array_id": "a1x8",
        "run": str(run),
        "solution": {"setup": "Setup2 : LastAdaptive", **s_info},
        "touchstone": {"path": str(touchstone), "bytes": touchstone.stat().st_size, "sha256": sha256_file(touchstone)},
        "eep": eep,
        "direct_audit": direct,
    }
    report_path = run / "FULL_GATE_REPORT.json"
    if report_path.exists():
        history = run / "stage_history"
        history.mkdir(exist_ok=True)
        index = 1
        while (history / f"FULL_GATE_REPORT_previous_{index}.json").exists():
            index += 1
        shutil.copy2(report_path, history / f"FULL_GATE_REPORT_previous_{index}.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("Direct-combination thresholds failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
