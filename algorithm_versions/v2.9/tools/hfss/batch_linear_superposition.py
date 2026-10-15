"""Batch v2.9 far-field synthesis from HFSS embedded element patterns.

This tool never opens or saves an AEDT project.  It combines the already
exported embedded element patterns using complex source coefficients and
derives RealizedGainTotal only from the A01 direct-HFSS calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PORTS = 16
EXPECTED_FREQ_HZ = 47.0e9
GAIN_FACTOR = 1.0 / 60.0
GRID_POINTS = 180 * 360
FLOAT_FMT = ".12g"


def parse_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError(f"FFD header is too short: {path}")
    theta_spec = lines[0].split()
    phi_spec = lines[1].split()
    theta = np.linspace(float(theta_spec[0]), float(theta_spec[1]), int(theta_spec[2]))
    phi = np.linspace(float(phi_spec[0]), float(phi_spec[1]), int(phi_spec[2]))
    freq_match = re.search(r"([0-9.eE+-]+)\s*$", lines[3])
    if not freq_match:
        raise ValueError(f"Cannot parse FFD frequency: {path}")
    freq_hz = float(freq_match.group(1))
    values = np.loadtxt(path, skiprows=4, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (theta.size * phi.size, 4):
        raise ValueError(f"Unexpected FFD shape {values.shape} for {path}")
    return theta, phi, values, freq_hz


def load_ffds(ffd_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fields = []
    hashes = []
    theta = phi = None
    frequency = None
    for port in range(1, EXPECTED_PORTS + 1):
        path = ffd_dir / f"{prefix}_{port}.ffd"
        if not path.is_file():
            raise FileNotFoundError(path)
        this_theta, this_phi, values, this_frequency = parse_grid(path)
        if theta is None:
            theta, phi, frequency = this_theta, this_phi, this_frequency
        elif not (np.array_equal(theta, this_theta) and np.array_equal(phi, this_phi)):
            raise ValueError(f"FFD grid mismatch at port {port}")
        elif not math.isclose(frequency, this_frequency, rel_tol=0, abs_tol=1.0):
            raise ValueError(f"FFD frequency mismatch at port {port}")
        fields.append(values[:, 0] + 1j * values[:, 1])
        fields.append(values[:, 2] + 1j * values[:, 3])
        hashes.append({"port": port, "file": str(path), "bytes": path.stat().st_size})
    # Split back into port x point arrays, retaining [re theta, im theta,
    # re phi, im phi] as the documented HFSS FFD convention.
    raw = np.asarray(fields).reshape(EXPECTED_PORTS, 2, -1)
    return raw[:, 0, :], raw[:, 1, :], {
        "theta_deg": theta.tolist(),
        "phi_deg": phi.tolist(),
        "frequency_hz": frequency,
        "files": hashes,
        "ffd_prefix": prefix,
    }


def load_sources(path: Path) -> tuple[str, list[dict[str, str]], np.ndarray, float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_PORTS:
        raise ValueError(f"{path} has {len(rows)} sources, expected {EXPECTED_PORTS}")
    command_ids = {row["command_id"] for row in rows}
    if len(command_ids) != 1:
        raise ValueError(f"Multiple command IDs in {path}: {command_ids}")
    rows.sort(key=lambda row: int(row["port_index"]))
    expected_ports = [f"{i}:1" for i in range(1, EXPECTED_PORTS + 1)]
    if [row["source_name"] for row in rows] != expected_ports:
        raise ValueError(f"Port order is not 1:1..16:1 in {path}")
    weights = np.asarray(
        [float(row["amplitude_coefficient"]) * np.exp(1j * np.deg2rad(float(row["phase_deg"]))) for row in rows],
        dtype=np.complex128,
    )
    total_power = float(sum(float(row["power_w"]) for row in rows))
    return rows[0]["command_id"], rows, weights, total_power


def write_complex(path: Path, theta: np.ndarray, phi: np.ndarray, et: np.ndarray, ep: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["theta_deg", "phi_deg", "rETheta_re_V", "rETheta_im_V", "rEPhi_re_V", "rEPhi_im_V"])
        for idx in range(theta.size * phi.size):
            writer.writerow([
                format(theta[idx // phi.size], FLOAT_FMT),
                format(phi[idx % phi.size], FLOAT_FMT),
                format(et[idx].real, FLOAT_FMT),
                format(et[idx].imag, FLOAT_FMT),
                format(ep[idx].real, FLOAT_FMT),
                format(ep[idx].imag, FLOAT_FMT),
            ])


def write_gain(path: Path, theta: np.ndarray, phi: np.ndarray, gain: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["theta_deg", "phi_deg", "RealizedGainTotal_linear", "RealizedGainTotal_dB"])
        for idx in range(theta.size * phi.size):
            db = 10.0 * math.log10(max(float(gain[idx]), np.finfo(float).tiny))
            writer.writerow([
                format(theta[idx // phi.size], FLOAT_FMT),
                format(phi[idx % phi.size], FLOAT_FMT),
                format(gain[idx], FLOAT_FMT),
                format(db, FLOAT_FMT),
            ])


def read_wide_report(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    headers = rows[0]
    data = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)
    theta = data[:, 0]
    groups: dict[str, np.ndarray] = {}
    for expression in ("re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)", "RealizedGainTotal"):
        indices = [idx for idx, header in enumerate(headers) if header.startswith(expression)]
        if len(indices) != 361:
            raise ValueError(f"Direct report group {expression} has {len(indices)} phi columns")
        groups[expression] = data[:, indices]
    return theta, groups, np.arange(-180.0, 181.0, 1.0)


def verify_a01(integer_et: np.ndarray, integer_ep: np.ndarray, integer_meta: dict[str, Any], report_path: Path,
               sources_path: Path) -> dict[str, Any]:
    command_id, _, weights, total_power = load_sources(sources_path)
    if command_id != "A01":
        raise ValueError("A01 verification was given a non-A01 source file")
    et = np.sum(weights[:, None] * integer_et, axis=0)
    ep = np.sum(weights[:, None] * integer_ep, axis=0)
    theta, groups, phi = read_wide_report(report_path)
    ntheta, nphi = len(theta), len(phi)
    et_direct = (groups["re(rETheta)"] + 1j * groups["im(rETheta)"]).reshape(-1)
    ep_direct = (groups["re(rEPhi)"] + 1j * groups["im(rEPhi)"]).reshape(-1)
    gain_direct = groups["RealizedGainTotal"].reshape(-1)
    gain_synth = GAIN_FACTOR * (np.abs(et) ** 2 + np.abs(ep) ** 2)
    complex_abs = np.concatenate([np.abs(et - et_direct), np.abs(ep - ep_direct)])
    phase_errors = np.concatenate([
        np.angle(np.exp(1j * (np.angle(et) - np.angle(et_direct))), deg=True),
        np.angle(np.exp(1j * (np.angle(ep) - np.angle(ep_direct))), deg=True),
    ])
    gain_db_error = 10.0 * np.log10(np.maximum(gain_synth, np.finfo(float).tiny)) - 10.0 * np.log10(np.maximum(gain_direct, np.finfo(float).tiny))
    ratio = gain_direct / np.maximum(np.abs(et_direct) ** 2 + np.abs(ep_direct) ** 2, np.finfo(float).tiny)
    result = {
        "passed": bool(np.max(complex_abs) < 1e-9 and np.max(np.abs(gain_db_error)) < 1e-9),
        "command_id": "A01",
        "direct_report": str(report_path),
        "source_csv": str(sources_path),
        "grid": {"theta_count": ntheta, "phi_count": nphi, "theta_start_deg": float(theta[0]), "theta_end_deg": float(theta[-1]), "phi_start_deg": float(phi[0]), "phi_end_deg": float(phi[-1])},
        "total_power_w": total_power,
        "complex_abs_error_max_V": float(np.max(complex_abs)),
        "complex_abs_error_rmse_V": float(np.sqrt(np.mean(complex_abs ** 2))),
        "phase_error_abs_p95_deg": float(np.percentile(np.abs(phase_errors), 95)),
        "realized_gain_factor_observed_min": float(np.min(ratio)),
        "realized_gain_factor_observed_max": float(np.max(ratio)),
        "realized_gain_factor_used": GAIN_FACTOR,
        "realized_gain_db_error_max": float(np.max(np.abs(gain_db_error))),
        "source_ffd_grid": integer_meta,
    }
    return result


def discover_commands(commands_dir: Path, priority: str, summary_path: Path | None) -> list[Path]:
    paths = sorted(commands_dir.glob("cmd_*_sources.csv"))
    if priority == "all" or summary_path is None:
        return paths
    allowed: set[str] = set()
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["priority"] == priority:
                allowed.add(row["command_id"])
    selected = []
    for path in paths:
        command_id, _, _, _ = load_sources(path)
        if command_id in allowed:
            selected.append(path)
    return selected


def synthesize(command_path: Path, et_ports: np.ndarray, ep_ports: np.ndarray, ffd_meta: dict[str, Any], out_root: Path) -> dict[str, Any]:
    command_id, rows, weights, total_power = load_sources(command_path)
    if not math.isclose(total_power, 1.0, rel_tol=0, abs_tol=2e-5):
        raise ValueError(f"{command_id} total power is {total_power}, expected 1 W")
    et = np.sum(weights[:, None] * et_ports, axis=0)
    ep = np.sum(weights[:, None] * ep_ports, axis=0)
    gain = GAIN_FACTOR * (np.abs(et) ** 2 + np.abs(ep) ** 2)
    if not (np.isfinite(et.real).all() and np.isfinite(et.imag).all() and np.isfinite(ep.real).all() and np.isfinite(ep.imag).all() and np.isfinite(gain).all()):
        raise ValueError(f"{command_id} synthesis contains NaN or Inf")
    theta = np.asarray(ffd_meta["theta_deg"], dtype=np.float64)
    phi = np.asarray(ffd_meta["phi_deg"], dtype=np.float64)
    complex_path = out_root / "exports" / "complex_farfield" / f"{command_id}_complex_farfield.csv"
    gain_path = out_root / "exports" / "realized_gain" / f"{command_id}_realized_gain.csv"
    write_complex(complex_path, theta, phi, et, ep)
    write_gain(gain_path, theta, phi, gain)
    if complex_path.stat().st_size == 0 or gain_path.stat().st_size == 0:
        raise IOError(f"{command_id} output is empty")
    meta = {
        "schema_version": 1,
        "command_id": command_id,
        "source_csv": str(command_path),
        "frequency_hz": ffd_meta["frequency_hz"],
        "grid": {"theta_start_deg": float(theta[0]), "theta_end_deg": float(theta[-1]), "theta_step_deg": float(theta[1] - theta[0]), "theta_count": int(theta.size), "phi_start_deg": float(phi[0]), "phi_end_deg": float(phi[-1]), "phi_step_deg": float(phi[1] - phi[0]), "phi_count": int(phi.size), "point_count": int(theta.size * phi.size)},
        "total_power_w": total_power,
        "weight_convention": "complex_weight_n = amplitude_coefficient_n * exp(j * phase_deg_n); power_w is not used as a complex amplitude",
        "source_ffd": ffd_meta["files"],
        "source_ffd_prefix": ffd_meta["ffd_prefix"],
        "realized_gain_calibration": {"formula": "RealizedGainTotal_linear = (|rETheta|^2 + |rEPhi|^2) / 60", "factor": GAIN_FACTOR, "calibrated_against": "A01 direct HFSS report on integer grid", "assumption": "same 47 GHz, same 1 W source normalization and same embedded FFD export convention"},
        "data_origin": "HFSS embedded-pattern linear superposition; not 24 direct HFSS reports",
        "direct_HFSS_or_linear_superposition": "linear_superposition",
        "outputs": {"complex_farfield": {"path": str(complex_path), "bytes": complex_path.stat().st_size}, "realized_gain": {"path": str(gain_path), "bytes": gain_path.stat().st_size}},
    }
    meta_path = out_root / "exports" / "metadata" / f"{command_id}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"command_id": command_id, "source_csv": str(command_path), "total_power_w": total_power, "point_count": int(theta.size * phi.size), "finite_values": True, "grid_unique_complete": True, "complex_path": str(complex_path), "gain_path": str(gain_path), "metadata_path": str(meta_path), "complex_bytes": complex_path.stat().st_size, "gain_bytes": gain_path.stat().st_size, "gain_min_linear": float(np.min(gain)), "gain_max_linear": float(np.max(gain))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v29-root", type=Path, required=True)
    parser.add_argument("--priority", choices=("core", "extended", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.v29_root.resolve()
    validation_root = root / "data" / "external_next" / "hfss_direct_validation"
    commands_dir = validation_root / "commands" / "sources"
    summary = validation_root / "commands" / "selected_commands_summary.csv"
    ffd_dir = root.parent / "v2.8" / "HFSS" / "exports" / "hfss_47ghz"
    out_root = validation_root
    out_root.joinpath("exports", "complex_farfield").mkdir(parents=True, exist_ok=True)
    out_root.joinpath("exports", "realized_gain").mkdir(parents=True, exist_ok=True)
    if not args.force and any(out_root.joinpath("exports", "complex_farfield").glob("*.csv")):
        raise RuntimeError("Outputs already exist; use --force only after checking the output directory")
    shifted_et, shifted_ep, shifted_meta = load_ffds(ffd_dir, "exportelement_new")
    integer_et, integer_ep, integer_meta = load_ffds(ffd_dir, "exportelement")
    if shifted_meta["frequency_hz"] != EXPECTED_FREQ_HZ or integer_meta["frequency_hz"] != EXPECTED_FREQ_HZ:
        raise ValueError("Expected 47 GHz FFDs")
    direct_report = validation_root / "automation_smoke" / "A01" / "V29_API_SMOKE_cmd_A01_uniform_0_sources.csv"
    a01_sources = commands_dir / "cmd_A01_uniform_0_sources.csv"
    calibration = verify_a01(integer_et, integer_ep, integer_meta, direct_report, a01_sources)
    if not calibration["passed"]:
        raise RuntimeError("A01 direct-HFSS calibration failed; no batch outputs generated")
    selected = discover_commands(commands_dir, args.priority, summary)
    records = []
    for path in selected:
        records.append(synthesize(path, shifted_et, shifted_ep, shifted_meta, out_root))
    # A01/A02 common phase invariant on the shifted grid, when both are in this run.
    invariance = None
    ids = {record["command_id"] for record in records}
    if {"A01", "A02"}.issubset(ids):
        def read_complex(path: Path) -> np.ndarray:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))[1:]
            values = np.asarray([[float(v) for v in row] for row in rows])
            return (values[:, 2] + 1j * values[:, 3]), (values[:, 4] + 1j * values[:, 5])
        a01_et, a01_ep = read_complex(out_root / "exports" / "complex_farfield" / "A01_complex_farfield.csv")
        a02_et, a02_ep = read_complex(out_root / "exports" / "complex_farfield" / "A02_complex_farfield.csv")
        invariance = {"passed": bool(np.max(np.abs(a02_et - 1j * a01_et)) < 1e-9 and np.max(np.abs(a02_ep - 1j * a01_ep)) < 1e-9), "complex_90deg_error_max_V": float(max(np.max(np.abs(a02_et - 1j * a01_et)), np.max(np.abs(a02_ep - 1j * a01_ep))))}
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "priority": args.priority, "calibration": calibration, "a01_a02_common_phase_invariance": invariance, "ffd_grid": shifted_meta, "records": records, "data_boundary": "All records are linear superpositions of HFSS embedded element patterns; they are not direct 24-state HFSS reports."}
    report_path = validation_root / "exports" / "batch_export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = validation_root / "exports" / "batch_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "records": records, "report": str(report_path), "calibration": calibration}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "priority": args.priority, "count": len(records), "calibration": calibration, "invariance": invariance, "report": str(report_path), "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
