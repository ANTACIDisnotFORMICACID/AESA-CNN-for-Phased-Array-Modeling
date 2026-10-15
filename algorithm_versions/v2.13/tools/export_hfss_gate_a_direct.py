"""Export direct HFSS complex rEPhi reports for the frozen Gate-A commands.

This uses HFSS Edit Sources and the HFSS Far Fields report engine.  It does not
form combined fields in Python.  The compact NPZ additionally stores the
single-port FFD basis solely so a later audit can compare the two code paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import traceback
from pathlib import Path

import numpy as np


def restore_machine_environment() -> dict[str, str]:
    import winreg

    restored: dict[str, str] = {}
    key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    for name in ("ANSYSEM_ROOT252", "ANSYSLMD_LICENSE_FILE"):
        if os.environ.get(name):
            continue
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            os.environ[name] = os.path.expandvars(str(value))
            restored[name] = "machine"
    os.environ.setdefault("ANSYSEM_ROOT252", r"D:\Program Files\ANSYS Inc\v252\AnsysEM")
    return restored


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_csv(path: Path) -> tuple[dict[str, tuple[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["port_index"]))
    if len(rows) != 16 or [row["source_name"] for row in rows] != [f"{i}:1" for i in range(1, 17)]:
        raise ValueError(f"invalid 16-port source CSV: {path}")
    source = {
        row["source_name"]: (f'{float(row["power_w"]):.12g}W', f'{float(row["phase_deg"]):.12g}deg')
        for row in rows
    }
    weight = np.asarray(
        [float(row["amplitude_coefficient"]) * np.exp(1j * np.deg2rad(float(row["phase_deg"]))) for row in rows],
        dtype=np.complex128,
    )
    if not math.isclose(float(np.vdot(weight, weight).real), 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError(f"source power is not 1 W: {path}")
    return source, weight


def parse_ffd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    theta_spec = lines[0].split()
    phi_spec = lines[1].split()
    theta = np.linspace(float(theta_spec[0]), float(theta_spec[1]), int(theta_spec[2]))
    phi = np.linspace(float(phi_spec[0]), float(phi_spec[1]), int(phi_spec[2]))
    frequency = float(re.search(r"([0-9.eE+-]+)\s*$", lines[3]).group(1))
    values = np.loadtxt(path, skiprows=4, dtype=np.float64)
    if values.shape != (len(theta) * len(phi), 4):
        raise ValueError(f"unexpected FFD shape {values.shape}: {path}")
    return theta, phi, values[:, 2] + 1j * values[:, 3], frequency


def load_single_port_ephi(ffd_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fields = []
    theta0 = phi0 = None
    frequency0 = None
    for port in range(1, 17):
        theta, phi, ephi, frequency = parse_ffd(ffd_dir / f"{prefix}_{port}.ffd")
        if theta0 is None:
            theta0, phi0, frequency0 = theta, phi, frequency
        elif not (np.array_equal(theta0, theta) and np.array_equal(phi0, phi)):
            raise ValueError(f"FFD grid mismatch at port {port}")
        fields.append(ephi)
    return theta0, phi0, np.stack(fields, axis=1), float(frequency0)


def parse_direct_report(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    headers = rows[0]
    data = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)
    theta = data[:, 0]

    def group(expression: str) -> tuple[np.ndarray, np.ndarray, str]:
        selected = [(i, h) for i, h in enumerate(headers) if h.startswith(expression)]
        if not selected:
            raise ValueError(f"{path} has no {expression} columns")
        phi = np.asarray([float(re.search(r"Phi='([-+0-9.eE]+)deg'", h).group(1)) for _, h in selected])
        units = {re.search(r"\[([^]]+)\]", h).group(1) for _, h in selected}
        if len(units) != 1:
            raise ValueError(f"mixed units in {expression}: {units}")
        unit = units.pop()
        scale = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}.get(unit)
        if scale is None:
            raise ValueError(f"unsupported electric-field unit {unit}: {path}")
        return phi, data[:, [i for i, _ in selected]] * scale, unit

    phi_re, real, unit_re = group("re(rEPhi)")
    phi_im, imag, unit_im = group("im(rEPhi)")
    if not np.array_equal(phi_re, phi_im) or unit_re != unit_im:
        raise ValueError(f"real/imag phi columns differ: {path}")
    return theta, phi_re, (real + 1j * imag).reshape(-1), unit_re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ffd-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--aedt-export-temp", required=True, type=Path)
    parser.add_argument("--reuse-h00-report", type=Path)
    parser.add_argument("--design", default="HFSSDesign1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--package-existing", action="store_true")
    parser.add_argument("--existing-report-pyaedt-version", default="0.9.6")
    args = parser.parse_args()
    raw_dir = args.output_root / "raw_direct_reports"
    prepared_dir = args.output_root / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    args.aedt_export_temp.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    project_hash_before = sha256(args.project)
    restored = restore_machine_environment()
    desktop = hfss = None
    records: list[dict[str, object]] = []
    weights: list[np.ndarray] = []
    direct_fields: list[np.ndarray] = []
    report: dict[str, object]
    try:
        pyaedt_version = (
            f"{args.existing_report_pyaedt_version} (existing direct reports; packaging-only run)"
            if args.package_existing else "unknown"
        )
        if not args.package_existing:
            import pyaedt
            from pyaedt import Desktop, Hfss

            pyaedt_version = getattr(pyaedt, "__version__", "unknown")
            desktop = Desktop(version="2025.2", non_graphical=True, new_desktop=True, close_on_exit=True)
            hfss = Hfss(
                project=str(args.project), design=args.design, version="2025.2",
                non_graphical=True, new_desktop=False, close_on_exit=True, remove_lock=False,
            )
            expected = [f"{i}:1" for i in range(1, 17)]
            if set(hfss.excitations) != set(expected):
                raise RuntimeError(f"HFSS source mismatch: {list(hfss.excitations)}")
        for item in manifest["commands"]:
            command_id = item["command_id"]
            source_csv = Path(item["source_csv"])
            source, weight = read_source_csv(source_csv)
            destination = raw_dir / f"{command_id}_hfss_direct.csv"
            reused = command_id == "H00" and args.reuse_h00_report is not None
            if destination.exists() and (not args.force or args.package_existing):
                pass
            elif args.package_existing:
                raise FileNotFoundError(f"package-existing report is missing: {destination}")
            elif reused:
                shutil.copy2(args.reuse_h00_report, destination)
            else:
                if not hfss.edit_sources(source, include_port_post_processing=True):
                    raise RuntimeError(f"HFSS edit_sources failed for {command_id}")
                plot_name = f"V213_GATEA_{command_id}"
                created = hfss.post.create_report(
                    expressions=["re(rEPhi)", "im(rEPhi)"],
                    setup_sweep_name=manifest["setup"],
                    variations={"Freq": ["47GHz"], "Theta": ["All"], "Phi": ["All"]},
                    primary_sweep_variable="Theta", secondary_sweep_variable="Phi",
                    report_category="Far Fields", plot_type="Data Table",
                    context=manifest["far_field_setup"], plot_name=plot_name,
                )
                if not created:
                    raise RuntimeError(f"HFSS create_report failed for {command_id}")
                exported = Path(hfss.post.export_report_to_file(str(args.aedt_export_temp), plot_name, ".csv", unique_file=False))
                if not exported.is_file() or exported.stat().st_size == 0:
                    raise RuntimeError(f"HFSS export is missing for {command_id}: {exported}")
                shutil.copy2(exported, destination)
                try:
                    hfss.post.delete_report(plot_name)
                except Exception:
                    pass
            theta, phi, ephi, source_unit = parse_direct_report(destination)
            weights.append(weight)
            direct_fields.append(ephi)
            records.append(
                {
                    "command_id": command_id,
                    "family": item["family"],
                    "provenance": "reused_prior_direct_HFSS_report" if reused else "new_direct_HFSS_report_via_EditSources",
                    "not_python_superposition": True,
                    "raw_report": str(destination.resolve()),
                    "bytes": destination.stat().st_size,
                    "theta_count": len(theta),
                    "phi_count": len(phi),
                    "point_count": int(len(ephi)),
                    "source_report_unit": source_unit,
                    "stored_complex_unit": "V",
                }
            )
        theta_ffd, phi_ffd, g_ports, frequency = load_single_port_ephi(args.ffd_dir, "exportelement")
        if not (np.array_equal(theta, theta_ffd) and np.array_equal(phi, phi_ffd)):
            raise RuntimeError("direct HFSS report grid and integer FFD basis grid differ")
        command_matrix = np.stack(weights)
        direct = np.stack(direct_fields, axis=1)
        synth = g_ports @ command_matrix.T
        error = direct - synth
        relative = float(np.vdot(error, error).real / np.vdot(direct, direct).real)
        per_command = []
        for index, item in enumerate(records):
            value = float(np.vdot(error[:, index], error[:, index]).real / np.vdot(direct[:, index], direct[:, index]).real)
            per_command.append({"command_id": item["command_id"], "complex_nmse": value, "complex_nmse_db": 10.0 * math.log10(max(value, 1e-300))})
        theta_grid = np.repeat(theta, len(phi))
        phi_grid = np.tile(phi, len(theta))
        theta_rad = np.deg2rad(theta_grid)
        phi_rad = np.deg2rad(phi_grid)
        direction_hfss = np.stack(
            [np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad), np.cos(theta_rad)], axis=1
        )
        np.savez_compressed(
            prepared_dir / "gate_a_hfss_direct_complex.npz",
            theta_deg=theta_grid,
            phi_deg=phi_grid,
            direction_hfss=direction_hfss,
            commands=command_matrix,
            ephi_complex=direct,
            g_single_port_ephi=g_ports,
            frequency_hz=np.asarray(frequency),
        )
        report = {
            "schema": "v213-hfss-gate-a-direct-export-1",
            "passed": True,
            "pyaedt_version": pyaedt_version,
            "aedt_version": "2025.2",
            "project": str(args.project.resolve()),
            "project_sha256_before": project_hash_before,
            "project_sha256_after": sha256(args.project),
            "project_unchanged_on_disk": project_hash_before == sha256(args.project),
            "restored_environment": restored,
            "data_boundary": "combined fields are direct HFSS report-engine outputs after Edit Sources; comparison fields alone are Python combinations of stored single-port FFDs",
            "command_count": len(records),
            "grid": {"theta_count": len(theta), "phi_count": len(phi), "point_count": int(len(direct))},
            "aggregate_complex_nmse": relative,
            "aggregate_complex_nmse_db": 10.0 * math.log10(max(relative, 1e-300)),
            "commands": records,
            "per_command": per_command,
            "prepared_npz": str((prepared_dir / "gate_a_hfss_direct_complex.npz").resolve()),
            "external_test_used": False,
            "not_for_training": True,
        }
    except Exception as exc:
        report = {
            "schema": "v213-hfss-gate-a-direct-export-1", "passed": False,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "project": str(args.project.resolve()), "project_sha256_before": project_hash_before,
        }
        raise
    finally:
        (args.output_root / "EXPORT_SUMMARY.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if hfss is not None:
            try:
                hfss.close_project(save=False)
            except Exception:
                pass
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
            except Exception:
                pass
    print(json.dumps({"passed": report["passed"], "command_count": report.get("command_count"), "aggregate_complex_nmse_db": report.get("aggregate_complex_nmse_db")}, indent=2))


if __name__ == "__main__":
    main()
