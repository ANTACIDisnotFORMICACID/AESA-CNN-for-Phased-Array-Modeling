"""Export 16 direct HFSS combinations and compare them with EEP synthesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.cross_array import ArraySpec, file_sha256, load_ffd_operator  # noqa: E402


def restore_environment() -> None:
    import winreg

    key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    for name in ("ANSYSEM_ROOT252", "ANSYSLMD_LICENSE_FILE"):
        if os.environ.get(name):
            continue
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, name)
                os.environ[name] = os.path.expandvars(str(value))
        except OSError:
            pass


def parse_report(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    values = np.asarray([[float(value) for value in row] for row in rows[1:]])
    theta = values[:, 0]

    def columns(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        selected = [(index, name) for index, name in enumerate(header) if name.startswith(prefix)]
        phi = np.asarray([float(re.search(r"Phi='([-+0-9.eE]+)deg'", name).group(1)) for _, name in selected])
        units = {re.search(r"\[([^]]+)\]", name).group(1) for _, name in selected}
        if len(units) != 1:
            raise ValueError(f"Mixed report units in {path}: {units}")
        scale = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[units.pop()]
        return phi, values[:, [index for index, _ in selected]] * scale

    phi_real, real = columns("re(rEPhi)")
    phi_imag, imag = columns("im(rEPhi)")
    if not np.array_equal(phi_real, phi_imag):
        raise ValueError("Real/imaginary report grids differ")
    return theta, phi_real, (real + 1j * imag).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--integer-mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphical", action="store_true")
    parser.add_argument(
        "--reuse-raw-reports",
        type=Path,
        help="Reuse a complete D00.csv..D15.csv directory and skip all AEDT calls.",
    )
    args = parser.parse_args()
    restore_environment()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spec = ArraySpec(manifest["array_id"], args.rows, args.cols, 0.015, 0.0068)
    eep = load_ffd_operator(args.integer_mapping, spec)
    weights = np.load(args.manifest.parent / "commands.npz", allow_pickle=False)["weights"]
    project_hash = file_sha256(args.project)
    raw = args.reuse_raw_reports.resolve() if args.reuse_raw_reports else args.output / "raw_reports"
    direct_fields = []
    if args.reuse_raw_reports:
        for item in manifest["commands"]:
            destination = raw / f"{item['command_id']}.csv"
            if not destination.is_file() or destination.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing reusable direct report {destination}")
            _, _, field = parse_report(destination)
            direct_fields.append(field)
    else:
        raw.mkdir(parents=True, exist_ok=True)
        from pyaedt import Desktop, Hfss

        desktop = hfss = None
        try:
            desktop = Desktop(version="2025.2", non_graphical=not args.graphical, new_desktop=True, close_on_exit=True)
            hfss = Hfss(project=str(args.project), design="HFSSDesign1", version="2025.2", non_graphical=not args.graphical, new_desktop=False, remove_lock=False)
            if set(hfss.excitations) != set(spec.port_names):
                raise RuntimeError(f"Port set mismatch: {list(hfss.excitations)}")
            for index, item in enumerate(manifest["commands"]):
                command_id = item["command_id"]
                sources = {
                    port: (f"{values['power_w']:.15g}W", f"{values['phase_deg']:.15g}deg")
                    for port, values in item["sources"].items()
                }
                if not hfss.edit_sources(sources, include_port_post_processing=True):
                    raise RuntimeError(f"edit_sources failed for {command_id}")
                plot = f"V217_DIRECT_{command_id}"
                created = hfss.post.create_report(
                    expressions=["re(rEPhi)", "im(rEPhi)"], setup_sweep_name="Setup2 : LastAdaptive",
                    variations={"Freq": ["47GHz"], "Theta": ["All"], "Phi": ["All"]},
                    primary_sweep_variable="Theta", secondary_sweep_variable="Phi",
                    report_category="Far Fields", plot_type="Data Table", context="SphereInteger", plot_name=plot,
                )
                if not created:
                    raise RuntimeError(f"Could not create direct report {plot}")
                exported = Path(hfss.post.export_report_to_file(str(raw), plot, ".csv", unique_file=False))
                destination = raw / f"{command_id}.csv"
                if exported != destination:
                    shutil.copy2(exported, destination)
                _, _, field = parse_report(destination)
                direct_fields.append(field)
                hfss.post.delete_report(plot)
        finally:
            if hfss is not None:
                hfss.close_project(save=False)
            if desktop is not None:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
    direct = np.stack(direct_fields, axis=1)
    synthesis = eep["g_ephi"] @ weights.T
    error = direct - synthesis
    aggregate = float(np.vdot(error, error).real / np.vdot(direct, direct).real)
    per_command = []
    for index, item in enumerate(manifest["commands"]):
        nmse = float(np.vdot(error[:, index], error[:, index]).real / np.vdot(direct[:, index], direct[:, index]).real)
        amplitude_relative = float(np.max(np.abs(np.abs(direct[:, index]) - np.abs(synthesis[:, index]))) / max(np.max(np.abs(direct[:, index])), 1e-300))
        per_command.append({"command_id": item["command_id"], "complex_nmse_db": 10 * math.log10(max(nmse, 1e-300)), "amplitude_max_relative_error": amplitude_relative})
    aggregate_db = 10 * math.log10(max(aggregate, 1e-300))
    passed = aggregate_db < -100 and all(item["complex_nmse_db"] < -80 and item["amplitude_max_relative_error"] < 1e-6 for item in per_command)
    report = {
        "passed": passed, "aggregate_complex_nmse_db": aggregate_db,
        "per_command": per_command, "project_unchanged": file_sha256(args.project) == project_hash,
        "raw_reports": str(raw),
        "raw_reports_reused": bool(args.reuse_raw_reports),
        "eep_phase_reference_to_global_applied": True,
        "not_for_training": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "AUDIT_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
