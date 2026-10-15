"""Read and audit an existing HFSS solution without solving or saving the project.

The caller must provide a recovery copy of both the AEDT project and its matching
``.aedtresults`` directory.  Outputs are written to a new run directory so the
known-good GUI solution and earlier receipts remain recoverable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np


DATASET_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = Path(__file__).resolve().parents[3]
V217_SRC = WORKTREE_ROOT / "algorithm_versions" / "v2.17" / "src"
sys.path.insert(0, str(V217_SRC))

from v217.cross_array import ArraySpec, load_ffd_operator, load_touchstone  # noqa: E402


FREQUENCY_HZ = 47e9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def restore_environment() -> dict[str, str]:
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
        os.environ[name] = os.path.expandvars(str(value))
        restored[name] = "machine"
    os.environ.setdefault("ANSYSEM_ROOT252", r"D:\Program Files\ANSYS Inc\v252\AnsysEM")
    return restored


def copy_eep_tree(mapping_paths: list[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing EEP output {output}")
    roots = {path.resolve().parent for path in mapping_paths}
    if len(roots) != 1:
        raise RuntimeError(f"Expected one EEP export directory, got {sorted(str(p) for p in roots)}")
    shutil.copytree(next(iter(roots)), output)
    mappings = sorted(output.glob("*.txt"))
    if len(mappings) != 1:
        raise RuntimeError(f"Expected one EEP mapping file in {output}, got {mappings}")
    return mappings[0]


def export_eep_native(hfss: Any, sphere: str, output: Path) -> Path:
    """Call the proven AEDT element-pattern exporter without PyAEDT's plot loader."""
    output.mkdir(parents=True, exist_ok=False)
    variations = hfss.available_variations.nominal_w_values_dict_w_dependent
    variation = " ".join(f"{key}='{value}'" for key, value in variations.items())
    hfss.oradfield.ExportElementPatternToFile(
        [
            "ExportFileName:=",
            str(output / "eep.ffd"),
            "SetupName:=",
            sphere,
            "IntrinsicVariationKey:=",
            f"Freq='{FREQUENCY_HZ}'",
            "DesignVariationKey:=",
            variation,
            "SolutionName:=",
            "Setup2 : LastAdaptive",
        ]
    )
    mapping = output / "eep.txt"
    if not mapping.is_file() or mapping.stat().st_size <= 0:
        raise RuntimeError(f"{sphere} native EEP export did not create a readable mapping")
    return mapping


def validate_eep(mapping: Path, spec: ArraySpec, expected_points: int) -> dict[str, Any]:
    operator = load_ffd_operator(mapping, spec)
    g_ephi = operator["g_ephi"]
    g_etheta = operator["g_etheta"]
    if g_ephi.shape != (expected_points, spec.element_count):
        raise RuntimeError(
            f"Unexpected EEP shape {g_ephi.shape}; expected {(expected_points, spec.element_count)}"
        )
    if not all(np.isfinite(value.real).all() and np.isfinite(value.imag).all() for value in (g_ephi, g_etheta)):
        raise RuntimeError("EEP contains NaN or Inf")
    source_files = [Path(value) for value in operator["source_files"]]
    records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    ]
    if len(records) != spec.element_count or any(item["bytes"] <= 0 for item in records):
        raise RuntimeError("EEP file count or size gate failed")
    return {
        "mapping": str(mapping),
        "mapping_bytes": mapping.stat().st_size,
        "mapping_sha256": sha256_file(mapping),
        "point_count": expected_points,
        "port_count": spec.element_count,
        "finite": True,
        "files": records,
    }


def frequency_values_hz(solution: Any) -> np.ndarray:
    values = np.asarray(solution.primary_sweep_values, dtype=np.float64)
    sweep = str(solution.primary_sweep)
    unit = str(getattr(solution, "units_sweeps", {}).get(sweep, "Hz"))
    scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(unit)
    if scale is None:
        raise RuntimeError(f"Unsupported solution frequency unit {unit!r}")
    return values * scale


def write_s4p_from_readback(hfss: Any, spec: ArraySpec, output: Path) -> dict[str, Any]:
    expressions = [
        f"S({receive},{source})"
        for source in range(1, spec.element_count + 1)
        for receive in range(1, spec.element_count + 1)
    ]
    solution = hfss.post.get_solution_data(
        expressions=expressions, setup_sweep_name="Setup2 : LastAdaptive"
    )
    if not solution:
        raise RuntimeError("Full S-parameter readback returned no solution data")
    frequencies = frequency_values_hz(solution)
    columns: list[np.ndarray] = []
    for expression in expressions:
        real = np.asarray(solution.data_real(expression), dtype=np.float64)
        imag = np.asarray(solution.data_imag(expression), dtype=np.float64)
        if real.shape != frequencies.shape or imag.shape != frequencies.shape:
            raise RuntimeError(
                f"{expression} shape mismatch: real={real.shape}, imag={imag.shape}, freq={frequencies.shape}"
            )
        columns.append(real + 1j * imag)
    matrix = np.stack(columns, axis=1)
    if matrix.shape != (frequencies.size, spec.element_count * spec.element_count):
        raise RuntimeError(f"Full S-parameter matrix is incomplete: {matrix.shape}")
    if not (np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all()):
        raise RuntimeError("Full S-parameter readback contains NaN or Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("! Reconstructed from HFSS Setup2 : LastAdaptive readable S traces\n")
        stream.write("# Hz S RI R 50\n")
        for frequency, row in zip(frequencies, matrix):
            tokens = [f"{frequency:.15g}"]
            for value in row:
                tokens.extend((f"{value.real:.15g}", f"{value.imag:.15g}"))
            stream.write(" ".join(tokens) + "\n")
    return {
        "method": f"all_{spec.element_count ** 2}_hfss_s_trace_readback",
        "expressions": expressions,
        "frequency_values_hz": frequencies.tolist(),
    }


def parse_direct_report(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2:
        raise RuntimeError(f"Direct report is empty: {path}")
    header = rows[0]
    values = np.asarray([[float(value) for value in row] for row in rows[1:]], dtype=np.float64)
    theta = values[:, 0]

    def columns(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        selected = [(index, name) for index, name in enumerate(header) if name.startswith(prefix)]
        if not selected:
            raise RuntimeError(f"No {prefix} columns in {path}")
        phi = np.asarray(
            [float(re.search(r"Phi='([-+0-9.eE]+)deg'", name).group(1)) for _, name in selected],
            dtype=np.float64,
        )
        units = {re.search(r"\[([^]]+)\]", name).group(1) for _, name in selected}
        scales = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "nV": 1e-9}
        if len(units) != 1 or next(iter(units)) not in scales:
            raise ValueError(f"Unsupported report units in {path}: {units}")
        scale = scales[next(iter(units))]
        return phi, values[:, [index for index, _ in selected]] * scale

    phi_real, real = columns("re(rEPhi)")
    phi_imag, imag = columns("im(rEPhi)")
    if not np.array_equal(phi_real, phi_imag):
        raise RuntimeError("Direct report real and imaginary grids differ")
    return theta, phi_real, (real + 1j * imag).reshape(-1)


def export_direct_audit(hfss: Any, spec: ArraySpec, integer_mapping: Path, manifest_path: Path,
                        output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = np.load(manifest_path.parent / "commands.npz", allow_pickle=False)["weights"]
    if weights.shape != (16, spec.element_count):
        raise RuntimeError(f"Unexpected direct-audit command shape {weights.shape}")
    eep = load_ffd_operator(integer_mapping, spec)
    raw = output / "raw_reports"
    raw.mkdir(parents=True, exist_ok=False)
    direct_fields: list[np.ndarray] = []
    for index, item in enumerate(manifest["commands"]):
        command_id = item["command_id"]
        sources = {
            port: (f"{values['power_w']:.15g}W", f"{values['phase_deg']:.15g}deg")
            for port, values in item["sources"].items()
        }
        if not hfss.edit_sources(sources, include_port_post_processing=True):
            raise RuntimeError(f"edit_sources failed for {command_id}")
        plot = f"V217_RECOVERY_{command_id}_{int(time.time())}"
        created = hfss.post.create_report(
            expressions=["re(rEPhi)", "im(rEPhi)"],
            setup_sweep_name="Setup2 : LastAdaptive",
            variations={"Freq": ["47GHz"], "Theta": ["All"], "Phi": ["All"]},
            primary_sweep_variable="Theta",
            secondary_sweep_variable="Phi",
            report_category="Far Fields",
            plot_type="Data Table",
            context="SphereInteger",
            plot_name=plot,
        )
        if not created:
            raise RuntimeError(f"Could not create direct report {plot}")
        try:
            exported = Path(hfss.post.export_report_to_file(str(raw), plot, ".csv", unique_file=False))
            destination = raw / f"{command_id}.csv"
            if exported.resolve() != destination.resolve():
                shutil.copy2(exported, destination)
            theta, phi, field = parse_direct_report(destination)
            if theta.size != 181 or phi.size != 361 or field.size != 181 * 361:
                raise RuntimeError(
                    f"{command_id} direct grid is {(theta.size, phi.size, field.size)}, expected (181,361,65341)"
                )
            if not (np.isfinite(field.real).all() and np.isfinite(field.imag).all()):
                raise RuntimeError(f"{command_id} direct report contains NaN or Inf")
            direct_fields.append(field)
        finally:
            hfss.post.delete_report(plot)

    direct = np.stack(direct_fields, axis=1)
    synthesis = eep["g_ephi"] @ weights.T
    error = direct - synthesis
    aggregate = float(np.vdot(error, error).real / np.vdot(direct, direct).real)
    aggregate_db = 10 * math.log10(max(aggregate, 1e-300))
    per_command = []
    for index, item in enumerate(manifest["commands"]):
        denominator = float(np.vdot(direct[:, index], direct[:, index]).real)
        nmse = float(np.vdot(error[:, index], error[:, index]).real / denominator)
        amplitude_relative = float(
            np.max(np.abs(np.abs(direct[:, index]) - np.abs(synthesis[:, index])))
            / max(float(np.max(np.abs(direct[:, index]))), 1e-300)
        )
        per_command.append(
            {
                "command_id": item["command_id"],
                "complex_nmse_db": 10 * math.log10(max(nmse, 1e-300)),
                "amplitude_max_relative_error": amplitude_relative,
            }
        )
    passed = bool(
        aggregate_db < -100.0
        and all(
            item["complex_nmse_db"] < -80.0
            and item["amplitude_max_relative_error"] < 1e-6
            for item in per_command
        )
    )
    report = {
        "passed": passed,
        "aggregate_complex_nmse_db": aggregate_db,
        "per_command": per_command,
        "thresholds": manifest["thresholds"],
        "not_for_training": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "AUDIT_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Direct-combination audit failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite recovery run {args.output}")
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    restored = restore_environment()
    spec = ArraySpec(args.array_id, args.rows, args.cols, 0.015, 0.0068)
    project_hash_before = sha256_file(args.project)
    desktop = hfss = None
    started = time.time()
    report: dict[str, Any]
    try:
        from pyaedt import Desktop, Hfss
        import pyaedt

        desktop = Desktop(
            version="2025.2", non_graphical=False, new_desktop=True, close_on_exit=True
        )
        hfss = Hfss(
            project=str(args.project), design="HFSSDesign1", version="2025.2",
            non_graphical=False, new_desktop=False, close_on_exit=True, remove_lock=False,
        )
        if set(hfss.excitations) != set(spec.port_names):
            raise RuntimeError(f"Port set mismatch: {list(hfss.excitations)}")

        solution = hfss.post.get_solution_data(
            expressions="S(1,1)", setup_sweep_name="Setup2 : LastAdaptive"
        )
        if not solution:
            raise RuntimeError("Setup2 : LastAdaptive did not return S(1,1)")
        real = np.asarray(solution.data_real(), dtype=np.float64)
        imag = np.asarray(solution.data_imag(), dtype=np.float64)
        if real.size == 0 or real.shape != imag.shape or not np.isfinite(real).all() or not np.isfinite(imag).all():
            raise RuntimeError("S(1,1) readback is empty, mismatched, or non-finite")
        frequencies = frequency_values_hz(solution)
        if frequencies.size != real.size or not np.isfinite(frequencies).all():
            raise RuntimeError("S(1,1) frequency axis is empty, mismatched, or non-finite")

        s4p = args.output / f"{args.array_id}_47ghz.s{spec.element_count}p"
        exported = hfss.export_touchstone(
            setup="Setup2", sweep="LastAdaptive", output_file=str(s4p)
        )
        native_touchstone_export = bool(exported and s4p.is_file() and s4p.stat().st_size > 0)
        if native_touchstone_export:
            touchstone_method = {"method": "hfss_export_touchstone", "native_api_passed": True}
        else:
            s4p = args.output / f"{args.array_id}_47ghz_readback.s{spec.element_count}p"
            touchstone_method = write_s4p_from_readback(hfss, spec, s4p)
            touchstone_method["native_api_passed"] = False
            touchstone_method["native_api_error"] = (
                "HFSS ExportNetworkData rejected the saved solution; used all readable S traces"
            )
        s_frequency, s_matrix, reference = load_touchstone(s4p, spec.element_count)
        if s_matrix.shape[1:] != (spec.element_count, spec.element_count):
            raise RuntimeError(f"Touchstone matrix is incomplete: {s_matrix.shape}")
        if not (np.isfinite(s_matrix.real).all() and np.isfinite(s_matrix.imag).all()):
            raise RuntimeError("Touchstone contains NaN or Inf")
        nearest = int(np.argmin(np.abs(s_frequency - FREQUENCY_HZ)))
        if abs(float(s_frequency[nearest]) - FREQUENCY_HZ) > 1.0:
            raise RuntimeError(f"Touchstone has no 47 GHz sample: {s_frequency.tolist()}")

        eep_reports: dict[str, Any] = {}
        mappings: dict[str, Path] = {}
        if not hfss.set_phase_center_per_port():
            raise RuntimeError("Could not set the per-port phase centers before EEP export")
        for sphere, folder, points in (
            ("SphereInteger", "integer", 181 * 361),
            ("SphereOffset", "offset", 180 * 360),
        ):
            mapping = export_eep_native(hfss, sphere, args.output / "eep" / folder)
            mappings[folder] = mapping
            eep_reports[folder] = {"sphere": sphere, **validate_eep(mapping, spec, points)}

        audit = export_direct_audit(
            hfss, spec, mappings["integer"], args.audit_manifest.resolve(),
            args.output / "direct_audit",
        )
        messages = list(hfss._desktop.GetMessages(hfss.project_name, hfss.design_name, 0))
        project_hash_after = sha256_file(args.project)
        if project_hash_after != project_hash_before:
            raise RuntimeError("Recovery project changed even though no save was requested")
        report = {
            "passed": True,
            "stage": "existing_solution_full_gate",
            "array_id": args.array_id,
            "project": str(args.project),
            "project_sha256_before": project_hash_before,
            "project_sha256_after": project_hash_after,
            "project_unchanged": True,
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "solution": {
                "setup": "Setup2 : LastAdaptive",
                "expression": "S(1,1)",
                "sample_count": int(real.size),
                "frequency_values": frequencies.tolist(),
                "finite": True,
            },
            "touchstone": {
                "path": str(s4p),
                "bytes": s4p.stat().st_size,
                "sha256": sha256_file(s4p),
                "frequency_count": int(len(s_frequency)),
                "port_count": spec.element_count,
                "reference_impedance_ohm": reference,
                "finite": True,
                "has_47ghz": True,
                **touchstone_method,
            },
            "eep": eep_reports,
            "direct_audit": audit,
            "aedt_messages_tail": messages[-20:],
            "elapsed_seconds": time.time() - started,
            "restored_environment": restored,
        }
    except Exception as exc:
        report = {
            "passed": False,
            "stage": "existing_solution_full_gate",
            "array_id": args.array_id,
            "project": str(args.project),
            "project_sha256_before": project_hash_before,
            "project_sha256_after": sha256_file(args.project),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "restored_environment": restored,
        }
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
