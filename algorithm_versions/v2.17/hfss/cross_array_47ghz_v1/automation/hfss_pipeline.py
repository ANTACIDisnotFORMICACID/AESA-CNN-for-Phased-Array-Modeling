"""Recoverable HFSS build/solve/export stages for the V2.17 array campaign.

Run through run_hfss_stage.ps1 so AEDT only sees the ASCII W: path.  Each
stage writes its own JSON report and never removes successful earlier output.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any


AEDT_ROOT = Path(r"D:\Program Files\ANSYS Inc\v252\AnsysEM")
SOURCE_SHA256 = "D647839C759D84C58D9A0251EFB814C29790F0D5D03287462284B3A2B683D350"
FREQUENCY_HZ = 47e9
ROW_PITCH_MM = 15.0
COL_PITCH_MM = 6.8
LAMBDA_MM = 299_792_458.0 / FREQUENCY_HZ * 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def identity() -> str:
    try:
        return subprocess.check_output(["whoami"], text=True, encoding="utf-8", errors="replace").strip()
    except Exception:
        return getpass.getuser()


def require_session() -> str:
    value = identity()
    if not value.lower().endswith(r"\asus"):
        raise PermissionError(f"HFSS must run in the interactive FORMIC-ACID\\ASUS session; got {value!r}")
    return value


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
    os.environ.setdefault("ANSYSEM_ROOT252", str(AEDT_ROOT))
    return restored


def assigned_objects(boundary: Any, modeler: Any) -> list[str]:
    raw = boundary.props.get("Objects", [])
    raw = [raw] if isinstance(raw, str) else raw
    result = []
    for item in raw:
        text = str(item)
        if text.isdigit():
            match = next((str(name) for name in modeler.object_names if int(modeler[name].id) == int(text)), None)
            if match is None:
                raise RuntimeError(f"Unknown object id {text} on boundary {boundary.name}")
            result.append(match)
        else:
            result.append(text)
    return result


def open_hfss(project: Path, graphical: bool):
    from pyaedt import Desktop, Hfss

    desktop = Desktop(version="2025.2", non_graphical=not graphical, new_desktop=True, close_on_exit=True)
    hfss = Hfss(
        project=str(project), design="HFSSDesign1", version="2025.2",
        non_graphical=not graphical, new_desktop=False, remove_lock=False,
    )
    return desktop, hfss


def close_hfss(desktop: Any, hfss: Any, save: bool) -> None:
    if hfss is not None:
        try:
            hfss.close_project(save=save)
        except Exception:
            pass
    if desktop is not None:
        try:
            desktop.release_desktop(close_projects=True, close_on_exit=True)
        except Exception:
            pass


def _port_records(hfss: Any, rows: int, cols: int) -> tuple[list[dict[str, object]], float]:
    boundaries = {str(item.name): item for item in hfss.boundaries}
    records = []
    for mode in hfss.excitations:
        port = str(mode).split(":")[0]
        boundary = boundaries.get(port)
        if boundary is None or str(boundary.type).lower() != "wave port":
            raise RuntimeError(f"Missing Wave Port boundary {port!r}")
        objects = assigned_objects(boundary, hfss.modeler)
        if len(objects) != 1:
            raise RuntimeError(f"Port {port} does not map to one sheet: {objects}")
        box = [float(value) for value in hfss.modeler[objects[0]].bounding_box]
        center = [(box[index] + box[index + 3]) / 2 for index in range(3)]
        records.append({"port": port, "mode": str(mode), "sheet": objects[0], "center_mm": center})
    if len(records) != rows * cols:
        raise RuntimeError(f"Expected {rows * cols} ports, got {len(records)}")
    y_values = sorted({round(float(item["center_mm"][1]), 6) for item in records})
    z_values = sorted({round(float(item["center_mm"][2]), 6) for item in records})
    if len(y_values) != cols or len(z_values) != rows:
        raise RuntimeError(f"Port centers are not a {rows}x{cols} grid: y={y_values}, z={z_values}")
    expected_y = [y_values[0] + index * COL_PITCH_MM for index in range(cols)]
    expected_z = [z_values[0] + index * ROW_PITCH_MM for index in range(rows)]
    maximum_error = 0.0
    for item in records:
        y = float(item["center_mm"][1])
        z = float(item["center_mm"][2])
        col = min(range(cols), key=lambda index: abs(expected_y[index] - y))
        row = min(range(rows), key=lambda index: abs(expected_z[index] - z))
        item["row"] = row
        item["col"] = col
        maximum_error = max(maximum_error, abs(expected_y[col] - y), abs(expected_z[row] - z))
    records.sort(key=lambda item: (int(item["row"]), int(item["col"])))
    expected_ports = [str(index) for index in range(1, rows * cols + 1)]
    actual_ports = [str(item["port"]) for item in records]
    if actual_ports != expected_ports:
        raise RuntimeError(f"Ports are not row-major: expected={expected_ports}, actual={actual_ports}")
    if maximum_error >= 0.01:
        raise RuntimeError(f"Maximum port position error {maximum_error} mm is not <0.01 mm")
    return records, maximum_error


def _radiation_record(hfss: Any) -> dict[str, object]:
    radiation = [item for item in hfss.boundaries if str(item.type).lower() == "radiation"]
    if len(radiation) != 1:
        raise RuntimeError(f"Expected one Radiation boundary, got {[item.name for item in radiation]}")
    objects = assigned_objects(radiation[0], hfss.modeler)
    if len(objects) != 1:
        raise RuntimeError(f"Radiation boundary does not map to one region: {objects}")
    region = hfss.modeler[objects[0]]
    if len(region.faces) != 6:
        raise RuntimeError(f"Radiation region is not a six-face box: {objects[0]}")
    region_box = np_box = [float(value) for value in region.bounding_box]
    array_boxes = [
        [float(value) for value in hfss.modeler[name].bounding_box]
        for name in hfss.modeler.model_objects if str(name) != objects[0]
    ]
    array_box = [
        min(value[0] for value in array_boxes), min(value[1] for value in array_boxes), min(value[2] for value in array_boxes),
        max(value[3] for value in array_boxes), max(value[4] for value in array_boxes), max(value[5] for value in array_boxes),
    ]
    margins = [
        array_box[0] - region_box[0], array_box[1] - region_box[1], array_box[2] - region_box[2],
        region_box[3] - array_box[3], region_box[4] - array_box[4], region_box[5] - array_box[5],
    ]
    if min(margins) + 0.05 * LAMBDA_MM < LAMBDA_MM / 4:
        raise RuntimeError(f"Radiation margin below lambda/4: {margins}")
    return {"boundary": str(radiation[0].name), "object": objects[0], "array_box_mm": array_box, "region_box_mm": np_box, "margins_mm": margins}


def _sphere_records(hfss: Any) -> list[dict[str, object]]:
    expected = {
        "SphereInteger": ("0deg", "180deg", "1deg", "-180deg", "180deg", "1deg"),
        "SphereOffset": ("0.5deg", "179.5deg", "1deg", "-179.5deg", "179.5deg", "1deg"),
    }
    found = {str(item.name): item for item in hfss.field_setups}
    records = []
    keys = ("ThetaStart", "ThetaStop", "ThetaStep", "PhiStart", "PhiStop", "PhiStep")
    for name, wanted in expected.items():
        if name not in found:
            raise RuntimeError(f"Missing far-field sphere {name}")
        actual = tuple(str(found[name].props.get(key, "")).replace(" ", "").lower() for key in keys)
        if actual != tuple(value.lower() for value in wanted):
            raise RuntimeError(f"Sphere {name} grid mismatch: {dict(zip(keys, actual))}")
        records.append({"name": name, "sampling": dict(zip(keys, actual))})
    return records


def _validate_design(hfss: Any, port_count: int) -> dict[str, object]:
    """Normalize PyAEDT 0.9.6's ``(messages, passed)`` validation result."""
    result = hfss.validate_full_design(ports=port_count)
    if isinstance(result, tuple) and len(result) == 2:
        messages, passed = result
    else:
        messages, passed = [], bool(result)
    messages = [str(item) for item in messages]
    if not bool(passed):
        errors = [item for item in messages if "[error]" in item.lower()]
        raise RuntimeError(
            "HFSS Design Validation failed: " + (" | ".join(errors) if errors else str(messages))
        )
    return {"passed": True, "messages": messages}


def _unite_touching_same_material(hfss: Any) -> list[list[str]]:
    """Union only same-material solids with coincident faces and nonzero overlap."""
    modeler = hfss.modeler
    names = [str(name) for name in modeler.model_objects]
    objects = []
    for name in names:
        obj = modeler[name]
        if str(obj.object_type).lower() != "solid":
            continue
        material = str(obj.material_name).strip('"').lower()
        if material not in {"copper", "rogers ro4350 (tm)"}:
            continue
        objects.append((name, material, [float(value) for value in obj.bounding_box]))
    groups: list[list[str]] = []
    used: set[str] = set()
    tolerance = 1e-5
    for name, material, box in objects:
        if name in used:
            continue
        group = [name]
        for other_name, other_material, other in objects:
            if other_name == name or other_name in used or other_material != material:
                continue
            touch_y = abs(box[4] - other[1]) < tolerance or abs(other[4] - box[1]) < tolerance
            overlap_x = min(box[3], other[3]) - max(box[0], other[0]) > tolerance
            overlap_z = min(box[5], other[5]) - max(box[2], other[2]) > tolerance
            if touch_y and overlap_x and overlap_z:
                group.append(other_name)
        if len(group) > 1:
            if not modeler.unite(group, purge=True):
                raise RuntimeError(f"Failed to unite touching same-material objects {group}")
            groups.append(group)
            used.update(group)
    return groups


def build(source: Path, project: Path, rows: int, cols: int, graphical: bool) -> dict[str, object]:
    if (rows, cols) not in {(1, 4), (2, 4), (1, 8)}:
        raise ValueError("Build stage supports a1x4, a2x4 and a1x8; a4x4 reuses frozen exports")
    if sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("Frozen 1x4 source fingerprint changed")
    if project.exists():
        raise FileExistsError(f"Refusing to overwrite existing project {project}")
    project.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, project)
    desktop = hfss = None
    try:
        desktop, hfss = open_hfss(project, graphical)
        radiation = [item for item in hfss.boundaries if str(item.type).lower() == "radiation"]
        if len(radiation) != 1:
            raise RuntimeError("Frozen source no longer has exactly one Radiation boundary")
        old_regions = assigned_objects(radiation[0], hfss.modeler)
        duplicate_objects = [str(name) for name in hfss.modeler.model_objects if str(name) not in old_regions]
        if "NF_Plane_Z65_75" in hfss.modeler.object_names:
            hfss.modeler.delete("NF_Plane_Z65_75")
        new_objects: list[str] = []
        construction = "frozen_source_revalidated"
        if rows == 2:
            success, new_objects = hfss.modeler.duplicate_along_line(
                duplicate_objects, ["0mm", "0mm", "15mm"], clones=2,
                attach=False, duplicate_assignment=True,
            )
            construction = "source_module_duplicated_along_+Z"
        elif cols == 8:
            success, new_objects = hfss.modeler.duplicate_along_line(
                duplicate_objects, ["0mm", "27.2mm", "0mm"], clones=2,
                attach=False, duplicate_assignment=True,
            )
            construction = "two_1x4_modules_duplicated_along_+Y"
        else:
            success = True
        if not success:
            raise RuntimeError("HFSS duplication failed")
        # Keep the duplicated 1x4 modules as separate solids.  Union/purge on
        # the 1x8 build rewired the entity references used by the integration
        # lines of ports 1 and 5, moving one endpoint off each port sheet.
        united: list[list[str]] = []
        if not radiation[0].delete() or not hfss.modeler.delete(old_regions):
            raise RuntimeError("Could not remove the old Radiation region")
        if not hfss.create_open_region(frequency="47GHz", boundary="Radiation"):
            raise RuntimeError("Could not create the new open region")
        for setup in list(hfss.field_setups):
            try:
                setup.delete()
            except Exception:
                pass
        integer = hfss.insert_infinite_sphere(
            definition="Theta-Phi", x_start=0, x_stop=180, x_step=1,
            y_start=-180, y_stop=180, y_step=1, units="deg", name="SphereInteger",
        )
        offset = hfss.insert_infinite_sphere(
            definition="Theta-Phi", x_start=0.5, x_stop=179.5, x_step=1,
            y_start=-179.5, y_stop=179.5, y_step=1, units="deg", name="SphereOffset",
        )
        if not integer or not offset:
            raise RuntimeError("Could not create both far-field spheres")
        ports, maximum_error = _port_records(hfss, rows, cols)
        setup_names = [str(item.name) for item in hfss.setups]
        if "Setup2" not in setup_names:
            raise RuntimeError(f"Setup2 missing: {setup_names}")
        setup2 = next(item for item in hfss.setups if str(item.name) == "Setup2")
        frequency_setting = str(setup2.props.get("Frequency", ""))
        if "47" not in frequency_setting:
            raise RuntimeError(f"Setup2 frequency is not 47 GHz: {frequency_setting!r}")
        radiation_record = _radiation_record(hfss)
        spheres = _sphere_records(hfss)
        validation = _validate_design(hfss, rows * cols)
        if not hfss.save_project(str(project), overwrite=True, refresh_ids=True):
            raise RuntimeError("AEDT refused to save the built project")
        return {
            "passed": True, "stage": "build", "identity": identity(), "array_shape": [rows, cols],
            "construction": construction, "source": str(source), "source_sha256": SOURCE_SHA256,
            "project": str(project), "project_sha256": sha256_file(project),
            "duplicated_object_count": len(new_objects), "united_same_material_groups": united,
            "port_count": len(ports), "port_position_max_error_mm": maximum_error,
            "ports": ports, "radiation": radiation_record, "far_field_spheres": spheres,
            "setup2_frequency": frequency_setting, "validation": validation, "solve_performed": False,
        }
    finally:
        close_hfss(desktop, hfss, save=False)


def verify_prerequisite(path: Path, stage: str, project: Path, *, require_hash: bool = True) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite report {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("stage") != stage:
        raise RuntimeError(f"Prerequisite {stage} has not passed")
    expected = report.get("project_sha256_after_solve") or report.get("project_sha256")
    if require_hash and expected and sha256_file(project) != expected:
        raise RuntimeError(f"Project fingerprint differs from the passed {stage} report")
    return report


def solve(project: Path, build_report: Path, rows: int, cols: int, graphical: bool, cores: int) -> dict[str, object]:
    # A failed or interrupted solve can legitimately rewrite AEDT metadata.
    # Re-audit all static invariants below instead of requiring the original
    # build-stage project hash on a recovery solve.
    verify_prerequisite(build_report, "build", project, require_hash=False)
    desktop = hfss = None
    started = time.time()
    try:
        desktop, hfss = open_hfss(project, graphical)
        _port_records(hfss, rows, cols)
        _radiation_record(hfss)
        _sphere_records(hfss)

        # A GUI solve is a valid predecessor for the recoverable pipeline.  Read
        # it before requesting another solve so a known-good result is never
        # overwritten merely to create the stage receipt.
        solution = hfss.post.get_solution_data(
            expressions="S(1,1)", setup_sweep_name="Setup2 : LastAdaptive"
        )
        if solution and len(solution.data_real()) > 0:
            aedt_messages = list(hfss._desktop.GetMessages(hfss.project_name, hfss.design_name, 0))
            close_hfss(desktop, hfss, save=False)
            desktop = hfss = None
            return {
                "passed": True, "stage": "solve", "array_shape": [rows, cols], "project": str(project),
                "project_sha256_after_solve": sha256_file(project), "setup": "Setup2 : LastAdaptive",
                "elapsed_seconds": time.time() - started, "solve_performed": False,
                "solution_reused": True,
                "hpc_configuration": {"type": "existing_solution"},
                "aedt_messages_tail": aedt_messages[-12:],
                "solution_readback": "existing S(1,1) readable from Setup2 : LastAdaptive",
            }

        # Use AEDT's native Analyze call without asking PyAEDT to rewrite the
        # active DSO/HPC registry.  The same project solves from the GUI, while
        # analyze_setup() repeatedly crashed HFSSCOMENGINE during its generated
        # DSO distribution path even with NumEngines=1.
        analyze_return = hfss.odesign.Analyze("Setup2", True)
        aedt_messages = list(hfss._desktop.GetMessages(hfss.project_name, hfss.design_name, 0))
        # AEDT documents 0 as success and a non-zero integer as an error.  Some
        # automation bridges return None for a successful void-style call, so
        # keep that value as evidence and let solution readback remain the
        # authoritative gate.  Never replace the actual return with True.
        if isinstance(analyze_return, int) and not isinstance(analyze_return, bool) and analyze_return != 0:
            raise RuntimeError(
                f"HFSS Analyze Setup2 returned error code {analyze_return}; "
                f"messages={aedt_messages[-12:]}"
            )
        solution = hfss.post.get_solution_data(
            expressions="S(1,1)", setup_sweep_name="Setup2 : LastAdaptive"
        )
        if not solution or len(solution.data_real()) == 0:
            raise RuntimeError(
                f"HFSS Analyze Setup2 returned {analyze_return!r} but no readable S(1,1) solution exists; "
                f"treating this as a solver-engine failure; messages={aedt_messages[-12:]}"
            )
        if not hfss.save_project(str(project), overwrite=True, refresh_ids=True):
            raise RuntimeError("Could not save the solved project")
        # The project is already durably saved.  Closing with save=True would
        # rewrite AEDT metadata after the fingerprint below is frozen.
        close_hfss(desktop, hfss, save=False)
        desktop = hfss = None
        return {
            "passed": True, "stage": "solve", "array_shape": [rows, cols], "project": str(project),
            "project_sha256_after_solve": sha256_file(project), "setup": "Setup2 : LastAdaptive",
            "elapsed_seconds": time.time() - started, "solve_performed": True,
            "solution_reused": False,
            "analyze_return": analyze_return,
            "hpc_configuration": {"type": "aedt_active_config", "requested_cores_ignored": cores},
            "aedt_messages_tail": aedt_messages[-12:],
            "solution_readback": "S(1,1) readable from Setup2 : LastAdaptive",
        }
    finally:
        close_hfss(desktop, hfss, save=False)


def export_s(project: Path, solve_report: Path, output: Path, rows: int, cols: int, graphical: bool) -> dict[str, object]:
    verify_prerequisite(solve_report, "solve", project)
    desktop = hfss = None
    try:
        desktop, hfss = open_hfss(project, graphical)
        _port_records(hfss, rows, cols)
        output.parent.mkdir(parents=True, exist_ok=True)
        exported = hfss.export_touchstone(setup="Setup2", sweep="LastAdaptive", output_file=str(output))
        if not exported or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Touchstone API export failed; use the documented GUI fallback and rerun verification")
        return {
            "passed": True, "stage": "export_s", "project": str(project),
            "project_sha256_after_solve": sha256_file(project), "touchstone": str(output),
            "bytes": output.stat().st_size, "sha256": sha256_file(output), "port_count": rows * cols,
        }
    finally:
        close_hfss(desktop, hfss, save=False)


def _copy_eep_tree(paths: list[str], output: Path) -> list[dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    roots = {Path(path).absolute().parent for path in paths}
    for root in roots:
        shutil.copytree(root, output, dirs_exist_ok=True)
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    ]


def export_eep(project: Path, solve_report: Path, output_root: Path, rows: int, cols: int, graphical: bool) -> dict[str, object]:
    verify_prerequisite(solve_report, "solve", project)
    desktop = hfss = None
    grids = {}
    try:
        desktop, hfss = open_hfss(project, graphical)
        _port_records(hfss, rows, cols)
        _sphere_records(hfss)
        for name, folder in (("SphereInteger", "integer"), ("SphereOffset", "offset")):
            solution = hfss.get_antenna_ffd_solution_data(
                FREQUENCY_HZ, setup="Setup2 : LastAdaptive", sphere=name,
                overwrite=True, link_to_hfss=True,
            )
            paths = [str(value) for value in solution.eep_files]
            files = _copy_eep_tree(paths, output_root / folder)
            ffd_count = sum(item["name"].lower().endswith(".ffd") for item in files)
            if ffd_count != rows * cols:
                raise RuntimeError(f"{name}: expected {rows * cols} FFDs, got {ffd_count}")
            mappings = [item for item in files if str(item["name"]).lower().endswith(".txt")]
            if not mappings:
                raise RuntimeError(f"{name}: mapping file missing")
            grids[folder] = {"sphere": name, "ffd_count": ffd_count, "files": files}
        return {
            "passed": True, "stage": "export_eep", "project": str(project),
            "project_sha256_after_solve": sha256_file(project), "grids": grids,
        }
    finally:
        close_hfss(desktop, hfss, save=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "solve", "export_s", "export_eep"), required=True)
    parser.add_argument("--array-id", choices=("a1x4", "a2x4", "a1x8"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--graphical", action="store_true")
    args = parser.parse_args()
    require_session()
    restored = restore_environment()
    shapes = {"a1x4": (1, 4), "a2x4": (2, 4), "a1x8": (1, 8)}
    rows, cols = shapes[args.array_id]
    root = args.root.absolute()
    project = root / "projects" / f"{args.array_id}_47ghz.aedt"
    stage_dir = root / "raw_exports" / args.array_id / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    report_path = stage_dir / f"{args.stage}.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("passed"):
            print(json.dumps({"reused_passed_stage": str(report_path), **existing}, ensure_ascii=False, indent=2))
            return
    try:
        if args.stage == "build":
            if args.source is None:
                raise ValueError("--source is required for build")
            report = build(args.source.absolute(), project, rows, cols, args.graphical)
        elif args.stage == "solve":
            report = solve(project, stage_dir / "build.json", rows, cols, args.graphical, args.cores)
        elif args.stage == "export_s":
            report = export_s(
                project, stage_dir / "solve.json",
                root / "raw_exports" / args.array_id / f"{args.array_id}_47ghz.s{rows * cols}p",
                rows, cols, args.graphical,
            )
        else:
            report = export_eep(
                project, stage_dir / "solve.json", root / "raw_exports" / args.array_id / "eep",
                rows, cols, args.graphical,
            )
        report["restored_environment"] = restored
    except Exception as exc:
        report = {
            "passed": False, "stage": args.stage, "array_id": args.array_id,
            "project": str(project), "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "restored_environment": restored,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report["array_id"] = args.array_id
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
