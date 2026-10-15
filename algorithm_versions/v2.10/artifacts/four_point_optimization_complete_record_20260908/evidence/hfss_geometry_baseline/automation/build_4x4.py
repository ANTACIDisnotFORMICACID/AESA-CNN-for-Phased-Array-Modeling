"""Build the V2.8 4x4 HFSS project from the frozen 1x4 copy.

Run this script only from the interactive FORMIC-ACID\ASUS Windows session.
It creates a new project, duplicates the complete 1x4 model row along +Z with
15 mm pitch, duplicates the four wave-port assignments, and rebuilds the open
region around the full array. It does not solve the project.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any, Iterable


AEDT_ROOT = Path(r"D:\Program Files\ANSYS Inc\v252\AnsysEM")
EXPECTED_SOURCE_SHA256 = "D647839C759D84C58D9A0251EFB814C29790F0D5D03287462284B3A2B683D350"
ROW_PITCH_MM = 15.0
COLUMN_PITCH_MM = 6.8
WAVELENGTH_MM = 299_792_458.0 / 47.0e9 * 1000.0
MINIMUM_RADIATION_MARGIN_MM = WAVELENGTH_MM / 4.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def current_identity() -> str:
    try:
        return subprocess.check_output(
            ["whoami"], text=True, encoding="utf-8", errors="replace"
        ).strip()
    except Exception:
        return getpass.getuser()


def require_interactive_user() -> str:
    identity = current_identity()
    if not identity.lower().endswith("\\asus"):
        raise PermissionError(
            "AEDT build requires FORMIC-ACID\\ASUS; current identity is {!r}".format(identity)
        )
    return identity


def restore_machine_environment(names: Iterable[str]) -> dict[str, str]:
    restored = {}
    import winreg

    key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    for name in names:
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
    os.environ.setdefault("ANSYSEM_ROOT252", str(AEDT_ROOT))
    return restored


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    try:
        return json_ready(dict(value))
    except (TypeError, ValueError):
        return str(value)


def assigned_objects(boundary: Any, modeler: Any) -> list[str]:
    raw = boundary.props.get("Objects", [])
    if isinstance(raw, str):
        raw = [raw]
    resolved = []
    for item in raw:
        text = str(item)
        if text.isdigit():
            object_id = int(text)
            match = next(
                (
                    str(name)
                    for name in modeler.object_names
                    if int(modeler[name].id) == object_id
                ),
                None,
            )
            if match is None:
                raise RuntimeError(
                    "Boundary {!r} references unknown object ID {}".format(
                        boundary.name, object_id
                    )
                )
            resolved.append(match)
        else:
            resolved.append(text)
    return resolved


def far_field_sphere_record(sphere: Any, available_names: list[str]) -> dict[str, Any]:
    """Verify and serialize one full-sphere export definition."""

    wanted = "Infinite Sphere1"
    if str(sphere.name) != wanted:
        raise RuntimeError("Unexpected far-field sphere name {!r}".format(sphere.name))
    expected = {
        "ThetaStart": "0deg",
        "ThetaStop": "180deg",
        "ThetaStep": "1deg",
        "PhiStart": "-180deg",
        "PhiStop": "180deg",
        "PhiStep": "1deg",
    }
    actual = {key: str(sphere.props.get(key, "")) for key in expected}
    mismatched = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key].replace(" ", "").lower()
        != expected[key].replace(" ", "").lower()
    }
    if mismatched:
        raise RuntimeError("Infinite Sphere1 sampling changed: {}".format(mismatched))
    return {
        "name": wanted,
        "coordinate_system": str(sphere.props.get("UseLocalCS", "Global")),
        "sampling": actual,
        "available_field_setups": available_names,
        "selected_name_is_unique": True,
    }


def far_field_sphere_audit(hfss: Any) -> dict[str, Any]:
    """Select and verify the explicitly named full-sphere export definition."""

    wanted = "Infinite Sphere1"
    field_setups = list(hfss.field_setups)
    names = [str(item.name) for item in field_setups]
    matches = [item for item in field_setups if str(item.name) == wanted]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one explicitly named {!r}, got {}".format(wanted, names)
        )
    return far_field_sphere_record(matches[0], names)


def radiation_box_audit(hfss: Any) -> dict[str, Any]:
    """Require one six-face rectangular radiation box around the full array."""

    modeler = hfss.modeler
    radiation_boundaries = [
        item for item in hfss.boundaries if str(item.type).lower() == "radiation"
    ]
    if len(radiation_boundaries) != 1:
        raise RuntimeError(
            "Expected exactly one radiation boundary, got {}".format(
                [str(item.name) for item in radiation_boundaries]
            )
        )
    radiation_objects = assigned_objects(radiation_boundaries[0], modeler)
    if len(radiation_objects) != 1:
        raise RuntimeError(
            "Radiation boundary must reference one box, got {}".format(radiation_objects)
        )
    radiation_name = radiation_objects[0]
    if radiation_name == "Box9" or "Box9" in modeler.object_names:
        raise RuntimeError("The old 1x4 Box9 radiation region still exists")
    radiation_object = modeler[radiation_name]
    face_count = len(radiation_object.faces)
    if str(radiation_object.object_type).lower() != "solid" or face_count != 6:
        raise RuntimeError(
            "Radiation object must be a six-face solid cuboid, got type={} faces={}".format(
                radiation_object.object_type, face_count
            )
        )
    radiation_bounds = [float(item) for item in radiation_object.bounding_box]
    array_objects = [
        str(name)
        for name in modeler.model_objects
        if str(name) != radiation_name
    ]
    if not array_objects:
        raise RuntimeError("No array model objects remain inside the radiation box")
    boxes = [
        [float(item) for item in modeler[name].bounding_box]
        for name in array_objects
    ]
    array_bounds = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        max(box[4] for box in boxes),
        max(box[5] for box in boxes),
    ]
    margins = {
        "x_min": array_bounds[0] - radiation_bounds[0],
        "x_max": radiation_bounds[3] - array_bounds[3],
        "y_min": array_bounds[1] - radiation_bounds[1],
        "y_max": radiation_bounds[4] - array_bounds[4],
        "z_min": array_bounds[2] - radiation_bounds[2],
        "z_max": radiation_bounds[5] - array_bounds[5],
    }
    tolerance = 0.05 * MINIMUM_RADIATION_MARGIN_MM
    too_small = {
        key: value
        for key, value in margins.items()
        if value + tolerance < MINIMUM_RADIATION_MARGIN_MM
    }
    if too_small:
        raise RuntimeError(
            "Radiation box margin is below lambda/4 ({:.6f} mm): {}".format(
                MINIMUM_RADIATION_MARGIN_MM, too_small
            )
        )
    return {
        "boundary_name": str(radiation_boundaries[0].name),
        "object_name": radiation_name,
        "object_type": str(radiation_object.object_type),
        "material": str(radiation_object.material_name),
        "face_count": face_count,
        "array_object_count": len(array_objects),
        "array_bounding_box_mm": array_bounds,
        "radiation_bounding_box_mm": radiation_bounds,
        "radiation_box_size_mm": [
            radiation_bounds[3] - radiation_bounds[0],
            radiation_bounds[4] - radiation_bounds[1],
            radiation_bounds[5] - radiation_bounds[2],
        ],
        "margins_mm": margins,
        "minimum_required_margin_mm": MINIMUM_RADIATION_MARGIN_MM,
        "old_box9_absent": True,
        "single_radiation_boundary": True,
        "single_radiation_object": True,
        "is_six_face_cuboid": True,
    }


def port_map(hfss: Any) -> list[dict[str, Any]]:
    boundaries = {str(item.name): item for item in hfss.boundaries}
    records = []
    port_names = sorted({str(item).split(":")[0] for item in hfss.excitations})
    for port_name in port_names:
        boundary = boundaries.get(port_name)
        if boundary is None:
            raise RuntimeError("Missing wave-port boundary {!r}".format(port_name))
        objects = assigned_objects(boundary, hfss.modeler)
        if len(objects) != 1:
            raise RuntimeError(
                "Port {!r} must reference exactly one sheet, got {}".format(port_name, objects)
            )
        obj = hfss.modeler[objects[0]]
        box = [float(item) for item in obj.bounding_box]
        center = [(box[index] + box[index + 3]) / 2.0 for index in range(3)]
        records.append(
            {
                "hfss_port": port_name,
                "mode_name": next(
                    (str(item) for item in hfss.excitations if str(item).split(":")[0] == port_name),
                    port_name,
                ),
                "sheet": objects[0],
                "center_mm": center,
            }
        )
    if len(records) != 16:
        raise RuntimeError("Expected 16 unique wave ports, got {}".format(len(records)))
    y_values = sorted({round(item["center_mm"][1], 6) for item in records})
    z_values = sorted({round(item["center_mm"][2], 6) for item in records})
    if len(y_values) != 4 or len(z_values) != 4:
        raise RuntimeError("Port centers do not form a 4x4 grid: y={}, z={}".format(y_values, z_values))
    y_spacing = [y_values[index + 1] - y_values[index] for index in range(3)]
    z_spacing = [z_values[index + 1] - z_values[index] for index in range(3)]
    if any(abs(value - COLUMN_PITCH_MM) > 0.01 for value in y_spacing):
        raise RuntimeError("Unexpected column pitch: {}".format(y_spacing))
    if any(abs(value - ROW_PITCH_MM) > 0.01 for value in z_spacing):
        raise RuntimeError("Unexpected row pitch: {}".format(z_spacing))
    for item in records:
        item["row_z"] = min(range(4), key=lambda index: abs(z_values[index] - item["center_mm"][2]))
        item["col_y"] = min(range(4), key=lambda index: abs(y_values[index] - item["center_mm"][1]))
    records.sort(key=lambda item: (item["row_z"], item["col_y"]))
    return records


def build(source: Path, target: Path, graphical: bool) -> dict[str, Any]:
    identity = require_interactive_user()
    restored = restore_machine_environment(("ANSYSEM_ROOT252", "ANSYSLMD_LICENSE_FILE"))
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source AEDT hash does not match the frozen 1x4 project")
    if target.exists() and sha256_file(target) != source_hash:
        raise FileExistsError(
            "Refusing to overwrite a modified target: {}".format(target)
        )
    shutil.copy2(source, target)

    from pyaedt import Desktop, Hfss
    import pyaedt

    desktop = None
    hfss = None
    try:
        desktop = Desktop(
            version="2025.2",
            non_graphical=not graphical,
            new_desktop=True,
            close_on_exit=True,
        )
        hfss = Hfss(
            project=str(target),
            version="2025.2",
            non_graphical=not graphical,
            new_desktop=False,
            remove_lock=False,
        )
        if len(hfss.excitations) != 4:
            raise RuntimeError("Expected four source ports, got {}".format(hfss.excitations))
        modeler = hfss.modeler
        radiation_boundaries = [
            item for item in hfss.boundaries if str(item.type).lower() == "radiation"
        ]
        if len(radiation_boundaries) != 1:
            raise RuntimeError(
                "Expected one source radiation boundary, got {}".format(
                    [item.name for item in radiation_boundaries]
                )
            )
        old_region_objects = assigned_objects(radiation_boundaries[0], modeler)
        if old_region_objects != ["Box9"]:
            raise RuntimeError(
                "Frozen source radiation object changed: {}".format(old_region_objects)
            )
        source_model_objects = [str(item) for item in modeler.model_objects]
        duplicate_objects = [
            name for name in source_model_objects if name not in set(old_region_objects)
        ]
        if "NF_Plane_Z65_75" in duplicate_objects:
            raise RuntimeError("Non-model near-field plane entered the duplication selection")
        if len(duplicate_objects) != 45:
            raise RuntimeError(
                "Expected 45 model objects excluding Box9, got {}".format(len(duplicate_objects))
            )
        hfss["v28_row_pitch"] = "15mm"
        duplicated, new_objects = modeler.duplicate_along_line(
            duplicate_objects,
            ["0mm", "0mm", "v28_row_pitch"],
            clones=4,
            attach=False,
            duplicate_assignment=True,
        )
        if not duplicated or len(new_objects) != 135:
            raise RuntimeError(
                "Expected 135 duplicated objects, got success={} count={}".format(
                    duplicated, len(new_objects)
                )
            )
        if len(hfss.excitations) != 16:
            raise RuntimeError(
                "Port assignments were not duplicated to 16: {}".format(hfss.excitations)
            )
        if not radiation_boundaries[0].delete():
            raise RuntimeError("Could not delete the old radiation boundary")
        if not modeler.delete(old_region_objects):
            raise RuntimeError("Could not delete the old 1x4 radiation box")
        if not hfss.create_open_region(frequency="47GHz", boundary="Radiation"):
            raise RuntimeError("Could not create the full-array open region")
        # HFSS deletes far-field setups as soon as their old radiation boundary
        # is removed, so recreate the selected sphere after the new region exists.
        sphere = hfss.insert_infinite_sphere(
            definition="Theta-Phi",
            x_start=0,
            x_stop=180,
            x_step=1,
            y_start=-180,
            y_stop=180,
            y_step=1,
            units="deg",
            custom_coordinate_system=None,
            name="Infinite Sphere1",
        )
        if not sphere:
            raise RuntimeError("Could not recreate Infinite Sphere1")
        far_field_sphere = far_field_sphere_record(sphere, ["Infinite Sphere1"])
        radiation_box = radiation_box_audit(hfss)
        ports = port_map(hfss)
        validation = hfss.validate_full_design(ports=16)
        if not hfss.save_project(str(target), overwrite=True, refresh_ids=True):
            raise RuntimeError("AEDT did not save the 4x4 project")
        return {
            "schema_version": 1,
            "passed": True,
            "identity": identity,
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "aedt_version": "2025.2",
            "source_project": str(source),
            "source_sha256": source_hash,
            "target_project": str(target),
            "target_sha256": sha256_file(target),
            "design": str(hfss.design_name),
            "frequency_hz": 47.0e9,
            "array_axes": {"row": "+Z", "column": "+Y", "boresight": "+X"},
            "row_pitch_mm": ROW_PITCH_MM,
            "column_pitch_mm": COLUMN_PITCH_MM,
            "logical_shape": [4, 4],
            "duplicated_source_object_count": len(duplicate_objects),
            "new_object_count": len(new_objects),
            "ports": ports,
            "far_field_sphere": far_field_sphere,
            "radiation_box": radiation_box,
            "validation": json_ready(validation),
            "restored_environment": restored,
            "solve_performed": False,
        }
    finally:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--graphical", action="store_true")
    args = parser.parse_args()
    # Keep a substituted ASCII drive path intact. Path.resolve() expands V:\\
    # back to the physical Unicode path, which AEDT 2025 R2 gRPC cannot save.
    source = args.source.absolute()
    target = args.target.absolute()
    report_path = args.report.absolute()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build(source, target, args.graphical)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "source": str(source),
            "target": str(target),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
