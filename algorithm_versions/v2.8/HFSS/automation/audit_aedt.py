"""Read-only AEDT inventory for the V2.8 4x4 build.

This script opens a project without solving or saving it and records the object,
boundary, port, setup, variable, and far-field configuration needed to build the
4x4 project safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

from build_4x4 import current_identity, restore_machine_environment


AEDT_ROOT = Path(r"D:\Program Files\ANSYS Inc\v252\AnsysEM")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def object_record(modeler: Any, name: str) -> dict[str, Any]:
    obj = modeler[name]
    record: dict[str, Any] = {"name": name}
    for attribute in (
        "id",
        "object_type",
        "material_name",
        "group_name",
        "model",
        "solve_inside",
        "display_wireframe",
    ):
        try:
            record[attribute] = json_ready(getattr(obj, attribute))
        except Exception as exc:
            record[f"{attribute}_error"] = str(exc)
    try:
        record["bounding_box"] = [float(item) for item in obj.bounding_box]
    except Exception as exc:
        record["bounding_box_error"] = str(exc)
    return record


def inventory(project: Path) -> dict[str, Any]:
    restored = restore_machine_environment(("ANSYSEM_ROOT252", "ANSYSLMD_LICENSE_FILE"))
    from pyaedt import Desktop, Hfss
    import pyaedt

    desktop = None
    hfss = None
    try:
        desktop = Desktop(
            version="2025.2",
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        hfss = Hfss(
            project=str(project),
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            remove_lock=False,
        )
        modeler = hfss.modeler
        object_names = list(modeler.object_names)
        boundaries = []
        for boundary in hfss.boundaries:
            boundaries.append(
                {
                    "name": str(boundary.name),
                    "type": str(boundary.type),
                    "props": json_ready(boundary.props),
                }
            )
        setups = []
        for setup in hfss.setups:
            setups.append(
                {
                    "name": str(setup.name),
                    "props": json_ready(setup.props),
                }
            )
        field_setups = []
        for field_setup in hfss.field_setups:
            field_setups.append(
                {
                    "name": str(field_setup.name),
                    "props": json_ready(field_setup.props),
                }
            )
        variables = {}
        for name in hfss.variable_manager.variable_names:
            try:
                variable = hfss.variable_manager[name]
                variables[str(name)] = {
                    "expression": str(variable.expression),
                    "evaluated_value": str(variable.evaluated_value),
                }
            except Exception as exc:
                variables[str(name)] = {"error": str(exc)}
        return {
            "schema_version": 1,
            "project": str(project),
            "project_sha256": sha256_file(project),
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "aedt_version": "2025.2",
            "launcher_identity": current_identity(),
            "restored_environment": restored,
            "project_name": str(hfss.project_name),
            "design_name": str(hfss.design_name),
            "solution_type": str(hfss.solution_type),
            "model_units": str(modeler.model_units),
            "excitations": [str(item) for item in hfss.excitations],
            "object_count": len(object_names),
            "objects": [object_record(modeler, name) for name in object_names],
            "boundaries": boundaries,
            "setups": setups,
            "field_setups": field_setups,
            "variables": variables,
            "no_solve_performed": True,
            "project_saved": False,
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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Preserve an ASCII substituted-drive path for AEDT gRPC.
    project = args.project.absolute()
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = inventory(project)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "project": str(project),
            "project_sha256": sha256_file(project),
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "no_solve_performed": True,
            "project_saved": False,
        }
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    report["passed"] = True
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
