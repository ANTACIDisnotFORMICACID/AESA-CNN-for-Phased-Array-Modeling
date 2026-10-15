"""Solve the validated V2.8 4x4 HFSS project and export spherical EEP data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any

from build_4x4 import (
    current_identity,
    far_field_sphere_audit,
    radiation_box_audit,
    require_interactive_user,
    restore_machine_environment,
)


AEDT_ROOT = Path(r"D:\Program Files\ANSYS Inc\v252\AnsysEM")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy_eep_tree(eep_maps: list[str], output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    copied_roots = set()
    for mapping in eep_maps:
        mapping_path = Path(mapping).resolve()
        source_root = mapping_path.parent
        if source_root not in copied_roots:
            shutil.copytree(source_root, output, dirs_exist_ok=True)
            copied_roots.add(source_root)
    for path in sorted(output.glob("*")):
        if path.is_file():
            records.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def solve_and_export(
    project: Path,
    build_report: Path,
    export_dir: Path,
    graphical: bool,
    cores: int,
) -> dict[str, Any]:
    identity = require_interactive_user()
    restored = restore_machine_environment(("ANSYSEM_ROOT252", "ANSYSLMD_LICENSE_FILE"))
    build = json.loads(build_report.read_text(encoding="utf-8"))
    if not build.get("passed") or len(build.get("ports", [])) != 16:
        raise RuntimeError("The 4x4 build report has not passed")
    if sha256_file(project) != build.get("target_sha256"):
        raise RuntimeError("The AEDT project hash differs from the validated build report")

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
            project=str(project),
            version="2025.2",
            non_graphical=not graphical,
            new_desktop=False,
            remove_lock=False,
        )
        if len(hfss.excitations) != 16:
            raise RuntimeError("Expected 16 ports before solving, got {}".format(hfss.excitations))
        setup_names = [str(item.name) for item in hfss.setups]
        if "Setup2" not in setup_names:
            raise RuntimeError("Setup2 is missing: {}".format(setup_names))
        far_field_sphere = far_field_sphere_audit(hfss)
        radiation_box = radiation_box_audit(hfss)
        validation = hfss.validate_full_design(ports=16)
        solved = hfss.analyze_setup(
            "Setup2",
            cores=cores,
            tasks=1,
            gpus=0,
            use_auto_settings=True,
            blocking=True,
        )
        if not solved:
            raise RuntimeError("HFSS Analyze Setup2 did not complete successfully")
        export_dir.mkdir(parents=True, exist_ok=True)
        touchstone = export_dir / "v2_8_array_4x4.s16p"
        if not hfss.export_touchstone(
            setup="Setup2",
            sweep="LastAdaptive",
            output_file=str(touchstone),
        ):
            raise RuntimeError("Could not export the 16-port Touchstone file")
        ffd = hfss.get_antenna_ffd_solution_data(
            47.0e9,
            setup="Setup2 : LastAdaptive",
            sphere="Infinite Sphere1",
            overwrite=True,
            link_to_hfss=True,
        )
        eep_maps = [str(item) for item in ffd.eep_files]
        eep_files = copy_eep_tree(eep_maps, export_dir / "element_patterns")
        pattern_count = len([item for item in eep_files if item["name"].lower().endswith(".ffd")])
        if pattern_count != 16:
            raise RuntimeError("Expected 16 FFD files, got {}".format(pattern_count))
        if not hfss.save_project(str(project), overwrite=True, refresh_ids=True):
            raise RuntimeError("Could not save solved AEDT project")
        return {
            "schema_version": 1,
            "passed": True,
            "identity": identity,
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "project": str(project),
            "project_sha256_after_solve": sha256_file(project),
            "design": str(hfss.design_name),
            "setup": "Setup2 : LastAdaptive",
            "frequency_hz": 47.0e9,
            "far_field_setup": "Infinite Sphere1",
            "far_field_sphere": far_field_sphere,
            "radiation_box": radiation_box,
            "port_count": len(hfss.excitations),
            "touchstone": {
                "path": str(touchstone),
                "bytes": touchstone.stat().st_size,
                "sha256": sha256_file(touchstone),
            },
            "eep_map_files": eep_maps,
            "exported_files": eep_files,
            "ffd_count": pattern_count,
            "validation": str(validation),
            "restored_environment": restored,
            "solve_performed": True,
        }
    finally:
        if hfss is not None:
            try:
                hfss.close_project(save=True)
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
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--graphical", action="store_true")
    args = parser.parse_args()
    # Preserve the substituted ASCII drive path for AEDT gRPC save/export calls.
    report_path = args.report.absolute()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = solve_and_export(
            args.project.absolute(),
            args.build_report.absolute(),
            args.export_dir.absolute(),
            args.graphical,
            args.cores,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "identity": current_identity(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
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
