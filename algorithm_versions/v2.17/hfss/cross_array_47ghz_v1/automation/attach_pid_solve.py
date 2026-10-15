"""Attach to an ordinary AEDT GUI process and solve without rewriting DSO settings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


PYAEDT_SITE_PACKAGES = os.environ.get("AESA_PYAEDT_SITE_PACKAGES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("ANSYSEM_ROOT252", r"D:\Program Files\ANSYS Inc\v252\AnsysEM")
    if PYAEDT_SITE_PACKAGES and PYAEDT_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, PYAEDT_SITE_PACKAGES)

    from pyaedt import Desktop
    import pyaedt

    desktop = None
    started = time.time()
    report: dict[str, object]
    try:
        desktop = Desktop(
            version="2025.2", non_graphical=False, new_desktop=False,
            close_on_exit=False, aedt_process_id=args.pid,
        )
        odesktop = desktop.odesktop
        deadline = time.time() + 120.0
        project_names: list[str] = []
        while time.time() < deadline:
            project_names = list(odesktop.GetProjectList())
            if args.project_name in project_names:
                break
            time.sleep(1.0)
        if args.project_name not in project_names:
            raise RuntimeError(
                f"Expected project {args.project_name!r} was not open; projects={project_names}"
            )
        project = odesktop.SetActiveProject(args.project_name)
        design = project.SetActiveDesign("HFSSDesign1")
        design.Analyze("Setup2")
        messages = list(odesktop.GetMessages(args.project_name, "HFSSDesign1", 0))
        project.Save()
        report = {
            "passed": True,
            "route": "ordinary_gui_plus_pyaedt_pid_attach",
            "aedt_process_id": args.pid,
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "project": args.project_name,
            "setup": "Setup2 : LastAdaptive",
            "elapsed_seconds": time.time() - started,
            "solution_readback": "PENDING independent post-solve readback",
            "messages_tail": messages[-12:],
        }
    except Exception as exc:
        report = {
            "passed": False,
            "route": "ordinary_gui_plus_pyaedt_pid_attach",
            "aedt_process_id": args.pid,
            "project": args.project_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=False, close_on_exit=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
