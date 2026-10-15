"""Non-destructive PyAEDT smoke test for source weights and far-field export."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path


def read_sources(path: Path) -> dict[str, tuple[str, str]]:
    sources: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["source_name"].strip()
            sources[name] = (f'{float(row["power_w"]):.12g}W', f'{float(row["phase_deg"]):.12g}deg')
    if len(sources) != 16:
        raise ValueError(f"Expected 16 sources, got {len(sources)}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--attach-pid", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = read_sources(args.source_csv)

    # PyAEDT 0.9.x discovers AEDT through this environment variable. The
    # dedicated automation shell does not inherit it reliably from Windows.
    os.environ.setdefault("ANSYSEM_ROOT252", r"D:\Program Files\ANSYS Inc\v252\AnsysEM")

    desktop = None
    hfss = None
    report: dict[str, object]
    try:
        from pyaedt import Desktop, Hfss
        import pyaedt

        attach = args.attach_pid is not None
        desktop = Desktop(
            version="2025.2",
            non_graphical=False if attach else True,
            new_desktop=False if attach else True,
            close_on_exit=False if attach else True,
            aedt_process_id=args.attach_pid,
        )
        hfss = Hfss(
            project=args.project.stem if attach else str(args.project),
            version="2025.2",
            non_graphical=False if attach else True,
            new_desktop=False,
            close_on_exit=False if attach else True,
            aedt_process_id=args.attach_pid,
            remove_lock=False,
        )
        excitations = list(hfss.excitations)
        if set(excitations) != set(sources):
            raise RuntimeError(f"Source mismatch. HFSS={excitations}, CSV={sorted(sources)}")

        source_api_ok = bool(hfss.edit_sources(sources, include_port_post_processing=True))
        setup = "Setup2 : LastAdaptive"
        sphere = "Infinite Sphere1"
        categories = list(hfss.post.available_report_types)
        displays = list(hfss.post.available_display_types("Far Fields"))
        quantities = list(
            hfss.post.available_report_quantities(
                report_category="Far Fields",
                display_type="Data Table",
                solution=setup,
                context=sphere,
            )
        )

        expressions = ["re(rETheta)", "im(rETheta)", "re(rEPhi)", "im(rEPhi)", "RealizedGainTotal"]
        plot_name = f"V29_API_SMOKE_{args.source_csv.stem[:40]}"
        created = bool(
            hfss.post.create_report(
                expressions=expressions,
                setup_sweep_name=setup,
                variations={"Freq": ["47GHz"], "Theta": ["All"], "Phi": ["All"]},
                primary_sweep_variable="Theta",
                secondary_sweep_variable="Phi",
                report_category="Far Fields",
                plot_type="Data Table",
                context=sphere,
                plot_name=plot_name,
            )
        )
        if not created:
            raise RuntimeError("create_report returned False")
        exported_path = Path(
            hfss.post.export_report_to_file(
                str(args.output_dir), plot_name, ".csv", unique_file=False
            )
        )
        if not exported_path.is_file() or exported_path.stat().st_size == 0:
            raise RuntimeError(f"Report export missing or empty: {exported_path}")
        with exported_path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            line_count = sum(1 for _ in stream)
        if line_count < 2:
            raise RuntimeError(f"Report has too few lines: {line_count}")

        report = {
            "schema_version": 1,
            "passed": True,
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "project": str(args.project),
            "design": str(hfss.design_name),
            "attached_to_existing_desktop": attach,
            "aedt_process_id": args.attach_pid,
            "source_csv": str(args.source_csv),
            "source_api_ok": source_api_ok,
            "sources": sources,
            "available_report_types": categories,
            "available_far_field_displays": displays,
            "available_far_field_quantities": quantities,
            "requested_expressions": expressions,
            "report_api_ok": created,
            "exported_csv": str(exported_path),
            "exported_bytes": exported_path.stat().st_size,
            "exported_line_count": line_count,
        }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "project": str(args.project),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if hfss is not None and args.attach_pid is None:
            try:
                hfss.close_project(save=False)
            except Exception:
                pass
        if desktop is not None:
            try:
                desktop.release_desktop(
                    close_projects=args.attach_pid is None,
                    close_on_exit=args.attach_pid is None,
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
