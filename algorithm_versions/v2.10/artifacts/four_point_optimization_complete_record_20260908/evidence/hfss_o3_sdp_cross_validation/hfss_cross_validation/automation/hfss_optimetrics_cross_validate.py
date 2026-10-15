"""Run an HFSS Optimetrics phase-only cross-check for one four-target case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import traceback
from pathlib import Path


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


def load_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16:
        raise ValueError(f"Expected 16 source rows, got {len(rows)}")
    return rows


def scalar(data, expression: str) -> float:
    values = data.data_real(expression)
    if not values:
        raise RuntimeError(f"No data for {expression}")
    return float(values[0])


def read_target_gains(hfss, case: dict[str, object]) -> list[float]:
    gains: list[float] = []
    for theta, phi in zip(case["hfss_theta_deg"], case["hfss_phi_deg"]):
        data = hfss.post.get_solution_data(
            expressions="RealizedGainTotal",
            setup_sweep_name="Setup2 : LastAdaptive",
            variations={
                "Freq": ["47GHz"],
                "Theta": [f"{float(theta):.12g}deg"],
                "Phi": [f"{float(phi):.12g}deg"],
            },
            primary_sweep_variable="Theta",
            report_category="Far Fields",
            context="Infinite Sphere1",
        )
        if not data:
            raise RuntimeError("HFSS returned no target gain")
        gains.append(10 * math.log10(max(scalar(data, "RealizedGainTotal"), 1e-30)))
    return gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive-copy", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--goal-db", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    source_rows = load_source_rows(args.source_csv)
    restored = restore_machine_environment()
    desktop = None
    hfss = None
    working_copy_active = False
    report: dict[str, object]
    try:
        import pyaedt
        from pyaedt import Desktop, Hfss

        desktop = Desktop(version="2025.2", non_graphical=True, new_desktop=True, close_on_exit=True)
        hfss = Hfss(
            project=str(args.project),
            design="HFSSDesign1",
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            close_on_exit=True,
            remove_lock=False,
        )
        args.archive_copy.parent.mkdir(parents=True, exist_ok=True)
        if not hfss.archive_project(
            str(args.archive_copy),
            include_external_files=True,
            include_results_file=True,
            notes="Temporary archive for O3/HFSS Optimetrics cross-validation",
        ):
            raise RuntimeError("HFSS project archive returned False")
        hfss.close_project(save=False)
        hfss = Hfss(
            project=str(args.archive_copy),
            design="HFSSDesign1",
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            close_on_exit=True,
            remove_lock=False,
        )
        working_copy_active = True
        phase_names = [f"cv_phase_{index:02d}" for index in range(1, 17)]
        initial_phase = [float(row["phase_deg"]) for row in source_rows]
        for name, phase in zip(phase_names, initial_phase):
            hfss[name] = f"{phase:.12g}deg"
        sources = {
            row["source_name"]: (f'{float(row["power_w"]):.12g}W', phase_names[index])
            for index, row in enumerate(source_rows)
        }
        if not hfss.edit_sources(sources, include_port_post_processing=True):
            raise RuntimeError("Variable-driven edit_sources returned False")
        initial_gain = read_target_gains(hfss, case)

        setup_name = "CV_O3_PHASE_ONLY"
        for existing in list(hfss.optimizations.setups):
            if existing.name == setup_name:
                existing.delete()
        common = {
            "solution": "Setup2 : LastAdaptive",
            "context": "Infinite Sphere1",
            "report_type": "Far Fields",
            "condition": ">=",
            "goal_value": args.goal_db,
            "goal_weight": 1,
        }
        ranges = lambda index: {
            "Freq": "47GHz",
            "Theta": f'{float(case["hfss_theta_deg"][index]):.12g}deg',
            "Phi": f'{float(case["hfss_phi_deg"][index]):.12g}deg',
        }
        setup = hfss.optimizations.add(
            calculation="dB(RealizedGainTotal)",
            ranges=ranges(0),
            variables=phase_names,
            optim_type="Optimization",
            parametricname=setup_name,
            **common,
        )
        if not setup:
            raise RuntimeError("Could not create HFSS Optimetrics setup")
        for index in range(1, 4):
            if not setup.add_goal(
                calculation="dB(RealizedGainTotal)",
                ranges=ranges(index),
                variables=phase_names,
                **common,
            ):
                raise RuntimeError(f"Could not add goal {index + 1}")
        for name, phase in zip(phase_names, initial_phase):
            if not setup.add_variation(
                name,
                -180.0,
                180.0,
                starting_point=phase,
                min_step=0.25,
                max_step=30.0,
            ):
                raise RuntimeError(f"Could not add phase variation {name}")
        setup.props["Optimizer"] = "Quasi Newton"
        setup.props["CostFuncNormType"] = "L2"
        setup.props["AnalysisStopOptions"]["MaxNumIteration"] = args.iterations
        setup.props["AnalysisStopOptions"]["MinNumIteration"] = min(5, args.iterations)
        setup.props["AnalysisStopOptions"]["MaxSolTimeInSec"] = 1800
        setup.props["AnalysisStopOptions"]["StopForSlowImprovement"] = True
        setup.props["UpdateDesignWhenDone"] = True
        if not setup.update():
            raise RuntimeError("Could not update HFSS Optimetrics setup")
        hfss.save_project()
        analyzed = bool(setup.analyze(cores=2, tasks=1, gpus=0))
        if not analyzed:
            raise RuntimeError("HFSS Optimetrics analyze returned False")
        final_gain = read_target_gains(hfss, case)
        final_phase_expressions = {name: str(hfss[name]) for name in phase_names}
        table_path = args.output_dir / "optimetrics_table.csv"
        table_exported = False
        try:
            table_exported = bool(setup.export_to_csv(str(table_path)))
        except Exception:
            table_exported = False
        hfss.save_project()
        report = {
            "passed": True,
            "aedt_version": "2025.2",
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "restored_environment": restored,
            "source_project": str(args.project),
            "archive_copy": str(args.archive_copy),
            "working_project": str(hfss.project_file),
            "setup": setup_name,
            "optimizer": "HFSS Optimetrics Quasi Newton",
            "objective_note": "Four equal-weight dB(RealizedGainTotal)>=goal goals; independent proxy for max-min, not an exact min operator.",
            "goal_db": args.goal_db,
            "maximum_iterations": args.iterations,
            "initial_source": "O3 phase vector",
            "initial_target_gain_db": initial_gain,
            "initial_gmin_db": min(initial_gain),
            "final_target_gain_db": final_gain,
            "final_gmin_db": min(final_gain),
            "gmin_change_db": min(final_gain) - min(initial_gain),
            "offline_o3_target_gain_db": case["target_gain_db"],
            "offline_o3_gmin_db": min(case["target_gain_db"]),
            "final_phase_expressions": final_phase_expressions,
            "optimetrics_table_exported": table_exported,
            "optimetrics_table": str(table_path) if table_exported else None,
        }
    except Exception as exc:
        report = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        (args.output_dir / "optimetrics_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if hfss is not None:
            try:
                hfss.close_project(save=working_copy_active)
            except Exception:
                pass
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
