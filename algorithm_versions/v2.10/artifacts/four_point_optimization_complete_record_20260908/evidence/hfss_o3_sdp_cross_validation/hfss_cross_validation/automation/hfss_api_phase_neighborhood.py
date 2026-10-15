"""HFSS-evaluated phase-only neighborhood search around one O3 solution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
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


def read_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16:
        raise ValueError(f"Expected 16 source rows, got {len(rows)}")
    return rows


def wrap_phase(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts-per-scale", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    source_rows = read_source_rows(args.source_csv)
    names = [row["source_name"] for row in source_rows]
    powers = [float(row["power_w"]) for row in source_rows]
    initial_phase = [float(row["phase_deg"]) for row in source_rows]
    scales = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    rng = random.Random(args.seed)
    restored = restore_machine_environment()
    desktop = None
    hfss = None
    history: list[dict[str, object]] = []
    started = time.time()
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

        def evaluate(phases: list[float]) -> list[float]:
            sources = {
                name: (f"{power:.12g}W", f"{phase:.12g}deg")
                for name, power, phase in zip(names, powers, phases)
            }
            if not hfss.edit_sources(sources, include_port_post_processing=True):
                raise RuntimeError("HFSS edit_sources returned False")
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
                if not data or not data.data_real("RealizedGainTotal"):
                    raise RuntimeError("HFSS returned no RealizedGainTotal data")
                linear = float(data.data_real("RealizedGainTotal")[0])
                gains.append(10.0 * math.log10(max(linear, 1e-30)))
            return gains

        best_phase = initial_phase[:]
        best_gain = evaluate(best_phase)
        history.append({"evaluation": 0, "scale_deg": 0.0, "accepted": True, "gmin_db": min(best_gain), "gains_db": best_gain})
        evaluation = 0
        for scale in scales:
            for _ in range(args.attempts_per_scale):
                evaluation += 1
                # Element 1 is fixed because a common phase rotation does not
                # change any gain. Perturb the remaining 15 independent phases.
                candidate = [best_phase[0]] + [
                    wrap_phase(value + rng.gauss(0.0, scale)) for value in best_phase[1:]
                ]
                candidate_gain = evaluate(candidate)
                accepted = min(candidate_gain) > min(best_gain) + 1e-10
                if accepted:
                    best_phase = candidate
                    best_gain = candidate_gain
                history.append(
                    {
                        "evaluation": evaluation,
                        "scale_deg": scale,
                        "accepted": accepted,
                        "gmin_db": min(candidate_gain),
                        "gains_db": candidate_gain,
                    }
                )

        csv_path = args.output_dir / "neighborhood_history.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["evaluation", "scale_deg", "accepted", "gmin_db", "gain_1_db", "gain_2_db", "gain_3_db", "gain_4_db"])
            for row in history:
                writer.writerow([row["evaluation"], row["scale_deg"], row["accepted"], row["gmin_db"], *row["gains_db"]])
        report = {
            "passed": True,
            "method": "HFSS API numeric phase neighborhood search",
            "method_scope": "Independent local check around O3; not HFSS Optimetrics and not a global-optimality proof.",
            "aedt_version": "2025.2",
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "project": str(args.project),
            "restored_environment": restored,
            "seed": args.seed,
            "scales_deg": scales,
            "attempts_per_scale": args.attempts_per_scale,
            "evaluations": len(history),
            "initial_phase_deg": initial_phase,
            "initial_target_gain_db": history[0]["gains_db"],
            "initial_gmin_db": history[0]["gmin_db"],
            "best_phase_deg": best_phase,
            "best_target_gain_db": best_gain,
            "best_gmin_db": min(best_gain),
            "gmin_improvement_db": min(best_gain) - float(history[0]["gmin_db"]),
            "accepted_moves": sum(bool(row["accepted"]) for row in history[1:]),
            "elapsed_seconds": time.time() - started,
            "history_csv": str(csv_path),
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
        (args.output_dir / "neighborhood_report.json").write_text(
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


if __name__ == "__main__":
    main()
