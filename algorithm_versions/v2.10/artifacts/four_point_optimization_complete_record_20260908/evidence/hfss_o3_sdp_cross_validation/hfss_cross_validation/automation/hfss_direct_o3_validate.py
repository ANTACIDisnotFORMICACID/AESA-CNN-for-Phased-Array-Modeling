"""Evaluate one O3 phase vector directly in HFSS at the same four directions."""

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


def read_sources(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16:
        raise ValueError(f"Expected 16 sources, got {len(rows)}")
    return {
        row["source_name"]: (
            f'{float(row["power_w"]):.12g}W',
            f'{float(row["phase_deg"]):.12g}deg',
        )
        for row in rows
    }


def scalar(data, expression: str, imaginary: bool = False) -> float:
    values = data.data_imag(expression) if imaginary else data.data_real(expression)
    if not values:
        raise RuntimeError(f"HFSS returned no data for {expression}")
    return float(values[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    case = json.loads(args.case_json.read_text(encoding="utf-8"))
    sources = read_sources(args.source_csv)
    restored = restore_machine_environment()
    desktop = None
    hfss = None
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
        if set(hfss.excitations) != set(sources):
            raise RuntimeError(f"Excitation mismatch: {list(hfss.excitations)}")
        if not hfss.edit_sources(sources, include_port_post_processing=True):
            raise RuntimeError("HFSS edit_sources returned False")
        points: list[dict[str, float]] = []
        expressions = ["rETheta", "rEPhi", "GainTotal", "RealizedGainTotal"]
        for index, (theta, phi) in enumerate(
            zip(case["hfss_theta_deg"], case["hfss_phi_deg"])
        ):
            data = hfss.post.get_solution_data(
                expressions=expressions,
                setup_sweep_name="Setup2 : LastAdaptive",
                variations={
                    "Freq": ["47GHz"],
                    "Theta": [f"{theta:.12g}deg"],
                    "Phi": [f"{phi:.12g}deg"],
                },
                primary_sweep_variable="Theta",
                report_category="Far Fields",
                context="Infinite Sphere1",
            )
            if not data:
                raise RuntimeError(f"No solution data returned for target {index + 1}")
            # PyAEDT returns the far-field electric-field components in mV for
            # this report. Convert to volts before applying G = |E|^2 / 60.
            et_mv = complex(scalar(data, "rETheta"), scalar(data, "rETheta", True))
            ep_mv = complex(scalar(data, "rEPhi"), scalar(data, "rEPhi", True))
            et = et_mv * 1e-3
            ep = ep_mv * 1e-3
            field_gain = (abs(et) ** 2 + abs(ep) ** 2) / 60.0
            gain = scalar(data, "GainTotal")
            realized = scalar(data, "RealizedGainTotal")
            points.append(
                {
                    "target": index + 1,
                    "theta_deg": float(theta),
                    "phi_deg": float(phi),
                    "field_component_source_unit": "mV",
                    "rETheta_real_v": et.real,
                    "rETheta_imag_v": et.imag,
                    "rEPhi_real_v": ep.real,
                    "rEPhi_imag_v": ep.imag,
                    "field_gain_linear": field_gain,
                    "field_gain_db": 10 * math.log10(max(field_gain, 1e-30)),
                    "GainTotal_linear": gain,
                    "GainTotal_db": 10 * math.log10(max(gain, 1e-30)),
                    "RealizedGainTotal_linear": realized,
                    "RealizedGainTotal_db": 10 * math.log10(max(realized, 1e-30)),
                    "offline_o3_gain_db": float(case["target_gain_db"][index]),
                }
            )
        field_values = [row["field_gain_db"] for row in points]
        gain_values = [row["GainTotal_db"] for row in points]
        realized_values = [row["RealizedGainTotal_db"] for row in points]
        offline_values = case["target_gain_db"]
        report = {
            "passed": True,
            "aedt_version": "2025.2",
            "pyaedt_version": getattr(pyaedt, "__version__", "unknown"),
            "restored_environment": restored,
            "project": str(args.project),
            "case": case,
            "points": points,
            "minimum_db": {
                "offline_o3": min(offline_values),
                "hfss_field_recompute": min(field_values),
                "hfss_GainTotal": min(gain_values),
                "hfss_RealizedGainTotal": min(realized_values),
            },
            "maximum_absolute_point_error_db": {
                "field_recompute_vs_offline": max(abs(a - b) for a, b in zip(field_values, offline_values)),
                "GainTotal_vs_offline": max(abs(a - b) for a, b in zip(gain_values, offline_values)),
                "RealizedGainTotal_vs_offline": max(abs(a - b) for a, b in zip(realized_values, offline_values)),
            },
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


if __name__ == "__main__":
    main()
