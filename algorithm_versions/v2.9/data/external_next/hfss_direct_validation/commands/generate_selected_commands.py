"""Generate reproducible HFSS source tables for the V2.9 direct-validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_grid(row_step: float, col_step: float, offset: float = 0.0) -> np.ndarray:
    row = np.arange(4, dtype=np.float64)[:, None]
    col = np.arange(4, dtype=np.float64)[None, :]
    return np.mod(offset + row_step * row + col_step * col, 360.0).reshape(16)


def equal_amplitude(phase_deg: np.ndarray) -> np.ndarray:
    return 0.25 * np.exp(1j * np.deg2rad(phase_deg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized-npz", type=Path, required=True)
    parser.add_argument("--diagnostic-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    sources = output / "commands" / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    definitions: list[dict[str, object]] = []

    structured = [
        ("A01", "uniform_0", "core", "等幅同相0度", phase_grid(0, 0, 0)),
        ("A02", "uniform_90", "core", "A01公共相位加90度", phase_grid(0, 0, 90)),
        ("A03", "checkerboard_180", "core", "行列棋盘0/180度", phase_grid(180, 180)),
        ("A04", "row_plus_45", "core", "行方向每行增加45度", phase_grid(45, 0)),
        ("A05", "row_minus_45", "extended", "行方向每行减少45度", phase_grid(-45, 0)),
        ("A06", "col_plus_45", "core", "列方向每列增加45度", phase_grid(0, 45)),
        ("A07", "col_minus_45", "extended", "列方向每列减少45度", phase_grid(0, -45)),
        ("A08", "diag_plus_plus_45", "core", "行列方向均增加45度", phase_grid(45, 45)),
        ("A09", "diag_plus_minus_45", "extended", "行加45度列减45度", phase_grid(45, -45)),
        ("A10", "mixed_row90_col45", "extended", "行加90度列加45度", phase_grid(90, 45)),
    ]
    for command_id, slug, priority, description, phase in structured:
        definitions.append(
            {
                "command_id": command_id,
                "slug": slug,
                "group": "structured_equal_amplitude",
                "priority": priority,
                "description": description,
                "weights": equal_amplitude(phase),
                "historical_case": None,
            }
        )

    generator = np.random.default_rng(47029)
    for index in range(1, 7):
        codes = generator.integers(0, 16, size=16)
        phase = 22.5 * codes.astype(np.float64)
        definitions.append(
            {
                "command_id": f"B{index:02d}",
                "slug": f"random_4bit_{index:02d}",
                "group": "random_equal_amplitude_4bit",
                "priority": "core" if index <= 2 else "extended",
                "description": f"固定seed=47029的4-bit等幅随机相位{index}",
                "weights": equal_amplitude(phase),
                "historical_case": None,
            }
        )

    with np.load(args.optimized_npz, allow_pickle=False) as archive:
        optimized = np.asarray(archive["optimized"], dtype=np.complex128)
    diagnostic = json.loads(args.diagnostic_json.read_text(encoding="utf-8"))
    case_by_id = {int(row["case"]): row for row in diagnostic["target_cases"]}
    selected_cases = [2, 14, 26, 22]
    for index, case_id in enumerate(selected_cases, start=1):
        row = case_by_id[case_id]
        weights = optimized[case_id]
        priority = "core" if case_id in (26, 22) else "extended"
        description = (
            f"历史四目标case {case_id}连续幅相; "
            f"regret={100.0 * float(row['relative_regret']):.3f}%; "
            f"最小间距={int(row['minimum_separation_deg'])}度"
        )
        definitions.append(
            {
                "command_id": f"C{index:02d}",
                "slug": f"optimized_case{case_id:02d}_amp_phase",
                "group": "historical_optimized_amplitude_phase",
                "priority": priority,
                "description": description,
                "weights": weights,
                "historical_case": case_id,
            }
        )
        definitions.append(
            {
                "command_id": f"D{index:02d}",
                "slug": f"optimized_case{case_id:02d}_phase_only",
                "group": "historical_optimized_phase_only_projection",
                "priority": priority,
                "description": description + "; 等幅phase-only投影",
                "weights": equal_amplitude(np.mod(np.rad2deg(np.angle(weights)), 360.0)),
                "historical_case": case_id,
            }
        )

    definitions.sort(key=lambda row: str(row["command_id"]))
    summary_fields = [
        "command_id",
        "group",
        "priority",
        "description",
        "historical_case",
        "total_power_w",
        "amplitude_control_required",
        "sources_file",
    ]
    matrix_fields = ["command_id", "group", "priority"]
    for port in range(1, 17):
        matrix_fields.extend([f"port_{port:02d}_power_w", f"port_{port:02d}_phase_deg"])

    summary_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for definition in definitions:
        weights = np.asarray(definition["weights"], dtype=np.complex128).reshape(16)
        amplitude = np.abs(weights)
        power = amplitude**2
        phase = np.mod(np.rad2deg(np.angle(weights)), 360.0)
        total_power = float(power.sum())
        if not np.isclose(total_power, 1.0, atol=2e-6):
            raise ValueError(f"{definition['command_id']} total power is {total_power}")
        source_name = f"cmd_{definition['command_id']}_{definition['slug']}_sources.csv"
        with (sources / source_name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "command_id",
                    "source_name",
                    "port_index",
                    "row",
                    "col",
                    "amplitude_coefficient",
                    "power_w",
                    "phase_deg",
                ],
            )
            writer.writeheader()
            for offset in range(16):
                writer.writerow(
                    {
                        "command_id": definition["command_id"],
                        "source_name": f"{offset + 1}:1",
                        "port_index": offset + 1,
                        "row": offset // 4 + 1,
                        "col": offset % 4 + 1,
                        "amplitude_coefficient": f"{amplitude[offset]:.9f}",
                        "power_w": f"{power[offset]:.9f}",
                        "phase_deg": f"{phase[offset]:.6f}",
                    }
                )
        summary_rows.append(
            {
                "command_id": definition["command_id"],
                "group": definition["group"],
                "priority": definition["priority"],
                "description": definition["description"],
                "historical_case": "" if definition["historical_case"] is None else definition["historical_case"],
                "total_power_w": f"{total_power:.9f}",
                "amplitude_control_required": "yes" if float(amplitude.max() - amplitude.min()) > 1e-7 else "no",
                "sources_file": f"sources/{source_name}",
            }
        )
        matrix_row: dict[str, object] = {
            "command_id": definition["command_id"],
            "group": definition["group"],
            "priority": definition["priority"],
        }
        for offset in range(16):
            matrix_row[f"port_{offset + 1:02d}_power_w"] = f"{power[offset]:.9f}"
            matrix_row[f"port_{offset + 1:02d}_phase_deg"] = f"{phase[offset]:.6f}"
        matrix_rows.append(matrix_row)

    commands_dir = output / "commands"
    with (commands_dir / "selected_commands_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (commands_dir / "selected_commands_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=matrix_fields)
        writer.writeheader()
        writer.writerows(matrix_rows)

    manifest = {
        "schema_version": "v2.9-hfss-direct-validation-commands-1",
        "frequency_hz": 47.0e9,
        "array_shape": [4, 4],
        "port_order": "row-major 1:1..16:1",
        "total_incident_power_w": 1.0,
        "hfss_source_rule": "power_w=abs(w)^2; phase_deg=arg(w)",
        "structured_and_random_equal_amplitude": 0.25,
        "random_phase_bits": 4,
        "random_seed": 47029,
        "selected_historical_cases": selected_cases,
        "command_count": len(definitions),
        "core_count": sum(row["priority"] == "core" for row in definitions),
        "optimized_npz_sha256": sha256(args.optimized_npz),
        "diagnostic_json_sha256": sha256(args.diagnostic_json),
        "commands": summary_rows,
    }
    (commands_dir / "commands_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
