"""Create the frozen V2.13 Gate-A HFSS command set and source CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PORTS = 16
SEED = 21313


def _normalise(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def build_commands(seed: int = SEED) -> list[tuple[str, str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    commands: list[tuple[str, str, np.ndarray]] = []
    base = np.full(PORTS, 0.25 + 0.0j, dtype=np.complex128)
    commands.append(("H00", "uniform_phase_control_0deg", base))
    commands.append(("H90", "uniform_phase_control_90deg", 1j * base))
    for index in range(4):
        phase = rng.uniform(-math.pi, math.pi, PORTS)
        commands.append(
            (f"P{index + 1:02d}", "random_continuous_phase_only", 0.25 * np.exp(1j * phase))
        )
    for index in range(2):
        magnitude = rng.uniform(0.35, 1.0, PORTS)
        phase = rng.uniform(-math.pi, math.pi, PORTS)
        commands.append(
            (f"A{index + 1:02d}", "random_amplitude_phase", _normalise(magnitude * np.exp(1j * phase)))
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    commands_dir = args.output_root / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    records = []
    command_matrix = []
    for command_id, family, weights in build_commands(args.seed):
        if not np.isclose(np.vdot(weights, weights).real, 1.0, atol=1e-12):
            raise RuntimeError(f"{command_id} is not unit L2 norm")
        path = commands_dir / f"{command_id}_sources.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["command_id", "family", "port_index", "source_name", "power_w", "phase_deg", "amplitude_coefficient"]
            )
            for port, value in enumerate(weights, 1):
                writer.writerow(
                    [command_id, family, port, f"{port}:1", f"{abs(value) ** 2:.17g}", f"{np.angle(value, deg=True):.17g}", f"{abs(value):.17g}"]
                )
        records.append(
            {
                "command_id": command_id,
                "family": family,
                "source_csv": str(path.resolve()),
                "reuse_existing_direct_report": command_id == "H00",
                "l2_norm": float(np.linalg.norm(weights)),
                "total_power_w": float(np.vdot(weights, weights).real),
            }
        )
        command_matrix.append(weights)
    matrix = np.asarray(command_matrix)
    gram = np.abs(matrix @ matrix.conj().T)
    np.fill_diagonal(gram, 0.0)
    manifest = {
        "schema": "v213-hfss-gate-a-command-set-1",
        "seed": args.seed,
        "command_count": len(records),
        "port_count": PORTS,
        "frequency_hz": 47.0e9,
        "setup": "Setup2 : LastAdaptive",
        "far_field_setup": "Infinite Sphere1",
        "receiver_component": "rEPhi",
        "normalisation": "sum_n |w_n|^2 = 1 W; HFSS edit_sources receives power_w and phase_deg",
        "purpose": "independent HFSS report-path validation of Ephi(q,w)=sum_n w_n Ephi(q,e_n)",
        "not_for_training": True,
        "command_cross_coherence_max": float(gram.max()),
        "commands": records,
    }
    (args.output_root / "command_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"command_count": len(records), "max_cross_coherence": float(gram.max())}, indent=2))


if __name__ == "__main__":
    main()
