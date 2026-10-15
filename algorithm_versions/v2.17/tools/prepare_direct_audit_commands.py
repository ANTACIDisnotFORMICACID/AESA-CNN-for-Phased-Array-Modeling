from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.cross_array import ArraySpec, generate_direct_audit_commands  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21740)
    args = parser.parse_args()
    spec = ArraySpec(args.array_id, args.rows, args.cols, 0.015, 0.0068)
    weights, family = generate_direct_audit_commands(spec, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "commands.npz", weights=weights, family=family)
    commands = []
    for index, (weight, name) in enumerate(zip(weights, family)):
        commands.append({
            "command_id": f"D{index:02d}", "family": str(name),
            "sources": {
                spec.port_names[port]: {
                    "power_w": float(abs(value) ** 2),
                    "phase_deg": float(np.rad2deg(np.angle(value))),
                }
                for port, value in enumerate(weight)
            },
        })
    manifest = {
        "schema": "v217-direct-combination-audit-v1", "array_id": args.array_id,
        "seed": args.seed, "normalization": "sum(abs(w)^2)=1", "commands": commands,
        "thresholds": {
            "aggregate_complex_nmse_db_max": -100.0,
            "per_command_complex_nmse_db_max": -80.0,
            "amplitude_max_relative_error_max": 1e-6,
        },
        "not_for_training": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"array_id": args.array_id, "commands": 16, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
