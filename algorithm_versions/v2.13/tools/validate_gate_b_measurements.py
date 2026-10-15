"""Validate a canonical Gate-B package and estimate train-command noise covariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))

from v212.codebook import MeasurementCodebook  # noqa: E402
from v213.gate_b import (  # noqa: E402
    load_gate_b_package,
    pooled_relative_noise_covariance,
    validate_gate_b_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    codebook = MeasurementCodebook.load(args.codebook)
    package = load_gate_b_package(args.package)
    report = validate_gate_b_package(package, codebook)
    output = args.output_dir or args.package
    output.mkdir(parents=True, exist_ok=True)
    if report["valid"]:
        covariance = pooled_relative_noise_covariance(
            package.power_repeats, np.arange(64, dtype=np.int64)
        )
        report["train_command_noise_covariance"] = {
            key: value for key, value in covariance.items() if np.isscalar(value)
        }
        np.savez_compressed(
            output / "TRAIN_COMMAND_NOISE_COVARIANCE.npz",
            pooled_relative_covariance=covariance["pooled_relative_covariance"],
            per_direction_diagonal_variance=covariance["per_direction_diagonal_variance"],
            command_indices=package.command_indices[:64],
        )
    (output / "VALIDATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
