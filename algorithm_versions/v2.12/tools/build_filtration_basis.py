"""Fit and audit L4/L5 filtration SSH transforms from training directions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--degrees", type=int, nargs="+", default=[4, 5])
    args = parser.parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    direction = torch.as_tensor(dataset.direction[split.query_train], dtype=torch.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "v212-filtration-basis-audit-1",
        "training_direction_count": len(direction),
        "measure": "equal empirical loss weights",
        "degrees": {},
    }
    for degree in args.degrees:
        transform = FiltrationSSHTransform.fit(direction, degree)
        transform.save(args.output_dir / f"filtration_ssh_l{degree}.npz")
        report["degrees"][str(degree)] = {
            "raw_channels": (degree + 1) ** 2,
            "retained_channels": transform.output_channels,
            "ranks": list(transform.ranks),
            "discarded_counts": [
                int(vectors.shape[1]) for vectors in transform.discarded_mode_vectors
            ],
            "raw_gram_condition": transform.raw_gram_condition,
            "retained_condition": transform.retained_condition,
            "transformed_condition": transform.transformed_condition,
        }
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
