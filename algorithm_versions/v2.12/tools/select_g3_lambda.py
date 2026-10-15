"""Select diagnostic G3 lambda using validation-D only, never external test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v212.statistics import paired_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows: dict[str, dict[str, object]] = {}
    payloads_by_lambda: dict[str, list[dict[str, object]]] = {}
    for lambda_q in (0.01, 0.1, 1.0):
        label = str(lambda_q).replace(".", "p")
        paths = sorted(args.runs_dir.glob(f"G3_lambda{label}_seed*/result.json"))
        if len(paths) != 10:
            raise ValueError(f"lambda={lambda_q} requires exactly 10 completed seeds")
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        payloads_by_lambda[str(lambda_q)] = payloads
        values = np.asarray(
            [payload["best_validation_nmse_db"] for payload in payloads], dtype=np.float64
        )
        rows[str(lambda_q)] = {
            "mean_validation_D_nmse_db": float(values.mean()),
            "median_validation_D_nmse_db": float(np.median(values)),
            "seeds": sorted(int(payload["seed"]) for payload in payloads),
        }
    selected = min(rows, key=lambda key: rows[key]["mean_validation_D_nmse_db"])
    g2_paths = sorted(args.runs_dir.glob("G2_seed*/result.json"))
    if len(g2_paths) != 10:
        raise ValueError("G2 comparison requires exactly 10 completed paired seeds")
    g2_by_seed = {
        int(payload["seed"]): payload
        for payload in (
            json.loads(path.read_text(encoding="utf-8")) for path in g2_paths
        )
    }
    selected_by_seed = {
        int(payload["seed"]): payload for payload in payloads_by_lambda[selected]
    }
    seeds = sorted(set(g2_by_seed).intersection(selected_by_seed))
    if len(seeds) != 10:
        raise ValueError("selected G3 and G2 do not have ten matching seeds")
    comparison = paired_comparison(
        np.asarray([g2_by_seed[seed]["best_validation_nmse_db"] for seed in seeds]),
        np.asarray([selected_by_seed[seed]["best_validation_nmse_db"] for seed in seeds]),
    )
    result = {
        "schema": "v212-g3-lambda-selection-1",
        "selection_metric": "mean validation-D amplitude NMSE dB over ten paired seeds",
        "external_test_used": False,
        "selected_lambda_q": float(selected),
        "candidates": rows,
        "diagnostic_only": True,
        "selected_g3_vs_g2_paired_validation": asdict(comparison),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
