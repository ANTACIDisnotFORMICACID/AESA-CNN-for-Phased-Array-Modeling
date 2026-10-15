"""Summarize the sealed three-seed V2.12-S spatial-gap diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (21201, 21202, 21203)
NAMES = {
    "G1-M5": "G1-M5-LR240_seed{seed}_gap",
    "G2-J": "G2-J-LR240_seed{seed}_gap",
    "G2-J-sep0.001": "G2-J-LR240_sep0p001_seed{seed}_gap",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records: dict[str, list[dict]] = {}
    for model, template in NAMES.items():
        rows = []
        for seed in SEEDS:
            result = json.loads(
                (args.runs / template.format(seed=seed) / "result.json").read_text(encoding="utf-8")
            )
            if result.get("test_status") != "sealed":
                raise ValueError(f"external test is not sealed for {model}, seed {seed}")
            if int(result["learnable_parameter_count"]) != 29142:
                raise ValueError(f"parameter mismatch for {model}, seed {seed}")
            metric = result["validation_metrics"]
            rows.append({
                "seed": seed,
                "nmse_db": float(metric["nmse_db"]),
                "mae_db": float(metric["mae_db"]),
                "p95_abs_db": float(metric["p95_abs_db"]),
                "best_epoch": int(result["best_epoch"]),
            })
        records[model] = rows

    def comparison(reference: str) -> dict:
        delta = np.asarray([
            ref["nmse_db"] - candidate["nmse_db"]
            for ref, candidate in zip(records[reference], records["G2-J-sep0.001"])
        ])
        return {
            "paired_improvement_db": delta.tolist(),
            "mean_improvement_db": float(delta.mean()),
            "wins": int(np.sum(delta > 0)),
        }

    payload = {
        "schema": "v212s-spatial-gap-three-seed-1",
        "seeds": list(SEEDS),
        "external_test_used": False,
        "records": records,
        "mean": {
            model: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("nmse_db", "mae_db", "p95_abs_db")
            }
            for model, rows in records.items()
        },
        "candidate_vs_G1-M5": comparison("G1-M5"),
        "candidate_vs_G2-J": comparison("G2-J"),
    }
    payload["passes_gap_rule_vs_G1-M5"] = bool(
        payload["candidate_vs_G1-M5"]["mean_improvement_db"] >= 0.2
        and payload["candidate_vs_G1-M5"]["wins"] == 3
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
