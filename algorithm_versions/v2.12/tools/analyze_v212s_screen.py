"""Apply the frozen three-seed V2.12-S screening rules without opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RECIPES = (
    "G1-LR240", "G1-M5-LR240", "G2-J-LR240", "G2-I-LR240",
    "G2-T", "G2-TI", "SSH-only",
)
CANDIDATES = ("G2-I-LR240", "G2-T", "G2-TI")
SEEDS = (21201, 21202, 21203)


def _load(root: Path, recipe: str, seed: int) -> tuple[dict, dict | None, dict | None]:
    run = root / f"{recipe}_seed{seed}"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    audit_path = run / "operator_audit" / "REPORT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    decomposition_path = run / "decomposition_audit.json"
    decomposition = (
        json.loads(decomposition_path.read_text(encoding="utf-8"))
        if decomposition_path.exists() else None
    )
    return result, audit, decomposition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records: dict[str, list[tuple[dict, dict | None]]] = {}
    for recipe in RECIPES:
        records[recipe] = [_load(args.runs, recipe, seed) for seed in SEEDS]
        if any(result.get("test_status") != "sealed" for result, _, _ in records[recipe]):
            raise ValueError(f"{recipe} contains an unsealed result")

    summary = {}
    for recipe, rows in records.items():
        results = [row[0] for row in rows]
        audits = [row[1] for row in rows]
        decompositions = [row[2] for row in rows]
        summary[recipe] = {
            "mean_validation_nmse_db": float(np.mean([
                float(value["validation_metrics"]["nmse_db"]) for value in results
            ])),
            "mean_mainlobe_mae_db": float(np.mean([
                float(value["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"])
                for value in results
            ])),
            "learnable_parameter_count": sorted({
                int(value["learnable_parameter_count"]) for value in results
            }),
            "mean_fourier_low_filtration_leakage": (
                None if any(
                    value is None or value["fourier_low_filtration_leakage"] is None
                    for value in decompositions
                )
                else float(np.mean([
                    float(value["fourier_low_filtration_leakage"])
                    for value in decompositions
                ]))
            ),
            "mean_nested_spectral_distance": (
                None if any(value is None for value in audits)
                else float(np.mean([
                    float(value["nested_filtration"]["spectral_distance_to_truth"])
                    for value in audits
                ]))
            ),
        }
    matched_count = summary["G1-M5-LR240"]["learnable_parameter_count"]
    ssh_count = summary["G2-J-LR240"]["learnable_parameter_count"]
    if matched_count != ssh_count:
        raise ValueError("G1-M5 and G2-J parameter counts do not match")

    reference_nmse = summary["G2-J-LR240"]["mean_validation_nmse_db"]
    reference_main = summary["G2-J-LR240"]["mean_mainlobe_mae_db"]
    candidate_rules = {}
    for recipe in CANDIDATES:
        row = summary[recipe]
        leakage = row["mean_fourier_low_filtration_leakage"]
        candidate_rules[recipe] = {
            "nmse_improvement_vs_G2_J_LR240_db": reference_nmse - row["mean_validation_nmse_db"],
            "mainlobe_mae_change_db": row["mean_mainlobe_mae_db"] - reference_main,
            "leakage": leakage,
            "passes": bool(
                reference_nmse - row["mean_validation_nmse_db"] >= -0.1
                and row["mean_mainlobe_mae_db"] - reference_main <= 0.05
                and leakage is not None and leakage <= 0.715
            ),
        }
    eligible = [name for name, value in candidate_rules.items() if value["passes"]]
    selected = min(eligible, key=lambda name: summary[name]["mean_validation_nmse_db"]) if eligible else None
    payload = {
        "schema": "v212s-three-seed-screen-1",
        "seeds": list(SEEDS),
        "external_test_used": False,
        "summary": summary,
        "candidate_rules": candidate_rules,
        "selected_candidate": selected,
        "soft_separation_required": selected is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
