"""Paired ten-seed V2.12-S analysis without opening the external test."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


SEEDS = tuple(range(21201, 21211))


def _run_dir(
    screen: Path,
    soft: Path,
    formal: Path,
    candidate_fix: Path | None,
    model: str,
    seed: int,
) -> Path:
    root = screen if seed <= 21203 and model != "G2-J-sep0.001" else formal
    if model == "G1-M5":
        name = f"G1-M5-LR240_seed{seed}"
    elif model == "G2-J":
        name = f"G2-J-LR240_seed{seed}"
    elif model == "G2-J-sep0.001":
        root = soft if seed <= 21203 else formal
        if seed >= 21209 and candidate_fix is not None:
            root = candidate_fix
        name = f"G2-J-LR240_sep0p001_seed{seed}"
    else:
        raise ValueError(model)
    return root / name


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run: Path) -> dict:
    result = _read(run / "result.json")
    if result.get("test_status") != "sealed":
        raise ValueError(f"external test is not sealed: {run}")
    decomposition = _read(run / "decomposition_audit.json")
    operator = _read(run / "operator_audit" / "REPORT.json")
    if operator.get("test_split_opened") is not False:
        raise ValueError(f"operator audit opened test: {run}")
    branch_path = run / "branch_reliance.json"
    branch = _read(branch_path) if branch_path.exists() else None
    nested = operator["nested_filtration"]
    return {
        "nmse_db": float(result["validation_metrics"]["nmse_db"]),
        "mainlobe_mae_db": float(
            result["validation_metrics"]["field_strength_bins"]["-3_to_0_db"]["mae_db"]
        ),
        "parameters": int(result["learnable_parameter_count"]),
        "leakage": decomposition["fourier_low_filtration_leakage"],
        "d_spec": float(nested["spectral_distance_to_truth"]),
        "gradient_error": abs(
            float(nested["normalized_gradient_energy"])
            - float(nested["truth_normalized_gradient_energy"])
        ),
        "graph_error": abs(float(operator["graph_roughness"]["relative_error"])),
        "nested_condition": float(nested["transformed_condition"]),
        "q4_nmse_db": None if branch is None else float(
            branch["angular_distance_quartiles"]["Q4"]["nmse_db"]
        ),
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 21290) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(200_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _paired_permutation_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def _holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * raw[name]))
        adjusted[name] = running
    return adjusted


def _comparison(candidate: list[dict], reference: list[dict], metric: str) -> dict:
    # Positive values mean the candidate is better for all error metrics here.
    delta = np.asarray([r[metric] - c[metric] for c, r in zip(candidate, reference)])
    return {
        "paired_improvements": delta.tolist(),
        "mean_improvement": float(delta.mean()),
        "bootstrap_95_ci": _bootstrap_ci(delta),
        "two_sided_exact_permutation_p": _paired_permutation_p(delta),
        "wins": int(np.sum(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--soft-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--candidate-fix-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = {
        model: [
            _load_run(_run_dir(
                args.screen_root,
                args.soft_root,
                args.formal_root,
                args.candidate_fix_root,
                model,
                seed,
            ))
            for seed in SEEDS
        ]
        for model in ("G1-M5", "G2-J", "G2-J-sep0.001")
    }
    if len({row["parameters"] for rows in records.values() for row in rows}) != 1:
        raise ValueError("formal controls do not have exactly matched parameter counts")

    summary = {}
    for model, rows in records.items():
        summary[model] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("nmse_db", "mainlobe_mae_db", "leakage", "d_spec", "gradient_error", "graph_error")
        }
        summary[model]["nested_condition_max"] = float(max(row["nested_condition"] for row in rows))

    candidate = records["G2-J-sep0.001"]
    comparisons = {
        "vs_G1-M5_nmse": _comparison(candidate, records["G1-M5"], "nmse_db"),
        "vs_G2-J_nmse": _comparison(candidate, records["G2-J"], "nmse_db"),
        "vs_G1-M5_d_spec": _comparison(candidate, records["G1-M5"], "d_spec"),
        "vs_G1-M5_gradient_error": _comparison(candidate, records["G1-M5"], "gradient_error"),
        "vs_G1-M5_graph_error": _comparison(candidate, records["G1-M5"], "graph_error"),
    }
    if all(row["q4_nmse_db"] is not None for rows in records.values() for row in rows):
        comparisons["vs_G1-M5_q4_nmse"] = _comparison(candidate, records["G1-M5"], "q4_nmse_db")

    raw_p = {
        name: value["two_sided_exact_permutation_p"]
        for name, value in comparisons.items()
        if name in {"vs_G1-M5_nmse", "vs_G2-J_nmse"}
    }
    holm = _holm(raw_p)
    for name, value in holm.items():
        comparisons[name]["holm_adjusted_p"] = value

    predictive = comparisons["vs_G1-M5_nmse"]
    base = comparisons["vs_G2-J_nmse"]
    level1 = predictive["bootstrap_95_ci"][0] > 0 and predictive["wins"] >= 8
    level2 = bool(
        level1
        and summary["G2-J-sep0.001"]["leakage"] <= 0.715
        and base["bootstrap_95_ci"][0] >= -0.1
    )
    condition_ok = all(value["nested_condition_max"] <= 1e4 for value in summary.values())
    d_spec = comparisons["vs_G1-M5_d_spec"]
    d_spec_relative = d_spec["mean_improvement"] / max(summary["G1-M5"]["d_spec"], 1e-15)
    q4_ok = "vs_G1-M5_q4_nmse" in comparisons and comparisons["vs_G1-M5_q4_nmse"]["bootstrap_95_ci"][0] > 0
    grad_relative = comparisons["vs_G1-M5_gradient_error"]["mean_improvement"] / max(
        summary["G1-M5"]["gradient_error"], 1e-15
    )
    graph_relative = comparisons["vs_G1-M5_graph_error"]["mean_improvement"] / max(
        summary["G1-M5"]["graph_error"], 1e-15
    )
    level3 = bool(
        level2 and condition_ok
        and d_spec_relative >= 0.10 and d_spec["wins"] >= 8 and q4_ok
        and grad_relative >= -0.05 and graph_relative >= -0.05
        and max(grad_relative, graph_relative) >= 0.10
    )
    payload = {
        "schema": "v212s-ten-seed-formal-1",
        "seeds": list(SEEDS),
        "external_test_used": False,
        "summary": summary,
        "comparisons": comparisons,
        "criteria": {
            "level1_ssh_predictive_value": level1,
            "level2_effective_branch_separation": level2,
            "level3_smooth_angular_inductive_bias": level3,
            "physical_audit_condition_ok": condition_ok,
            "d_spec_relative_improvement": d_spec_relative,
            "gradient_error_relative_improvement": grad_relative,
            "graph_error_relative_improvement": graph_relative,
            "q4_ci_positive": q4_ok,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
