"""Run Gate-B synthetic power-noise sensitivity on validation directions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))

from v212.codebook import MeasurementCodebook  # noqa: E402
from v213.gate_b import (  # noqa: E402
    recover_gate_b_rank1,
    synthetic_power_repeats,
)


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-30)))


def _ratio_report(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    ratio = numerator / np.maximum(denominator, 1e-30)
    aggregate = float(numerator.sum() / max(denominator.sum(), 1e-30))
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        selected = generator.integers(0, len(ratio), size=len(ratio))
        bootstrap[index] = numerator[selected].sum() / max(denominator[selected].sum(), 1e-30)
    return {
        "aggregate_nmse": aggregate,
        "aggregate_nmse_db": _db(aggregate),
        "aggregate_nmse_db_bootstrap_95ci": [
            _db(float(np.quantile(bootstrap, 0.025))),
            _db(float(np.quantile(bootstrap, 0.975))),
        ],
        "per_direction_nmse_db": {
            "p05": _db(float(np.quantile(ratio, 0.05))),
            "median": _db(float(np.median(ratio))),
            "p95": _db(float(np.quantile(ratio, 0.95))),
            "maximum": _db(float(np.max(ratio))),
        },
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _failure_angles(
    direction: np.ndarray,
    operator_ratio: np.ndarray,
    power_ratio: np.ndarray,
    jacobian_condition: np.ndarray,
    *,
    cell_deg: float = 10.0,
    minimum_count: int = 5,
) -> list[dict[str, float | int | list[float]]]:
    theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0]))
    theta_bin = np.floor(theta / cell_deg).astype(int)
    phi_bin = np.floor((phi + 180.0) / cell_deg).astype(int)
    rows: list[dict[str, float | int | list[float]]] = []
    for left, right in sorted(set(zip(theta_bin.tolist(), phi_bin.tolist()))):
        mask = (theta_bin == left) & (phi_bin == right)
        count = int(mask.sum())
        if count < minimum_count:
            continue
        operator_db = 10.0 * np.log10(np.maximum(operator_ratio[mask], 1e-30))
        power_db = 10.0 * np.log10(np.maximum(power_ratio[mask], 1e-30))
        rows.append(
            {
                "theta_deg": [left * cell_deg, (left + 1) * cell_deg],
                "phi_deg": [right * cell_deg - 180.0, (right + 1) * cell_deg - 180.0],
                "count": count,
                "operator_nmse_db_median": float(np.median(operator_db)),
                "operator_nmse_db_p95": float(np.quantile(operator_db, 0.95)),
                "heldout_power_nmse_db_median": float(np.median(power_db)),
                "noise_weighted_jacobian_condition_median": float(np.median(jacobian_condition[mask])),
                "noise_weighted_jacobian_condition_p95": float(np.quantile(jacobian_condition[mask], 0.95)),
                "operator_worse_than_minus10_db_fraction": float(np.mean(operator_db > -10.0)),
                "operator_worse_than_minus20_db_fraction": float(np.mean(operator_db > -20.0)),
            }
        )
    rows.sort(key=lambda row: float(row["operator_nmse_db_p95"]), reverse=True)
    return rows[:10]


def _field_strength_bins(clean_power: np.ndarray, operator_ratio: np.ndarray) -> dict[str, object]:
    rms = np.sqrt(np.mean(clean_power**2, axis=1))
    relative_db = 10.0 * np.log10(np.maximum(rms / max(float(rms.max()), 1e-30), 1e-30))
    definitions = {
        "minus3_to_0_db": (relative_db > -3.0),
        "minus10_to_minus3_db": (relative_db > -10.0) & (relative_db <= -3.0),
        "minus20_to_minus10_db": (relative_db > -20.0) & (relative_db <= -10.0),
        "below_minus20_db": relative_db <= -20.0,
    }
    report: dict[str, object] = {}
    operator_db = 10.0 * np.log10(np.maximum(operator_ratio, 1e-30))
    for name, mask in definitions.items():
        if not np.any(mask):
            report[name] = {"count": 0}
        else:
            report[name] = {
                "count": int(mask.sum()),
                "operator_nmse_db_median": float(np.median(operator_db[mask])),
                "operator_nmse_db_p95": float(np.quantile(operator_db[mask], 0.95)),
                "operator_worse_than_minus10_db_fraction": float(np.mean(operator_db[mask] > -10.0)),
            }
    return report


def _plot(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    models = list(report["noise_models"])
    colours = {models[0]: "#1f77b4", models[1]: "#ff7f0e", models[2]: "#2ca02c"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for model in models:
        rows = [row for row in report["scenarios"] if row["noise_model"] == model]
        rows.sort(key=lambda row: row["snr_db"])
        snr = [row["snr_db"] for row in rows]
        axes[0].plot(snr, [row["operator"]["aggregate_nmse_db"] for row in rows], marker="o", label=model, color=colours[model])
        axes[1].plot(snr, [row["heldout_validation_power_vs_clean"]["aggregate_nmse_db"] for row in rows], marker="o", label=model, color=colours[model])
        axes[2].plot(snr, [100.0 * row["failure_fractions"]["operator_worse_than_minus10_db"] for row in rows], marker="o", label=model, color=colours[model])
    axes[0].set_ylabel("Operator NMSE (dB)")
    axes[1].set_ylabel("Held-out power NMSE (dB)")
    axes[2].set_ylabel("Directions worse than -10 dB (%)")
    for axis in axes:
        axis.set_xlabel("Power-domain SNR (dB)")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.suptitle("V2.13 Gate B: ideal-synthetic-only sensitivity")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-truth", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[20.0, 30.0, 40.0])
    parser.add_argument("--measurement-repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=21320)
    parser.add_argument("--query-limit", type=int)
    args = parser.parse_args()
    if args.query_limit is not None and args.query_limit <= 0:
        raise ValueError("query-limit must be positive")

    codebook = MeasurementCodebook.load(args.codebook)
    with np.load(args.complex_truth, allow_pickle=False) as archive:
        validation_index = np.asarray(archive["query_validation"], dtype=np.int64)
        if args.query_limit is not None:
            selected = np.linspace(0, len(validation_index) - 1, args.query_limit, dtype=np.int64)
            validation_index = validation_index[selected]
        direction = np.asarray(archive["direction"], dtype=np.float32)[validation_index]
        g = np.asarray(archive["g"], dtype=np.complex64)[validation_index]
    truth_g = torch.from_numpy(g.copy())
    active_index = torch.cat((codebook.indices("train"), codebook.indices("validation")))
    active_commands = codebook.weights[active_index]
    clean_power = torch.einsum("qn,cn->qc", truth_g, active_commands).abs().square()

    noise_models = (
        "additive_gaussian_power",
        "multiplicative_lognormal_power",
        "mixed_equal_variance_power",
    )
    scenarios: list[dict[str, object]] = []
    scenario_index = 0
    for noise_model in noise_models:
        for snr_db in args.snr_db:
            scenario_seed = args.seed + 1000 * scenario_index
            repeats = synthetic_power_repeats(
                clean_power,
                noise_model=noise_model,
                snr_db=snr_db,
                repeats=args.measurement_repeats,
                seed=scenario_seed,
            )
            recovery = recover_gate_b_rank1(
                repeats,
                codebook,
                truth_g=truth_g,
                clean_validation_power=clean_power[:, 64:80],
                steps=args.steps,
                batch_size=args.batch_size,
                seed=scenario_seed + 1,
            )
            assert recovery.operator_error_numerator is not None
            assert recovery.operator_error_denominator is not None
            assert recovery.clean_power_error_numerator is not None
            assert recovery.clean_power_error_denominator is not None
            operator_num = recovery.operator_error_numerator.numpy()
            operator_den = recovery.operator_error_denominator.numpy()
            power_num = recovery.clean_power_error_numerator.numpy()
            power_den = recovery.clean_power_error_denominator.numpy()
            observed_num = recovery.observed_power_error_numerator.numpy()
            observed_den = recovery.observed_power_error_denominator.numpy()
            operator_ratio = operator_num / np.maximum(operator_den, 1e-30)
            power_ratio = power_num / np.maximum(power_den, 1e-30)
            scenario = {
                "noise_model": noise_model,
                "snr_db": float(snr_db),
                "synthetic_repeat_count": args.measurement_repeats,
                "operator": _ratio_report(
                    operator_num,
                    operator_den,
                    seed=scenario_seed + 2,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "heldout_validation_power_vs_clean": _ratio_report(
                    power_num,
                    power_den,
                    seed=scenario_seed + 3,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "heldout_validation_power_vs_noisy_repeat_mean": _ratio_report(
                    observed_num,
                    observed_den,
                    seed=scenario_seed + 4,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "jacobian_condition_unweighted": _percentiles(
                    recovery.jacobian_condition_unweighted.numpy()
                ),
                "jacobian_condition_noise_weighted": _percentiles(
                    recovery.jacobian_condition_noise_weighted.numpy()
                ),
                "minimum_jacobian_singular_value_noise_weighted": _percentiles(
                    recovery.minimum_jacobian_singular_value_noise_weighted.numpy()
                ),
                "failure_fractions": {
                    "operator_worse_than_minus10_db": float(np.mean(operator_ratio > 0.1)),
                    "operator_worse_than_minus20_db": float(np.mean(operator_ratio > 0.01)),
                    "heldout_power_worse_than_minus20_db": float(np.mean(power_ratio > 0.01)),
                },
                "worst_10deg_angle_cells": _failure_angles(
                    direction,
                    operator_ratio,
                    power_ratio,
                    recovery.jacobian_condition_noise_weighted.numpy(),
                ),
                "field_strength_failure_bins": _field_strength_bins(
                    clean_power.numpy(), operator_ratio
                ),
            }
            scenarios.append(scenario)
            print(
                json.dumps(
                    {
                        "noise_model": noise_model,
                        "snr_db": snr_db,
                        "operator_nmse_db": scenario["operator"]["aggregate_nmse_db"],
                        "power_nmse_db": scenario["heldout_validation_power_vs_clean"]["aggregate_nmse_db"],
                    }
                ),
                flush=True,
            )
            scenario_index += 1

    report = {
        "schema": "v213-gate-b-synthetic-noise-audit-1",
        "gate_b_status": "ideal_synthetic_only_hardware_blocked",
        "hardware_identifiability_claim": False,
        "external_test_used": False,
        "new_model_trained": False,
        "cnn_evaluated": False,
        "source": str(args.complex_truth.resolve()),
        "truth": "privileged HFSS complex g; never exposed as an amplitude-only measurement",
        "direction_split": "validation only",
        "query_count": len(direction),
        "commands": {
            "recovery": "64 frozen train commands",
            "heldout_evaluation": "16 frozen validation commands",
            "external_test": "no test observations loaded or evaluated; test command weights excluded",
        },
        "noise_models": list(noise_models),
        "snr_definition": (
            "20*log10(RMS clean power / RMS power-domain noise); mixed assigns equal variance "
            "to additive Gaussian and multiplicative lognormal components"
        ),
        "synthetic_repeat_count": args.measurement_repeats,
        "optimization_steps": args.steps,
        "bootstrap_confidence": 0.95,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "scenarios": scenarios,
        "claim_boundary": (
            "The scan is a measurement-design and failure-boundary baseline only. Synthetic "
            "noise drawn from one HFSS g cannot establish hardware amplitude-only identifiability."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "GATE_B_SYNTHETIC_NOISE_AUDIT.json"
    plot_path = args.output_dir / "GATE_B_SYNTHETIC_NOISE_AUDIT.png"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(report, plot_path)
    print(json.dumps({"report": str(json_path), "plot": str(plot_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
