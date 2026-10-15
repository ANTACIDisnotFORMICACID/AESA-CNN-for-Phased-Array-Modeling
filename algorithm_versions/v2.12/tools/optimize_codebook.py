"""Run diagnostic O1 optimization without changing the frozen baseline codebook."""

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

from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.codebook_optimization import (  # noqa: E402
    information_metrics,
    optimize_supplemental_commands,
)
from v212.audits import rank_one_q_matrix  # noqa: E402
from v212.inversion import graph_synchronization_initialization, recover_rank_one  # noqa: E402


def _summary(g: torch.Tensor, commands: torch.Tensor) -> dict[str, float]:
    chunks_logdet: list[torch.Tensor] = []
    chunks_minimum: list[torch.Tensor] = []
    chunks_condition: list[torch.Tensor] = []
    for start in range(0, len(g), 512):
        logdet, minimum, condition = information_metrics(
            g[start : start + 512], commands, None
        )
        chunks_logdet.append(logdet.detach().cpu())
        chunks_minimum.append(minimum.detach().cpu())
        chunks_condition.append(condition.detach().cpu())
    logdet = torch.cat(chunks_logdet)
    minimum = torch.cat(chunks_minimum)
    condition = torch.cat(chunks_condition)
    return {
        "minimum_p05": float(torch.quantile(minimum, 0.05)),
        "minimum_median": float(minimum.median()),
        "mean_logdet": float(logdet.mean()),
        "condition_median": float(condition.median()),
        "condition_p95": float(torch.quantile(condition, 0.95)),
    }


def _monte_carlo_nmse_db(
    g: torch.Tensor,
    codebook: MeasurementCodebook,
    *,
    snr_db: float | None,
    trials: int,
    steps: int,
    seed: int,
    repeat_noise_standard_deviation: torch.Tensor | None = None,
) -> float:
    generator = torch.Generator(device=g.device).manual_seed(seed)
    commands = codebook.weights.to(g.device)
    values: list[float] = []
    for direction_index, true_g in enumerate(g):
        noiseless = (commands @ true_g).abs().square()
        if snr_db is None:
            if repeat_noise_standard_deviation is None:
                raise ValueError("repeat noise is required when snr_db is None")
            noise_std = repeat_noise_standard_deviation[direction_index].to(g.device)
        else:
            scalar = noiseless.square().mean().sqrt() / (10.0 ** (snr_db / 20.0))
            noise_std = torch.ones_like(noiseless) * scalar
        for trial in range(trials):
            noise = torch.randn(
                len(commands), generator=generator, device=g.device, dtype=noiseless.dtype
            ) * noise_std
            measured = (noiseless + noise).clamp_min(0)
            initial = graph_synchronization_initialization(measured, codebook)
            recovered = recover_rank_one(
                commands,
                measured,
                steps=steps,
                seed=seed + 1000 * direction_index + trial,
                initial_g=initial,
            )
            target_q = rank_one_q_matrix(true_g)
            error = (recovered.q_matrix - target_q).abs().square().sum()
            scale = target_q.abs().square().sum().clamp_min(1e-12)
            values.append(float(10.0 * torch.log10((error / scale).clamp_min(1e-12))))
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-audit", required=True, type=Path)
    parser.add_argument("--baseline-codebook", required=True, type=Path)
    parser.add_argument("--power-repeats", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=21310)
    parser.add_argument("--mc-directions", type=int, default=8)
    parser.add_argument("--mc-trials", type=int, default=5)
    parser.add_argument("--inversion-steps", type=int, default=200)
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with np.load(args.complex_audit, allow_pickle=False) as archive:
        g = torch.from_numpy(np.asarray(archive["g"], dtype=np.complex64).copy()).to(args.device)
        q_train = torch.from_numpy(np.asarray(archive["query_train"], dtype=np.int64))
        q_validation = torch.from_numpy(np.asarray(archive["query_validation"], dtype=np.int64))
        q_test = torch.from_numpy(np.asarray(archive["query_test"], dtype=np.int64))
    baseline = MeasurementCodebook.load(args.baseline_codebook)
    result = optimize_supplemental_commands(
        baseline,
        g[q_train.to(args.device)],
        g[q_validation.to(args.device)],
        restarts=args.restarts,
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.codebook.save(args.output_dir / "optimized_codebook.json")
    report = {
        "schema": "v212-o1-codebook-optimization-1",
        "diagnostic_only": True,
        "selected_restart": result.restart,
        "validation": {
            "minimum_p05": result.validation_minimum_p05,
            "minimum_median": result.validation_median_minimum,
            "mean_logdet": result.validation_mean_logdet,
            "condition_median": result.validation_condition_median,
            "condition_p95": result.validation_condition_p95,
        },
        "test_split_opened": bool(args.open_test),
        "validation_baseline": _summary(
            g[q_validation.to(args.device)], baseline.weights.to(args.device)
        ),
    }
    baseline_validation = report["validation_baseline"]
    p05_gain = (
        result.validation_minimum_p05 / max(baseline_validation["minimum_p05"], 1e-12) - 1.0
    )
    median_gain = (
        result.validation_median_minimum
        / max(baseline_validation["minimum_median"], 1e-12)
        - 1.0
    )
    report["validation_conditioning_gate"] = {
        "minimum_p05_relative_gain": p05_gain,
        "minimum_median_relative_gain": median_gain,
        "passes_p05_20_percent_and_median_non_degradation": p05_gain >= 0.20
        and median_gain >= 0.0,
    }
    if args.open_test:
        test_g = g[q_test.to(args.device)]
        report["test_baseline"] = _summary(test_g, baseline.weights.to(args.device))
        report["test_optimized"] = _summary(test_g, result.codebook.weights.to(args.device))
        if args.mc_directions <= 0 or args.mc_trials <= 0 or args.inversion_steps <= 0:
            raise ValueError("Monte Carlo settings must be positive")
        selection_generator = torch.Generator().manual_seed(args.seed + 99)
        selected = torch.randperm(len(test_g), generator=selection_generator)[
            : min(args.mc_directions, len(test_g))
        ].to(test_g.device)
        mc_g = test_g[selected]
        monte_carlo: dict[str, dict[str, float]] = {}
        improvements: list[float] = []
        for snr_db in (20.0, 30.0, 40.0):
            baseline_nmse = _monte_carlo_nmse_db(
                mc_g,
                baseline,
                snr_db=snr_db,
                trials=args.mc_trials,
                steps=args.inversion_steps,
                seed=args.seed + int(snr_db),
            )
            optimized_nmse = _monte_carlo_nmse_db(
                mc_g,
                result.codebook,
                snr_db=snr_db,
                trials=args.mc_trials,
                steps=args.inversion_steps,
                seed=args.seed + int(snr_db),
            )
            improvement = baseline_nmse - optimized_nmse
            improvements.append(improvement)
            monte_carlo[str(int(snr_db))] = {
                "baseline_operator_nmse_db": baseline_nmse,
                "optimized_operator_nmse_db": optimized_nmse,
                "improvement_db": improvement,
            }
        if args.power_repeats is not None:
            repeats = np.load(args.power_repeats, allow_pickle=False)
            if repeats.shape[:2] != (len(g), 96) or repeats.shape[2] < 3:
                raise ValueError("power repeats must have shape [Q,96,at least 3]")
            repeat_std = torch.from_numpy(
                np.asarray(repeats[q_test.numpy()][selected.cpu().numpy()].std(axis=-1), dtype=np.float32)
            ).to(args.device)
            if torch.any(repeat_std > 0):
                baseline_repeat = _monte_carlo_nmse_db(
                    mc_g,
                    baseline,
                    snr_db=None,
                    trials=args.mc_trials,
                    steps=args.inversion_steps,
                    seed=args.seed + 91,
                    repeat_noise_standard_deviation=repeat_std,
                )
                optimized_repeat = _monte_carlo_nmse_db(
                    mc_g,
                    result.codebook,
                    snr_db=None,
                    trials=args.mc_trials,
                    steps=args.inversion_steps,
                    seed=args.seed + 91,
                    repeat_noise_standard_deviation=repeat_std,
                )
                repeat_improvement = baseline_repeat - optimized_repeat
                monte_carlo["measured_repeatability"] = {
                    "baseline_operator_nmse_db": baseline_repeat,
                    "optimized_operator_nmse_db": optimized_repeat,
                    "improvement_db": repeat_improvement,
                }
                improvements.append(repeat_improvement)
            else:
                report["repeatability_comparison"] = (
                    "zero_noise_hfss_repeat_baseline_not_used_as_acceptance_gate"
                )
        else:
            report["repeatability_comparison"] = "not_run_no_power_repeats_supplied"
        report["test_monte_carlo"] = monte_carlo
        report["recommend_for_future_measurement"] = bool(
            p05_gain >= 0.20 and median_gain >= 0.0 and min(improvements) >= 0.2
        )
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
