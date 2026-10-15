"""Final V2.18 Stage-2 validation audit and engineering gates."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .audit import _branch_diagnostics, load_run
from .canonical import PreparedData, atomic_json
from .experiment import PHASE_AUX_ARMS, SEEDS, architecture_contract
from .stage2 import amplitude_metrics, operator_metrics, predict_model


STAGE2_ARM_ORDER = (
    "control_v212_canonical",
    "control_v212_phase_aux",
    "dcar_gated_pm_canonical",
    "dcar_gated_pm_phase_aux",
)


def _run_dir(version_root: Path, arm: str, seed: int) -> Path:
    folder = "stage2_training_4x4_20261014" if arm in PHASE_AUX_ARMS else "training_4x4_20261014"
    return version_root / "artifacts" / folder / f"{arm}_seed{seed}"


def estimate_macs(model: torch.nn.Module) -> int:
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        value = output[0] if isinstance(output, tuple) else output
        if isinstance(module, torch.nn.Conv2d):
            total += int(value.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size))
        elif isinstance(module, torch.nn.Linear):
            total += int(value.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    try:
        from v210.contracts import QueryBatch
        with torch.no_grad():
            model.operator(QueryBatch(torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)))
    finally:
        for item in hooks:
            item.remove()
    return total


@torch.no_grad()
def latency_profile(model: torch.nn.Module, data: PreparedData) -> dict[str, object]:
    model = model.to("cpu").eval()
    commands = torch.from_numpy(data.train["commands"])
    result: dict[str, object] = {}
    original_threads = torch.get_num_threads()
    try:
        for label, threads, batch in (("single", 1, 1), ("throughput", 12, 256)):
            torch.set_num_threads(threads)
            direction = torch.from_numpy(data.validation["direction"][:batch]).float()
            for _ in range(20):
                model.forward_command_grid(commands, direction)
            samples = []
            for _ in range(100):
                started = time.perf_counter()
                model.forward_command_grid(commands, direction)
                samples.append((time.perf_counter() - started) * 1000)
            result[label] = {
                "threads": threads,
                "batch": batch,
                "median_ms": float(statistics.median(samples)),
                "p95_ms": float(np.quantile(samples, 0.95)),
            }
    finally:
        torch.set_num_threads(original_threads)
    return result


def audit_one(version_root: Path, arm: str, seed: int, data: PreparedData, device: str) -> dict[str, object]:
    run_dir = _run_dir(version_root, arm, seed)
    model, checkpoint = load_run(run_dir, data, device)
    prediction, operator = predict_model(
        model, data.validation["direction"], data.validation["commands"], device=device
    )
    report = {
        "arm": arm,
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "budget_sensitive": int(checkpoint["best_epoch"]) >= 217,
        "amplitude": amplitude_metrics(
            prediction,
            data.validation["amplitude"],
            direction_weights=data.validation["solid_angle_weights"],
            command_family=data.validation["command_family"],
            direction=data.validation["direction"],
        ),
        "operator": operator_metrics(operator, data.validation["g_ephi"], data.validation["solid_angle_weights"]),
        "architecture": architecture_contract(model, arm),
        "checkpoint_bytes": (run_dir / "checkpoint.pt").stat().st_size,
        "estimated_operator_macs_per_direction": estimate_macs(model.to("cpu")),
        "latency": latency_profile(model, data),
        "diagnostics": _branch_diagnostics(model, data, "cpu"),
        "all_outputs_finite": bool(np.isfinite(prediction).all() and np.isfinite(operator).all()),
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    }
    atomic_json(run_dir / "stage2_validation_audit.json", report)
    return report


def _metric(report: dict, path: tuple[str, ...]) -> float:
    value = report
    for key in path:
        value = value[key]
    return float(value)


def _gain(candidate: dict[int, dict], reference: dict[int, dict], path: tuple[str, ...]) -> list[float]:
    return [_metric(reference[seed], path) - _metric(candidate[seed], path) for seed in SEEDS]


def _degradation(candidate: dict[int, dict], reference: dict[int, dict], path: tuple[str, ...]) -> list[float]:
    return [_metric(candidate[seed], path) - _metric(reference[seed], path) for seed in SEEDS]


def _mechanism_gate(candidate: dict[int, dict], reference: dict[int, dict]) -> dict[str, object]:
    phase = _gain(candidate, reference, ("operator", "phase_aligned_complex_nmse_db"))
    q_gain = _gain(candidate, reference, ("operator", "q_nmse_db"))
    amplitude = _gain(candidate, reference, ("amplitude", "solid_angle_micro_nmse_db"))
    strongest = _degradation(candidate, reference, ("amplitude", "fixed_field_bins", "strong_-3_to_0_db", "nmse_db"))
    weakest = _degradation(candidate, reference, ("amplitude", "fixed_field_bins", "weak_below_-20_db", "nmse_db"))
    latency = np.mean([candidate[s]["latency"]["throughput"]["median_ms"] for s in SEEDS]) / np.mean(
        [reference[s]["latency"]["throughput"]["median_ms"] for s in SEEDS]
    ) - 1
    gates = {
        "phase_aligned_mean_gain_ge_0p30_db": bool(np.mean(phase) >= 0.30),
        "phase_aligned_at_least_2_of_3_positive": sum(x > 0 for x in phase) >= 2,
        "q_mean_gain_ge_0p30_db": bool(np.mean(q_gain) >= 0.30),
        "q_at_least_2_of_3_positive": sum(x > 0 for x in q_gain) >= 2,
        "amplitude_mean_degradation_le_0p10_db": bool(np.mean(amplitude) >= -0.10),
        "amplitude_at_least_2_of_3_nonnegative": sum(x >= 0 for x in amplitude) >= 2,
        "strongest_mean_degradation_le_0p10_db": bool(np.mean(strongest) <= 0.10),
        "weakest_mean_degradation_le_0p10_db": bool(np.mean(weakest) <= 0.10),
        "worst_seed_field_degradation_le_0p30_db": bool(max(strongest + weakest) <= 0.30),
        "parameter_count_identical": all(candidate[s]["architecture"]["parameter_count"] == reference[s]["architecture"]["parameter_count"] for s in SEEDS),
        "model_bytes_delta_le_5_percent": bool(np.mean([candidate[s]["checkpoint_bytes"] for s in SEEDS]) / np.mean([reference[s]["checkpoint_bytes"] for s in SEEDS]) - 1 <= 0.05),
        "latency_delta_le_5_percent": bool(latency <= 0.05),
        "all_outputs_finite": all(candidate[s]["all_outputs_finite"] for s in SEEDS),
    }
    return {"pass": all(gates.values()), "gates": gates, "gain_db": {"phase_aligned": phase, "q": q_gain, "amplitude": amplitude}, "field_degradation_db": {"strongest": strongest, "weak_below_-20": weakest}, "latency_delta_fraction": float(latency)}


def _engineering_gate(candidate: dict[int, dict], control: dict[int, dict]) -> dict[str, object]:
    macro = _gain(candidate, control, ("amplitude", "command_family_macro_nmse_db"))
    micro = _gain(candidate, control, ("amplitude", "solid_angle_micro_nmse_db"))
    phase = _gain(candidate, control, ("operator", "phase_aligned_complex_nmse_db"))
    q_gain = _gain(candidate, control, ("operator", "q_nmse_db"))
    weak = _degradation(candidate, control, ("amplitude", "fixed_field_bins", "weak_below_-20_db", "nmse_db"))
    deep = _degradation(candidate, control, ("amplitude", "fixed_field_bins", "deep_null_below_-30_db", "nmse_db"))
    strong = _degradation(candidate, control, ("amplitude", "fixed_field_bins", "strong_-3_to_0_db", "nmse_db"))
    hard_null = _gain(candidate, control, ("amplitude", "command_family", "hard_null", "solid_angle_nmse_db"))
    family_degradation = {}
    for family in ("hard_null", "random_constant_modulus", "scan", "sparse", "taper"):
        family_degradation[family] = _degradation(candidate, control, ("amplitude", "command_family", family, "solid_angle_nmse_db"))
    candidate_std = float(np.std([_metric(candidate[s], ("amplitude", "solid_angle_micro_nmse_db")) for s in SEEDS], ddof=1))
    control_std = float(np.std([_metric(control[s], ("amplitude", "solid_angle_micro_nmse_db")) for s in SEEDS], ddof=1))
    gates = {
        "macro_mean_gain_ge_0p30_db": bool(np.mean(macro) >= 0.30),
        "macro_3_of_3_positive": all(x > 0 for x in macro),
        "micro_mean_gain_ge_0p20_db": bool(np.mean(micro) >= 0.20),
        "micro_at_least_2_of_3_positive": sum(x > 0 for x in micro) >= 2,
        "phase_aligned_mean_gain_ge_0p30_db": bool(np.mean(phase) >= 0.30),
        "phase_aligned_at_least_2_of_3_positive": sum(x > 0 for x in phase) >= 2,
        "q_mean_gain_ge_0p30_db": bool(np.mean(q_gain) >= 0.30),
        "q_at_least_2_of_3_positive": sum(x > 0 for x in q_gain) >= 2,
        "weak_mean_non_degraded": bool(np.mean(weak) <= 0),
        "deep_null_mean_non_degraded": bool(np.mean(deep) <= 0),
        "weak_and_deep_worst_seed_degradation_le_0p30_db": bool(max(weak + deep) <= 0.30),
        "hard_null_mean_non_degraded": bool(np.mean(hard_null) >= 0),
        "hard_null_worst_seed_degradation_le_0p30_db": bool(min(hard_null) >= -0.30),
        "no_family_mean_degradation_gt_0p20_db": all(np.mean(values) <= 0.20 for values in family_degradation.values()),
        "strong_mean_degradation_le_0p10_db": bool(np.mean(strong) <= 0.10),
        "cross_seed_std_le_control_plus_0p10_db": bool(candidate_std <= control_std + 0.10),
        "all_outputs_finite": all(candidate[s]["all_outputs_finite"] for s in SEEDS),
    }
    return {"pass": all(gates.values()), "gates": gates, "gain_db": {"macro": macro, "micro": micro, "phase_aligned": phase, "q": q_gain, "hard_null": hard_null}, "degradation_db": {"weak": weak, "deep_null": deep, "strong": strong, "by_family": family_degradation}, "cross_seed_std_db": {"candidate": candidate_std, "control": control_std}}


def final_gate(reports: dict[str, dict[int, dict]]) -> dict[str, object]:
    a, b, c, d = (reports[name] for name in STAGE2_ARM_ORDER)
    mechanism = {"B_vs_A": _mechanism_gate(b, a), "D_vs_C": _mechanism_gate(d, c)}
    engineering = {
        "B_vs_A": _engineering_gate(b, a),
        "C_vs_A": _engineering_gate(c, a),
        "D_vs_A": _engineering_gate(d, a),
    }
    d_vs_b_macro = _gain(d, b, ("amplitude", "command_family_macro_nmse_db"))
    d_vs_b_q = _gain(d, b, ("operator", "q_nmse_db"))
    d_vs_b_weak = _degradation(d, b, ("amplitude", "fixed_field_bins", "weak_below_-20_db", "nmse_db"))
    d_vs_b_deep = _degradation(d, b, ("amplitude", "fixed_field_bins", "deep_null_below_-30_db", "nmse_db"))
    dcar_value = {
        "gates": {
            "macro_mean_gain_ge_0p50_db": bool(np.mean(d_vs_b_macro) >= 0.50),
            "macro_3_of_3_positive": all(x > 0 for x in d_vs_b_macro),
            "q_mean_non_degraded": bool(np.mean(d_vs_b_q) >= 0),
            "weak_mean_non_degraded": bool(np.mean(d_vs_b_weak) <= 0),
            "deep_null_mean_non_degraded": bool(np.mean(d_vs_b_deep) <= 0),
        },
        "gain_db": {"macro": d_vs_b_macro, "q": d_vs_b_q},
        "degradation_db": {"weak": d_vs_b_weak, "deep_null": d_vs_b_deep},
    }
    dcar_value["pass"] = all(dcar_value["gates"].values())
    eligible = []
    if mechanism["B_vs_A"]["pass"] and engineering["B_vs_A"]["pass"]:
        eligible.append("B")
    if engineering["C_vs_A"]["pass"]:
        eligible.append("C")
    if mechanism["D_vs_C"]["pass"] and engineering["D_vs_A"]["pass"]:
        eligible.append("D")
    if "B" in eligible and "D" in eligible and not dcar_value["pass"]:
        eligible.remove("D")
    candidate = "NONE"
    if eligible:
        arm_for = {"B": STAGE2_ARM_ORDER[1], "C": STAGE2_ARM_ORDER[2], "D": STAGE2_ARM_ORDER[3]}
        macro_mean = {
            label: float(np.mean([_metric(reports[arm_for[label]][s], ("amplitude", "command_family_macro_nmse_db")) for s in SEEDS]))
            for label in eligible
        }
        best_macro = min(macro_mean.values())
        shortlist = [label for label in eligible if macro_mean[label] <= best_macro + 0.10]
        candidate = min(shortlist, key=lambda label: (
            max(_metric(reports[arm_for[label]][s], ("amplitude", "command_family_macro_nmse_db")) for s in SEEDS),
            np.mean([reports[arm_for[label]][s]["latency"]["throughput"]["median_ms"] for s in SEEDS]),
            reports[arm_for[label]][SEEDS[0]]["architecture"]["parameter_count"],
            label,
        ))
    return {
        "schema": "v218-stage2-final-audit-1",
        "migration_attribution_status": "MIGRATION_ATTRIBUTION_PASS",
        "phase_aux_experiment_status": "PHASE_AUX_EXPERIMENT_PASS" if any(value["pass"] for value in mechanism.values()) else "PHASE_AUX_EXPERIMENT_FAIL",
        "engineering_candidate": candidate,
        "engineering_candidate_status": f"ENGINEERING_CANDIDATE_{candidate}",
        "selected_canonical_baseline": candidate if candidate != "NONE" else "A",
        "mechanism_gates": mechanism,
        "engineering_gates": engineering,
        "dcar_engineering_value": dcar_value,
        "budget_sensitive_arms": [arm for arm in STAGE2_ARM_ORDER if all(reports[arm][seed]["budget_sensitive"] for seed in SEEDS)],
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    }


def run_final_audit(*, version_root: Path, data: PreparedData, output_dir: Path, device: str) -> dict[str, object]:
    reports: dict[str, dict[int, dict]] = {}
    for arm in STAGE2_ARM_ORDER:
        reports[arm] = {}
        for seed in SEEDS:
            reports[arm][seed] = audit_one(version_root, arm, seed, data, device)
    receipt = final_gate(reports)
    receipt["reports"] = reports
    atomic_json(output_dir / "STAGE2_FINAL_AUDIT.json", receipt)
    return receipt
