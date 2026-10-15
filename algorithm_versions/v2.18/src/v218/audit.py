"""Validation-only V2.18 metrics, diagnostics, and preregistered gates."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from v210.contracts import QueryBatch

from .canonical import PreparedData, atomic_json
from .experiment import OPAUX_ARM, SEEDS, architecture_contract, build_model
from .training import _operator_complex, complex_operator_nmse


def _db(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-12))


def load_run(run_dir: Path, data: PreparedData, device: str):
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("blind_test_status") != "sealed_not_opened":
        raise ValueError("checkpoint did not preserve the confirmation seal")
    model = build_model(
        checkpoint["arm"], transform=data.transform,
        envelope_scale=float(checkpoint["envelope_scale"]), seed=int(checkpoint["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _predictions(model, data: PreparedData, device: str, chunk: int = 256):
    direction = torch.from_numpy(data.validation["direction"]).float().to(device)
    commands = torch.from_numpy(data.validation["commands"]).to(device)
    amplitude_parts, operator_parts = [], []
    for start in range(0, len(direction), chunk):
        selected = direction[start:start + chunk]
        amplitude_parts.append(model.forward_command_grid(commands, selected).T.cpu())
        operator, _ = _operator_complex(model, selected)
        operator_parts.append(operator.cpu())
    return torch.cat(amplitude_parts), torch.cat(operator_parts)


def _amplitude_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    error = prediction - target
    nmse = error.square().sum() / target.square().sum().clamp_min(1e-12)
    error_db = 20 * torch.log10(prediction.clamp_min(1e-6)) - 20 * torch.log10(target.clamp_min(1e-6))
    normalized = target / target.max(dim=0, keepdim=True).values.clamp_min(1e-12)
    cuts = torch.quantile(normalized.flatten(), torch.tensor([0.25, 0.5, 0.75]))
    bounds = (0.0, float(cuts[0]), float(cuts[1]), float(cuts[2]), float("inf"))
    buckets = {}
    for index, name in enumerate(("weakest", "weak_mid", "strong_mid", "strongest")):
        mask = (normalized >= bounds[index]) & (normalized < bounds[index + 1])
        bucket_nmse = error[mask].square().sum() / target[mask].square().sum().clamp_min(1e-12)
        buckets[name] = {
            "count": int(mask.sum()), "nmse": float(bucket_nmse),
            "nmse_db": _db(float(bucket_nmse)),
            "mae_db": float(error_db[mask].abs().mean()),
        }
    return {
        "nmse": float(nmse), "nmse_db": _db(float(nmse)),
        "mae_db": float(error_db.abs().mean()), "field_strength_quartiles": buckets,
    }


def _operator_metrics(prediction, target, weights) -> dict[str, float]:
    complex_nmse = complex_operator_nmse(prediction, target, weights)
    prediction_norm = prediction.abs().square().sum(dim=1)
    target_norm = target.abs().square().sum(dim=1)
    inner = (prediction.conj() * target).sum(dim=1).abs().square()
    q_error = prediction_norm.square() + target_norm.square() - 2 * inner
    normalized = weights / weights.sum().clamp_min(1e-12)
    q_nmse = (normalized * q_error).sum() / (normalized * target_norm.square()).sum().clamp_min(1e-12)
    return {
        "complex_nmse": float(complex_nmse), "complex_nmse_db": _db(float(complex_nmse)),
        "q_nmse": float(q_nmse), "q_nmse_db": _db(float(q_nmse)),
    }


@torch.no_grad()
def _branch_diagnostics(model, data: PreparedData, device: str) -> dict[str, object]:
    direction = torch.from_numpy(data.validation["direction"][:1024]).float().to(device)
    _, terms = model.envelope(QueryBatch(direction[None]))
    energies = {}
    for key in ("baseline_iq", "ssh_iq", "fourier_residual_iq"):
        energies[key] = float(terms[key].square().sum(dim=2).mean())
    residual = torch.complex(terms["fourier_residual_iq"][0, :, 0], terms["fourier_residual_iq"][0, :, 1]).reshape(len(direction), 16)
    basis = data.transform.evaluate(direction.double()).to(residual)
    weights = torch.full((len(direction),), 1 / len(direction), dtype=basis.real.dtype, device=device)
    gram = basis.conj().T @ (weights[:, None] * basis)
    rhs = basis.conj().T * weights[None]
    analysis = torch.linalg.pinv(gram) @ rhs
    projection = basis @ (analysis @ residual)
    projection_ratio = float((weights[:, None] * projection.abs().square()).sum() / (weights[:, None] * residual.abs().square()).sum().clamp_min(1e-12))
    messages = []
    module = model.residual
    if not all(hasattr(module, name) for name in ("encoder", "geometry_maps", "stem", "blocks", "gate_values")):
        return {
            "complex_branch_energy": energies,
            "residual_ssh_projection_ratio": projection_ratio,
            "effective_messages": messages,
            "effective_message_note": "not applicable to the V2.12 isotropic control",
        }
    encoded = module.encoder(direction[None])
    maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
    geometry = module.geometry_maps.to(direction)[None, None].expand(1, len(direction), -1, -1, -1)
    hidden = module.activation(module.stem(torch.cat((maps, geometry), dim=2).reshape(len(direction), 31, 4, 4)))
    gates = module.gate_values(direction[None]).reshape(len(direction), module.depth, 2)
    for index, block in enumerate(module.blocks):
        row = gates[:, index, 0, None, None, None] * block.row_conv2(block.activation(block.row_conv1(hidden)))
        col = gates[:, index, 1, None, None, None] * block.col_conv2(block.activation(block.col_conv1(hidden)))
        messages.append({"block": index, "row_effective_energy": float(row.square().mean()), "column_effective_energy": float(col.square().mean())})
        hidden = hidden + row + col
    return {"complex_branch_energy": energies, "residual_ssh_projection_ratio": projection_ratio, "effective_messages": messages}


@torch.no_grad()
def _cpu_latency_ms(model, data: PreparedData) -> float:
    cpu = model.to("cpu")
    direction = torch.from_numpy(data.validation["direction"][:256]).float()
    commands = torch.from_numpy(data.train["commands"])
    for _ in range(3):
        cpu.forward_command_grid(commands, direction)
    samples = []
    for _ in range(10):
        start = time.perf_counter()
        cpu.forward_command_grid(commands, direction)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def audit_run(run_dir: Path, data: PreparedData, device: str) -> dict[str, object]:
    model, checkpoint = load_run(run_dir, data, device)
    prediction, operator = _predictions(model, data, device)
    target = torch.from_numpy(data.validation["amplitude"])
    operator_target = torch.from_numpy(data.validation["g_ephi"])
    weights = torch.from_numpy(data.validation["solid_angle_weights"]).float()
    family = np.asarray(data.validation["command_family"]).astype(str)
    family_metrics = {}
    for label in sorted(set(family)):
        indices = torch.from_numpy(np.flatnonzero(family == label))
        family_metrics[label] = _amplitude_metrics(prediction[:, indices], target[:, indices])
    cancellation_null = np.flatnonzero(
        np.asarray([("null" in label.lower()) or ("cancel" in label.lower()) for label in family])
    )
    report = {
        "arm": checkpoint["arm"], "seed": int(checkpoint["seed"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "architecture": architecture_contract(model, checkpoint["arm"]),
        "amplitude": _amplitude_metrics(prediction, target),
        "operator": _operator_metrics(operator, operator_target, weights),
        "command_family": family_metrics,
        "cancellation_null_subset": (
            _amplitude_metrics(
                prediction[:, torch.from_numpy(cancellation_null)],
                target[:, torch.from_numpy(cancellation_null)],
            ) if len(cancellation_null) else {"status": "no_labeled_commands"}
        ),
        "diagnostics": _branch_diagnostics(model, data, device),
        "cpu_latency_ms": _cpu_latency_ms(model, data),
        "all_outputs_finite": bool(torch.isfinite(prediction).all() and torch.isfinite(operator).all()),
        "blind_test_status": "sealed_not_opened",
    }
    atomic_json(run_dir / "validation_audit.json", report)
    return report


def _improvements(candidate, reference, key1, key2=None):
    key2 = key2 or key1
    return [
        reference[seed][key1] - candidate[seed][key2]
        for seed in SEEDS
    ]


def baseline_gate(reports: dict[str, dict[int, dict]]) -> dict[str, object]:
    control = {s: reports["control_v212_canonical"][s]["amplitude"]["nmse_db"] for s in SEEDS}
    static = {s: reports["dcar_static_pm_canonical"][s]["amplitude"]["nmse_db"] for s in SEEDS}
    gated = {s: reports["dcar_gated_pm_canonical"][s]["amplitude"]["nmse_db"] for s in SEEDS}
    gated_control = [control[s] - gated[s] for s in SEEDS]
    gated_static = [static[s] - gated[s] for s in SEEDS]
    complex_gain = [reports["control_v212_canonical"][s]["operator"]["complex_nmse_db"] - reports["dcar_gated_pm_canonical"][s]["operator"]["complex_nmse_db"] for s in SEEDS]
    q_gain = [reports["control_v212_canonical"][s]["operator"]["q_nmse_db"] - reports["dcar_gated_pm_canonical"][s]["operator"]["q_nmse_db"] for s in SEEDS]
    gates = {
        "gated_vs_control_mean_ge_0p30_db": bool(np.mean(gated_control) >= 0.30),
        "gated_vs_control_3_of_3_positive": sum(value > 0 for value in gated_control) == 3,
        "gated_vs_static_mean_ge_0p10_db": bool(np.mean(gated_static) >= 0.10),
        "gated_vs_static_at_least_2_of_3_positive": sum(value > 0 for value in gated_static) >= 2,
        "complex_operator_mean_not_degraded": bool(np.mean(complex_gain) >= 0),
        "q_mean_not_degraded": bool(np.mean(q_gain) >= 0),
        "parameter_match_within_0p5_percent": bool(abs(np.mean([reports["dcar_gated_pm_canonical"][s]["architecture"]["parameter_count"] for s in SEEDS]) / np.mean([reports["control_v212_canonical"][s]["architecture"]["parameter_count"] for s in SEEDS]) - 1) <= 0.005),
        "all_outputs_finite": all(reports[arm][s]["all_outputs_finite"] for arm in reports for s in SEEDS),
    }
    return {
        "status": "BASELINE_REPRODUCTION_PASS" if all(gates.values()) else "BASELINE_REPRODUCTION_FAIL",
        "gates": gates,
        "paired_improvement_db": {"gated_vs_control_amplitude": gated_control, "gated_vs_static_amplitude": gated_static, "gated_vs_control_complex": complex_gain, "gated_vs_control_q": q_gain},
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    }


def optimization_gate(reports: dict[str, dict[int, dict]]) -> dict[str, object]:
    base = reports["dcar_gated_pm_canonical"]
    candidate = reports[OPAUX_ARM]
    amp = [base[s]["amplitude"]["nmse_db"] - candidate[s]["amplitude"]["nmse_db"] for s in SEEDS]
    complex_gain = [base[s]["operator"]["complex_nmse_db"] - candidate[s]["operator"]["complex_nmse_db"] for s in SEEDS]
    q_gain = [base[s]["operator"]["q_nmse_db"] - candidate[s]["operator"]["q_nmse_db"] for s in SEEDS]
    weakest = [candidate[s]["amplitude"]["field_strength_quartiles"]["weakest"]["nmse_db"] - base[s]["amplitude"]["field_strength_quartiles"]["weakest"]["nmse_db"] for s in SEEDS]
    strongest = [candidate[s]["amplitude"]["field_strength_quartiles"]["strongest"]["nmse_db"] - base[s]["amplitude"]["field_strength_quartiles"]["strongest"]["nmse_db"] for s in SEEDS]
    latency_delta = np.mean([candidate[s]["cpu_latency_ms"] for s in SEEDS]) / np.mean([base[s]["cpu_latency_ms"] for s in SEEDS]) - 1
    gates = {
        "amplitude_mean_ge_0p20_db": bool(np.mean(amp) >= 0.20),
        "amplitude_at_least_2_of_3_positive": sum(value > 0 for value in amp) >= 2,
        "complex_mean_ge_0p30_db": bool(np.mean(complex_gain) >= 0.30),
        "complex_at_least_2_of_3_positive": sum(value > 0 for value in complex_gain) >= 2,
        "q_mean_ge_0p30_db": bool(np.mean(q_gain) >= 0.30),
        "q_at_least_2_of_3_positive": sum(value > 0 for value in q_gain) >= 2,
        "weakest_bucket_mean_degradation_le_0p10_db": bool(np.mean(weakest) <= 0.10),
        "strongest_bucket_mean_degradation_le_0p10_db": bool(np.mean(strongest) <= 0.10),
        "parameter_count_identical": all(candidate[s]["architecture"]["parameter_count"] == base[s]["architecture"]["parameter_count"] for s in SEEDS),
        "cpu_latency_delta_le_5_percent": bool(latency_delta <= 0.05),
        "all_outputs_finite": all(candidate[s]["all_outputs_finite"] for s in SEEDS),
    }
    return {
        "status": "OPTIMIZATION_PASS" if all(gates.values()) else "OPTIMIZATION_FAIL",
        "gates": gates,
        "paired_improvement_db": {"amplitude": amp, "complex": complex_gain, "q": q_gain},
        "bucket_degradation_db": {"weakest": weakest, "strongest": strongest},
        "cpu_latency_fractional_delta": float(latency_delta),
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    }
