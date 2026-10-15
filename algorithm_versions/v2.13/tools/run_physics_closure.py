"""Run the V2.13 validation-only physics closure on existing non-test artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V212_ROOT = ROOT.parent / "v2.12"
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V212_ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.contracts import ArrayGeometry, QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.distillation import OperatorTarget  # noqa: E402
from v212.evaluation import amplitude_metrics  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import operator_from_checkpoint  # noqa: E402
from v213.physics_audit import (  # noqa: E402
    compare_operators,
    deembed_geometric_phase,
    gradient_alignment,
    nested_vector_spectrum,
    operator_contract_metrics,
    phase_only_maximin,
    rank_one_q,
    vector_graph_roughness,
)


def _nmse(prediction: torch.Tensor, truth: torch.Tensor, epsilon: float = 1e-12) -> dict[str, float]:
    value = (prediction - truth).abs().square().sum() / truth.abs().square().sum().clamp_min(epsilon)
    return {"nmse": float(value), "nmse_db": float(10.0 * torch.log10(value.clamp_min(epsilon)))}


@torch.no_grad()
def _predict_g(model, direction: torch.Tensor, chunk_size: int) -> torch.Tensor:
    result = []
    for start in range(0, len(direction), chunk_size):
        operator_iq, _ = model.operator(QueryBatch(direction[start : start + chunk_size][None]))
        result.append(torch.complex(operator_iq[0, :, 0], operator_iq[0, :, 1]).reshape(-1, 16).cpu())
    return torch.cat(result)


def _load_independent_combined(path: Path, g: torch.Tensor) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        commands = torch.from_numpy(np.asarray(archive["commands"], dtype=np.complex128).copy())
        field = torch.from_numpy(np.asarray(archive["ephi_complex"], dtype=np.complex128).copy())
        if "g_single_port_ephi" in archive:
            comparison_g = torch.from_numpy(
                np.asarray(archive["g_single_port_ephi"], dtype=np.complex128).copy()
            )
            provenance = "independent_HFSS_report_vs_stored_single_port_FFD"
        else:
            direction = torch.from_numpy(np.asarray(archive["direction"], dtype=np.float64).copy())
            if len(direction) != len(g):
                raise ValueError("independent combined export must use the same direction grid")
            comparison_g = g.to(torch.complex128)
            provenance = "independent_HFSS_report_vs_main_complex_audit_g"
    if field.shape != (len(comparison_g), len(commands)):
        raise ValueError("independent ephi_complex must have shape [Q,C]")
    if comparison_g.shape[1] != commands.shape[1]:
        raise ValueError("single-port and command port counts differ")
    synth = comparison_g @ commands.T
    error = _nmse(synth, field)
    return {
        "status": "pass_hfss_report_path" if error["nmse"] < 1e-12 else "fail_hfss_report_path",
        "independent_combined_complex_field_available": True,
        "provenance": provenance,
        "complex_field_error": error,
        "query_count": len(comparison_g),
        "command_count": len(commands),
        "scope_limitation": "validates HFSS linear source/report path, not nonlinear RF hardware",
    }


def _controlled_perturbations(g: torch.Tensor) -> dict[str, float]:
    sample = g[:128].to(torch.complex128)
    base = rank_one_q(sample)
    gain = sample.clone()
    gain[:, 6] *= 0.8
    gain_q = rank_one_q(gain)
    gain_ratio = gain_q[:, 6, 6].real / base[:, 6, 6].real.clamp_min(1e-12)
    phase = sample.clone()
    phase[:, 4] *= torch.exp(torch.tensor(1j * math.radians(20), dtype=torch.complex128))
    phase_q = rank_one_q(phase)
    valid = base[:, 4, 0].abs() > 1e-9
    phase_ratio = phase_q[valid, 4, 0] / base[valid, 4, 0]
    expected = torch.exp(torch.tensor(-1j * math.radians(20), dtype=torch.complex128))
    common = sample * torch.exp(torch.linspace(0, 2 * math.pi, len(sample), dtype=torch.float64)[:, None] * 1j)
    common_q = rank_one_q(common)
    return {
        "gain_0p8_q77_ratio_mean": float(gain_ratio.mean()),
        "gain_0p8_q77_expected": 0.64,
        "phase_20deg_relative_entry_error_max": float((phase_ratio - expected).abs().max()),
        "direction_dependent_common_phase_q_relative_error": float(
            (common_q - base).abs().square().sum().sqrt() / base.abs().square().sum().sqrt()
        ),
    }


def _decision_audit(
    prediction_q: torch.Tensor,
    truth_q: torch.Tensor,
    *,
    groups: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(truth_q), generator=generator)[: groups * 4]
    predicted_group = prediction_q[order].reshape(groups, 4, 16, 16)
    truth_group = truth_q[order].reshape(groups, 4, 16, 16)
    device = prediction_q.device
    predicted_weight, _ = phase_only_maximin(
        predicted_group, starts=8, steps=200, learning_rate=0.05, seed=seed
    )
    oracle_weight, oracle_objective = phase_only_maximin(
        truth_group, starts=16, steps=320, learning_rate=0.05, seed=seed + 1
    )
    executed = torch.einsum(
        "bn,bknm,bm->bk", predicted_weight.conj(), truth_group, predicted_weight
    ).real.min(-1).values
    oracle_executed = torch.einsum(
        "bn,bknm,bm->bk", oracle_weight.conj(), truth_group, oracle_weight
    ).real.min(-1).values
    # The non-convex truth solve is only a numerical reference, not a certified
    # global oracle.  Include the surrogate-selected feasible point in that
    # reference set so reported regret/gap cannot become negative merely because
    # the higher-budget truth solve reached a worse local optimum.
    numerical_reference = torch.maximum(oracle_executed, executed)
    denominator = numerical_reference.clamp_min(1e-12)
    regret = (numerical_reference - executed) / denominator
    gap_db = 10.0 * torch.log10(denominator / executed.clamp_min(1e-12))
    return {
        "status": "validation_only_numerical_reference",
        "groups": groups,
        "directions_per_group": 4,
        "constraint": "phase_only_constant_modulus_unit_l2",
        "oracle_is_global": False,
        "reference_definition": "best feasible value among truth-optimized and surrogate-selected weights",
        "surrogate_outperformed_truth_solver_fraction": float((executed > oracle_executed).float().mean()),
        "oracle_solver_reported_objective_consistency_max": float(
            (oracle_executed - oracle_objective).abs().max()
        ),
        "relative_regret_median": float(torch.quantile(regret, 0.50)),
        "relative_regret_p95": float(torch.quantile(regret, 0.95)),
        "decision_gap_db_median": float(torch.quantile(gap_db, 0.50)),
        "decision_gap_db_p95": float(torch.quantile(gap_db, 0.95)),
        "regret_values": regret.detach().cpu().tolist(),
        "gap_db_values": gap_db.detach().cpu().tolist(),
    }


def _plot(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    validation = report["gate_D_deembedding"]["validation"]
    degree = np.arange(13)
    axes[0, 0].plot(degree, [validation["g"]["cumulative_fraction"][str(i)] for i in degree], marker="o", label="raw g")
    axes[0, 0].plot(degree, [validation["a"]["cumulative_fraction"][str(i)] for i in degree], marker="o", label="de-embedded a")
    axes[0, 0].set(title="Privileged HFSS vector filtration", xlabel="L", ylabel="cumulative energy")
    axes[0, 0].legend()

    axes[0, 1].bar(
        ["raw g", "de-embedded a"],
        [validation["g_graph_roughness"], validation["a_graph_roughness"]],
    )
    axes[0, 1].set(title="Validation graph roughness", ylabel="normalized roughness")

    gate_b = report["gate_B_amplitude_identifiability"]
    axes[1, 0].bar(
        ["Q recovery", "held-out power"],
        [gate_b["q_recovery"]["aggregate_nmse_db"], gate_b["heldout_power"]["nmse_db"]],
    )
    axes[1, 0].set(title="Ideal synthetic inversion checks", ylabel="NMSE (dB)")

    decision = np.asarray(report["gate_E_decision"]["gap_db_values"])
    axes[1, 1].hist(decision, bins=16)
    axes[1, 1].axvline(np.quantile(decision, .95), color="red", linestyle="--", label="P95")
    axes[1, 1].set(title="Validation four-point decision gap", xlabel="truth numerical reference / surrogate choice (dB)")
    axes[1, 1].legend()
    fig.suptitle("V2.13 validation-only physics closure")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-audit", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--operator-target", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--independent-combined-complex", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--decision-groups", type=int, default=64)
    parser.add_argument("--spectrum-train-limit", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=21301)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with np.load(args.complex_audit, allow_pickle=False) as archive:
        direction = torch.from_numpy(np.asarray(archive["direction"], dtype=np.float64).copy())
        g = torch.from_numpy(np.asarray(archive["g"], dtype=np.complex128).copy())
        solid_angle = torch.from_numpy(np.asarray(archive["solid_angle_weights"], dtype=np.float64).copy())
        audit_query_test = np.asarray(archive["query_test"], dtype=np.int64).copy()
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    commands = codebook.weights.to(torch.complex128)
    if len(direction) != len(dataset.direction):
        raise ValueError("complex audit and amplitude dataset direction counts differ")
    if not np.array_equal(np.asarray(split.query_test), audit_query_test):
        raise ValueError("complex audit and frozen test indices differ")
    non_test = torch.from_numpy(np.concatenate((split.query_train, split.query_validation))).long()
    q_train = torch.from_numpy(np.asarray(split.query_train, dtype=np.int64)).long()
    q_validation = torch.from_numpy(np.asarray(split.query_validation, dtype=np.int64)).long()

    synthesized_amplitude = (g.to(torch.complex128) @ commands.T).abs().float()
    stored_amplitude = torch.from_numpy(np.asarray(dataset.amplitude.T, dtype=np.float32).copy())
    pipeline_check = _nmse(synthesized_amplitude[non_test], stored_amplitude[non_test])
    gate_a = {
        "status": "blocked_missing_independent_combined_port_hfss_export",
        "independent_combined_complex_field_available": False,
        "reason": "stored amplitude labels were synthesized from the same single-port g; agreement is circular",
        "circular_pipeline_self_consistency_only": pipeline_check,
    }
    if args.independent_combined_complex is not None:
        gate_a = _load_independent_combined(args.independent_combined_complex, g)

    contract = operator_contract_metrics(g[non_test], commands)
    truth_q_train = rank_one_q(g[q_train]).to(torch.complex64)
    operator_target = OperatorTarget.load(args.operator_target)
    if not torch.equal(operator_target.query_indices, q_train):
        raise ValueError("operator target query indices are not the frozen train split")
    if not torch.equal(operator_target.command_indices, codebook.indices("train")):
        raise ValueError("operator target did not use only frozen train commands")
    q_recovery = compare_operators(operator_target.q_hat, truth_q_train)
    validation_command = codebook.indices("validation")
    reconstructed_power = torch.einsum(
        "cn,qnm,cm->qc", commands[validation_command].conj(), operator_target.q_hat.to(torch.complex128), commands[validation_command]
    ).real
    heldout_power = stored_amplitude[q_train][:, validation_command].double().square()
    gate_b = {
        "status": "ideal_synthetic_pass_not_hardware_identifiability",
        "q_recovery": q_recovery,
        "heldout_power": _nmse(reconstructed_power, heldout_power),
        "conditioning": json.loads((args.operator_target.parent / "SUMMARY.json").read_text(encoding="utf-8")),
        "limitation": "zero-noise labels and three identical repeats were generated from the same g",
    }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("test_split_opened") is not False:
        raise ValueError("checkpoint does not preserve the sealed-test contract")
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    transform = FiltrationSSHTransform.load(transform_path)
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=direction[q_train].float(),
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(args.device).eval()
    predicted_g = _predict_g(model, direction[q_validation].float().to(args.device), args.chunk_size)
    predicted_q = rank_one_q(predicted_g).to(torch.complex64)
    truth_q_validation = rank_one_q(g[q_validation]).to(torch.complex64)
    predicted_amplitude = torch.einsum(
        "cn,qnm,cm->qc", commands[validation_command].conj(), predicted_q.to(torch.complex128), commands[validation_command]
    ).real.clamp_min(0).sqrt().float()
    gate_c = {
        "status": "architecture_pass_validation_prediction_diagnostic",
        "test_split_opened": False,
        "operator_contract": operator_contract_metrics(predicted_g.to(torch.complex128), commands[validation_command]),
        "operator_error_vs_complex_truth": compare_operators(predicted_q, truth_q_validation),
        "validation_D_amplitude": amplitude_metrics(
            predicted_amplitude, stored_amplitude[q_validation][:, validation_command]
        ),
        "power_gradient_alignment": gradient_alignment(
            predicted_q, truth_q_validation, commands[validation_command].to(torch.complex64)
        ),
        "checkpoint": str(args.checkpoint.resolve()),
    }

    geometry = ArrayGeometry()
    positions = torch.from_numpy(geometry.positions_m.reshape(16, 3)).double()
    wavenumber = 2.0 * math.pi / geometry.wavelength_m
    a = deembed_geometric_phase(g, direction, positions, wavenumber)
    generator = torch.Generator().manual_seed(args.seed)
    train_take = q_train[torch.randperm(len(q_train), generator=generator)[: args.spectrum_train_limit]]

    def complexity(indices: torch.Tensor) -> dict[str, object]:
        weights = solid_angle[indices]
        g_spectrum = nested_vector_spectrum(g[indices], direction[indices], weights=weights)
        a_spectrum = nested_vector_spectrum(a[indices], direction[indices], weights=weights)
        return {
            "query_count": len(indices),
            "g": g_spectrum,
            "a": a_spectrum,
            "g_graph_roughness": vector_graph_roughness(g[indices], direction[indices]),
            "a_graph_roughness": vector_graph_roughness(a[indices], direction[indices]),
        }

    gate_d = {
        "status": "evaluated_conditional_on_frozen_nominal_geometry",
        "interpretation": "analytic nominal phase de-embedding, not unique physical element envelope",
        "geometry": {
            "frequency_hz": geometry.frequency_hz,
            "row_pitch_m": geometry.row_pitch_m,
            "col_pitch_m": geometry.col_pitch_m,
            "positions_embedded_in_complex_audit_metadata": False,
        },
        "train_sample": complexity(train_take),
        "validation": complexity(q_validation),
    }

    decision = _decision_audit(
        predicted_q.to(args.device), truth_q_validation.to(args.device),
        groups=args.decision_groups, seed=args.seed,
    )
    gate_a_status = (
        "pass_hfss_linear_report_path"
        if gate_a.get("status") == "pass_hfss_report_path"
        else "failed_or_blocked"
    )
    report = {
        "schema": "v213-validation-physics-closure-1",
        "external_test_used": False,
        "new_model_trained": False,
        "gate_A_physics_closure": gate_a,
        "gate_A_rank1_operator_identity": contract,
        "gate_B_amplitude_identifiability": gate_b,
        "gate_C_network_physics": gate_c,
        "gate_D_deembedding": gate_d,
        "gate_D_controlled_perturbations": _controlled_perturbations(g[q_validation]),
        "gate_E_decision": decision,
        "gate_status": {
            "A_independent_HFSS_LTI": gate_a_status,
            "B_amplitude_only_identifiability": "ideal_synthetic_only",
            "C_network_structure": "pass_by_construction_and_validation",
            "D_structure_interpretation": "evaluated_with_geometry_provenance_limitation",
            "E_final_decision": "validation_diagnostic_only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(report, args.plot)
    print(json.dumps({
        "gate_status": report["gate_status"],
        "gate_B": gate_b["q_recovery"],
        "gate_C_gradient": gate_c["power_gradient_alignment"],
        "gate_D_validation": {
            "g_C5": gate_d["validation"]["g"]["cumulative_fraction"]["5"],
            "a_C5": gate_d["validation"]["a"]["cumulative_fraction"]["5"],
            "g_roughness": gate_d["validation"]["g_graph_roughness"],
            "a_roughness": gate_d["validation"]["a_graph_roughness"],
        },
        "gate_E": {key: value for key, value in decision.items() if not key.endswith("_values")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
