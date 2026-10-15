"""Run the preregistered V2.14 Ephi physical residual ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.13", ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v212.codebook import MeasurementCodebook  # noqa: E402
from v213.physics_audit import nested_vector_spectrum, vector_graph_roughness  # noqa: E402
from v214.angular_sampling import deterministic_spatial_folds, nearest_angle  # noqa: E402
from v214.physics_ladder import fit_common_state, fit_stage, element_classes  # noqa: E402
from v214.physics_metrics import amplitude_nmse_db, operator_metrics, polarization_diagnostic, spearman_correlation, transverse_diagnostic  # noqa: E402


def _metrics(prediction, target, weights, commands):
    return {
        "operator_solid_angle": operator_metrics(prediction, target, weights),
        "operator_equal_direction": operator_metrics(prediction, target),
        "validation_command_amplitude_nmse_db": amplitude_nmse_db(prediction, target, commands),
    }


def _fold_scores(direction, target, theta, phi, weights, positions, frequency, s_matrix, degree):
    fold = deterministic_spatial_folds(theta, phi, 3)
    scores, states = [], []
    for held in range(3):
        train = fold != held
        validation = fold == held
        state = fit_common_state(direction[train], target[train], positions, frequency, weights[train], degree)
        prediction = state.predict(direction[validation], positions, frequency, s_matrix)
        scores.append(operator_metrics(prediction, target[validation], weights[validation])["nmse_db"])
        states.append((state, train, validation))
    return scores, states


def _stage_cv(states, stage, direction, target, weights, positions, frequency, s_matrix, rank=None):
    candidate_states, improvements, rows = [], [], []
    for state, train, validation in states:
        base = state.predict(direction[validation], positions, frequency, s_matrix)
        candidate = fit_stage(
            state, stage, direction[train], target[train], positions, frequency,
            s_matrix, weights[train], rank=rank,
        )
        prediction = candidate.predict(direction[validation], positions, frequency, s_matrix)
        before = operator_metrics(base, target[validation], weights[validation])["nmse_db"]
        after = operator_metrics(prediction, target[validation], weights[validation])["nmse_db"]
        improvements.append(before - after)
        rows.append({"baseline_nmse_db": before, "candidate_nmse_db": after, "improvement_db": before - after})
        candidate_states.append((candidate, train, validation))
    passed = float(np.mean(improvements)) >= 0.5 and all(value > 0 for value in improvements)
    return candidate_states, {
        "folds": rows, "mean_improvement_db": float(np.mean(improvements)),
        "minimum_improvement_db": float(np.min(improvements)), "three_of_three_positive": bool(all(v > 0 for v in improvements)),
        "passed": bool(passed),
    }


def _parameter_count(state) -> int:
    count = 2 * len(state.common.coefficient)
    if state.gain is not None:
        count += 30
    if state.phase_center_m is not None:
        count += 45
    if state.coupling_alpha is not None:
        count += 4
    if state.class_coefficient is not None:
        count += 2 * (state.class_coefficient.size - state.class_coefficient.shape[1])
    if state.low_rank is not None:
        count += 2 * (state.low_rank["coefficient"].size + state.low_rank["aperture_modes"].size)
    return int(count)


def _state_arrays(state):
    arrays = {"common_coefficient": state.common.coefficient, "common_degree": np.asarray(state.common.degree)}
    for name in ("gain", "phase_center_m", "coupling_alpha", "class_coefficient"):
        value = getattr(state, name)
        if value is not None:
            arrays[name] = value
    arrays["gain_scale"] = np.asarray(state.gain_scale)
    if state.low_rank is not None:
        arrays.update({f"low_rank_{key}": value for key, value in state.low_rank.items()})
    return arrays


def _make_plot(report, output):
    import matplotlib.pyplot as plt

    decisions = report["selection"]["decisions"]
    stage_names = ["P1 gain", "P2 phase", "P3 coupling", "P4 class", "P5 rank-1"]
    stage_values = [
        decisions["P1_element_gain"]["mean_improvement_db"],
        decisions["P2_phase_center"]["mean_improvement_db"],
        decisions["P3_s_coupling"]["mean_improvement_db"],
        decisions["P4_element_class"]["mean_improvement_db"],
        decisions["P5_low_rank"]["mean_improvement_db"],
    ]
    diagnostics = report["final_residual_diagnostics"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    axes[0, 0].bar(stage_names, stage_values, color="#4e79a7")
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("3-fold operator improvement (dB)")
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    class_energy = diagnostics["class_residual_energy"]
    axes[0, 1].bar(list(class_energy), list(class_energy.values()), color="#59a14f")
    axes[0, 1].set_ylabel("mean residual energy")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    correlations = {
        "angle/error": diagnostics["nearest_train_angle_error_spearman"],
        "adjacency": diagnostics["residual_covariance_adjacency_correlation"],
        "|Sij|": diagnostics["residual_covariance_s_parameter_correlation"],
    }
    axes[1, 0].bar(list(correlations), list(correlations.values()), color="#e15759")
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_ylabel("correlation")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    spectrum = report["final_residual_diagnostics"]["ssh_spectrum"]["shell_fraction"]
    axes[1, 1].bar(np.arange(len(spectrum)), spectrum, color="#f28e2b")
    axes[1, 1].set_xlabel("SSH degree")
    axes[1, 1].set_ylabel("residual energy fraction")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.14 physical residual ladder")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--cnn-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    direction = data["direction_train"].astype(np.float64)
    target = data["g_ephi_train"].astype(np.complex128)
    theta, phi = data["theta_deg_train"], data["phi_deg_train"]
    weights = data["solid_angle_weights_train"].astype(np.float64)
    positions = data["positions_m"].astype(np.float64)
    frequency = float(data["frequency_hz"])
    s_matrix = data["s_parameters_47ghz"].astype(np.complex128)
    codebook = MeasurementCodebook.load(args.codebook)
    commands = codebook.weights[codebook.indices("validation")].numpy()

    degree_trials = {}
    for degree in (1, 3, 5, 8, 12):
        scores, _ = _fold_scores(direction, target, theta, phi, weights, positions, frequency, s_matrix, degree)
        degree_trials[str(degree)] = {"fold_nmse_db": scores, "mean_nmse_db": float(np.mean(scores))}
    degree = min((1, 3, 5, 8, 12), key=lambda value: degree_trials[str(value)]["mean_nmse_db"])
    _, states = _fold_scores(direction, target, theta, phi, weights, positions, frequency, s_matrix, degree)
    decisions = {"P0_common_pattern": {"selected_degree": degree, "trials": degree_trials, "passed": True}}
    accepted = []
    for stage in ("P1_element_gain", "P2_phase_center", "P3_s_coupling", "P4_element_class"):
        candidate, result = _stage_cv(states, stage, direction, target, weights, positions, frequency, s_matrix)
        decisions[stage] = result
        if result["passed"]:
            states = candidate
            accepted.append(stage)
    low_rank_trials = {}
    candidates = {}
    for rank in (1, 2, 4, 8):
        candidate, result = _stage_cv(states, "P5_low_rank", direction, target, weights, positions, frequency, s_matrix, rank)
        low_rank_trials[str(rank)] = result
        candidates[rank] = candidate
    best_rank = max((1, 2, 4, 8), key=lambda value: low_rank_trials[str(value)]["mean_improvement_db"])
    low_rank_result = {"selected_rank": best_rank, "trials": low_rank_trials, **{k: v for k, v in low_rank_trials[str(best_rank)].items() if k != "folds"}}
    decisions["P5_low_rank"] = low_rank_result
    if low_rank_result["passed"]:
        states = candidates[best_rank]
        accepted.append("P5_low_rank")

    full_state = fit_common_state(direction, target, positions, frequency, weights, degree)
    validation_direction = data["direction_validation"].astype(np.float64)
    validation_target = data["g_ephi_validation"].astype(np.complex128)
    validation_weights = data["solid_angle_weights_validation"].astype(np.float64)
    outer = {"P0_common_pattern": _metrics(full_state.predict(validation_direction, positions, frequency, s_matrix), validation_target, validation_weights, commands)}
    for stage in accepted:
        full_state = fit_stage(full_state, stage, direction, target, positions, frequency, s_matrix, weights, rank=best_rank if stage == "P5_low_rank" else None)
        outer[stage] = _metrics(full_state.predict(validation_direction, positions, frequency, s_matrix), validation_target, validation_weights, commands)
    final_prediction = full_state.predict(validation_direction, positions, frequency, s_matrix)
    residual = validation_target - final_prediction
    distance = np.rad2deg(nearest_angle(validation_direction, direction))
    relative_error = np.sum(np.abs(residual) ** 2, axis=1) / np.maximum(np.sum(np.abs(validation_target) ** 2, axis=1), 1e-300)
    classes = element_classes()
    class_energy = {
        name: float(np.mean(np.abs(residual[:, classes == class_id]) ** 2))
        for class_id, name in enumerate(("corner", "edge", "interior"))
    }
    covariance = np.abs(residual.conj().T @ residual)
    mask = ~np.eye(16, dtype=bool)
    row = np.arange(16) // 4
    col = np.arange(16) % 4
    adjacency = ((np.abs(row[:, None] - row[None, :]) + np.abs(col[:, None] - col[None, :])) == 1).astype(float)
    s_abs = np.abs(s_matrix).copy()
    np.fill_diagonal(s_abs, 0.0)
    ssh = nested_vector_spectrum(torch.as_tensor(residual), torch.as_tensor(validation_direction), degree=12, weights=torch.as_tensor(validation_weights))
    diagnostics = {
        "nearest_train_angle_error_spearman": spearman_correlation(distance, 10.0 * np.log10(np.maximum(relative_error, 1e-300))),
        "nearest_train_angle_deg": {"median": float(np.median(distance)), "p95": float(np.quantile(distance, 0.95)), "maximum": float(distance.max())},
        "class_residual_energy": class_energy,
        "residual_covariance_adjacency_correlation": float(np.corrcoef(covariance[mask], adjacency[mask])[0, 1]),
        "residual_covariance_s_parameter_correlation": float(np.corrcoef(covariance[mask], s_abs[mask])[0, 1]),
        "ssh_spectrum": ssh,
        "graph_roughness": vector_graph_roughness(torch.as_tensor(residual), torch.as_tensor(validation_direction)),
    }

    theta_state = fit_common_state(direction, data["g_etheta_train"], positions, frequency, weights, degree)
    for stage in accepted:
        theta_state = fit_stage(theta_state, stage, direction, data["g_etheta_train"], positions, frequency, s_matrix, weights, rank=best_rank if stage == "P5_low_rank" else None)
    theta_prediction = theta_state.predict(validation_direction, positions, frequency, s_matrix)
    vector_diagnostics = {
        "etheta_final_operator": operator_metrics(theta_prediction, data["g_etheta_validation"], validation_weights),
        "transverse_validation": transverse_diagnostic(validation_direction, data["h_cartesian_validation"]),
        "polarization_validation": polarization_diagnostic(validation_target, data["g_etheta_validation"]),
    }
    if full_state.phase_center_m is not None and theta_state.phase_center_m is not None:
        vector_diagnostics["phase_center_ephi_etheta_rms_difference_m"] = float(np.sqrt(np.mean((full_state.phase_center_m - theta_state.phase_center_m) ** 2)))

    cnn_audit = json.loads(args.cnn_audit.read_text(encoding="utf-8"))
    report = {
        "schema": "v214-physics-residual-ladder-1", "status": "physics_ladder_complete",
        "external_test_used": False, "complex_labels_used_for_cnn_training": False,
        "selection": {"folds": 3, "threshold_db": 0.5, "accepted_stages": accepted, "decisions": decisions},
        "outer_validation": outer, "final_parameter_count_real": _parameter_count(full_state),
        "final_residual_diagnostics": diagnostics, "vector_diagnostics": vector_diagnostics,
        "cnn_benchmark": {
            "source": str(args.cnn_audit.resolve()),
            "ten_seed_amplitude_nmse_db_mean": cnn_audit["baseline"]["candidate_nmse_db"]["mean"],
            "ten_seed_amplitude_nmse_db_standard_deviation": cnn_audit["baseline"]["candidate_nmse_db"]["standard_deviation"],
            "note": "Frozen amplitude-only CNN benchmark; no complex labels were exposed to CNN training.",
        },
    }
    (args.output_dir / "LADDER_SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "LADDER_PARAMETERS.npz", **_state_arrays(full_state))
    np.savez_compressed(
        args.output_dir / "residual_atlas.npz", direction=validation_direction,
        residual_ephi=residual, operator_relative_error=relative_error,
        nearest_train_angle_deg=distance, element_class=classes,
    )
    _make_plot(report, args.output_dir / "PHYSICS_RESIDUAL_LADDER.png")
    print(json.dumps({
        "status": report["status"], "accepted_stages": accepted,
        "selected_degree": degree, "selected_low_rank": best_rank,
        "outer_validation": outer, "final_parameter_count_real": report["final_parameter_count_real"],
        "residual_diagnostics": diagnostics, "vector_diagnostics": vector_diagnostics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
