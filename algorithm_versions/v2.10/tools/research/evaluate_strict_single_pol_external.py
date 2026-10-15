"""Evaluate an amplitude-only checkpoint on untouched HFSS directions.

Complex HFSS fields are opened only here to create external reference labels;
they are never passed to the trained model or its loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from v210.coordinates import spherical_basis  # noqa: E402
from v210.io import load_amplitude_dataset  # noqa: E402
from v210.models import model_from_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--external-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    linear_nmse = np.mean(error**2) / max(float(np.mean(target**2)), 1e-30)
    pred_power = prediction**2
    target_power = target**2
    power_nmse = np.mean((pred_power - target_power) ** 2) / max(
        float(np.mean(target_power**2)), 1e-30
    )
    floor = max(float(np.max(target)) * 1e-8, 1e-30)
    valid = target >= floor
    db_error = 20.0 * np.log10(np.maximum(prediction, floor)) - 20.0 * np.log10(
        np.maximum(target, floor)
    )
    return {
        "observation_count": int(target.size),
        "linear_nmse": float(linear_nmse),
        "linear_nmse_db": float(10.0 * np.log10(max(linear_nmse, 1e-30))),
        "power_nmse": float(power_nmse),
        "power_nmse_db": float(10.0 * np.log10(max(power_nmse, 1e-30))),
        "amplitude_db_mae": float(np.mean(np.abs(db_error[valid]))),
        "amplitude_db_error_p95": float(np.quantile(np.abs(db_error[valid]), 0.95)),
        "mean_command_correlation": float(
            np.mean(
                [
                    np.corrcoef(prediction[index], target[index])[0, 1]
                    for index in range(target.shape[0])
                ]
            )
        ),
    }


def _level_bins(prediction: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    peak = np.max(target, axis=1, keepdims=True)
    relative_db = 20.0 * np.log10(np.maximum(target / np.maximum(peak, 1e-30), 1e-12))
    floor = max(float(np.max(target)) * 1e-8, 1e-30)
    db_error = np.abs(
        20.0 * np.log10(np.maximum(prediction, floor))
        - 20.0 * np.log10(np.maximum(target, floor))
    )
    bounds = (
        ("0_to_-3_dB", -3.0, 1e-9),
        ("-3_to_-6_dB", -6.0, -3.0),
        ("-6_to_-10_dB", -10.0, -6.0),
        ("-10_to_-20_dB", -20.0, -10.0),
        ("below_-20_dB", -np.inf, -20.0),
    )
    result: dict[str, dict[str, float]] = {}
    top_ten = relative_db >= -10.0
    top_ten_error = db_error[top_ten]
    result["top_10_dB_combined"] = {
        "count": int(top_ten_error.size),
        "fraction": float(top_ten_error.size / target.size),
        "db_mae": float(np.mean(top_ten_error)),
        "db_error_p95": float(np.quantile(top_ten_error, 0.95)),
    }
    for name, low, high in bounds:
        mask = (relative_db >= low) & (relative_db < high)
        values = db_error[mask]
        result[name] = {
            "count": int(values.size),
            "fraction": float(values.size / target.size),
            "db_mae": float(np.mean(values)) if values.size else float("nan"),
            "db_error_p95": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        }
    return result


@torch.no_grad()
def _predict(model: torch.nn.Module, commands: np.ndarray, q: np.ndarray, batch_size: int) -> np.ndarray:
    parameter = next(model.parameters())
    command_tensor = torch.as_tensor(commands, dtype=parameter.dtype, device=parameter.device)
    chunks: list[np.ndarray] = []
    for start in range(0, q.shape[0], batch_size):
        direction = torch.as_tensor(
            q[start : start + batch_size], dtype=parameter.dtype, device=parameter.device
        )
        chunks.append(model.forward_command_grid(command_tensor, direction).cpu().numpy())
    return np.concatenate(chunks, axis=1)


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    contract = checkpoint.get("input_contract", {})
    if contract.get("measured_phase_used") is not False or contract.get("s_parameters_used") is not False:
        raise ValueError("checkpoint violates strict amplitude-only contract")
    receiver_channel = dataset.manifest.receiver_channel
    if receiver_channel not in {"etheta", "ephi"}:
        raise ValueError("external evaluator currently requires etheta or ephi")

    with np.load(args.external_source, allow_pickle=False) as source:
        q = np.asarray(source["q"], dtype=np.float64)
        h = np.asarray(source["h"])
        theta = np.asarray(source["theta"], dtype=np.float64)
        phi = np.asarray(source["phi"], dtype=np.float64)
    front = q[:, 0] > 0.0
    q, h, theta, phi = q[front], h[front], theta[front], phi[front]
    e_theta, e_phi = spherical_basis(theta, phi)
    receiver = e_theta if receiver_channel == "etheta" else e_phi
    scalar_operator = np.einsum("qx,qxn->qn", receiver, h, optimize=True)

    command_indices = checkpoint["command_split_indices"]["test"].cpu().numpy().astype(np.int64)
    command_iq = dataset.command_iq[command_indices]
    complex_commands = command_iq[:, 0] + 1j * command_iq[:, 1]
    target = np.abs(
        np.einsum(
            "qn,cn->cq", scalar_operator, complex_commands.reshape(complex_commands.shape[0], 16), optimize=True
        )
    ).astype(np.float32) / dataset.manifest.reference_distance_m

    model = model_from_config(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    trained = _predict(model, command_iq, q, args.batch_size)

    ideal = model_from_config(checkpoint["model_config"]).to(args.device)
    ideal.eval()
    baseline = _predict(ideal, command_iq, q, args.batch_size)

    # Give the analytic ideal array a fair scalar calibration, but fit that
    # scalar exclusively on the amplitude-only training partition.  No
    # external holdout label is used for calibration.
    training_command_indices = (
        checkpoint["command_split_indices"]["train"].cpu().numpy().astype(np.int64)
    )
    training_query_indices = (
        checkpoint["query_split_indices"]["train"].cpu().numpy().astype(np.int64)
    )
    calibration_commands = dataset.command_iq[training_command_indices]
    calibration_target = dataset.amplitude[
        np.ix_(training_command_indices, training_query_indices)
    ]
    calibration_prediction = _predict(
        ideal,
        calibration_commands,
        dataset.direction[training_query_indices],
        args.batch_size,
    )
    ideal_scale = float(
        np.sum(calibration_prediction * calibration_target, dtype=np.float64)
        / max(float(np.sum(calibration_prediction**2, dtype=np.float64)), 1e-30)
    )
    calibrated_baseline = ideal_scale * baseline

    payload = {
        "experiment": "strict_positive_hemisphere_single_polarization_amplitude",
        "receiver_channel": receiver_channel,
        "hemisphere": "q_x > 0 only",
        "external_source": str(args.external_source.resolve()),
        "external_direction_count": int(q.shape[0]),
        "unseen_command_count": int(command_iq.shape[0]),
        "training_contract": {
            "input_arrays": [
                "direction.npy",
                "command_iq.npy",
                "amplitude.npy",
                "command_family.npy",
                "theta_deg.npy",
                "phi_deg.npy",
                "manifest.json",
            ],
            "measured_phase_used": False,
            "complex_h_used": False,
            "q_matrix_used": False,
            "s_parameters_used": False,
        },
        "ideal_phase_baseline": _metrics(baseline, target),
        "training_amplitude_calibrated_ideal_phase_baseline": {
            "fitted_scalar": ideal_scale,
            "fit_partition": "training commands x training angular blocks only",
            **_metrics(calibrated_baseline, target),
        },
        "trained_amplitude_only": _metrics(trained, target),
        "trained_error_by_true_relative_level": _level_bins(trained, target),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
