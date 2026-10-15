"""Evaluate a V2.11 encoding checkpoint on the reused HFSS reserved holdout."""

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

from v210.coordinates import spherical_basis  # noqa: E402
from v210.io import load_amplitude_dataset  # noqa: E402
from v211.training_bridge import load_v211_checkpoint_model  # noqa: E402


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    linear_nmse = np.mean(error**2) / max(float(np.mean(target**2)), 1e-30)
    pred_power = prediction**2
    target_power = target**2
    power_nmse = np.mean((pred_power - target_power) ** 2) / max(
        float(np.mean(target_power**2)), 1e-30
    )
    floor = max(float(np.max(target)) * 1e-8, 1e-30)
    db_error = 20.0 * np.log10(np.maximum(prediction, floor)) - 20.0 * np.log10(
        np.maximum(target, floor)
    )
    return {
        "observation_count": int(target.size),
        "linear_nmse": float(linear_nmse),
        "linear_nmse_db": float(10.0 * np.log10(max(linear_nmse, 1e-30))),
        "power_nmse": float(power_nmse),
        "power_nmse_db": float(10.0 * np.log10(max(power_nmse, 1e-30))),
        "amplitude_db_mae": float(np.mean(np.abs(db_error))),
        "amplitude_db_error_p95": float(np.quantile(np.abs(db_error), 0.95)),
        "mean_command_correlation": float(
            np.mean(
                [np.corrcoef(prediction[index], target[index])[0, 1] for index in range(target.shape[0])]
            )
        ),
    }


def level_bins(prediction: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
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
    top_ten = db_error[relative_db >= -10.0]
    result["top_10_dB_combined"] = {
        "count": int(top_ten.size),
        "fraction": float(top_ten.size / target.size),
        "db_mae": float(np.mean(top_ten)),
        "db_error_p95": float(np.quantile(top_ten, 0.95)),
    }
    for name, low, high in bounds:
        values = db_error[(relative_db >= low) & (relative_db < high)]
        result[name] = {
            "count": int(values.size),
            "fraction": float(values.size / target.size),
            "db_mae": float(np.mean(values)) if values.size else float("nan"),
            "db_error_p95": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        }
    return result


@torch.no_grad()
def predict(model: torch.nn.Module, commands: np.ndarray, direction: np.ndarray, batch_size: int) -> np.ndarray:
    parameter = next(model.parameters())
    command = torch.as_tensor(commands, dtype=parameter.dtype, device=parameter.device)
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, direction.shape[0], batch_size):
        query = torch.as_tensor(
            direction[start : start + batch_size], dtype=parameter.dtype, device=parameter.device
        )
        chunks.append(model.forward_command_grid(command, query).cpu().numpy())
    return np.concatenate(chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--external-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = checkpoint["input_contract"]
    if contract["measured_phase_used"] is not False or contract["s_parameters_used"] is not False:
        raise ValueError("checkpoint violates amplitude-only training contract")
    with np.load(args.external_source, allow_pickle=False) as source:
        q = np.asarray(source["q"], dtype=np.float64)
        h = np.asarray(source["h"])
        theta = np.asarray(source["theta"], dtype=np.float64)
        phi = np.asarray(source["phi"], dtype=np.float64)
    front = q[:, 0] > 0.0
    q, h, theta, phi = q[front], h[front], theta[front], phi[front]
    e_theta, e_phi = spherical_basis(theta, phi)
    receiver = e_theta if dataset.manifest.receiver_channel == "etheta" else e_phi
    scalar_operator = np.einsum("qx,qxn->qn", receiver, h, optimize=True)
    command_indices = checkpoint["command_split_indices"]["test"].numpy().astype(np.int64)
    command_iq = dataset.command_iq[command_indices]
    command_complex = command_iq[:, 0] + 1j * command_iq[:, 1]
    target = np.abs(
        np.einsum(
            "qn,cn->cq",
            scalar_operator,
            command_complex.reshape(command_complex.shape[0], 16),
            optimize=True,
        )
    ).astype(np.float32) / dataset.manifest.reference_distance_m
    model = load_v211_checkpoint_model(checkpoint, args.device)
    prediction = predict(model, command_iq, q, args.batch_size)
    payload = {
        "variant": checkpoint["encoding_config"]["kind"],
        "seed": checkpoint["seed"],
        "external_direction_count": int(q.shape[0]),
        "unseen_command_count": int(command_iq.shape[0]),
        "holdout_status": "reused: opened previously by V2.10; not blind for V2.11 selection",
        "training_contract": contract,
        "metrics": metrics(prediction, target),
        "error_by_true_relative_level": level_bins(prediction, target),
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
