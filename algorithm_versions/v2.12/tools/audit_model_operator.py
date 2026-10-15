"""Audit a trained G1/G2/G3 model in gauge-invariant operator coordinates."""

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

from v210.contracts import QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.audits import (  # noqa: E402
    dephase_q_matrix,
    fourier_low_order_leakage,
    gauge_invariant_spectrum,
    nested_filtration_spectrum,
    nested_operator_comparison,
    operator_graph_roughness,
    rank_one_q_matrix,
)
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.operator import operator_from_checkpoint  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--solid-angle-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--truth-audit", type=Path)
    parser.add_argument(
        "--direction-split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.direction_split in {"test", "all"} and not args.open_test:
        raise ValueError("test/all operator audit requires explicit --open-test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "v212-rank1-checkpoint-1":
        raise ValueError("unsupported checkpoint schema")
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    transform = None
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    if transform_path.exists():
        transform = FiltrationSSHTransform.load(transform_path)
    observation = ReceiverObservation(**checkpoint["observation"])
    q_train = torch.as_tensor(
        checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
    )
    all_direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=all_direction[q_train],
        transform=transform,
        observation=observation,
    ).to(args.device)
    model.eval()
    if args.direction_split == "all":
        indices = np.arange(len(dataset.direction), dtype=np.int64)
    else:
        indices = np.asarray(getattr(split, f"query_{args.direction_split}"), dtype=np.int64)
    direction = torch.as_tensor(dataset.direction[indices], dtype=torch.float32)
    solid_angle = torch.from_numpy(
        np.asarray(np.load(args.solid_angle_weights, allow_pickle=False)[indices], dtype=np.float64)
    )
    q_tilde_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    degree_parts: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(direction), args.chunk_size):
            q = direction[start : start + args.chunk_size].to(args.device)
            query = QueryBatch(q[None])
            q_tilde_parts.append(model.dephased_q_matrix(query)[0].cpu())
            _, terms = model.envelope(query)
            residual_iq = terms["fourier_residual_iq"][0]
            residual_parts.append(
                torch.complex(residual_iq[:, 0], residual_iq[:, 1]).reshape(len(q), 16).cpu()
            )
            for name, value in terms.items():
                if name.startswith("ssh_l"):
                    degree_parts.setdefault(name, []).append(value[0].cpu())
    q_tilde = torch.cat(q_tilde_parts)
    spectrum = gauge_invariant_spectrum(
        q_tilde,
        direction.double(),
        degree=args.degree,
        solid_angle_weights=solid_angle,
    )
    leakage = None
    if transform is not None:
        phi = transform.evaluate(direction.double())
        leakage = float(
            fourier_low_order_leakage(phi, torch.cat(residual_parts)).detach().cpu()
        )
    degree_rms = {
        name: float(torch.cat(parts).square().mean().sqrt())
        for name, parts in degree_parts.items()
    }
    nested_comparison = None
    graph_truth = None
    if args.truth_audit is not None:
        with np.load(args.truth_audit, allow_pickle=False) as archive:
            truth_g = torch.from_numpy(np.asarray(archive["g"])[indices].copy())
        truth_q = rank_one_q_matrix(truth_g)
        truth_q_tilde = dephase_q_matrix(
            truth_q,
            direction,
            model.positions_m.cpu(),
            model.wavenumber_rad_m,
        )
        nested_comparison = nested_operator_comparison(
            q_tilde,
            truth_q_tilde,
            direction.double(),
            degree=args.degree,
            solid_angle_weights=solid_angle,
        )
        nested = nested_comparison.prediction
        graph_truth = operator_graph_roughness(truth_q_tilde, direction.double())
    else:
        nested = nested_filtration_spectrum(
            q_tilde, direction.double(), degree=args.degree, solid_angle_weights=solid_angle
        )
    graph_prediction = operator_graph_roughness(q_tilde, direction.double())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "model_q_tilde_spectrum.npz",
        degree_energy=spectrum.degree_energy.numpy(),
        coefficient_real=spectrum.coefficients.real.numpy(),
        coefficient_imag=spectrum.coefficients.imag.numpy(),
        nested_cumulative_fraction=nested.cumulative_fraction.numpy(),
        nested_shell_fraction=nested.shell_fraction.numpy(),
        nested_shell_error_fraction=(
            np.asarray([]) if nested_comparison is None
            else nested_comparison.shell_error_fraction.numpy()
        ),
    )
    report = {
        "schema": "v212-trained-model-operator-audit-1",
        "recipe": checkpoint["recipe"],
        "direction_split": args.direction_split,
        "test_split_opened": args.direction_split in {"test", "all"},
        "quantity": "gauge-invariant dephased Q_tilde",
        "cumulative_energy_fraction": {
            str(key): value for key, value in spectrum.cumulative_fraction.items()
        },
        "degree_energy": spectrum.degree_energy.tolist(),
        "normalized_gradient_energy": spectrum.normalized_gradient_energy,
        "fourier_residual_low_filtration_leakage": leakage,
        "leakage_is_diagnostic_not_physical_truth": True,
        "ssh_degree_output_rms": degree_rms,
        "nested_filtration": {
            "cumulative_fraction": nested.cumulative_fraction.tolist(),
            "shell_fraction": nested.shell_fraction.tolist(),
            "normalized_gradient_energy": nested.normalized_gradient_energy,
            "retained_ranks": list(nested.retained_ranks),
            "transformed_condition": nested.transformed_condition,
            "spectral_distance_to_truth": (
                None if nested_comparison is None else nested_comparison.spectral_distance
            ),
            "truth_cumulative_fraction": (
                None if nested_comparison is None
                else nested_comparison.truth.cumulative_fraction.tolist()
            ),
            "truth_shell_fraction": (
                None if nested_comparison is None
                else nested_comparison.truth.shell_fraction.tolist()
            ),
            "truth_normalized_gradient_energy": (
                None if nested_comparison is None
                else nested_comparison.truth.normalized_gradient_energy
            ),
            "shell_error_fraction": (
                None if nested_comparison is None
                else nested_comparison.shell_error_fraction.tolist()
            ),
            "tail_error_fraction": (
                None if nested_comparison is None else nested_comparison.tail_error_fraction
            ),
        },
        "graph_roughness": {
            "prediction": graph_prediction,
            "truth": graph_truth,
            "relative_error": (
                None if graph_truth is None
                else abs(graph_prediction - graph_truth) / max(abs(graph_truth), 1e-12)
            ),
        },
    }
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
