"""Run cross-module V2.12 acceptance checks on a small train-only HFSS slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.contracts import QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.operator import CoherentRank1Operator  # noqa: E402
from v212.stage1 import Stage1EncodingConfig, build_stage1_model  # noqa: E402


def _base_config(dataset) -> dict[str, object]:
    return {
        "architecture_version": "v210_nominal_coupling_residual_1",
        "basis_type": "ideal_phase",
        "source_type": dataset.manifest.source_type,
        "direction_fourier_bands": 4,
        "nominal_width": 24,
        "nominal_depth": 2,
        "nominal_max_log_amplitude": 3.0,
        "enable_coupling": True,
        "coupling_low_rank": 1,
        "enable_residual": True,
        "residual_width": 12,
        "residual_depth": 1,
        "residual_maximum_magnitude": 2.0,
        "reference_distance_m": dataset.manifest.reference_distance_m,
    }


def _parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--l4-transform", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    transform = FiltrationSSHTransform.load(args.l4_transform)
    q_train = torch.as_tensor(dataset.direction[split.query_train], dtype=torch.float32)
    q = q_train[:8]
    command_iq = torch.as_tensor(dataset.command_iq[split.command_train[:4]])
    dataset_commands = torch.complex(
        torch.as_tensor(dataset.command_iq[:, 0].reshape(96, 16)),
        torch.as_tensor(dataset.command_iq[:, 1].reshape(96, 16)),
    )
    torch.testing.assert_close(dataset_commands, codebook.weights, atol=1e-6, rtol=1e-6)

    base = _base_config(dataset)
    torch.manual_seed(21201)
    e0, _ = build_stage1_model(base, Stage1EncodingConfig("E0", seed=21201), q_train)
    torch.manual_seed(21201)
    e2, _ = build_stage1_model(
        base, Stage1EncodingConfig("E2-F4", seed=21201), q_train, transform=transform
    )
    torch.manual_seed(21201)
    matched, _ = build_stage1_model(
        base, Stage1EncodingConfig("E0-M4", seed=21201), q_train, transform=transform
    )
    with torch.no_grad():
        e0_prediction = e0.forward_command_grid(command_iq, q)
        e2_prediction = e2.forward_command_grid(command_iq, q)
    torch.testing.assert_close(e2_prediction, e0_prediction, atol=0.0, rtol=0.0)
    if _parameters(e2) != _parameters(matched):
        raise AssertionError("E2-F4 and E0-M4 parameter counts differ")

    torch.manual_seed(21201)
    g1 = CoherentRank1Operator()
    torch.manual_seed(21201)
    g2 = CoherentRank1Operator(ssh_transform=transform, ssh_seed=21201)
    with torch.no_grad():
        g1_prediction = g1.forward_command_grid(codebook.weights[:4], q)
        g2_prediction = g2.forward_command_grid(codebook.weights[:4], q)
        q_matrix = g2.q_matrix(QueryBatch(q[None]))
    torch.testing.assert_close(g2_prediction, g1_prediction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(q_matrix, q_matrix.mH, atol=1e-5, rtol=1e-5)

    payload = {
        "schema": "v212-real-data-smoke-acceptance-1",
        "train_directions_used": len(q_train),
        "test_split_opened": False,
        "checks": {
            "dataset_codebook_exact": True,
            "e2_initial_equals_e0": True,
            "e2_e0m_parameter_count_equal": True,
            "g2_initial_equals_g1": True,
            "q_hermitian": True,
        },
        "l4_basis": {
            "ranks": list(transform.ranks),
            "raw_gram_condition": transform.raw_gram_condition,
            "retained_condition": transform.retained_condition,
            "transformed_condition": transform.transformed_condition,
        },
        "parameter_counts": {
            "E0": _parameters(e0),
            "E0-M4": _parameters(matched),
            "E2-F4": _parameters(e2),
            "G1": _parameters(g1),
            "G2": _parameters(g2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
