"""Ablate V2.10 physical layers on a held-out amplitude split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..contracts import QueryBatch
from ..io import amplitude_dataset_fingerprint, load_amplitude_dataset
from ..models import model_from_config
from ..training import AmplitudeDirectionDataset, evaluate_amplitude_grid_model
from .train_forward import CHECKPOINT_SCHEMA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("structure diagnosis requires a V2.10 checkpoint")
    dataset = load_amplitude_dataset(args.dataset)
    if checkpoint.get("dataset_fingerprint") != amplitude_dataset_fingerprint(args.dataset):
        raise ValueError("checkpoint fingerprint does not match the amplitude dataset")
    model = model_from_config(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    command_indices = checkpoint["command_split_indices"][args.split].cpu().numpy().astype(np.int64)
    query_indices = checkpoint["query_split_indices"][args.split].cpu().numpy().astype(np.int64)
    evaluation_data = AmplitudeDirectionDataset(dataset, command_indices, query_indices)
    loader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=False)

    variants = {
        "full": (True, True),
        "without_coupling": (False, True),
        "without_residual": (True, False),
        "nominal_only": (False, False),
    }
    metrics: dict[str, object] = {}
    for name, (coupling, residual) in variants.items():
        model.enable_coupling = coupling
        model.enable_residual = residual
        metrics[name] = evaluate_amplitude_grid_model(
            model,
            loader,
            evaluation_data.command_iq,
            noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
        )
    model.enable_coupling = True
    model.enable_residual = True

    sample_direction = torch.as_tensor(
        dataset.direction[query_indices[: min(2048, query_indices.size)]],
        dtype=torch.float32,
        device=args.device,
    )
    with torch.no_grad():
        _, terms = model.operator(QueryBatch(sample_direction[None]))
        term_rms = {
            name: float(torch.sqrt(value.square().mean()).cpu())
            for name, value in terms.items()
        }
        coupling_regularization = {
            name: float(value.cpu())
            for name, value in model.port_coupling.regularization_terms().items()
        }
        matrix = model.port_coupling.matrix_iq()
        identity_iq = torch.stack(
            (torch.eye(16, device=matrix.device, dtype=matrix.dtype), torch.zeros(16, 16, device=matrix.device, dtype=matrix.dtype)),
            dim=0,
        )
        coupling_delta_rms = float(torch.sqrt((matrix - identity_iq).square().mean()).cpu())

    payload = {
        "schema_version": "v2.9-structure-diagnosis-1",
        "split": args.split,
        "measured_phase_used": False,
        "s_parameters_used": False,
        "metrics": metrics,
        "term_rms_on_query_sample": term_rms,
        "coupling_delta_rms": coupling_delta_rms,
        "coupling_regularization": coupling_regularization,
        "parameter_count_by_module": {
            "nominal_pattern": sum(p.numel() for p in model.nominal_pattern.parameters()),
            "port_coupling": sum(p.numel() for p in model.port_coupling.parameters()),
            "angular_residual": sum(p.numel() for p in model.angular_residual.parameters()),
            "total": sum(p.numel() for p in model.parameters()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

