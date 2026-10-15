"""Evaluate V2.9 on held-out commands and angular blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..io import amplitude_dataset_fingerprint, load_amplitude_dataset
from ..models import model_from_config
from ..training import AmplitudeDirectionDataset, evaluate_amplitude_grid_model
from .train_forward import CHECKPOINT_SCHEMA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("only V2.9 amplitude-operator checkpoints are accepted")
    contract = checkpoint.get("input_contract", {})
    if contract.get("measured_phase_used") is not False or contract.get("s_parameters_used") is not False:
        raise ValueError("checkpoint violates the amplitude-only deployment contract")
    dataset = load_amplitude_dataset(args.dataset)
    if checkpoint.get("dataset_fingerprint") != amplitude_dataset_fingerprint(args.dataset):
        raise ValueError("checkpoint fingerprint does not match the amplitude dataset")
    model = model_from_config(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    command_indices = checkpoint["command_split_indices"][args.split].cpu().numpy().astype(np.int64)
    query_indices = checkpoint["query_split_indices"][args.split].cpu().numpy().astype(np.int64)
    evaluation_data = AmplitudeDirectionDataset(dataset, command_indices, query_indices)
    loader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_amplitude_grid_model(
        model,
        loader,
        evaluation_data.command_iq,
        noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
    )
    families = dataset.command_family[command_indices]
    metrics_by_command_family: dict[str, object] = {}
    for family in np.unique(families):
        family_slots = np.flatnonzero(families == family)
        family_data = AmplitudeDirectionDataset(dataset, command_indices[family_slots], query_indices)
        family_loader = DataLoader(family_data, batch_size=args.batch_size, shuffle=False)
        metrics_by_command_family[str(family)] = {
            "command_count": int(family_slots.size),
            "metrics": evaluate_amplitude_grid_model(
                model,
                family_loader,
                family_data.command_iq,
                noise_floor_amplitude=dataset.manifest.noise_floor_amplitude,
            ),
        }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "split": args.split,
        "unseen_commands": int(command_indices.size),
        "unseen_queries": int(query_indices.size),
        "observation_count": int(command_indices.size * query_indices.size),
        "measured_phase_used": False,
        "s_parameters_used": False,
        "metrics": metrics,
        "metrics_by_command_family": metrics_by_command_family,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
