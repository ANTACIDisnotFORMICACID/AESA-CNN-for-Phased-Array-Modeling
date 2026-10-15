"""Optimize 4x4 complex excitation for exactly four Cartesian far-field targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..contracts import ArrayGeometry
from ..coordinates import query_from_cartesian_targets
from ..models import AmplitudeOnlyForwardModel, AmplitudeOperatorResidualCNN
from ..optimization import FourTargetProblem, optimize_four_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path, help="JSON array with shape [4,3], metres")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature-db", type=float, default=0.25)
    parser.add_argument("--total-power", type=float, default=1.0)
    parser.add_argument("--amplitude-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=28001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = torch.as_tensor(
        json.loads(args.targets.read_text(encoding="utf-8-sig")),
        dtype=torch.float32,
        device=args.device,
    )
    if targets.shape != (4, 3):
        raise ValueError("targets JSON must have shape [4,3]")
    geometry = ArrayGeometry()
    query = query_from_cartesian_targets(targets[None], geometry, enforce_far_field=True)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("schema_version") != "v2.8-amplitude-1":
        raise ValueError("optimization requires an amplitude-only v2.8 checkpoint")
    contract = checkpoint.get("input_contract", {})
    if contract.get("measured_phase_used") is not False or contract.get("s_parameters_used") is not False:
        raise ValueError("checkpoint violates the amplitude-only deployment contract")
    config = checkpoint["model_config"]
    model = AmplitudeOnlyForwardModel(
        geometry=geometry,
        correction=AmplitudeOperatorResidualCNN(
            fourier_bands=config["direction_fourier_bands"],
            directional_width=config["directional_width"],
            edge_width=config["edge_width"],
            element_pattern_width=config.get("element_pattern_width", 20),
            multihop_width=config["multihop_width"],
            enable_element_pattern=config.get("enable_element_pattern", False),
            enable_multihop=config["enable_multihop"],
        ),
        reference_distance_m=config["reference_distance_m"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    result = optimize_four_targets(
        model,
        FourTargetProblem(
            query,
            total_command_power=args.total_power,
            per_element_amplitude_max=args.amplitude_max,
        ),
        steps=args.steps,
        learning_rate=args.learning_rate,
        softmin_temperature_db=args.temperature_db,
        random_restarts=args.restarts,
        seed=args.seed,
    )
    command = result.command_iq.cpu().numpy()
    complex_command = command[0] + 1j * command[1]
    payload = {
        "schema_version": "v2.8-amplitude-1",
        "objective": "maximize minimum of four predicted received powers",
        "model_observable": "scalar amplitude only",
        "measured_phase_used_for_training": False,
        "receiver_channel": checkpoint["dataset_manifest"]["receiver_channel"],
        "reference_distance_m": config["reference_distance_m"],
        "training_source_type": checkpoint["dataset_manifest"]["source_type"],
        "training_array_id": checkpoint["dataset_manifest"]["array_id"],
        "deployment_eligible": checkpoint["dataset_manifest"]["source_type"]
        == "anechoic_chamber_amplitude",
        "amplitude_4x4": np.abs(complex_command).tolist(),
        "phase_deg_4x4": np.rad2deg(np.angle(complex_command)).tolist(),
        "target_power": result.target_power.cpu().tolist(),
        "min_target_power": result.min_target_power,
        "constraints": {
            "total_command_power": args.total_power,
            "per_element_amplitude_max": args.amplitude_max,
        },
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved optimized 4x4 amplitudes/phases to {args.output.resolve()}")


if __name__ == "__main__":
    main()
