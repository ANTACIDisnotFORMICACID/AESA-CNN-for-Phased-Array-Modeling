"""Re-evaluate optimized 4x4 weights directly on imported HFSS FFD basis data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..contracts import ArrayGeometry
from ..coordinates import query_from_cartesian_targets
from ..io import load_basis_dataset
from ..io import amplitude_for_commands_from_basis
from ..optimization import nearest_basis_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-angular-error-deg", type=float, default=0.51)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = torch.as_tensor(json.loads(args.targets.read_text(encoding="utf-8-sig")), dtype=torch.float32)
    if targets.shape != (4, 3):
        raise ValueError("targets JSON must have shape [4,3]")
    query = query_from_cartesian_targets(targets[None], ArrayGeometry(), enforce_far_field=True)
    payload = json.loads(args.weights.read_text(encoding="utf-8-sig"))
    amplitude = np.asarray(payload["amplitude_4x4"], dtype=np.float32)
    phase_rad = np.deg2rad(np.asarray(payload["phase_deg_4x4"], dtype=np.float32))
    if amplitude.shape != (4, 4) or phase_rad.shape != (4, 4):
        raise ValueError("weights must contain amplitude_4x4 and phase_deg_4x4")
    command = torch.as_tensor(
        np.stack((amplitude * np.cos(phase_rad), amplitude * np.sin(phase_rad)), axis=0)
    )[None]
    basis = load_basis_dataset(args.dataset)
    indices, angular_error = nearest_basis_indices(
        query.unit_direction[0].numpy(),
        basis.direction,
        max_angular_error_deg=args.max_angular_error_deg,
    )
    amplitude_at_grid = amplitude_for_commands_from_basis(
        basis,
        command.numpy(),
        payload["receiver_channel"],
    )[0, indices]
    target_range = query.range_m[0].numpy()
    amplitude = amplitude_at_grid * float(payload["reference_distance_m"]) / target_range
    target_power = amplitude**2
    result = {
        "schema_version": "v2.8-amplitude-1",
        "verification_source": "direct_hfss_ffd_receiver_projection",
        "receiver_channel": payload["receiver_channel"],
        "grid_indices": indices.tolist(),
        "angular_error_deg": angular_error.tolist(),
        "target_amplitude": amplitude.tolist(),
        "target_power": target_power.tolist(),
        "min_target_power": float(target_power.min()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
