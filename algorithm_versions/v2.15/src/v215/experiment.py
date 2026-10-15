"""Frozen V2.15 experiment constants, packages and model construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from v212.basis import FiltrationSSHTransform
from v212.operator import CoherentRank1Operator

from .graph_operator import GRAPH_MODES, V215GraphOperator


ARMS = ("control", *GRAPH_MODES)
SEEDS = (21501, 21502, 21503)
PRIMARY_CANDIDATE = "graph_s_true"


def complex_commands(iq: np.ndarray) -> torch.Tensor:
    value = np.asarray(iq, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (2, 16):
        raise ValueError("command_iq must have shape [C,2,16]")
    return torch.complex(torch.from_numpy(value[:, 0]), torch.from_numpy(value[:, 1]))


def load_amplitude_package(path: str | Path, *, expected_commands: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"direction", "amplitude", "command_indices", "command_iq"}
    if not required.issubset(data):
        raise ValueError(f"missing package fields: {sorted(required - set(data))}")
    query_count = len(data["direction"])
    if data["direction"].shape != (query_count, 3):
        raise ValueError("direction must have shape [Q,3]")
    if data["amplitude"].shape != (query_count, expected_commands):
        raise ValueError("amplitude package shape mismatch")
    if data["command_iq"].shape != (expected_commands, 2, 16):
        raise ValueError("command IQ package shape mismatch")
    if not np.isfinite(data["direction"]).all() or not np.isfinite(data["amplitude"]).all():
        raise ValueError("package contains non-finite values")
    if np.any(data["amplitude"] < 0):
        raise ValueError("amplitudes must be non-negative")
    forbidden = {name for name in data if "test" in name.lower() or name.startswith("g_")}
    if forbidden:
        raise ValueError(f"training/validation package contains forbidden fields: {sorted(forbidden)}")
    return data


def build_model(
    arm: str,
    *,
    transform: FiltrationSSHTransform,
    s_matrix: np.ndarray,
    envelope_scale: float,
    seed: int,
) -> torch.nn.Module:
    if arm not in ARMS:
        raise ValueError(f"unknown V2.15 arm: {arm}")
    common = {
        "ssh_transform": transform,
        "ssh_seed": seed,
        "envelope_scale": envelope_scale,
        "fourier_input_mode": "raw",
    }
    torch.manual_seed(seed)
    if arm == "control":
        return CoherentRank1Operator(residual_width=24, residual_depth=2, **common)
    return V215GraphOperator(
        s_matrix=s_matrix,
        coupling_mode=arm,
        graph_width=32,
        graph_top_k=4,
        graph_shuffle_seed=21500,
        **common,
    )


def load_model_from_run(run_dir: str | Path, s_matrix: np.ndarray, device: str = "cpu"):
    run = Path(run_dir)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("test_split_opened") is not False:
        raise ValueError(f"checkpoint is not test-sealed: {run}")
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    model = build_model(
        str(checkpoint["arm"]),
        transform=transform,
        s_matrix=s_matrix,
        envelope_scale=float(checkpoint["envelope_scale"]),
        seed=int(checkpoint["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint

