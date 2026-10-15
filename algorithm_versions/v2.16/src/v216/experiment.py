"""Frozen V2.16 experiment arms, data contracts and model construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from v212.basis import FiltrationSSHTransform
from v212.operator import CoherentRank1Operator

from .axial_operator import (
    V216AxialOperator,
    V216ControlledDCAROperator,
    V216IsotropicOperator,
)


ARMS = ("control_v212", "axial_static", "axial_gated", "iso_control")
SEEDS = (21601, 21602, 21603)
PRIMARY_CANDIDATE = "axial_gated"
STAGE1_ARMS = (
    "control_v212",
    "dcar_static_native",
    "dcar_gated_native",
    "dcar_static_pm",
    "dcar_gated_pm",
)
STAGE1_TRAIN_ARMS = STAGE1_ARMS[1:]
STAGE1_PRIMARY_CANDIDATE = "dcar_gated_pm"


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
        raise ValueError(f"package contains forbidden fields: {sorted(forbidden)}")
    return data


def build_model(
    arm: str,
    *,
    transform: FiltrationSSHTransform,
    envelope_scale: float,
    seed: int,
) -> torch.nn.Module:
    if arm not in ARMS:
        raise ValueError(f"unknown V2.16 arm: {arm}")
    common = {
        "ssh_transform": transform,
        "ssh_seed": seed,
        "envelope_scale": envelope_scale,
        "fourier_input_mode": "raw",
    }
    torch.manual_seed(seed)
    if arm == "control_v212":
        return CoherentRank1Operator(residual_width=24, residual_depth=2, **common)
    if arm in {"axial_static", "axial_gated"}:
        return V216AxialOperator(axial_mode=arm, axial_width=68, **common)
    return V216IsotropicOperator(isotropic_width=56, **common)


def build_stage1_model(
    arm: str,
    *,
    transform: FiltrationSSHTransform,
    envelope_scale: float,
    seed: int,
) -> torch.nn.Module:
    if arm not in STAGE1_ARMS:
        raise ValueError(f"unknown V2.16 Stage-1 arm: {arm}")
    if arm == "control_v212":
        return build_model(
            arm, transform=transform, envelope_scale=envelope_scale, seed=seed
        )
    mode = "gated" if "gated" in arm else "static"
    branch_widths = ((36, 35), (35, 36)) if arm.endswith("_pm") else None
    torch.manual_seed(seed)
    return V216ControlledDCAROperator(
        ssh_transform=transform,
        ssh_seed=seed,
        envelope_scale=envelope_scale,
        fourier_input_mode="raw",
        dcar_mode=mode,
        residual_width=24,
        residual_depth=2,
        branch_widths=branch_widths,
    )


def load_model_from_run(run_dir: str | Path, device: str = "cpu"):
    run = Path(run_dir)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("old_v215_test_used") is not False:
        raise ValueError(f"checkpoint did not preserve the old-test seal: {run}")
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    model = build_model(
        str(checkpoint["arm"]),
        transform=transform,
        envelope_scale=float(checkpoint["envelope_scale"]),
        seed=int(checkpoint["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def load_stage1_model_from_run(run_dir: str | Path, device: str = "cpu"):
    run = Path(run_dir)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("old_v215_test_used") is not False:
        raise ValueError(f"checkpoint did not preserve the old-test seal: {run}")
    transform = FiltrationSSHTransform.load(run / "filtration_transform.npz")
    model = build_stage1_model(
        str(checkpoint["arm"]),
        transform=transform,
        envelope_scale=float(checkpoint["envelope_scale"]),
        seed=int(checkpoint["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
