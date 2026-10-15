"""Frozen V2.18 model and loss contracts."""

from __future__ import annotations

import torch

from v212.basis import FiltrationSSHTransform
from v216.experiment import build_stage1_model


BASELINE_ARMS = (
    "control_v212_canonical",
    "dcar_static_pm_canonical",
    "dcar_gated_pm_canonical",
)
OPAUX_ARM = "dcar_gated_pm_opaux"
PHASE_AUX_ARMS = (
    "control_v212_phase_aux",
    "dcar_gated_pm_phase_aux",
)
STAGE2_ARMS = (
    "control_v212_canonical",
    "control_v212_phase_aux",
    "dcar_gated_pm_canonical",
    "dcar_gated_pm_phase_aux",
)
ARMS = (*BASELINE_ARMS, OPAUX_ARM, *PHASE_AUX_ARMS)
SEEDS = (21801, 21802, 21803)

_V216_ARM = {
    "control_v212_canonical": "control_v212",
    "dcar_static_pm_canonical": "dcar_static_pm",
    "dcar_gated_pm_canonical": "dcar_gated_pm",
    "dcar_gated_pm_opaux": "dcar_gated_pm",
    "control_v212_phase_aux": "control_v212",
    "dcar_gated_pm_phase_aux": "dcar_gated_pm",
}

LOSS_CONTRACT = {
    "amplitude_linear_nmse": 1.0,
    "power_nmse": 0.1,
    "db_huber": 0.03,
    "residual_energy": 1e-4,
    "separation": 0.001,
    "complex_operator_nmse": 0.1,
    "q_loss": 0.0,
}


def build_model(
    arm: str,
    *,
    transform: FiltrationSSHTransform,
    envelope_scale: float,
    seed: int,
) -> torch.nn.Module:
    """Build through the frozen V2.16 Stage-1 implementation."""
    if arm not in ARMS:
        raise ValueError(f"unknown V2.18 arm: {arm}")
    return build_stage1_model(
        _V216_ARM[arm],
        transform=transform,
        envelope_scale=envelope_scale,
        seed=seed,
    )


def architecture_contract(model: torch.nn.Module, arm: str) -> dict[str, object]:
    count = sum(parameter.numel() for parameter in model.parameters())
    metadata = model.architecture_metadata() if hasattr(model, "architecture_metadata") else {}
    return {
        "v218_arm": arm,
        "v216_builder_arm": _V216_ARM[arm],
        "parameter_count": count,
        "metadata": metadata,
        "operator_auxiliary_enabled": arm == OPAUX_ARM,
        "phase_aligned_auxiliary_enabled": arm in PHASE_AUX_ARMS,
    }
