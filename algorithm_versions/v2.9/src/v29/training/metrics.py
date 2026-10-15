"""Independent forward-operator metrics for held-out angular blocks."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..contracts import QueryBatch
from ..io import BasisDataset
from ..models import PhysicsStructuredForwardModel
from .datasets import OperatorDataset


def operator_metrics(
    prediction_iq: torch.Tensor,
    target_iq: torch.Tensor,
    direction: torch.Tensor,
    *,
    phase_floor_db: float = -60.0,
) -> dict[str, object]:
    if prediction_iq.shape != target_iq.shape or prediction_iq.ndim != 5:
        raise ValueError("operators must share shape [Q,3,2,4,4]")
    if direction.shape != (prediction_iq.shape[0], 3):
        raise ValueError("direction must have shape [Q,3]")
    error_energy = (prediction_iq - target_iq).square().sum()
    target_energy = target_iq.square().sum().clamp_min(1e-20)
    nmse = error_energy / target_energy

    predicted_amplitude = torch.linalg.vector_norm(prediction_iq, dim=2)
    target_amplitude = torch.linalg.vector_norm(target_iq, dim=2)
    amplitude_floor = target_amplitude.max() * (10.0 ** (phase_floor_db / 20.0))
    valid = target_amplitude >= amplitude_floor
    amplitude_error_db = torch.abs(
        20.0 * torch.log10(predicted_amplitude.clamp_min(1e-20))
        - 20.0 * torch.log10(target_amplitude.clamp_min(1e-20))
    )[valid]

    predicted_phase = torch.atan2(prediction_iq[:, :, 1], prediction_iq[:, :, 0])
    target_phase = torch.atan2(target_iq[:, :, 1], target_iq[:, :, 0])
    phase_delta = torch.atan2(
        torch.sin(predicted_phase - target_phase),
        torch.cos(predicted_phase - target_phase),
    ).abs()
    phase_error_deg = torch.rad2deg(phase_delta)[valid]

    per_port_error = (prediction_iq - target_iq).square().sum(dim=(0, 1, 2))
    per_port_target = target_iq.square().sum(dim=(0, 1, 2)).clamp_min(1e-20)
    per_port_nmse = per_port_error / per_port_target
    radial = torch.einsum("qx,qxirc->qirc", direction, prediction_iq)
    transverse_ratio = radial.square().sum() / prediction_iq.square().sum().clamp_min(1e-20)

    def percentile(values: torch.Tensor, probability: float) -> float:
        if values.numel() == 0:
            return float("nan")
        return float(torch.quantile(values, probability).cpu())

    return {
        "complex_nmse": float(nmse.cpu()),
        "complex_nmse_db": float((10.0 * torch.log10(nmse.clamp_min(1e-20))).cpu()),
        "amplitude_error_db_p50": percentile(amplitude_error_db, 0.50),
        "amplitude_error_db_p95": percentile(amplitude_error_db, 0.95),
        "phase_error_deg_p50": percentile(phase_error_deg, 0.50),
        "phase_error_deg_p95": percentile(phase_error_deg, 0.95),
        "valid_phase_fraction": float(valid.to(torch.float32).mean().cpu()),
        "transverse_energy_ratio": float(transverse_ratio.cpu()),
        "per_port_nmse_4x4": per_port_nmse.cpu().tolist(),
        "worst_port_nmse": float(per_port_nmse.max().cpu()),
    }


@torch.no_grad()
def evaluate_operator_model(
    model: PhysicsStructuredForwardModel,
    basis: BasisDataset,
    indices: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    dataset = OperatorDataset(basis, indices)
    if len(dataset) == 0:
        raise ValueError("evaluation split is empty")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    directions: list[torch.Tensor] = []
    model.eval()
    model.to(device)
    for direction, target in loader:
        direction = direction.to(device)
        prediction = model.operator(QueryBatch(direction[:, None])).effective_iq[:, 0]
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        directions.append(direction.cpu())
    return operator_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(directions, dim=0),
    )

