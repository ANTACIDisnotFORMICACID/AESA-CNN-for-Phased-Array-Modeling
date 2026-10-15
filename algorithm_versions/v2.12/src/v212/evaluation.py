"""Amplitude metrics and the mandatory direction-command cross-generalization matrix."""

from __future__ import annotations

import math

import torch


FIELD_BINS_DB = ((-3.0, 0.0), (-10.0, -3.0), (-20.0, -10.0), (-float("inf"), -20.0))


def amplitude_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, object]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching shape [Q,C]")
    if torch.any(prediction < 0) or torch.any(target < 0):
        raise ValueError("amplitudes must be non-negative")
    error = prediction - target
    nmse = error.square().sum() / target.square().sum().clamp_min(epsilon)
    target_peak = target.max(dim=0, keepdim=True).values.clamp_min(epsilon)
    relative_db = 20.0 * torch.log10((target / target_peak).clamp_min(epsilon))
    error_db = 20.0 * torch.log10(prediction.clamp_min(epsilon)) - 20.0 * torch.log10(
        target.clamp_min(epsilon)
    )
    bins: dict[str, dict[str, float | int]] = {}
    for lower, upper in FIELD_BINS_DB:
        if lower == -float("inf"):
            mask = relative_db < upper
            name = "below_-20_db"
        else:
            mask = (relative_db >= lower) & (relative_db < upper + (1e-9 if upper == 0 else 0))
            name = f"{int(lower)}_to_{int(upper)}_db"
        values = error_db[mask]
        linear_error = error[mask]
        linear_target = target[mask]
        bin_nmse = (
            linear_error.square().sum()
            / linear_target.square().sum().clamp_min(epsilon)
            if values.numel()
            else torch.tensor(float("nan"), device=target.device)
        )
        bins[name] = {
            "count": int(values.numel()),
            "linear_nmse": float(bin_nmse),
            "linear_nmse_db": float(10.0 * torch.log10(bin_nmse.clamp_min(epsilon))),
            "mae_db": float(values.abs().mean()) if values.numel() else float("nan"),
            "bias_db": float(values.mean()) if values.numel() else float("nan"),
            "p95_abs_db": float(torch.quantile(values.abs(), 0.95)) if values.numel() else float("nan"),
        }
    return {
        "nmse": float(nmse),
        "nmse_db": float(10.0 * torch.log10(nmse.clamp_min(epsilon))),
        "mae_db": float(error_db.abs().mean()),
        "bias_db": float(error_db.mean()),
        "p95_abs_db": float(torch.quantile(error_db.abs(), 0.95)),
        "field_strength_bins": bins,
    }


def cross_generalization_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    train_direction: torch.Tensor,
    test_direction: torch.Tensor,
    train_command: torch.Tensor,
    test_command: torch.Tensor,
) -> dict[str, dict[str, object]]:
    """Return A/B/C/D; model training is allowed to use only cell A."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have shape [direction,command]")
    train_direction = torch.as_tensor(train_direction, dtype=torch.long)
    test_direction = torch.as_tensor(test_direction, dtype=torch.long)
    train_command = torch.as_tensor(train_command, dtype=torch.long)
    test_command = torch.as_tensor(test_command, dtype=torch.long)
    if torch.isin(train_direction, test_direction).any():
        raise ValueError("seen and unseen direction indices overlap")
    if torch.isin(train_command, test_command).any():
        raise ValueError("seen and unseen command indices overlap")
    if any(
        torch.any(index < 0) or torch.any(index >= limit)
        for index, limit in (
            (train_direction, prediction.shape[0]),
            (test_direction, prediction.shape[0]),
            (train_command, prediction.shape[1]),
            (test_command, prediction.shape[1]),
        )
    ):
        raise ValueError("cross-generalization indices are out of range")
    q_sets = {"seen": train_direction, "unseen": test_direction}
    w_sets = {"seen": train_command, "unseen": test_command}
    cells = {
        "A_seen_q_seen_w": ("seen", "seen"),
        "B_seen_q_unseen_w": ("seen", "unseen"),
        "C_unseen_q_seen_w": ("unseen", "seen"),
        "D_unseen_q_unseen_w": ("unseen", "unseen"),
    }
    report: dict[str, dict[str, object]] = {}
    for name, (q_name, w_name) in cells.items():
        q_index = q_sets[q_name].to(device=prediction.device)
        w_index = w_sets[w_name].to(device=prediction.device)
        selected_prediction = prediction.index_select(0, q_index).index_select(1, w_index)
        selected_target = target.index_select(0, q_index).index_select(1, w_index)
        report[name] = amplitude_metrics(selected_prediction, selected_target)
    return report


def hemisphere_cells(unit_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equal-area x-polar rings, azimuth sectors and 24-cell ids."""
    if unit_direction.ndim != 2 or unit_direction.shape[1] != 3:
        raise ValueError("unit_direction must have shape [Q,3]")
    direction = unit_direction / unit_direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if torch.any(direction[:, 0] < -1e-7):
        raise ValueError("V2.12 hemisphere cells require q_x >= 0")
    ring = torch.floor(direction[:, 0].clamp(0, 1 - 1e-7) * 3).to(torch.long).clamp_max(2)
    azimuth = torch.remainder(torch.atan2(direction[:, 2], direction[:, 1]), 2 * math.pi)
    sector = torch.floor(azimuth / (2 * math.pi / 8)).to(torch.long).clamp_max(7)
    return ring, sector, ring * 8 + sector


def stratified_separation_subset(
    training_direction: torch.Tensor,
    *,
    per_cell: int = 40,
    seed: int = 21290,
    training_weights: torch.Tensor | None = None,
    require_all_cells: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select fixed q_sep and weights preserving each cell's train-measure mass."""
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    _, _, cell = hemisphere_cells(training_direction)
    count = len(training_direction)
    if training_weights is None:
        source_weights = torch.full((count,), 1.0 / count, dtype=torch.float64)
    else:
        source_weights = torch.as_tensor(training_weights, dtype=torch.float64).reshape(-1)
        if source_weights.shape != (count,) or torch.any(source_weights < 0):
            raise ValueError("training_weights are invalid")
        source_weights = source_weights / source_weights.sum().clamp_min(1e-12)
    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    selected_weights: list[torch.Tensor] = []
    for cell_id in range(24):
        members = torch.nonzero(cell == cell_id, as_tuple=False).flatten()
        if not len(members):
            if require_all_cells:
                raise ValueError(f"hemisphere cell {cell_id} is empty")
            continue
        take = min(per_cell, len(members))
        chosen = members[torch.randperm(len(members), generator=generator)[:take]]
        mass = source_weights[members].sum()
        selected.append(chosen)
        selected_weights.append(torch.full((take,), float(mass / take), dtype=torch.float64))
    if not selected:
        raise ValueError("no non-empty hemisphere cell is available")
    index = torch.cat(selected)
    weights = torch.cat(selected_weights)
    return index, weights / weights.sum()


def spatial_gap_partition(training_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic fold-0 gap split using the frozen 24-cell rule."""
    ring, sector, _ = hemisphere_cells(training_direction)
    gap = torch.remainder(sector + 2 * ring, 5) == 0
    return torch.nonzero(~gap, as_tuple=False).flatten(), torch.nonzero(gap, as_tuple=False).flatten()


def nearest_training_angle(
    validation_direction: torch.Tensor,
    training_direction: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Great-circle distance to the nearest training direction."""
    train = training_direction / training_direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    validation = validation_direction / validation_direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    result: list[torch.Tensor] = []
    for start in range(0, len(validation), chunk_size):
        cosine = validation[start : start + chunk_size] @ train.T
        result.append(torch.acos(cosine.max(dim=1).values.clamp(-1.0, 1.0)))
    return torch.cat(result)


def angular_distance_quartile_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validation_direction: torch.Tensor,
    training_direction: torch.Tensor,
) -> dict[str, object]:
    if prediction.shape != target.shape or prediction.shape[0] != len(validation_direction):
        raise ValueError("prediction/target rows must match validation directions")
    distance = nearest_training_angle(validation_direction, training_direction)
    boundaries = torch.quantile(distance, torch.tensor([0.25, 0.5, 0.75], device=distance.device))
    groups = torch.bucketize(distance, boundaries)
    report: dict[str, object] = {}
    for quartile in range(4):
        mask = groups == quartile
        report[f"Q{quartile + 1}"] = {
            "count": int(mask.sum()),
            "angle_min_deg": float(torch.rad2deg(distance[mask].min())),
            "angle_max_deg": float(torch.rad2deg(distance[mask].max())),
            **amplitude_metrics(prediction[mask], target[mask]),
        }
    return report
