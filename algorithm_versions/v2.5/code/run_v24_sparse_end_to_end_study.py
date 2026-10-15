"""V2.4：稀疏远场测点监督、稀疏校准链与整数输出参考。"""

from __future__ import annotations

import copy
import csv
import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    array_positions,
)
from physics_initialized_frontend import (
    PhysicsInitializedApertureEstimator,
    complex_to_iq,
    design_probe_grid,
    iq_to_complex,
    near_field_green_matrix,
    regularized_pseudoinverse,
)
from experiment_common import (
    SOURCE,
    TARGET,
    Samples,
    evaluate,
    make_samples,
    model_forward,
    target_gain,
    train_model,
)
from sparse_beamforming_v24 import (
    ALGORITHM_VERSION,
    ComplexDiskInt8Quantizer,
    PreFpgaAlgorithmContract,
    SparseFieldObjective,
)


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
RESULT_DIR = VERSION_DIR / "output" / "controller_training"
TRAIN_SEEDS = (24001, 24002, 24003, 24004, 24005)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _candidate_angles() -> torch.Tensor:
    theta = torch.deg2rad(torch.linspace(60.0, 120.0, 17))
    phi = torch.deg2rad(torch.linspace(-30.0, 30.0, 13))
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    return torch.stack((tt.flatten(), pp.flatten()), dim=1)


def _nominal_far_matrix(
    config: ArrayGeometryConfig, angles: torch.Tensor
) -> torch.Tensor:
    positions = array_positions(config, dtype=angles.dtype)
    theta, phi = angles[:, 0], angles[:, 1]
    directions = torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=1,
    )
    phase = (2.0 * math.pi / config.wavelength) * (directions @ positions.T)
    return torch.polar(torch.ones_like(phase), phase)


def d_optimal_fixed_angles(
    config: ArrayGeometryConfig, count: int
) -> torch.Tensor:
    """从名义几何候选网格离线选择固定测点；不使用测试数据。"""

    if count <= 0:
        return torch.empty(0, 2)
    candidates = _candidate_angles()
    matrix = _nominal_far_matrix(config, candidates).numpy()
    if count > matrix.shape[0]:
        raise ValueError("measurement count exceeds the candidate grid")

    # 贪心 D-optimal 行选择：每次最大化 log det(I + S^H S) 的边际增益。
    # 这是带主元 QR 的稳定等价用途，且在测点数超过阵元数时仍能继续优化。
    n_elements = matrix.shape[1]
    information_inv = np.eye(n_elements, dtype=np.complex128) / 1.0e-6
    remaining = np.ones(matrix.shape[0], dtype=bool)
    selected: List[int] = []
    for _ in range(count):
        rows = matrix[remaining]
        scores = np.real(np.einsum("bi,ij,bj->b", rows, information_inv, rows.conj()))
        available = np.flatnonzero(remaining)
        chosen = int(available[int(np.argmax(scores))])
        selected.append(chosen)
        vector = matrix[chosen].conj()[:, None]
        transformed = information_inv @ vector
        denominator = 1.0 + (vector.conj().T @ transformed).real.item()
        information_inv -= (transformed @ transformed.conj().T) / denominator
        remaining[chosen] = False
    return candidates[torch.as_tensor(selected, dtype=torch.long)]


def measurement_design_metrics(
    config: ArrayGeometryConfig, measurement_count: int, fixed_angles: torch.Tensor
) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(240000 + 31 * measurement_count + config.num_elements)
    dynamic = torch.empty(2000, 2)
    dynamic[:, 0] = math.radians(60.0) + math.radians(60.0) * torch.rand(
        2000, generator=generator
    )
    dynamic[:, 1] = math.radians(-30.0) + math.radians(60.0) * torch.rand(
        2000, generator=generator
    )
    fixed = _nominal_far_matrix(config, fixed_angles)
    dynamic_matrix = _nominal_far_matrix(config, dynamic)[:, None, :]
    matrix = torch.cat((dynamic_matrix, fixed[None].expand(2000, -1, -1)), dim=1)
    singular = torch.linalg.svdvals(matrix)
    tolerance = (
        max(measurement_count, config.num_elements)
        * torch.finfo(singular.dtype).eps
        * singular[:, :1]
    )
    rank = (singular > tolerance).sum(dim=1)
    full_rank = rank == config.num_elements
    finite_conditions = (
        singular[full_rank, 0] / singular[full_rank, config.num_elements - 1]
        if singular.shape[1] >= config.num_elements and bool(full_rank.any())
        else singular.new_empty(0)
    )
    condition = (
        {
            "median": float(torch.quantile(finite_conditions, 0.5)),
            "p90": float(torch.quantile(finite_conditions, 0.9)),
            "max": float(finite_conditions.max()),
        }
        if finite_conditions.numel() > 0
        else None
    )
    return {
        "topology": config.topology_name,
        "measurement_count": measurement_count,
        "dynamic_target_trials": 2000,
        "full_column_rank_rate": float(full_rank.float().mean()),
        "minimum_rank": int(rank.min()),
        "condition_number": condition,
        "selection": "nominal_geometry_greedy_d_optimal_plus_dynamic_target",
    }


def _directions_from_angles(angles: torch.Tensor) -> torch.Tensor:
    theta, phi = angles[..., 0], angles[..., 1]
    return torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=-1,
    )


def sparse_coefficients(
    config: ArrayGeometryConfig,
    samples: Samples,
    fixed_angles: torch.Tensor,
) -> torch.Tensor:
    batch = len(samples)
    fixed = fixed_angles.to(dtype=samples.targets.dtype)
    angles = torch.cat(
        (
            samples.targets[:, None, :],
            fixed[None, :, :].expand(batch, -1, -1),
        ),
        dim=1,
    )
    directions = _directions_from_angles(angles)
    nominal = array_positions(config, dtype=samples.targets.dtype)
    actual = nominal.unsqueeze(0) + samples.true_position.flatten(2).transpose(1, 2)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "bmc,bnc->bmn", directions, actual
    )
    propagation = torch.polar(torch.ones_like(phase), phase)
    response = torch.complex(samples.true_response[:, 0], samples.true_response[:, 1]).flatten(1)
    before_coupling = propagation * response[:, None, :]
    return torch.einsum("bmn,bnj->bmj", before_coupling, samples.coupling)


def calibrated_nominal_sparse_coefficients(
    config: ArrayGeometryConfig,
    samples: Samples,
    fixed_angles: torch.Tensor,
) -> torch.Tensor:
    """Only observable calibration I/Q and nominal geometry; no true error tensors."""

    batch = len(samples)
    fixed = fixed_angles.to(dtype=samples.targets.dtype)
    angles = torch.cat(
        (samples.targets[:, None, :], fixed[None].expand(batch, -1, -1)), dim=1
    )
    directions = _directions_from_angles(angles)
    nominal = array_positions(config, dtype=samples.targets.dtype)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "bmc,nc->bmn", directions, nominal
    )
    propagation = torch.polar(torch.ones_like(phase), phase)
    response = torch.complex(
        samples.calibration_input[:, 0], samples.calibration_input[:, 1]
    ).flatten(1)
    return propagation * response[:, None, :]


def _weights_complex(weights_iq: torch.Tensor) -> torch.Tensor:
    return torch.complex(weights_iq[:, 0], weights_iq[:, 1]).flatten(1)


def sparse_fields(coefficients: torch.Tensor, weights_iq: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bmn,bn->bm", coefficients, _weights_complex(weights_iq))


def train_sparse_controller(
    config: ArrayGeometryConfig,
    model: nn.Module,
    samples: Samples,
    fixed_angles: torch.Tensor,
    *,
    seed: int,
    mode: str,
    steps: int = 300,
    lr: float = 2.0e-3,
    snr_db: float = 30.0,
    random_common_phase: bool = False,
    coefficient_mode: str = "oracle_true",
) -> List[Dict[str, object]]:
    if coefficient_mode not in {"oracle_true", "calibrated_nominal"}:
        raise ValueError("unknown coefficient_mode")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    # 动态目标点始终占总权重25%，不因 M 改变损失尺度。
    target_weight = max(1.0, float(fixed_angles.shape[0]) / 3.0)
    objective = SparseFieldObjective(mode=mode, target_point_weight=target_weight)
    log: List[Dict[str, object]] = []
    checkpoints = {0, steps // 4 - 1, steps // 2 - 1, 3 * steps // 4 - 1, steps - 1}
    # One fixed acquisition per command: the data budget is P*M, not steps*batch*M.
    true_coefficients = sparse_coefficients(config, samples, fixed_angles)
    reference = sparse_fields(true_coefficients, samples.oracle).detach()
    signal_power = reference.abs().square().mean(dim=1, keepdim=True)
    noise_sigma = torch.sqrt(signal_power / (10.0 ** (snr_db / 10.0)) / 2.0)
    measured_all = reference + noise_sigma * (
        torch.randn(reference.shape, generator=generator)
        + 1j * torch.randn(reference.shape, generator=generator)
    )
    if random_common_phase:
        phase = 2.0 * math.pi * torch.rand(reference.shape[0], generator=generator)
        measured_all = measured_all * torch.polar(torch.ones_like(phase), phase)[:, None]
    if mode == "power_only":
        measured_all = measured_all.abs().square()
    model.train()
    for step in range(steps):
        indices = torch.randint(len(samples), (min(64, len(samples)),), generator=generator)
        batch = samples.take(indices)
        output = model_forward(model, batch)
        coefficients = (
            sparse_coefficients(config, batch, fixed_angles)
            if coefficient_mode == "oracle_true"
            else calibrated_nominal_sparse_coefficients(config, batch, fixed_angles)
        )
        predicted = sparse_fields(coefficients, output["weights_iq"])
        measured = measured_all[indices]
        loss_sparse = objective(predicted, measured)
        regularisation = output["delta_iq"].square().mean()
        loss = loss_sparse + 0.01 * regularisation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in checkpoints:
            log.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "sparse_loss": float(loss_sparse.detach()),
                    "regularisation": float(regularisation.detach()),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
    return log


def with_physics_sparse_calibration(
    config: ArrayGeometryConfig,
    samples: Samples,
    *,
    seed: int,
    snr_db: float = 30.0,
) -> Samples:
    first_by_profile: Dict[int, int] = {}
    for index, profile in enumerate(samples.profile_index.tolist()):
        first_by_profile.setdefault(int(profile), index)
    representative = samples.take(
        torch.tensor(list(first_by_profile.values()), dtype=torch.long)
    )
    probes = design_probe_grid(config)
    nominal_green = near_field_green_matrix(config, probes)
    inverse = regularized_pseudoinverse(nominal_green, ridge=1.0e-5)
    estimator = PhysicsInitializedApertureEstimator(
        inverse, adapter_rank=0, train_base=False
    )
    actual_green = near_field_green_matrix(
        config, probes, position_errors=representative.true_position
    )
    response = torch.complex(
        representative.true_response[:, 0], representative.true_response[:, 1]
    ).flatten(1)
    measurement = torch.einsum("bmn,bn->bm", actual_green, response)
    generator = torch.Generator().manual_seed(seed)
    power = measurement.abs().square().mean(dim=1, keepdim=True)
    sigma = torch.sqrt(power / (10.0 ** (snr_db / 10.0)) / 2.0)
    measurement = measurement + sigma * (
        torch.randn(measurement.shape, generator=generator)
        + 1j * torch.randn(measurement.shape, generator=generator)
    )
    estimate_per_profile = estimator(complex_to_iq(measurement)).reshape(
        -1, 2, config.x_num, config.y_num
    )
    estimate = estimate_per_profile[samples.profile_index]
    result = copy.deepcopy(samples)
    result.calibration_input = estimate.detach()
    # 单一复孔径不能唯一恢复三维位置；不要把真值位置泄漏给控制器。
    result.position_input = torch.zeros_like(samples.position_input)
    # 当前近场采样未独立识别互耦，因此也不把仿真真值 state 泄漏给控制器。
    result.state_input = torch.zeros_like(samples.state_input)
    return result


def _representative_profile_subset(samples: Samples) -> Samples:
    seen = set()
    indices: List[int] = []
    for index, profile in enumerate(samples.profile_index.tolist()):
        if profile not in seen:
            seen.add(profile)
            indices.append(index)
    return samples.take(torch.tensor(indices, dtype=torch.long))


def _full_grid_fields(
    config: ArrayGeometryConfig,
    samples: Samples,
    weights_iq: torch.Tensor,
) -> torch.Tensor:
    theta = torch.linspace(math.radians(60.0), math.radians(120.0), 21)
    phi = torch.linspace(math.radians(-30.0), math.radians(30.0), 21)
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    angles = torch.stack((tt.flatten(), pp.flatten()), dim=1)
    directions = _directions_from_angles(angles)
    nominal = array_positions(config)
    actual = nominal[None] + samples.true_position.flatten(2).transpose(1, 2)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "mc,bnc->bmn", directions, actual
    )
    propagation = torch.polar(torch.ones_like(phase), phase)
    response = torch.complex(
        samples.true_response[:, 0], samples.true_response[:, 1]
    ).flatten(1)
    coefficients = torch.einsum(
        "bmn,bnj->bmj", propagation * response[:, None, :], samples.coupling
    )
    return torch.einsum("bmn,bn->bm", coefficients, _weights_complex(weights_iq))


@torch.no_grad()
def evaluate_sparse_model(
    config: ArrayGeometryConfig,
    model: Optional[nn.Module],
    samples: Samples,
    *,
    name: str,
    seed: int,
    observation_mode: str = "coherent_iq",
) -> Dict[str, object]:
    if observation_mode not in {"coherent_iq", "phase_invariant_iq", "power_only"}:
        raise ValueError("unknown observation_mode")
    if model is None:
        weights = samples.initial
    else:
        model.eval()
        weights = model_forward(model, samples)["weights_iq"]
    gain = target_gain(config, weights, samples)
    analytic_gain = target_gain(config, samples.initial, samples).clamp_min(1.0e-8)
    oracle_gain = target_gain(config, samples.oracle, samples).clamp_min(1.0e-8)
    ratio = gain / oracle_gain

    weights_complex = _weights_complex(weights)
    oracle_complex = _weights_complex(samples.oracle)
    alignment = (torch.conj(weights_complex) * oracle_complex).sum(dim=1)
    alignment = alignment / alignment.abs().clamp_min(1.0e-12)
    aligned_weights_complex = weights_complex * alignment[:, None]
    metric_weights_complex = (
        weights_complex if observation_mode == "coherent_iq" else aligned_weights_complex
    )
    phase_error = torch.angle(metric_weights_complex * torch.conj(oracle_complex)).abs()

    representative = _representative_profile_subset(samples)
    representative_indices = []
    first_by_profile: Dict[int, int] = {}
    for index, profile in enumerate(samples.profile_index.tolist()):
        first_by_profile.setdefault(int(profile), index)
    representative_indices = list(first_by_profile.values())
    rep_weights = weights[torch.tensor(representative_indices, dtype=torch.long)]
    predicted_pattern = _full_grid_fields(config, representative, rep_weights)
    reference_pattern = _full_grid_fields(config, representative, representative.oracle)
    pattern_alignment = (torch.conj(predicted_pattern) * reference_pattern).sum(dim=1)
    pattern_alignment = pattern_alignment / pattern_alignment.abs().clamp_min(1.0e-12)
    aligned_pattern = predicted_pattern * pattern_alignment[:, None]
    coherent_nmse = (
        (predicted_pattern - reference_pattern).abs().square().sum(dim=1)
        / reference_pattern.abs().square().sum(dim=1).clamp_min(1.0e-12)
    )
    aligned_nmse = (
        (aligned_pattern - reference_pattern).abs().square().sum(dim=1)
        / reference_pattern.abs().square().sum(dim=1).clamp_min(1.0e-12)
    )
    predicted_power = predicted_pattern.abs().square()
    reference_power = reference_pattern.abs().square()
    power_nmse = (
        (predicted_power - reference_power).square().sum(dim=1)
        / reference_power.square().sum(dim=1).clamp_min(1.0e-12)
    )
    primary_nmse = {
        "coherent_iq": coherent_nmse,
        "phase_invariant_iq": aligned_nmse,
        "power_only": power_nmse,
    }[observation_mode]
    amplitude = weights_complex.abs()
    return {
        "topology": config.topology_name,
        "architecture": name,
        "seed": seed,
        "samples": len(samples),
        "evaluated_profiles": len(representative),
        "mean_oracle_fraction": float(ratio.mean()),
        "p05_oracle_fraction": float(torch.quantile(ratio, 0.05)),
        "mean_gain_gap_db": float((20.0 * torch.log10(ratio)).mean()),
        "mean_improvement_vs_analytic_db": float(
            (20.0 * torch.log10(gain / analytic_gain)).mean()
        ),
        "phase_mae_deg": float(torch.rad2deg(phase_error).mean()),
        "pattern_nmse_db": float(
            (10.0 * torch.log10(primary_nmse.clamp_min(1.0e-12))).mean()
        ),
        "coherent_pattern_nmse_db": float(
            (10.0 * torch.log10(coherent_nmse.clamp_min(1.0e-12))).mean()
        ),
        "aligned_pattern_nmse_db": float(
            (10.0 * torch.log10(aligned_nmse.clamp_min(1.0e-12))).mean()
        ),
        "power_pattern_nmse_db": float(
            (10.0 * torch.log10(power_nmse.clamp_min(1.0e-12))).mean()
        ),
        "primary_pattern_metric": {
            "coherent_iq": "coherent_complex_nmse",
            "phase_invariant_iq": "global_phase_aligned_complex_nmse",
            "power_only": "power_nmse",
        }[observation_mode],
        "amplitude_violation_rate": float((amplitude > 1.000001).float().mean()),
    }


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = (
        "mean_oracle_fraction",
        "p05_oracle_fraction",
        "mean_gain_gap_db",
        "mean_improvement_vs_analytic_db",
        "phase_mae_deg",
        "pattern_nmse_db",
        "amplitude_violation_rate",
    )
    keys = (
        "topology",
        "method",
        "measurement_count",
        "train_samples",
        "train_profiles",
        "label_mode",
    )
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result: Dict[str, object] = dict(zip(keys, group))
        result["seeds"] = len(values)
        for metric in metrics:
            numbers = [float(value[metric]) for value in values]
            mean = statistics.fmean(numbers)
            std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            # n=5 uses Student-t(0.975,4), not a large-sample normal interval.
            result[f"{metric}_ci95"] = (
                2.776 * std / math.sqrt(len(numbers)) if len(numbers) > 1 else 0.0
            )
        output.append(result)
    return output


class QuantizedOutputWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.quantizer = ComplexDiskInt8Quantizer()

    def forward(self, *args, **kwargs):
        result = self.base(*args, **kwargs)
        quantized = self.quantizer(result["weights_iq"])
        return {**result, "weights_iq": quantized["dequantized_iq"]}


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seed_everything(24000)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    per_seed: List[Dict[str, object]] = []
    training_log: List[Dict[str, object]] = []
    design_rows: List[Dict[str, object]] = []
    quantization: Dict[str, object] = {}
    contracts: Dict[str, object] = {}
    checkpoints: Dict[str, object] = {}

    for topology_index, config in enumerate(
        (ArrayGeometryConfig.theoretical_4x4(), ArrayGeometryConfig.hardware_1x4())
    ):
        counts = (8, 16, 24, 32, 48) if config.x_num > 1 else (4, 6, 8, 12, 16)
        power_counts = (32, 64, 96) if config.x_num > 1 else (8, 12, 16)
        recommended = 32 if config.x_num > 1 else 8
        source_train = make_samples(
            768, SOURCE, config, seed=7100 + topology_index, targets_per_profile=8
        )
        target_validation = make_samples(
            192, TARGET, config, seed=7200 + topology_index, targets_per_profile=8
        )
        target_test = make_samples(
            384, TARGET, config, seed=7250 + topology_index, targets_per_profile=8
        )
        single_board_raw = make_samples(
            256, TARGET, config, seed=7275 + topology_index, targets_per_profile=256
        )
        sparse_cal_train = with_physics_sparse_calibration(
            config, source_train, seed=7300 + topology_index
        )
        sparse_cal_test = with_physics_sparse_calibration(
            config, target_test, seed=7400 + topology_index
        )
        sparse_cal_validation = with_physics_sparse_calibration(
            config, target_validation, seed=7450 + topology_index
        )
        single_board = with_physics_sparse_calibration(
            config, single_board_raw, seed=7475 + topology_index
        )
        single_board_train = single_board.take(torch.arange(0, 128))
        single_board_validation = single_board.take(torch.arange(128, 192))
        single_board_test = single_board.take(torch.arange(192, 256))
        all_counts = tuple(sorted(set(counts + power_counts)))
        fixed_by_count = {
            count: d_optimal_fixed_angles(config, count - 1) for count in all_counts
        }
        for count, angles in fixed_by_count.items():
            design_rows.append(measurement_design_metrics(config, count, angles))
        contracts[config.topology_name] = PreFpgaAlgorithmContract.create(
            config, recommended, "coherent_iq"
        ).to_dict()

        analytic = evaluate_sparse_model(
            config, None, target_test, name="analytic_initial", seed=-1
        )
        per_seed.append(
            {
                **analytic,
                "method": "analytic_initial",
                "measurement_count": 0,
                "train_samples": 0,
                "train_profiles": 0,
                "label_mode": "none",
            }
        )

        best_score = -math.inf
        best_model = None
        best_seed = None
        best_calibrated_score = -math.inf
        best_calibrated_model = None
        best_calibrated_seed = None
        best_calibrated_regime = None
        best_board_score = -math.inf
        best_board_model = None
        best_board_seed = None
        source_pretrained_by_seed: Dict[int, nn.Module] = {}
        for seed in TRAIN_SEEDS:
            # 全权值监督作为同预算上界，不代表实测可获得完整标签。
            seed_everything(seed + topology_index * 1000)
            upper = TopologyAdaptiveBeamformingCNN(
                config, width=16, num_blocks=3, global_context="none", output_bits=None
            )
            upper_log = train_model(
                config, upper, source_train, seed=seed + 100, steps=300
            )
            for row in upper_log:
                training_log.append(
                    {
                        "topology": config.topology_name,
                        "method": "weight_supervised_upper",
                        "seed": seed,
                        **row,
                    }
                )
            metric = evaluate_sparse_model(
                config, upper, target_test, name="weight_supervised_upper", seed=seed
            )
            per_seed.append(
                {
                    **metric,
                    "method": "weight_supervised_upper",
                    "measurement_count": -1,
                    "train_samples": len(source_train),
                    "train_profiles": 96,
                    "label_mode": "full_weight",
                }
            )

            for count in counts:
                seed_everything(seed + topology_index * 1000 + count)
                model = TopologyAdaptiveBeamformingCNN(
                    config, width=16, num_blocks=3, global_context="none", output_bits=None
                )
                log = train_sparse_controller(
                    config,
                    model,
                    source_train,
                    fixed_by_count[count],
                    seed=seed + count * 10,
                    mode="coherent_iq",
                    steps=300,
                )
                for row in log:
                    training_log.append(
                        {
                            "topology": config.topology_name,
                            "method": "sparse_coherent",
                            "measurement_count": count,
                            "seed": seed,
                            **row,
                        }
                    )
                metric = evaluate_sparse_model(
                    config, model, target_test, name="sparse_coherent", seed=seed
                )
                per_seed.append(
                    {
                        **metric,
                        "method": "sparse_coherent",
                        "measurement_count": count,
                        "train_samples": len(source_train),
                        "train_profiles": 96,
                        "label_mode": "coherent_iq",
                    }
                )
                if count == recommended:
                    source_pretrained_by_seed[seed] = copy.deepcopy(model)
                    validation_metric = evaluate_sparse_model(
                        config,
                        model,
                        target_validation,
                        name="sparse_coherent_validation",
                        seed=seed,
                    )
                    if float(validation_metric["mean_oracle_fraction"]) > best_score:
                        best_score = float(validation_metric["mean_oracle_fraction"])
                        best_model = copy.deepcopy(model)
                        best_seed = seed

            # 推荐测点数下的未知公共相位边界。
            for method, mode, random_phase in (
                ("sparse_phase_invariant", "phase_invariant_iq", True),
                ("sparse_naive_unknown_phase", "coherent_iq", True),
            ):
                seed_everything(seed + topology_index * 2000 + len(method))
                model = TopologyAdaptiveBeamformingCNN(
                    config, width=16, num_blocks=3, global_context="none", output_bits=None
                )
                mode_log = train_sparse_controller(
                    config,
                    model,
                    source_train,
                    fixed_by_count[recommended],
                    seed=seed + len(method) * 11,
                    mode=mode,
                    steps=300,
                    random_common_phase=random_phase,
                )
                for row in mode_log:
                    training_log.append(
                        {
                            "topology": config.topology_name,
                            "method": method,
                            "measurement_count": recommended,
                            "seed": seed,
                            **row,
                        }
                    )
                metric = evaluate_sparse_model(
                    config,
                    model,
                    target_test,
                    name=method,
                    seed=seed,
                    observation_mode="phase_invariant_iq",
                )
                per_seed.append(
                    {
                        **metric,
                        "method": method,
                        "measurement_count": recommended,
                        "train_samples": len(source_train),
                        "train_profiles": 96,
                        "label_mode": mode,
                    }
                )

            # 只有功率的相位恢复需要更多方向点；分别覆盖低于和高于4N-4。
            for count in power_counts:
                method = "sparse_power_only"
                seed_everything(seed + topology_index * 2500 + count)
                model = TopologyAdaptiveBeamformingCNN(
                    config, width=16, num_blocks=3, global_context="none", output_bits=None
                )
                power_log = train_sparse_controller(
                    config,
                    model,
                    source_train,
                    fixed_by_count[count],
                    seed=seed + count * 17,
                    mode="power_only",
                    steps=300,
                )
                for row in power_log:
                    training_log.append(
                        {
                            "topology": config.topology_name,
                            "method": method,
                            "measurement_count": count,
                            "seed": seed,
                            **row,
                        }
                    )
                metric = evaluate_sparse_model(
                    config,
                    model,
                    target_test,
                    name=method,
                    seed=seed,
                    observation_mode="power_only",
                )
                per_seed.append(
                    {
                        **metric,
                        "method": method,
                        "measurement_count": count,
                        "train_samples": len(source_train),
                        "train_profiles": 96,
                        "label_mode": "power_only",
                    }
                )

            # 完整链：近场稀疏物理校准输入 + 远场稀疏监督。
            seed_everything(seed + topology_index * 3000)
            chain = copy.deepcopy(source_pretrained_by_seed[seed])
            chain_log = train_sparse_controller(
                config,
                chain,
                sparse_cal_train,
                fixed_by_count[recommended],
                seed=seed + 333,
                mode="coherent_iq",
                steps=300,
                coefficient_mode="calibrated_nominal",
            )
            for row in chain_log:
                training_log.append(
                    {
                        "topology": config.topology_name,
                        "method": "calibrated_surrogate_transfer",
                        "measurement_count": recommended,
                        "seed": seed,
                        **row,
                    }
                )
            metric = evaluate_sparse_model(
                config, chain, sparse_cal_test, name="sparse_calibration_chain", seed=seed
            )
            per_seed.append(
                {
                    **metric,
                    "method": "calibrated_surrogate_transfer",
                    "measurement_count": recommended,
                    "train_samples": len(source_train),
                    "train_profiles": 96,
                    "label_mode": "coherent_iq",
                }
            )
            validation_metric = evaluate_sparse_model(
                config,
                chain,
                sparse_cal_validation,
                name="calibrated_surrogate_validation",
                seed=seed,
            )
            if float(validation_metric["mean_oracle_fraction"]) > best_calibrated_score:
                best_calibrated_score = float(validation_metric["mean_oracle_fraction"])
                best_calibrated_model = copy.deepcopy(chain)
                best_calibrated_seed = seed
                best_calibrated_regime = "source_pretrain_calibrated_transfer"
            print(
                json.dumps(
                    {"stage": "core", "topology": config.topology_name, "seed": seed}
                ),
                flush=True,
            )

        # Q=1同一块阵列上只改变P；校准只做一次，方向在train/val/test间隔离。
        for train_size in (8, 16, 32, 64, 128):
            subset = single_board_train.take(torch.arange(train_size))
            for seed in TRAIN_SEEDS:
                for initialisation, model in (
                    ("pretrained", copy.deepcopy(source_pretrained_by_seed[seed])),
                    (
                        "scratch",
                        TopologyAdaptiveBeamformingCNN(
                            config,
                            width=16,
                            num_blocks=3,
                            global_context="none",
                            output_bits=None,
                        ),
                    ),
                ):
                    seed_everything(
                        seed
                        + train_size
                        + topology_index * 4000
                        + (0 if initialisation == "pretrained" else 50000)
                    )
                    method = f"single_board_{initialisation}"
                    curve_log = train_sparse_controller(
                        config,
                        model,
                        subset,
                        fixed_by_count[recommended],
                        seed=seed + train_size + (0 if initialisation == "pretrained" else 7000),
                        mode="coherent_iq",
                        steps=300,
                        coefficient_mode="calibrated_nominal",
                    )
                    for row in curve_log:
                        training_log.append(
                            {
                                "topology": config.topology_name,
                                "method": method,
                                "measurement_count": recommended,
                                "train_samples": train_size,
                                "train_profiles": 1,
                                "seed": seed,
                                **row,
                            }
                        )
                    metric = evaluate_sparse_model(
                        config,
                        model,
                        single_board_test,
                        name=method,
                        seed=seed,
                    )
                    per_seed.append(
                        {
                            **metric,
                            "method": method,
                            "measurement_count": recommended,
                            "train_samples": train_size,
                            "train_profiles": 1,
                            "label_mode": "coherent_iq",
                        }
                    )
                    if initialisation == "pretrained" and train_size == 128:
                        validation_metric = evaluate_sparse_model(
                            config,
                            model,
                            single_board_validation,
                            name="single_board_pretrained_validation",
                            seed=seed,
                        )
                        score = float(validation_metric["mean_oracle_fraction"])
                        if score > best_board_score:
                            best_board_score = score
                            best_board_model = copy.deepcopy(model)
                            best_board_seed = seed
            print(
                json.dumps(
                    {
                        "stage": "single_board_curve",
                        "topology": config.topology_name,
                        "train_commands": train_size,
                    }
                ),
                flush=True,
            )

        if (
            best_model is None
            or best_seed is None
            or best_calibrated_model is None
            or best_calibrated_seed is None
            or best_calibrated_regime is None
            or best_board_model is None
            or best_board_seed is None
        ):
            raise RuntimeError("未得到推荐稀疏模型")
        checkpoints[config.topology_name] = {}
        for regime, selected_model, selected_seed in (
            ("oracle_simulation", best_model, best_seed),
            ("calibrated_surrogate", best_calibrated_model, best_calibrated_seed),
            ("single_board_adapted", best_board_model, best_board_seed),
        ):
            checkpoint_path = (
                RESULT_DIR / f"v24_{config.topology_name}_{regime}_controller.pt"
            )
            torch.save(
                {
                    "schema_version": 2,
                    "algorithm_version": ALGORITHM_VERSION,
                    "array_config": asdict(config),
                    "model_config": {
                        "width": 16,
                        "num_blocks": 3,
                        "global_context": "none",
                        "state_dim": 4,
                        "output_bits": None,
                    },
                    "training_regime": regime,
                    "selected_calibrated_regime": (
                        best_calibrated_regime if regime == "calibrated_surrogate" else None
                    ),
                    "total_measurement_count": recommended,
                    "dynamic_target_measurement_count": 1,
                    "fixed_measurement_angles_rad": fixed_by_count[recommended],
                    "measurement_mode": "coherent_iq",
                    "selection_split": (
                        "single_board_validation"
                        if regime == "single_board_adapted"
                        else "target_validation"
                    ),
                    "selected_seed": selected_seed,
                    "state_dict": selected_model.state_dict(),
                },
                checkpoint_path,
            )
            checkpoints[config.topology_name][regime] = checkpoint_path.name

        # 部署候选使用合成预训练后在Q=1目标板稀疏适配的模型。
        deployment_model = best_board_model
        deployment_seed = best_board_seed
        fp32_metric = evaluate_sparse_model(
            config,
            deployment_model,
            single_board_test,
            name="best_single_board_sparse_fp32",
            seed=deployment_seed,
        )
        quantized_model = QuantizedOutputWrapper(deployment_model)
        int8_metric = evaluate_sparse_model(
            config,
            quantized_model,
            single_board_test,
            name="best_single_board_sparse_int8_output",
            seed=deployment_seed,
        )
        with torch.no_grad():
            fp32_output = model_forward(deployment_model, single_board_test)["weights_iq"]
            integer = ComplexDiskInt8Quantizer()(fp32_output)
        quantization[config.topology_name] = {
            "fp32": fp32_metric,
            "int8_output": int8_metric,
            "mean_oracle_fraction_delta": int8_metric["mean_oracle_fraction"]
            - fp32_metric["mean_oracle_fraction"],
            "gain_gap_delta_db": int8_metric["mean_gain_gap_db"]
            - fp32_metric["mean_gain_gap_db"],
            "max_integer_radius_sq": int(integer["max_integer_radius_sq"]),
            "integer_radius_limit_sq": 127 * 127,
            "projection_rate": float(integer["input_disk_projection_rate"]),
            "scope": "strict INT8 output disk reference; internal CNN remains FP32",
        }
        sample_indices = torch.arange(8)
        golden = single_board_test.take(sample_indices)
        with torch.no_grad():
            golden_fp32 = model_forward(deployment_model, golden)["weights_iq"]
            golden_int = ComplexDiskInt8Quantizer()(golden_fp32)["codes_iq"]
        np.savez_compressed(
            RESULT_DIR / f"golden_vectors_{config.topology_name}.npz",
            targets_rad=golden.targets.numpy(),
            initial_weights_iq=golden.initial.numpy(),
            calibration_iq=golden.calibration_input.numpy(),
            position_errors=golden.position_input.numpy(),
            state=golden.state_input.numpy(),
            output_fp32_iq=golden_fp32.numpy(),
            output_int8_iq=golden_int.numpy(),
        )

    aggregate = aggregate_rows([row for row in per_seed if int(row["seed"]) >= 0])
    payload = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "study": "sparse_far_field_supervision_and_prefpga_integer_output",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "measurement_selector": "numpy_greedy_d_optimal",
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "train_seeds": list(TRAIN_SEEDS),
        "measurement_design": design_rows,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "quantization": quantization,
        "contracts": contracts,
        "checkpoints": checkpoints,
        "protocol": {
            "source_train_commands": 768,
            "source_train_profiles": 96,
            "directions_per_profile": 8,
            "target_validation_commands": 192,
            "target_validation_profiles": 24,
            "target_test_commands": 384,
            "target_test_profiles": 48,
            "single_board_total_commands": 256,
            "single_board_profiles_Q": 1,
            "single_board_train_pool_commands": 128,
            "single_board_validation_commands": 64,
            "single_board_test_commands": 64,
            "sparse_label_snr_db": 30.0,
            "measurement_repeats_per_command": 1,
            "far_field_observation_budget_formula": "Q_profiles*P_commands_per_profile*M_directions*R_repeats",
            "near_field_calibration_budget_formula": "Q_profiles*M_calibration_points*R_calibration_repeats",
            "near_field_calibration_repeats": 1,
            "near_field_calibration_points": {"4x4": 32, "1x4": 12},
            "full_direction_grid_usage": "heldout evaluation only",
            "recommended_measurements": {"4x4": 32, "1x4": 8},
            "single_board_data_curve_commands_P": [8, 16, 32, 64, 128],
            "single_board_data_curve_profiles_Q": 1,
            "checkpoint_selection": {
                "source_and_calibrated": "independent target_validation",
                "single_board_and_low_update": "single_board_validation",
                "test_used_for_selection": False,
            },
        },
        "wall_time_seconds": time.perf_counter() - started,
        "limitations": [
            "oracle_true rows are simulation feasibility upper bounds and use exact differentiable error coefficients",
            "calibrated_surrogate rows predict through nominal geometry plus an independently estimated near-field aperture and do not use true error coefficients in the gradient path",
            "sparse teacher labels are generated from a simulation oracle; real acquisition needs teacher commands, a desired-field specification, or hardware-in-loop optimisation",
            "strict integer reference covers output I/Q only; full W8A8 internal equivalence belongs to FPGA implementation",
            "power-only labels are formed by squaring noisy coherent I/Q and are not yet a calibrated square-law detector noise model",
            "power-only supervision remains non-convex; below roughly 4N-4 points it cannot establish generic complex-aperture identifiability",
            "actual 1x4 coordinates and coherent receiver capability must replace the provisional geometry before deployment",
        ],
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / "contracts.json").write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(RESULT_DIR / "measurement_design.csv", design_rows)
    write_csv(RESULT_DIR / "per_seed.csv", per_seed)
    write_csv(RESULT_DIR / "aggregate.csv", aggregate)
    write_csv(RESULT_DIR / "training_log.csv", training_log)
    print(
        json.dumps(
            {
                "status": "ok",
                "version": ALGORITHM_VERSION,
                "wall_time_seconds": payload["wall_time_seconds"],
                "output": str(metrics_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
