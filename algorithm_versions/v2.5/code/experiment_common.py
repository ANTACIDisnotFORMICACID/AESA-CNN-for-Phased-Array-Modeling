"""V2.3 及后续实验共用的非理想阵列采样、训练与评价内核。

本模块没有结果目录、版本号、checkpoint 写入或旧版消融网络；固定随机种子
和发布决策由每个版本自己的实验入口负责。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    analytic_far_field_weights,
    array_positions,
    element_response_from_amp_phase,
)


@dataclass(frozen=True)
class ErrorDomain:
    name: str
    amp_rms: float
    phase_rms_deg: float
    position_rms_wavelength: float
    response_observation_rms: float
    position_observation_rms_wavelength: float
    horizontal_coupling_mean: float
    vertical_coupling_mean: float
    coupling_rms: float
    coupling_phase_deg: float
    coupling_observation_rms: float
    correlated: bool


@dataclass
class Samples:
    targets: torch.Tensor
    initial: torch.Tensor
    calibration_input: torch.Tensor
    position_input: torch.Tensor
    true_response: torch.Tensor
    true_position: torch.Tensor
    state_input: torch.Tensor
    true_state: torch.Tensor
    coupling: torch.Tensor
    oracle: torch.Tensor
    profile_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def take(self, indices: torch.Tensor) -> "Samples":
        return Samples(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})


def _unit_rms(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    return x / torch.sqrt(x.square().mean(dim=dims, keepdim=True).clamp_min(1.0e-8))


def _smooth(x: torch.Tensor, config: ArrayGeometryConfig) -> torch.Tensor:
    kernel = (3, 3) if config.x_num > 1 else (1, 3)
    padding = (1, 1) if config.x_num > 1 else (0, 1)
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=padding)


def _build_coupling(
    config: ArrayGeometryConfig, states: torch.Tensor
) -> torch.Tensor:
    """C[b,n,j]：命令 j 到实际激励 n 的局部互耦。"""

    batch = states.shape[0]
    n = config.num_elements
    matrix = torch.eye(n, dtype=torch.complex64).unsqueeze(0).repeat(batch, 1, 1)
    horizontal = torch.complex(states[:, 0], states[:, 1])
    vertical = torch.complex(states[:, 2], states[:, 3])
    for xi in range(config.x_num):
        for yi in range(config.y_num):
            index = xi * config.y_num + yi
            if xi + 1 < config.x_num:
                neighbour = (xi + 1) * config.y_num + yi
                matrix[:, index, neighbour] = horizontal
                matrix[:, neighbour, index] = horizontal
            if yi + 1 < config.y_num:
                neighbour = xi * config.y_num + yi + 1
                matrix[:, index, neighbour] = vertical
                matrix[:, neighbour, index] = vertical
    return matrix


def _sample_profiles(
    profile_count: int,
    domain: ErrorDomain,
    config: ArrayGeometryConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = config.spatial_shape
    amp = torch.randn(profile_count, 1, h, w, generator=generator)
    phase = torch.randn(profile_count, 1, h, w, generator=generator)
    position = torch.randn(profile_count, 3, h, w, generator=generator)
    if domain.correlated:
        amp = 0.7 * _smooth(amp, config) + 0.3 * amp
        phase = 0.7 * _smooth(phase, config) + 0.3 * phase
        position = 0.7 * _smooth(position, config) + 0.3 * position
        x_gradient = (
            torch.linspace(-1.0, 1.0, h).reshape(1, 1, h, 1)
            if h > 1
            else torch.zeros(1, 1, 1, 1)
        )
        y_gradient = torch.linspace(-1.0, 1.0, w).reshape(1, 1, 1, w)
        amp = amp + 0.4 * x_gradient - 0.25 * y_gradient
        phase = phase + 0.3 * x_gradient + 0.45 * y_gradient
        position[:, 1:2] = position[:, 1:2] + 0.35 * x_gradient
        position[:, 2:3] = position[:, 2:3] - 0.35 * y_gradient
    # 修复 V2.1：相关平滑后重新标定每个 realization 的实际 RMS。
    amp = _unit_rms(amp, (2, 3))
    phase = _unit_rms(phase, (2, 3))
    position = _unit_rms(position, (2, 3))
    amplitude = (1.0 + domain.amp_rms * amp[:, 0]).clamp(0.50, 1.50)
    phase_rad = math.radians(domain.phase_rms_deg) * phase[:, 0]
    true_position = domain.position_rms_wavelength * config.wavelength * position
    true_response = element_response_from_amp_phase(amplitude, phase_rad)

    coupling_noise_h = domain.coupling_rms * torch.randn(profile_count, generator=generator)
    coupling_noise_v = domain.coupling_rms * torch.randn(profile_count, generator=generator)
    phase_noise_h = math.radians(4.0) * torch.randn(profile_count, generator=generator)
    phase_noise_v = math.radians(4.0) * torch.randn(profile_count, generator=generator)
    base_phase = math.radians(domain.coupling_phase_deg)
    h_mag = torch.clamp(domain.horizontal_coupling_mean + coupling_noise_h, min=0.0)
    if config.x_num == 1:
        h_mag.zero_()
    v_mag = torch.clamp(domain.vertical_coupling_mean + coupling_noise_v, min=0.0)
    h_value = torch.polar(h_mag, base_phase + phase_noise_h)
    v_value = torch.polar(v_mag, -base_phase + phase_noise_v)
    true_state = torch.stack(
        (h_value.real, h_value.imag, v_value.real, v_value.imag), dim=1
    )
    state_input = true_state + domain.coupling_observation_rms * torch.randn(
        true_state.shape, generator=generator
    )
    coupling = _build_coupling(config, true_state)

    calibration_input = true_response + domain.response_observation_rms * torch.randn(
        true_response.shape, generator=generator
    )
    position_input = true_position + (
        domain.position_observation_rms_wavelength
        * config.wavelength
        * torch.randn(true_position.shape, generator=generator)
    )
    return (
        true_response,
        true_position,
        calibration_input,
        position_input,
        true_state,
        state_input,
        coupling,
    )


def _effective_coefficients(
    config: ArrayGeometryConfig,
    targets: torch.Tensor,
    response_iq: torch.Tensor,
    position_errors: torch.Tensor,
    coupling: torch.Tensor,
) -> torch.Tensor:
    batch = targets.shape[0]
    nominal = array_positions(config, dtype=targets.dtype)
    actual = nominal.unsqueeze(0) + position_errors.flatten(2).transpose(1, 2)
    theta, phi = targets[:, 0], targets[:, 1]
    directions = torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=1,
    )
    propagation = torch.polar(
        torch.ones(batch, config.num_elements),
        (2.0 * math.pi / config.wavelength)
        * torch.einsum("bnc,bc->bn", actual, directions),
    )
    response = torch.complex(response_iq[:, 0], response_iq[:, 1]).flatten(1)
    before_coupling = propagation * response
    return torch.einsum("bn,bnj->bj", before_coupling, coupling)


def _complex_to_iq_grid(value: torch.Tensor, config: ArrayGeometryConfig) -> torch.Tensor:
    return torch.stack((value.real, value.imag), dim=1).reshape(
        -1, 2, config.x_num, config.y_num
    )


def _oracle_weights(coefficients: torch.Tensor, config: ArrayGeometryConfig) -> torch.Tensor:
    unit = torch.conj(coefficients) / coefficients.abs().clamp_min(1.0e-8)
    return _complex_to_iq_grid(unit, config)


def make_samples(
    count: int,
    domain: ErrorDomain,
    config: ArrayGeometryConfig,
    *,
    seed: int,
    targets_per_profile: int = 8,
) -> Samples:
    generator = torch.Generator().manual_seed(seed)
    profile_count = math.ceil(count / targets_per_profile)
    profile_values = _sample_profiles(profile_count, domain, config, generator)
    profile_index = torch.arange(count) // targets_per_profile
    expanded = [value[profile_index] for value in profile_values]
    (
        true_response,
        true_position,
        calibration_input,
        position_input,
        true_state,
        state_input,
        coupling,
    ) = expanded
    targets = torch.empty(count, 2)
    targets[:, 0] = math.radians(60.0) + math.radians(60.0) * torch.rand(
        count, generator=generator
    )
    targets[:, 1] = math.radians(-30.0) + math.radians(60.0) * torch.rand(
        count, generator=generator
    )
    initial = analytic_far_field_weights(config, targets)
    coefficients = _effective_coefficients(
        config, targets, true_response, true_position, coupling
    )
    oracle = _oracle_weights(coefficients, config)
    return Samples(
        targets,
        initial,
        calibration_input,
        position_input,
        true_response,
        true_position,
        state_input,
        true_state,
        coupling,
        oracle,
        profile_index,
    )


def model_forward(model: nn.Module, samples: Samples) -> Dict[str, torch.Tensor]:
    return model(
        samples.initial,
        samples.targets,
        calibration_iq=samples.calibration_input,
        position_errors=samples.position_input,
        state=samples.state_input,
        quantize_output=False,
    )


def _weights_complex(weights_iq: torch.Tensor) -> torch.Tensor:
    return torch.complex(weights_iq[:, 0], weights_iq[:, 1]).flatten(1)


def target_gain(config: ArrayGeometryConfig, weights_iq: torch.Tensor, samples: Samples) -> torch.Tensor:
    coefficients = _effective_coefficients(
        config,
        samples.targets,
        samples.true_response,
        samples.true_position,
        samples.coupling,
    )
    return torch.abs((coefficients * _weights_complex(weights_iq)).sum(dim=1))


def batch_loss(
    config: ArrayGeometryConfig, model: nn.Module, samples: Samples
) -> torch.Tensor:
    weights = model_forward(model, samples)["weights_iq"]
    supervised = (weights - samples.oracle).square().mean()
    gain = target_gain(config, weights, samples)
    oracle_gain = target_gain(config, samples.oracle, samples).detach().clamp_min(1.0e-6)
    beam = (1.0 - gain / oracle_gain).square().mean()
    return supervised + 0.75 * beam


def train_model(
    config: ArrayGeometryConfig,
    model: nn.Module,
    samples: Samples,
    *,
    seed: int,
    steps: int = 300,
    lr: float = 2.0e-3,
) -> List[Dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    log = []
    checkpoints = {0, steps // 4 - 1, steps // 2 - 1, 3 * steps // 4 - 1, steps - 1}
    model.train()
    for step in range(steps):
        indices = torch.randint(len(samples), (min(64, len(samples)),), generator=generator)
        batch = samples.take(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(config, model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in checkpoints:
            log.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
    return log


def _pattern_nmse(
    config: ArrayGeometryConfig,
    weights: torch.Tensor,
    oracle: torch.Tensor,
    samples: Samples,
    count: int = 24,
) -> torch.Tensor:
    count = min(count, len(samples))
    theta = torch.linspace(math.radians(60.0), math.radians(120.0), 21)
    phi = torch.linspace(math.radians(-30.0), math.radians(30.0), 21)
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    directions = torch.stack(
        (
            torch.sin(tt) * torch.cos(pp),
            torch.sin(tt) * torch.sin(pp),
            torch.cos(tt),
        ),
        dim=-1,
    ).reshape(-1, 3)
    nominal = array_positions(config)
    actual = nominal.unsqueeze(0) + samples.true_position[:count].flatten(2).transpose(1, 2)
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "pc,bnc->bpn", directions, actual
    )
    propagation = torch.polar(torch.ones_like(phase), phase)
    response = torch.complex(
        samples.true_response[:count, 0], samples.true_response[:count, 1]
    ).flatten(1)
    before_coupling = propagation * response[:, None, :]
    coefficients = torch.einsum(
        "bpn,bnj->bpj", before_coupling, samples.coupling[:count]
    )
    predicted = torch.einsum("bpn,bn->bp", coefficients, _weights_complex(weights[:count]))
    reference = torch.einsum("bpn,bn->bp", coefficients, _weights_complex(oracle[:count]))
    return (
        (predicted - reference).abs().square().sum(dim=1)
        / reference.abs().square().sum(dim=1).clamp_min(1.0e-12)
    )


@torch.no_grad()
def evaluate(
    config: ArrayGeometryConfig,
    model: Optional[nn.Module],
    samples: Samples,
    *,
    name: str,
    seed: int,
) -> Dict[str, object]:
    if model is None:
        weights = samples.initial
    else:
        model.eval()
        weights = model_forward(model, samples)["weights_iq"]
    gain = target_gain(config, weights, samples)
    analytic_gain = target_gain(config, samples.initial, samples).clamp_min(1.0e-8)
    oracle_gain = target_gain(config, samples.oracle, samples).clamp_min(1.0e-8)
    ratio = gain / oracle_gain
    improvement = 20.0 * torch.log10(gain / analytic_gain)
    product = torch.complex(weights[:, 0], weights[:, 1]) * torch.conj(
        torch.complex(samples.oracle[:, 0], samples.oracle[:, 1])
    )
    phase_error = torch.atan2(product.imag, product.real).abs()
    amplitude = torch.sqrt(weights[:, 0].square() + weights[:, 1].square())
    nmse = _pattern_nmse(config, weights, samples.oracle, samples)
    return {
        "topology": config.topology_name,
        "architecture": name,
        "seed": seed,
        "samples": len(samples),
        "mean_oracle_fraction": float(ratio.mean()),
        "p05_oracle_fraction": float(torch.quantile(ratio, 0.05)),
        "mean_gain_gap_db": float((20.0 * torch.log10(ratio)).mean()),
        "mean_improvement_vs_analytic_db": float(improvement.mean()),
        "phase_mae_deg": float(torch.rad2deg(phase_error).mean()),
        "pattern_nmse_db": float((10.0 * torch.log10(nmse.clamp_min(1.0e-12))).mean()),
        "amplitude_violation_rate": float((amplitude > 1.000001).float().mean()),
    }


SOURCE = ErrorDomain(
    "source_nonideal",
    amp_rms=0.08,
    phase_rms_deg=12.0,
    position_rms_wavelength=0.010,
    response_observation_rms=0.008,
    position_observation_rms_wavelength=0.0015,
    horizontal_coupling_mean=0.040,
    vertical_coupling_mean=0.055,
    coupling_rms=0.010,
    coupling_phase_deg=20.0,
    coupling_observation_rms=0.003,
    correlated=False,
)

TARGET = ErrorDomain(
    "shifted_correlated_nonideal",
    amp_rms=0.13,
    phase_rms_deg=22.0,
    position_rms_wavelength=0.020,
    response_observation_rms=0.018,
    position_observation_rms_wavelength=0.0030,
    horizontal_coupling_mean=0.075,
    vertical_coupling_mean=0.100,
    coupling_rms=0.018,
    coupling_phase_deg=28.0,
    coupling_observation_rms=0.006,
    correlated=True,
)
