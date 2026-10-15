"""Direction encoders for the V2.11 Fourier/real-SSH controlled ablation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


EncodingKind = Literal["fourier", "ssh", "fourier_plus_ssh"]


def ssh_mode_index(degree: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return degree-major (l,m), with m=-l..l and l=0..degree."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    degrees: list[int] = []
    orders: list[int] = []
    for l_value in range(degree + 1):
        for m_value in range(-l_value, l_value + 1):
            degrees.append(l_value)
            orders.append(m_value)
    return torch.tensor(degrees), torch.tensor(orders)


def ssh_mode_labels(degree: int) -> list[str]:
    degrees, orders = ssh_mode_index(degree)
    return [f"Y_{int(l_value)}_{int(m_value):+d}" for l_value, m_value in zip(degrees, orders)]


def _associated_legendre(
    degree: int,
    order: int,
    cosine_theta: torch.Tensor,
) -> torch.Tensor:
    """Condon-Shortley associated Legendre polynomial P_degree^order."""
    if order < 0 or order > degree:
        raise ValueError("associated-Legendre order must satisfy 0 <= order <= degree")
    one_minus_x2 = (1.0 - cosine_theta.square()).clamp_min(0.0)
    p_mm = torch.ones_like(cosine_theta)
    for m_step in range(1, order + 1):
        p_mm = -(2 * m_step - 1) * torch.sqrt(one_minus_x2) * p_mm
    if degree == order:
        return p_mm
    p_m1m = (2 * order + 1) * cosine_theta * p_mm
    if degree == order + 1:
        return p_m1m
    p_lm2 = p_mm
    p_lm1 = p_m1m
    for l_value in range(order + 2, degree + 1):
        p_lm = (
            (2 * l_value - 1) * cosine_theta * p_lm1
            - (l_value + order - 1) * p_lm2
        ) / (l_value - order)
        p_lm2, p_lm1 = p_lm1, p_lm
    return p_lm1


def real_spherical_harmonics(unit_direction: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate an orthonormal real SSH basis from Cartesian xyz directions.

    Ordering is degree-major, m=-l..l. For m>0 the channel is sqrt(2) Re(Y_lm),
    and for m<0 it is sqrt(2) Im(Y_l|m|). The associated Legendre function
    includes the Condon-Shortley phase.
    """
    if unit_direction.shape[-1] != 3:
        raise ValueError("unit_direction must end with xyz")
    if not unit_direction.is_floating_point():
        raise TypeError("unit_direction must use a floating dtype")
    norm = torch.linalg.vector_norm(unit_direction, dim=-1, keepdim=True)
    if torch.any(norm <= 0):
        raise ValueError("unit_direction contains a zero vector")
    direction = unit_direction / norm
    cosine_theta = direction[..., 2].clamp(-1.0, 1.0)
    phi = torch.atan2(direction[..., 1], direction[..., 0])
    channels: list[torch.Tensor] = []
    for l_value in range(degree + 1):
        for m_value in range(-l_value, l_value + 1):
            abs_m = abs(m_value)
            log_norm = 0.5 * (
                math.log((2 * l_value + 1) / (4.0 * math.pi))
                + math.lgamma(l_value - abs_m + 1)
                - math.lgamma(l_value + abs_m + 1)
            )
            value = math.exp(log_norm) * _associated_legendre(
                l_value, abs_m, cosine_theta
            )
            if m_value < 0:
                value = math.sqrt(2.0) * value * torch.sin(abs_m * phi)
            elif m_value > 0:
                value = math.sqrt(2.0) * value * torch.cos(m_value * phi)
            channels.append(value)
    return torch.stack(channels, dim=-1)


class DirectionFourierEncoder(nn.Module):
    """Exact V2.10 Fourier encoder retained as the control branch."""

    def __init__(self, bands: int = 4) -> None:
        super().__init__()
        if bands < 0:
            raise ValueError("bands must be non-negative")
        self.bands = int(bands)
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    @property
    def output_channels(self) -> int:
        return 3 * (1 + 2 * self.bands)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        if self.bands == 0:
            return unit_direction
        angles = unit_direction[..., :, None] * self.frequencies.to(unit_direction)
        sine = torch.sin(angles).reshape(*unit_direction.shape[:2], -1)
        cosine = torch.cos(angles).reshape(*unit_direction.shape[:2], -1)
        return torch.cat((unit_direction, sine, cosine), dim=-1)


class RealSSHEncoder(nn.Module):
    """Real SSH, train-split standardization, then a learnable K-to-27 projection."""

    def __init__(
        self,
        degree: int,
        *,
        output_channels: int = 27,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if output_channels <= 0 or epsilon <= 0:
            raise ValueError("output_channels and epsilon must be positive")
        self.degree = int(degree)
        self.raw_channels = (self.degree + 1) ** 2
        self._output_channels = int(output_channels)
        self.epsilon = float(epsilon)
        if mean is None or std is None:
            raise ValueError("SSH normalization mean/std from the training split are required")
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.raw_channels or std.numel() != self.raw_channels:
            raise ValueError(f"SSH statistics must contain {self.raw_channels} channels")
        if not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(std)):
            raise ValueError("SSH statistics must be finite")
        if torch.any(std < 0):
            raise ValueError("SSH standard deviations must be non-negative")
        self.register_buffer("ssh_mean", mean)
        self.register_buffer("ssh_std", std)
        self.projection = nn.Linear(self.raw_channels, self._output_channels)

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def standardized_basis(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = real_spherical_harmonics(unit_direction, self.degree)
        return (raw - self.ssh_mean.to(raw)) / (self.ssh_std.to(raw) + self.epsilon)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        return self.projection(self.standardized_basis(unit_direction))


class FourierPlusSSHEncoder(nn.Module):
    """V2.10 Fourier features plus an SSH correction initialized exactly off."""

    def __init__(
        self,
        *,
        fourier_bands: int,
        ssh_degree: int,
        ssh_mean: torch.Tensor,
        ssh_std: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.fourier = DirectionFourierEncoder(fourier_bands)
        if self.fourier.output_channels != 27:
            raise ValueError("the controlled hybrid requires the V2.10 27-channel Fourier branch")
        self.ssh = RealSSHEncoder(
            ssh_degree,
            output_channels=27,
            mean=ssh_mean,
            std=ssh_std,
            epsilon=epsilon,
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    @property
    def output_channels(self) -> int:
        return 27

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        return self.fourier(unit_direction) + self.alpha * self.ssh(unit_direction)


@dataclass(frozen=True)
class DirectionEncodingConfig:
    kind: EncodingKind = "fourier"
    fourier_bands: int = 4
    ssh_degree: int = 4
    output_channels: int = 27
    normalization_epsilon: float = 1e-6


def load_ssh_statistics(path: str | Path, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the degree prefix from a training-only statistics NPZ."""
    channel_count = (degree + 1) ** 2
    with np.load(Path(path), allow_pickle=False) as archive:
        fitted_degree = int(np.asarray(archive["degree"]).item())
        if fitted_degree < degree:
            raise ValueError(f"statistics degree {fitted_degree} is below requested degree {degree}")
        mean = np.asarray(archive["mean"], dtype=np.float32)[:channel_count]
        std = np.asarray(archive["std"], dtype=np.float32)[:channel_count]
    return torch.from_numpy(mean.copy()), torch.from_numpy(std.copy())


def build_direction_encoder(
    config: DirectionEncodingConfig,
    *,
    ssh_mean: torch.Tensor | None = None,
    ssh_std: torch.Tensor | None = None,
) -> nn.Module:
    if config.output_channels != 27:
        raise ValueError("V2.11 controlled encoders must emit 27 channels")
    if config.kind == "fourier":
        encoder = DirectionFourierEncoder(config.fourier_bands)
        if encoder.output_channels != config.output_channels:
            raise ValueError("Fourier settings do not produce the required 27 channels")
        return encoder
    if ssh_mean is None or ssh_std is None:
        raise ValueError("SSH and hybrid encoders require training-split statistics")
    if config.kind == "ssh":
        return RealSSHEncoder(
            config.ssh_degree,
            output_channels=config.output_channels,
            mean=ssh_mean,
            std=ssh_std,
            epsilon=config.normalization_epsilon,
        )
    if config.kind == "fourier_plus_ssh":
        return FourierPlusSSHEncoder(
            fourier_bands=config.fourier_bands,
            ssh_degree=config.ssh_degree,
            ssh_mean=ssh_mean,
            ssh_std=ssh_std,
            epsilon=config.normalization_epsilon,
        )
    raise ValueError(f"unsupported direction encoding kind: {config.kind}")
