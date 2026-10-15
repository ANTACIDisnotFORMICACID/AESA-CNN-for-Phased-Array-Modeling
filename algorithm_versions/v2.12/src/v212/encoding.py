"""Controlled 27-channel direction encoders for the V2.12 Stage-1 matrix."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .basis import FiltrationSSHTransform, real_spherical_harmonics


def _validate_direction(unit_direction: torch.Tensor) -> None:
    if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
        raise ValueError("unit_direction must have shape [B,Q,3]")


def _normalised_loss_weights(
    count: int,
    loss_weights: torch.Tensor | None,
) -> torch.Tensor:
    if loss_weights is None:
        return torch.full((count,), 1.0 / count, dtype=torch.float64)
    weights = torch.as_tensor(loss_weights, dtype=torch.float64).reshape(-1)
    if weights.shape != (count,) or torch.any(weights < 0) or not torch.isfinite(weights).all():
        raise ValueError("loss_weights must be finite, non-negative and length N")
    if float(weights.sum()) <= 0:
        raise ValueError("loss_weights must have positive sum")
    return weights / weights.sum()


def weighted_rms(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("values/weights shapes are incompatible")
    return torch.sqrt((weights[:, None] * values.square()).sum(dim=0).clamp_min(1e-16))


@dataclass(frozen=True)
class WeightedProjectionFit:
    coefficients: torch.Tensor
    gram: torch.Tensor
    gram_condition: float
    identity_error: float
    orthogonality_residual: float
    used_fast_path: bool


def fit_weighted_projection(
    basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    identity_tolerance: float = 1e-6,
) -> WeightedProjectionFit:
    """Fit ``basis @ coefficients`` under an explicit discrete measure."""
    if basis.ndim != 2 or target.ndim != 2 or basis.shape[0] != target.shape[0]:
        raise ValueError("basis and target must be matrices with the same row count")
    weights = _normalised_loss_weights(len(basis), weights).to(basis)
    gram = basis.T @ (weights[:, None] * basis)
    rhs = basis.T @ (weights[:, None] * target)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    identity_error = float(torch.linalg.matrix_norm(gram - identity, ord=2))
    eigenvalues = torch.linalg.eigvalsh(gram).real
    positive = eigenvalues[eigenvalues > 1e-14 * eigenvalues.max().clamp_min(1e-14)]
    condition = float(positive.max() / positive.min()) if positive.numel() else float("inf")
    fast = identity_error < identity_tolerance
    if fast:
        coefficients = rhs
    else:
        factor, info = torch.linalg.cholesky_ex(gram)
        coefficients = (
            torch.cholesky_solve(rhs, factor)
            if int(info.max()) == 0
            else torch.linalg.lstsq(gram, rhs).solution
        )
    residual = target - basis @ coefficients
    cross = basis.T @ (weights[:, None] * residual)
    orthogonality = float(cross.abs().max()) if cross.numel() else 0.0
    return WeightedProjectionFit(
        coefficients=coefficients,
        gram=gram,
        gram_condition=condition,
        identity_error=identity_error,
        orthogonality_residual=orthogonality,
        used_fast_path=fast,
    )


def _linear_without_bias(input_channels: int, output_channels: int, seed: int) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = nn.Linear(input_channels, output_channels, bias=False)
    return layer


class DirectionFourierEncoder(nn.Module):
    """Exact V2.10 Fourier mapping."""

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
        _validate_direction(unit_direction)
        if self.bands == 0:
            return unit_direction
        angles = unit_direction[..., :, None] * self.frequencies.to(unit_direction)
        sine = torch.sin(angles).reshape(*unit_direction.shape[:2], -1)
        cosine = torch.cos(angles).reshape(*unit_direction.shape[:2], -1)
        return torch.cat((unit_direction, sine, cosine), dim=-1)


class ResidualizedDirectionFourierEncoder(nn.Module):
    """Fourier-27 with its train-measure filtration projection removed."""

    def __init__(
        self,
        transform: FiltrationSSHTransform,
        training_direction: torch.Tensor,
        *,
        loss_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.degree = transform.degree
        self.fourier = DirectionFourierEncoder(4)
        direction = training_direction.detach().to(torch.float64)
        weights = _normalised_loss_weights(len(direction), loss_weights)
        with torch.no_grad():
            phi = transform.evaluate(direction)
            fourier = self.fourier(direction[None])[0]
            fit = fit_weighted_projection(phi, fourier, weights)
            residual = fourier - phi @ fit.coefficients
            residual_rms = weighted_rms(residual, weights)
            original_rms = weighted_rms(fourier, weights)
            scale = torch.where(
                residual_rms > 1e-3 * original_rms.clamp_min(1e-12),
                original_rms / residual_rms.clamp_min(1e-12),
                torch.zeros_like(residual_rms),
            )
        self.register_buffer("ssh_transform", transform.transform.to(torch.float32))
        self.register_buffer("projection_coefficients", fit.coefficients.to(torch.float32))
        self.register_buffer("projection_gram", fit.gram.to(torch.float32))
        self.register_buffer("residual_scale", scale.to(torch.float32))
        self.gram_condition = fit.gram_condition
        self.identity_error = fit.identity_error
        self.orthogonality_residual = fit.orthogonality_residual
        self.used_fast_path = fit.used_fast_path

    @property
    def output_channels(self) -> int:
        return 27

    def filtration_basis(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = real_spherical_harmonics(unit_direction, self.degree)
        return raw @ self.ssh_transform.to(raw)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier(unit_direction)
        phi = self.filtration_basis(unit_direction)
        residual = fourier - phi @ self.projection_coefficients.to(fourier)
        return residual * self.residual_scale.to(residual)


class FiltrationSSHEncoder(nn.Module):
    """Per-filtration-block projection with interpretable, zero-start gates."""

    def __init__(
        self,
        transform: FiltrationSSHTransform,
        *,
        include_fourier: bool = True,
        output_channels: int = 27,
        gate_initial: float = 0.0,
        seed: int = 21200,
        minimum_degree: int = 0,
    ) -> None:
        super().__init__()
        if output_channels != 27:
            raise ValueError("the controlled CNN contract requires 27 channels")
        if minimum_degree < 0 or minimum_degree > transform.degree:
            raise ValueError("minimum_degree is outside the fitted transform")
        self.degree = transform.degree
        self.ranks = transform.ranks
        self.minimum_degree = int(minimum_degree)
        self.include_fourier = bool(include_fourier)
        self.fourier = DirectionFourierEncoder(4) if include_fourier else None
        self.register_buffer("ssh_transform", transform.transform.to(torch.float32))
        projections: list[nn.Module] = []
        active_degrees: list[int] = []
        for l_value, rank in enumerate(transform.ranks):
            if l_value < minimum_degree or rank == 0:
                continue
            projections.append(_linear_without_bias(rank, output_channels, seed + l_value))
            active_degrees.append(l_value)
        self.projections = nn.ModuleList(projections)
        self.active_degrees = tuple(active_degrees)
        self.alpha = nn.Parameter(torch.full((len(active_degrees),), float(gate_initial)))
        self._block_slices = transform.block_slices
        self._output_channels = int(output_channels)

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def filtration_basis(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = real_spherical_harmonics(unit_direction, self.degree)
        return raw @ self.ssh_transform.to(raw)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        _validate_direction(unit_direction)
        if self.include_fourier:
            output = self.fourier(unit_direction)
        else:
            output = torch.zeros(
                *unit_direction.shape[:2],
                self.output_channels,
                device=unit_direction.device,
                dtype=unit_direction.dtype,
            )
        phi = self.filtration_basis(unit_direction)
        for gate_index, (degree, projection) in enumerate(
            zip(self.active_degrees, self.projections)
        ):
            output = output + self.alpha[gate_index] * projection(phi[..., self._block_slices[degree]])
        return output


class RawSSHEncoder(nn.Module):
    """RMS-scaled restricted SSH control without centering or whitening."""

    def __init__(
        self,
        degree: int,
        rms: torch.Tensor,
        *,
        include_fourier: bool,
        output_channels: int = 27,
        gate_initial: float = 0.0,
        seed: int = 21220,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        channels = (degree + 1) ** 2
        rms = torch.as_tensor(rms, dtype=torch.float32).reshape(-1)
        if rms.shape != (channels,) or torch.any(rms < 0):
            raise ValueError("rms must contain one non-negative value per raw SSH channel")
        self.degree = int(degree)
        self.include_fourier = bool(include_fourier)
        self.fourier = DirectionFourierEncoder(4) if include_fourier else None
        self.register_buffer("rms", rms)
        self.epsilon = float(epsilon)
        self.projections = nn.ModuleList()
        self.block_slices: list[slice] = []
        start = 0
        for l_value in range(degree + 1):
            width = 2 * l_value + 1
            self.block_slices.append(slice(start, start + width))
            self.projections.append(_linear_without_bias(width, output_channels, seed + l_value))
            start += width
        initial = 0.0 if include_fourier else float(gate_initial)
        self.alpha = nn.Parameter(torch.full((degree + 1,), initial))
        self._output_channels = int(output_channels)

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def scaled_basis(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = real_spherical_harmonics(unit_direction, self.degree)
        return raw / (self.rms.to(raw) + self.epsilon)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        _validate_direction(unit_direction)
        if self.include_fourier:
            output = self.fourier(unit_direction)
        else:
            output = torch.zeros(
                *unit_direction.shape[:2], self.output_channels,
                device=unit_direction.device, dtype=unit_direction.dtype,
            )
        basis = self.scaled_basis(unit_direction)
        for l_value, (block, projection) in enumerate(zip(self.block_slices, self.projections)):
            output = output + self.alpha[l_value] * projection(basis[..., block])
        return output


class MatchedFourierEncoder(nn.Module):
    """Capacity-matched Fourier auxiliary branch with SSH-shaped gate blocks."""

    def __init__(
        self,
        training_direction: torch.Tensor,
        ranks: tuple[int, ...],
        *,
        extended_bands: int,
        loss_weights: torch.Tensor | None = None,
        output_channels: int = 27,
        seed: int = 21240,
    ) -> None:
        super().__init__()
        self.main = DirectionFourierEncoder(4)
        self.extended = DirectionFourierEncoder(extended_bands)
        dictionary_channels = self.extended.output_channels
        retained_channels = sum(ranks)
        if retained_channels > dictionary_channels:
            raise ValueError("extended Fourier dictionary is smaller than retained SSH capacity")
        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(
            dictionary_channels, retained_channels, generator=generator, dtype=torch.float64
        )
        fixed_projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        weights = _normalised_loss_weights(len(training_direction), loss_weights)
        with torch.no_grad():
            train = training_direction.detach().to(torch.float64)[None]
            auxiliary = self.extended(train)[0] @ fixed_projection
            rms = weighted_rms(auxiliary, weights)
        self.register_buffer("fixed_projection", fixed_projection.to(torch.float32))
        self.register_buffer("auxiliary_rms", rms.to(torch.float32))
        self.ranks = tuple(int(x) for x in ranks)
        self.block_slices: list[slice] = []
        self.projections = nn.ModuleList()
        active: list[int] = []
        start = 0
        for l_value, rank in enumerate(self.ranks):
            self.block_slices.append(slice(start, start + rank))
            if rank:
                self.projections.append(_linear_without_bias(rank, output_channels, seed + 100 + l_value))
                active.append(l_value)
            start += rank
        self.active_degrees = tuple(active)
        self.alpha = nn.Parameter(torch.zeros(len(active)))
        self._output_channels = int(output_channels)

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def auxiliary_features(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = self.extended(unit_direction) @ self.fixed_projection.to(unit_direction)
        return raw / self.auxiliary_rms.to(raw).clamp_min(1e-6)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        output = self.main(unit_direction)
        auxiliary = self.auxiliary_features(unit_direction)
        for gate_index, (degree, projection) in enumerate(
            zip(self.active_degrees, self.projections)
        ):
            output = output + self.alpha[gate_index] * projection(
                auxiliary[..., self.block_slices[degree]]
            )
        return output


class MatchedFourierBasis(nn.Module):
    """Fixed Fourier basis with the same non-constant block widths as SSH."""

    def __init__(
        self,
        training_direction: torch.Tensor,
        ranks: tuple[int, ...],
        *,
        loss_weights: torch.Tensor | None = None,
        extended_bands: int = 6,
        seed: int = 21280,
    ) -> None:
        super().__init__()
        self.extended = DirectionFourierEncoder(extended_bands)
        self.ranks = tuple(int(value) for value in ranks)
        nonconstant_ranks = self.ranks[1:]
        retained = sum(nonconstant_ranks)
        if retained > self.extended.output_channels:
            raise ValueError("extended Fourier dictionary is smaller than matched capacity")
        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(
            self.extended.output_channels, retained, generator=generator, dtype=torch.float64
        )
        fixed_projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        direction = training_direction.detach().to(torch.float64)
        weights = _normalised_loss_weights(len(direction), loss_weights)
        with torch.no_grad():
            raw = self.extended(direction[None])[0] @ fixed_projection
            rms = weighted_rms(raw, weights)
        self.register_buffer("fixed_projection", fixed_projection.to(torch.float32))
        self.register_buffer("rms", rms.to(torch.float32))
        slices: list[slice] = []
        start = 0
        for rank in nonconstant_ranks:
            slices.append(slice(start, start + rank))
            start += rank
        self.block_slices = tuple(slices)

    @property
    def output_channels(self) -> int:
        return int(self.fixed_projection.shape[1])

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = self.extended(unit_direction) @ self.fixed_projection.to(unit_direction)
        return raw / self.rms.to(raw).clamp_min(1e-6)


class ResidualizedFourierSSHEncoder(FiltrationSSHEncoder):
    """E3: input-space residualized Fourier plus filtration SSH."""

    def __init__(
        self,
        transform: FiltrationSSHTransform,
        training_direction: torch.Tensor,
        *,
        loss_weights: torch.Tensor | None = None,
        output_channels: int = 27,
        seed: int = 21260,
    ) -> None:
        super().__init__(
            transform,
            include_fourier=False,
            output_channels=output_channels,
            gate_initial=0.0,
            seed=seed,
        )
        self.fourier = ResidualizedDirectionFourierEncoder(
            transform, training_direction, loss_weights=loss_weights
        )

    def residualized_fourier(self, unit_direction: torch.Tensor) -> torch.Tensor:
        return self.fourier(unit_direction)

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        output = self.residualized_fourier(unit_direction)
        phi = self.filtration_basis(unit_direction)
        for gate_index, (degree, projection) in enumerate(
            zip(self.active_degrees, self.projections)
        ):
            output = output + self.alpha[gate_index] * projection(
                phi[..., self._block_slices[degree]]
            )
        return output


def raw_ssh_rms(
    training_direction: torch.Tensor,
    degree: int,
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    direction = training_direction.detach().to(torch.float64)
    weights = _normalised_loss_weights(len(direction), loss_weights)
    return weighted_rms(real_spherical_harmonics(direction, degree), weights).to(torch.float32)
