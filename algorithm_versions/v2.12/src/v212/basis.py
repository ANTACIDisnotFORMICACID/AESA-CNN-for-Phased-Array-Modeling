"""Filtration-preserving real scalar spherical-harmonic basis construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch


def ssh_mode_index(degree: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    return [f"Y_{int(l)}_{int(m):+d}" for l, m in zip(degrees, orders)]


def _associated_legendre(
    degree: int,
    order: int,
    cosine_theta: torch.Tensor,
) -> torch.Tensor:
    if order < 0 or order > degree:
        raise ValueError("associated-Legendre order must satisfy 0 <= m <= l")
    one_minus_x2 = (1.0 - cosine_theta.square()).clamp_min(0.0)
    p_mm = torch.ones_like(cosine_theta)
    for m_step in range(1, order + 1):
        p_mm = -(2 * m_step - 1) * torch.sqrt(one_minus_x2) * p_mm
    if degree == order:
        return p_mm
    p_m1m = (2 * order + 1) * cosine_theta * p_mm
    if degree == order + 1:
        return p_m1m
    p_lm2, p_lm1 = p_mm, p_m1m
    for l_value in range(order + 2, degree + 1):
        p_lm = (
            (2 * l_value - 1) * cosine_theta * p_lm1
            - (l_value + order - 1) * p_lm2
        ) / (l_value - order)
        p_lm2, p_lm1 = p_lm1, p_lm
    return p_lm1


def real_spherical_harmonics(unit_direction: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate orthonormal real SSH in degree-major, m=-l..l order."""
    if unit_direction.shape[-1] != 3:
        raise ValueError("unit_direction must end with xyz")
    if not unit_direction.is_floating_point():
        raise TypeError("unit_direction must use a floating dtype")
    norm = torch.linalg.vector_norm(unit_direction, dim=-1, keepdim=True)
    if torch.any(norm <= 0):
        raise ValueError("unit_direction contains a zero vector")
    direction = unit_direction / norm
    cosine_theta = direction[..., 2].clamp(-1.0, 1.0)
    azimuth = torch.atan2(direction[..., 1], direction[..., 0])
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
                value = math.sqrt(2.0) * value * torch.sin(abs_m * azimuth)
            elif m_value > 0:
                value = math.sqrt(2.0) * value * torch.cos(m_value * azimuth)
            channels.append(value)
    return torch.stack(channels, dim=-1)


def _condition_from_gram(gram: torch.Tensor, tolerance: float = 1e-14) -> float:
    eigenvalues = torch.linalg.eigvalsh(gram).real
    positive = eigenvalues[eigenvalues > tolerance * eigenvalues.max().clamp_min(tolerance)]
    if positive.numel() == 0:
        return float("inf")
    return float((positive.max() / positive.min()).cpu())


@dataclass(frozen=True)
class FiltrationSSHTransform:
    """Fixed raw-SSH to numerically identifiable filtration basis transform."""

    degree: int
    transform: torch.Tensor
    ranks: tuple[int, ...]
    singular_values: tuple[tuple[float, ...], ...]
    discarded_mode_vectors: tuple[torch.Tensor, ...]
    raw_gram_condition: float
    retained_condition: float
    transformed_condition: float
    truncation_ratio: float = 1e-4
    measure: str = "empirical_equal_loss"

    def __post_init__(self) -> None:
        raw_channels = (self.degree + 1) ** 2
        if self.transform.ndim != 2 or self.transform.shape[0] != raw_channels:
            raise ValueError("transform must have shape [(L+1)^2, retained_channels]")
        if len(self.ranks) != self.degree + 1 or sum(self.ranks) != self.transform.shape[1]:
            raise ValueError("ranks must contain one entry per degree and match transform")
        if len(self.singular_values) != self.degree + 1:
            raise ValueError("singular_values must contain one entry per degree")
        if len(self.discarded_mode_vectors) != self.degree + 1:
            raise ValueError("discarded_mode_vectors must contain one entry per degree")
        for l_value, vectors in enumerate(self.discarded_mode_vectors):
            if vectors.ndim != 2 or vectors.shape[0] != 2 * l_value + 1:
                raise ValueError("discarded vectors must use the raw order width for each degree")

    @property
    def output_channels(self) -> int:
        return int(self.transform.shape[1])

    @property
    def block_slices(self) -> tuple[slice, ...]:
        result: list[slice] = []
        start = 0
        for rank in self.ranks:
            result.append(slice(start, start + rank))
            start += rank
        return tuple(result)

    def evaluate(self, unit_direction: torch.Tensor) -> torch.Tensor:
        raw = real_spherical_harmonics(unit_direction, self.degree)
        return raw @ self.transform.to(device=raw.device, dtype=raw.dtype)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        flat_singular = [value for block in self.singular_values for value in block]
        offsets = np.cumsum([0, *(len(block) for block in self.singular_values)])
        metadata = {
            "schema": "v212-filtration-ssh-1",
            "degree": self.degree,
            "ranks": list(self.ranks),
            "raw_gram_condition": self.raw_gram_condition,
            "retained_condition": self.retained_condition,
            "transformed_condition": self.transformed_condition,
            "truncation_ratio": self.truncation_ratio,
            "measure": self.measure,
            "mode_labels": ssh_mode_labels(self.degree),
        }
        arrays = {
            f"discarded_vectors_l{l_value}": vectors.detach().cpu().numpy()
            for l_value, vectors in enumerate(self.discarded_mode_vectors)
        }
        np.savez(
            target,
            transform=self.transform.detach().cpu().numpy(),
            singular_values=np.asarray(flat_singular, dtype=np.float64),
            singular_offsets=np.asarray(offsets, dtype=np.int64),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FiltrationSSHTransform":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            values = np.asarray(archive["singular_values"], dtype=np.float64)
            offsets = np.asarray(archive["singular_offsets"], dtype=np.int64)
            singular = tuple(
                tuple(float(x) for x in values[offsets[i] : offsets[i + 1]])
                for i in range(len(offsets) - 1)
            )
            transform = torch.from_numpy(
                np.asarray(archive["transform"], dtype=np.float64).copy()
            )
            discarded = tuple(
                torch.from_numpy(
                    np.asarray(
                        archive[f"discarded_vectors_l{l_value}"], dtype=np.float64
                    ).copy()
                )
                for l_value in range(int(metadata["degree"]) + 1)
            )
        return cls(
            degree=int(metadata["degree"]),
            transform=transform,
            ranks=tuple(int(x) for x in metadata["ranks"]),
            singular_values=singular,
            discarded_mode_vectors=discarded,
            raw_gram_condition=float(metadata["raw_gram_condition"]),
            retained_condition=float(metadata["retained_condition"]),
            transformed_condition=float(metadata["transformed_condition"]),
            truncation_ratio=float(metadata["truncation_ratio"]),
            measure=str(metadata["measure"]),
        )

    @classmethod
    def fit(
        cls,
        unit_direction: torch.Tensor,
        degree: int,
        *,
        loss_weights: torch.Tensor | None = None,
        truncation_ratio: float = 1e-4,
        measure: str = "empirical_equal_loss",
    ) -> "FiltrationSSHTransform":
        """Fit using the weights that actually multiply samples in the loss.

        Equal-loss samples use ``loss_weights=None``. Sampling probabilities must
        not be supplied again when the sample population already represents them.
        """
        if unit_direction.ndim != 2 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [N,3]")
        if degree < 0 or not 0 < truncation_ratio < 1:
            raise ValueError("degree and truncation_ratio are invalid")
        direction = unit_direction.detach().to(dtype=torch.float64, device="cpu")
        count = direction.shape[0]
        if count == 0:
            raise ValueError("at least one direction is required")
        if loss_weights is None:
            weights = torch.full((count,), 1.0 / count, dtype=torch.float64)
        else:
            weights = torch.as_tensor(loss_weights, dtype=torch.float64).reshape(-1)
            if weights.shape != (count,) or torch.any(weights < 0) or not torch.isfinite(weights).all():
                raise ValueError("loss_weights must be finite, non-negative and length N")
            if float(weights.sum()) <= 0:
                raise ValueError("loss_weights must have positive sum")
            weights = weights / weights.sum()

        raw = real_spherical_harmonics(direction, degree)
        raw_gram = raw.T @ (weights[:, None] * raw)
        raw_condition = _condition_from_gram(raw_gram)
        raw_channels = raw.shape[1]
        previous_phi = torch.empty(count, 0, dtype=torch.float64)
        previous_transform = torch.empty(raw_channels, 0, dtype=torch.float64)
        transforms: list[torch.Tensor] = []
        ranks: list[int] = []
        singular_values: list[tuple[float, ...]] = []
        discarded_mode_vectors: list[torch.Tensor] = []
        retained_conditions: list[float] = []
        start = 0

        for l_value in range(degree + 1):
            width = 2 * l_value + 1
            psi_l = raw[:, start : start + width]
            selector = torch.zeros(raw_channels, width, dtype=torch.float64)
            selector[start : start + width] = torch.eye(width, dtype=torch.float64)
            if previous_phi.shape[1]:
                cross = previous_phi.T @ (weights[:, None] * psi_l)
                residual = psi_l - previous_phi @ cross
                residual_transform = selector - previous_transform @ cross
            else:
                residual = psi_l
                residual_transform = selector

            _, sigma, vh = torch.linalg.svd(
                weights.sqrt()[:, None] * residual,
                full_matrices=False,
            )
            if sigma.numel() == 0 or float(sigma.max()) <= 0:
                keep = torch.zeros_like(sigma, dtype=torch.bool)
            else:
                keep = sigma.square() >= truncation_ratio * sigma.max().square()
            sigma_kept = sigma[keep]
            right_kept = vh.T[:, keep]
            discarded_mode_vectors.append(vh.T[:, ~keep].clone())
            if sigma_kept.numel():
                phi_l = residual @ right_kept @ torch.diag(sigma_kept.reciprocal())
                transform_l = (
                    residual_transform @ right_kept @ torch.diag(sigma_kept.reciprocal())
                )
                transforms.append(transform_l)
                previous_phi = torch.cat((previous_phi, phi_l), dim=1)
                previous_transform = torch.cat((previous_transform, transform_l), dim=1)
                retained_conditions.append(
                    float((sigma_kept.max().square() / sigma_kept.min().square()).cpu())
                )
            ranks.append(int(sigma_kept.numel()))
            singular_values.append(tuple(float(x) for x in sigma.cpu()))
            start += width

        if not transforms:
            raise ValueError("all SSH modes were truncated")
        transform = torch.cat(transforms, dim=1)
        phi = raw @ transform
        transformed_gram = phi.T @ (weights[:, None] * phi)
        return cls(
            degree=degree,
            transform=transform,
            ranks=tuple(ranks),
            singular_values=tuple(singular_values),
            discarded_mode_vectors=tuple(discarded_mode_vectors),
            raw_gram_condition=raw_condition,
            retained_condition=max(retained_conditions, default=float("inf")),
            transformed_condition=_condition_from_gram(transformed_gram),
            truncation_ratio=truncation_ratio,
            measure=measure,
        )
