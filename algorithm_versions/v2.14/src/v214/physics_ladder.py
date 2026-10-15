"""Nested, interpretable complex-field models used by the V2.14 residual ladder."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

import numpy as np
import torch

from v212.basis import real_spherical_harmonics


def ssh_basis(direction: np.ndarray, degree: int) -> np.ndarray:
    value = torch.as_tensor(np.asarray(direction), dtype=torch.float64)
    return real_spherical_harmonics(value, degree).numpy()


def weighted_complex_lstsq(design: np.ndarray, target: np.ndarray, weights: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    design = np.asarray(design, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    root = np.sqrt(weight / weight.sum())[:, None]
    x = root * design
    y = root * target
    gram = x.conj().T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.conj().T @ y)


def geometric_phase(direction: np.ndarray, positions_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    k = 2.0 * math.pi * float(frequency_hz) / 299_792_458.0
    return np.exp(1j * k * np.asarray(direction) @ np.asarray(positions_m).T)


def deembed(g: np.ndarray, direction: np.ndarray, positions_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    return np.asarray(g) / geometric_phase(direction, positions_m, frequency_hz)


def reembed(a: np.ndarray, direction: np.ndarray, positions_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    return np.asarray(a) * geometric_phase(direction, positions_m, frequency_hz)


@dataclass
class CommonPattern:
    degree: int
    coefficient: np.ndarray

    def predict(self, direction: np.ndarray, positions_m: np.ndarray, frequency_hz: float) -> np.ndarray:
        pattern = ssh_basis(direction, self.degree) @ self.coefficient
        return reembed(np.repeat(pattern[:, None], 16, axis=1), direction, positions_m, frequency_hz)


def fit_common_pattern(direction: np.ndarray, g: np.ndarray, positions_m: np.ndarray, frequency_hz: float, weights: np.ndarray, degree: int) -> CommonPattern:
    target = deembed(g, direction, positions_m, frequency_hz).mean(axis=1)
    coefficient = weighted_complex_lstsq(ssh_basis(direction, degree), target[:, None], weights)[:, 0]
    return CommonPattern(degree, coefficient)


def fit_element_gains(base: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, complex]:
    weight = np.asarray(weights, dtype=np.float64)[:, None]
    numerator = np.sum(weight * base.conj() * target, axis=0)
    denominator = np.maximum(np.sum(weight * np.abs(base) ** 2, axis=0), 1e-300)
    gains = numerator / denominator
    scale = complex(np.mean(gains))
    if abs(scale) > 1e-12:
        gains = gains / scale
    else:
        scale = 1.0 + 0.0j
    return base * (scale * gains)[None, :], gains, scale


def fit_phase_centers(
    base: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    weights: np.ndarray,
    frequency_hz: float,
    *,
    maximum_offset_m: float | None = None,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = maximum_offset_m or (299_792_458.0 / float(frequency_hz) / 4.0)
    q = torch.as_tensor(direction, dtype=torch.float64)
    source = torch.as_tensor(base, dtype=torch.complex128)
    truth = torch.as_tensor(target, dtype=torch.complex128)
    weight = torch.as_tensor(weights / np.sum(weights), dtype=torch.float64)[:, None]
    raw = torch.zeros((16, 3), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.7, max_iter=max_iter, line_search_fn="strong_wolfe")
    k = 2.0 * math.pi * float(frequency_hz) / 299_792_458.0

    def closure():
        optimizer.zero_grad()
        delta = maximum * torch.tanh(raw)
        delta = delta - delta.mean(dim=0, keepdim=True)
        prediction = source * torch.exp(torch.complex(torch.zeros_like(q @ delta.T), k * (q @ delta.T)))
        loss = (weight * (prediction - truth).abs().square()).sum() / (weight * truth.abs().square()).sum().clamp_min(1e-20)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        delta = maximum * torch.tanh(raw)
        delta = delta - delta.mean(dim=0, keepdim=True)
        phase = torch.exp(torch.complex(torch.zeros_like(q @ delta.T), k * (q @ delta.T)))
    return (source * phase).numpy(), delta.numpy()


def fit_s_guided_coupling(base: np.ndarray, target: np.ndarray, s_matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_matrix, dtype=np.complex128).copy()
    np.fill_diagonal(s, 0.0)
    norm = np.linalg.norm(s)
    if norm <= 0:
        raise ValueError("off-diagonal S matrix is zero")
    s /= norm
    features = np.stack((base @ s, base @ (s @ s)), axis=-1).reshape(-1, 2)
    residual = (target - base).reshape(-1, 1)
    expanded_weights = np.repeat(np.asarray(weights), 16)
    alpha = weighted_complex_lstsq(features, residual, expanded_weights)[:, 0]
    prediction = base + alpha[0] * (base @ s) + alpha[1] * (base @ (s @ s))
    return prediction, alpha


def element_classes() -> np.ndarray:
    result = []
    for row in range(4):
        for col in range(4):
            boundary = int(row in (0, 3)) + int(col in (0, 3))
            result.append(0 if boundary == 2 else (1 if boundary == 1 else 2))
    return np.asarray(result, dtype=np.int64)


def fit_class_residual(base: np.ndarray, target: np.ndarray, direction: np.ndarray, weights: np.ndarray, degree: int = 5) -> tuple[np.ndarray, np.ndarray]:
    basis = ssh_basis(direction, degree)
    classes = element_classes()
    coefficient = np.zeros((3, basis.shape[1]), dtype=np.complex128)
    residual = target - base
    for class_id in range(3):
        class_target = residual[:, classes == class_id].mean(axis=1)
        coefficient[class_id] = weighted_complex_lstsq(basis, class_target[:, None], weights)[:, 0]
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    addition = np.stack([basis @ coefficient[class_id] for class_id in classes], axis=1)
    return base + addition, coefficient


def fit_low_rank_residual(base: np.ndarray, target: np.ndarray, direction: np.ndarray, weights: np.ndarray, rank: int, degree: int = 12) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    residual = target - base
    normalized = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    weighted = np.sqrt(normalized)[:, None] * residual
    u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    rank = min(rank, len(singular))
    angular_sample = u[:, :rank] * singular[:rank][None, :] / np.sqrt(normalized)[:, None]
    basis = ssh_basis(direction, degree)
    coefficient = weighted_complex_lstsq(basis, angular_sample, normalized)
    angular = basis @ coefficient
    prediction = base + angular @ vh[:rank]
    return prediction, {"coefficient": coefficient, "aperture_modes": vh[:rank], "singular_values": singular[:rank]}


@dataclass
class LadderState:
    common: CommonPattern
    gain: np.ndarray | None = None
    gain_scale: complex = 1.0 + 0.0j
    phase_center_m: np.ndarray | None = None
    coupling_alpha: np.ndarray | None = None
    class_coefficient: np.ndarray | None = None
    low_rank: dict[str, np.ndarray] | None = None

    def clone(self) -> "LadderState":
        return deepcopy(self)

    def predict(self, direction: np.ndarray, positions_m: np.ndarray, frequency_hz: float, s_matrix: np.ndarray) -> np.ndarray:
        prediction = self.common.predict(direction, positions_m, frequency_hz)
        if self.gain is not None:
            prediction = prediction * (self.gain_scale * self.gain)[None, :]
        if self.phase_center_m is not None:
            k = 2.0 * math.pi * float(frequency_hz) / 299_792_458.0
            prediction = prediction * np.exp(1j * k * np.asarray(direction) @ self.phase_center_m.T)
        if self.coupling_alpha is not None:
            s = np.asarray(s_matrix, dtype=np.complex128).copy()
            np.fill_diagonal(s, 0.0)
            s /= np.linalg.norm(s)
            prediction = prediction + self.coupling_alpha[0] * (prediction @ s) + self.coupling_alpha[1] * (prediction @ (s @ s))
        if self.class_coefficient is not None:
            basis = ssh_basis(direction, 5)
            classes = element_classes()
            prediction = prediction + np.stack(
                [basis @ self.class_coefficient[class_id] for class_id in classes], axis=1
            )
        if self.low_rank is not None:
            basis = ssh_basis(direction, 12)
            prediction = prediction + (basis @ self.low_rank["coefficient"]) @ self.low_rank["aperture_modes"]
        return prediction


def fit_common_state(direction: np.ndarray, target: np.ndarray, positions_m: np.ndarray, frequency_hz: float, weights: np.ndarray, degree: int) -> LadderState:
    return LadderState(fit_common_pattern(direction, target, positions_m, frequency_hz, weights, degree))


def fit_stage(
    state: LadderState,
    stage: str,
    direction: np.ndarray,
    target: np.ndarray,
    positions_m: np.ndarray,
    frequency_hz: float,
    s_matrix: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int | None = None,
) -> LadderState:
    result = state.clone()
    base = result.predict(direction, positions_m, frequency_hz, s_matrix)
    if stage == "P1_element_gain":
        _, result.gain, result.gain_scale = fit_element_gains(base, target, weights)
    elif stage == "P2_phase_center":
        _, result.phase_center_m = fit_phase_centers(base, target, direction, weights, frequency_hz)
    elif stage == "P3_s_coupling":
        _, result.coupling_alpha = fit_s_guided_coupling(base, target, s_matrix, weights)
    elif stage == "P4_element_class":
        _, result.class_coefficient = fit_class_residual(base, target, direction, weights, degree=5)
    elif stage == "P5_low_rank":
        if rank is None:
            raise ValueError("low-rank stage requires rank")
        _, result.low_rank = fit_low_rank_residual(base, target, direction, weights, rank, degree=12)
    else:
        raise ValueError(f"unknown stage {stage}")
    return result
