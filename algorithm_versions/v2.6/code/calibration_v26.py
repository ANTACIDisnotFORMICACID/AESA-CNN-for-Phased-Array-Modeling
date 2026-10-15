"""Phase-invariant four-port calibration for the fixed HFSS physical backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hfss_physics_v26 import HfssPatternData, complex_to_iq, iq_to_complex


def dft_commands(count: int = 4) -> np.ndarray:
    q, n = np.meshgrid(np.arange(count), np.arange(count), indexing="ij")
    return np.exp(2j * np.pi * q * n / count).astype(np.complex128)


def normalize_global_phase(gains: np.ndarray, health: np.ndarray | None = None) -> np.ndarray:
    gains = np.asarray(gains, dtype=np.complex128)
    if health is None:
        health = np.ones(gains.shape, dtype=np.float64)
    health = np.asarray(health, dtype=np.float64)
    weighted = np.sum(health * gains, axis=-1, keepdims=True)
    fallback = np.sum(health * np.exp(1j * np.angle(gains + 1e-30)), axis=-1, keepdims=True)
    reference = np.where(np.abs(weighted) > 1e-10, weighted, fallback)
    return gains * np.exp(-1j * np.angle(reference))


@dataclass(frozen=True)
class CalibrationDesign:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    polarization_projection: np.ndarray  # [M, 2] Jones vectors
    scalar_response: np.ndarray  # [M, 4]
    commands: np.ndarray  # [Q, 4]
    condition_number: float

    def serializable(self) -> dict[str, Any]:
        def complex_pairs(value):
            value = np.asarray(value)
            return np.stack((value.real, value.imag), axis=-1).tolist()

        return {
            "theta_deg": self.theta_deg.tolist(),
            "phi_deg": self.phi_deg.tolist(),
            "polarization_projection_iq": complex_pairs(self.polarization_projection),
            "scalar_response_iq": complex_pairs(self.scalar_response),
            "commands_iq": complex_pairs(self.commands),
            "condition_number": self.condition_number,
            "measurement_iq_shape": ["B", 2, 4, 8],
        }


@dataclass(frozen=True)
class CalibrationEstimate:
    calibration_iq: np.ndarray
    confidence: np.ndarray
    residual_nmse_db: np.ndarray
    condition_number: float


def _scalar_response(data: HfssPatternData, theta_deg: float, phi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ht, hp = data.response_at(theta_deg, phi_deg)
    matrix = np.stack((ht, hp), axis=0)
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    projection = u[:, 0]
    scalar = projection.conj() @ matrix
    return projection, scalar


def build_d_optimal_design(
    data: HfssPatternData,
    measurement_count: int = 8,
    theta_candidates: np.ndarray | None = None,
    phi_candidates: np.ndarray | None = None,
) -> CalibrationDesign:
    """Greedy log-det selection of eight fixed, spatially separated measurements."""
    if theta_candidates is None:
        theta_candidates = np.arange(60.0, 121.0, 10.0)
    if phi_candidates is None:
        phi_candidates = np.arange(-30.0, 31.0, 10.0)
    commands = dft_commands(4)
    candidates = []
    for theta in theta_candidates:
        for phi in phi_candidates:
            projection, response = _scalar_response(data, float(theta), float(phi))
            block = commands * response[None, :]
            candidates.append((float(theta), float(phi), projection, response, block))
    selected: list[int] = []
    information = np.eye(4, dtype=np.complex128) * 1e-12
    while len(selected) < measurement_count:
        best_index = -1
        best_score = -np.inf
        for index, (_, _, _, _, block) in enumerate(candidates):
            if index in selected:
                continue
            trial = information + block.conj().T @ block
            sign, logdet = np.linalg.slogdet(trial)
            score = float(logdet.real) if sign.real > 0 else -np.inf
            # Avoid selecting essentially the same direction under grating-lobe symmetry.
            separation_penalty = 0.0
            for chosen in selected:
                dt = candidates[chosen][0] - candidates[index][0]
                dp = candidates[chosen][1] - candidates[index][1]
                if dt * dt + dp * dp < 99.0:
                    separation_penalty += 5.0
            score -= separation_penalty
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        information += candidates[best_index][4].conj().T @ candidates[best_index][4]
    rows = [candidates[index] for index in selected]
    response = np.stack([row[3] for row in rows])
    design_matrix = np.concatenate([commands * response[m][None, :] for m in range(measurement_count)], axis=0)
    return CalibrationDesign(
        theta_deg=np.asarray([row[0] for row in rows]),
        phi_deg=np.asarray([row[1] for row in rows]),
        polarization_projection=np.stack([row[2] for row in rows]),
        scalar_response=response,
        commands=commands,
        condition_number=float(np.linalg.cond(design_matrix)),
    )


def simulate_measurements(
    design: CalibrationDesign,
    calibration: np.ndarray,
    snr_db: float | np.ndarray = np.inf,
    global_phase_rad: float | np.ndarray = 0.0,
    seed: int = 26004,
) -> np.ndarray:
    """Return measurement_iq [B,2,Q,M] for deterministic tests or acquisition emulation."""
    calibration = np.asarray(calibration, dtype=np.complex128)
    if calibration.ndim == 1:
        calibration = calibration[None, :]
    if calibration.shape[1] != 4:
        raise ValueError("calibration must contain four ports")
    # y[b,q,m] = sum_n command[q,n] response[m,n] calibration[b,n]
    measurement = np.einsum("qn,mn,bn->bqm", design.commands, design.scalar_response, calibration, optimize=True)
    phase = np.asarray(global_phase_rad, dtype=np.float64).reshape(-1)
    if phase.size == 1:
        phase = np.repeat(phase, calibration.shape[0])
    measurement *= np.exp(1j * phase[:, None, None])
    snr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if snr.size == 1:
        snr = np.repeat(snr, calibration.shape[0])
    rng = np.random.default_rng(seed)
    for b in range(calibration.shape[0]):
        if np.isfinite(snr[b]):
            signal_power = float(np.mean(np.abs(measurement[b]) ** 2))
            noise_power = signal_power * 10.0 ** (-snr[b] / 10.0)
            noise = rng.normal(size=measurement[b].shape) + 1j * rng.normal(size=measurement[b].shape)
            measurement[b] += noise * np.sqrt(noise_power / 2.0)
    return np.stack((measurement.real, measurement.imag), axis=1).astype(np.float32)


class PhaseInvariantCalibrator:
    def __init__(self, design: CalibrationDesign, condition_limit: float = 20.0, nmse_limit_db: float = -20.0):
        self.design = design
        self.condition_limit = float(condition_limit)
        self.nmse_limit_db = float(nmse_limit_db)
        self._matrix = np.concatenate(
            [design.commands * design.scalar_response[m][None, :] for m in range(len(design.theta_deg))], axis=0
        )

    def calibrate(self, measurement_iq: np.ndarray, health_mask: np.ndarray | None = None) -> CalibrationEstimate:
        measurement_iq = np.asarray(measurement_iq)
        if measurement_iq.ndim == 3:
            measurement_iq = measurement_iq[None, ...]
        if measurement_iq.shape[1:] != (2, 4, len(self.design.theta_deg)):
            raise ValueError(f"expected [B,2,4,{len(self.design.theta_deg)}], got {measurement_iq.shape}")
        y = measurement_iq[:, 0] + 1j * measurement_iq[:, 1]
        y = np.transpose(y, (0, 2, 1)).reshape(y.shape[0], -1)  # [B,M*Q], matching matrix blocks
        estimate = np.stack([np.linalg.lstsq(self._matrix, row, rcond=None)[0] for row in y])
        if health_mask is None:
            health = np.ones_like(estimate.real)
        else:
            health = np.asarray(health_mask, dtype=np.float64)
            if health.ndim == 1:
                health = health[None, :]
        estimate = normalize_global_phase(estimate, health)
        predicted = np.einsum("rn,bn->br", self._matrix, estimate)
        # Residual is invariant to the global phase convention by fitting one scalar phase.
        alpha = np.sum(y * predicted.conj(), axis=1) / np.maximum(np.sum(np.abs(predicted) ** 2, axis=1), 1e-30)
        aligned = predicted * alpha[:, None]
        nmse = np.sum(np.abs(y - aligned) ** 2, axis=1) / np.maximum(np.sum(np.abs(y) ** 2, axis=1), 1e-30)
        nmse_db = 10.0 * np.log10(np.maximum(nmse, 1e-30))
        cond_score = np.clip(self.condition_limit / max(self.design.condition_number, 1e-12), 0.0, 1.0)
        residual_score = np.clip((self.nmse_limit_db - nmse_db) / 20.0 + 1.0, 0.0, 1.0)
        confidence = (cond_score * residual_score).astype(np.float32)[:, None]
        calibration_iq = np.stack((estimate.real, estimate.imag), axis=1)[:, :, None, :].astype(np.float32)
        return CalibrationEstimate(calibration_iq, confidence, nmse_db.astype(np.float32), self.design.condition_number)

    def holdout_nmse_db(
        self,
        data: HfssPatternData,
        estimated: np.ndarray,
        truth: np.ndarray,
        holdout_angles: np.ndarray,
    ) -> np.ndarray:
        estimated = normalize_global_phase(np.asarray(estimated, dtype=np.complex128))
        truth = normalize_global_phase(np.asarray(truth, dtype=np.complex128))
        if estimated.ndim == 1:
            estimated, truth = estimated[None, :], truth[None, :]
        responses = np.stack([_scalar_response(data, float(t), float(p))[1] for t, p in holdout_angles])
        y_est = np.einsum("mn,bn->bm", responses, estimated)
        y_true = np.einsum("mn,bn->bm", responses, truth)
        alpha = np.sum(y_true * y_est.conj(), axis=1) / np.maximum(np.sum(np.abs(y_est) ** 2, axis=1), 1e-30)
        error = np.sum(np.abs(y_true - alpha[:, None] * y_est) ** 2, axis=1)
        reference = np.maximum(np.sum(np.abs(y_true) ** 2, axis=1), 1e-30)
        return 10.0 * np.log10(np.maximum(error / reference, 1e-30))

