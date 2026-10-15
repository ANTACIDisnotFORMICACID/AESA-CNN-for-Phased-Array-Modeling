"""Deterministic phase-only, continuous disk, and INT8 physical oracles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from hfss_physics_v26 import HfssPatternData, PatternMetrics, project_complex_disk, quantize_int8_disk


@dataclass(frozen=True)
class OracleConfig:
    starts: int = 32
    max_steps: int = 300
    exclusion_deg: float = 8.0
    target_slack_db: float = 0.5
    seed: int = 26000


@dataclass
class OracleResult:
    target_theta_deg: float
    target_phi_deg: float
    phase_only_weights: np.ndarray
    continuous_weights: np.ndarray
    int8_codes: np.ndarray
    int8_weights: np.ndarray
    phase_only_metrics: PatternMetrics
    continuous_metrics: PatternMetrics
    int8_metrics: PatternMetrics
    reference_target_power: float
    optimizer_success: bool
    optimizer_message: str

    def serializable(self) -> dict[str, Any]:
        def weight_list(value):
            value = np.asarray(value)
            return [[float(v.real), float(v.imag)] for v in value]

        def metric_dict(value: PatternMetrics):
            payload = asdict(value)
            payload["feasible"] = bool(value.feasible)
            return payload

        return {
            "target_theta_deg": self.target_theta_deg,
            "target_phi_deg": self.target_phi_deg,
            "phase_only_weights_iq": weight_list(self.phase_only_weights),
            "continuous_weights_iq": weight_list(self.continuous_weights),
            "int8_codes_iq": self.int8_codes.astype(int).tolist(),
            "int8_weights_iq": weight_list(self.int8_weights),
            "phase_only_metrics": metric_dict(self.phase_only_metrics),
            "continuous_metrics": metric_dict(self.continuous_metrics),
            "int8_metrics": metric_dict(self.int8_metrics),
            "reference_target_power": self.reference_target_power,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


class PhysicalOracle:
    def __init__(self, data: HfssPatternData, config: OracleConfig = OracleConfig()):
        self.data = data
        self.config = config

    def _target_gram(self, theta_deg: float, phi_deg: float) -> np.ndarray:
        ht, hp = self.data.response_at(theta_deg, phi_deg)
        return np.outer(ht.conj(), ht) + np.outer(hp.conj(), hp)

    def target_power_phase_oracle(self, theta_deg: float, phi_deg: float) -> np.ndarray:
        """Coordinate phase ascent for the per-port unit-modulus target-power maximum."""
        gram = self._target_gram(theta_deg, phi_deg)
        values, vectors = np.linalg.eigh(gram)
        starts = [np.exp(1j * np.angle(vectors[:, -1]))]
        rng = np.random.default_rng(self.config.seed + int(round(theta_deg * 10)) * 1009 + int(round(phi_deg * 10)))
        starts.extend(np.exp(1j * rng.uniform(-np.pi, np.pi, size=4)) for _ in range(max(0, self.config.starts - 1)))
        best_w = starts[0]
        best_power = -np.inf
        for start in starts:
            w = start.copy()
            for _ in range(100):
                previous = w.copy()
                for n in range(4):
                    cross = gram[n] @ w - gram[n, n] * w[n]
                    if abs(cross) > 1e-15:
                        w[n] = np.exp(1j * np.angle(cross))
                if np.max(np.abs(w - previous)) < 1e-10:
                    break
            power = float(np.real(w.conj() @ gram @ w))
            if power > best_power:
                best_w, best_power = w.copy(), power
        return best_w

    @lru_cache(maxsize=256)
    def _outside_flat_indices(self, theta_key: float, phi_key: float) -> np.ndarray:
        theta, phi = self.data.flat_theta_phi
        from hfss_physics_v26 import angular_distance_deg
        return np.flatnonzero(angular_distance_deg(theta, phi, theta_key, phi_key) > self.config.exclusion_deg)

    @staticmethod
    def _decode(parameters: np.ndarray) -> np.ndarray:
        amplitudes = np.clip(parameters[:4], 0.0, 1.0)
        phases = parameters[4:]
        return amplitudes * np.exp(1j * phases)

    def _score(self, weights: np.ndarray, theta_deg: float, phi_deg: float, minimum_target: float) -> float:
        target = self.data.target_power(weights, theta_deg, phi_deg)
        power = self.data.power(weights).reshape(-1)
        outside = self._outside_flat_indices(float(theta_deg), float(phi_deg))
        outside_peak = float(np.max(power[outside]))
        margin = 10.0 * np.log10((target + 1e-30) / (outside_peak + 1e-30))
        shortfall = max(0.0, (minimum_target - target) / max(minimum_target, 1e-30))
        return float(margin - 100.0 * shortfall * shortfall)

    def continuous_disk_oracle(self, theta_deg: float, phi_deg: float, phase_weights: np.ndarray) -> tuple[np.ndarray, bool, str]:
        reference = self.data.target_power(phase_weights, theta_deg, phi_deg)
        minimum_target = reference * 10.0 ** (-self.config.target_slack_db / 10.0)
        rng = np.random.default_rng(self.config.seed + int(round(theta_deg * 100)) * 4099 + int(round(phi_deg * 100)))
        starts = [np.concatenate((np.ones(4), np.angle(phase_weights)))]
        for _ in range(max(0, self.config.starts - 1)):
            starts.append(np.concatenate((rng.uniform(0.35, 1.0, 4), rng.uniform(-np.pi, np.pi, 4))))
        best_w = phase_weights.copy()
        best_score = self._score(best_w, theta_deg, phi_deg, minimum_target)
        messages = []
        any_success = False
        bounds = [(0.0, 1.0)] * 4 + [(-np.pi, np.pi)] * 4
        outside = self._outside_flat_indices(float(theta_deg), float(phi_deg))
        flat_theta = self.data.etheta.reshape(4, -1)
        flat_phi = self.data.ephi.reshape(4, -1)

        def exact_top(weights: np.ndarray, count: int = 24) -> np.ndarray:
            et = weights @ flat_theta[:, outside]
            ep = weights @ flat_phi[:, outside]
            power = np.abs(et) ** 2 + np.abs(ep) ** 2
            local = np.argpartition(power, -min(count, len(power)))[-min(count, len(power)):]
            return outside[local]

        for start in starts:
            parameters = start.copy()
            active = np.unique(np.concatenate((exact_top(phase_weights), exact_top(self._decode(start)))))
            result = None
            # Exchange method: optimize a small active set, then add exact full-sphere violators.
            for _ in range(4):
                ht_active = flat_theta[:, active]
                hp_active = flat_phi[:, active]

                def smooth_ratio(p):
                    w = self._decode(p)
                    target = self.data.target_power(w, theta_deg, phi_deg)
                    et = w @ ht_active
                    ep = w @ hp_active
                    lobe = np.abs(et) ** 2 + np.abs(ep) ** 2
                    log_ratio = np.log(lobe + 1e-30) - np.log(target + 1e-30)
                    return float(logsumexp(12.0 * log_ratio) / 12.0)

                result = minimize(
                    smooth_ratio,
                    parameters,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=({"type": "ineq", "fun": lambda p: self.data.target_power(self._decode(p), theta_deg, phi_deg) - minimum_target},),
                    options={"maxiter": max(10, self.config.max_steps // 4), "ftol": 1e-8},
                )
                parameters = result.x
                new_active = np.unique(np.concatenate((active, exact_top(self._decode(parameters)))))
                if len(new_active) == len(active):
                    break
                active = new_active
            candidate = self._decode(parameters)
            target = self.data.target_power(candidate, theta_deg, phi_deg)
            score = self._score(candidate, theta_deg, phi_deg, minimum_target)
            if target >= minimum_target * (1.0 - 1e-6) and score > best_score:
                best_w, best_score = candidate, score
            if result is not None:
                any_success = any_success or bool(result.success)
                messages.append(str(result.message))
        return project_complex_disk(best_w), any_success, "; ".join(sorted(set(messages)))

    def int8_coordinate_oracle(
        self,
        continuous: np.ndarray,
        theta_deg: float,
        phi_deg: float,
        reference_target_power: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        codes, weights = quantize_int8_disk(continuous)
        codes16 = codes.astype(np.int16)
        minimum_target = reference_target_power * 10.0 ** (-self.config.target_slack_db / 10.0)
        current = self._score(weights, theta_deg, phi_deg, minimum_target)
        for _ in range(32):
            improved = False
            for n in range(4):
                best_codes = codes16.copy()
                best_score = current
                for di in (-2, -1, 0, 1, 2):
                    for dq in (-2, -1, 0, 1, 2):
                        candidate = codes16.copy()
                        candidate[n, 0] += di
                        candidate[n, 1] += dq
                        if np.any(candidate < -127) or np.any(candidate > 127):
                            continue
                        if int(candidate[n, 0]) ** 2 + int(candidate[n, 1]) ** 2 > 127 ** 2:
                            continue
                        candidate_w = candidate[:, 0] / 127.0 + 1j * candidate[:, 1] / 127.0
                        if self.data.target_power(candidate_w, theta_deg, phi_deg) < minimum_target:
                            continue
                        score = self._score(candidate_w, theta_deg, phi_deg, minimum_target)
                        if score > best_score + 1e-9:
                            best_codes, best_score = candidate, score
                if best_score > current + 1e-9:
                    codes16, current, improved = best_codes, best_score, True
            if not improved:
                break
        restored = codes16[:, 0] / 127.0 + 1j * codes16[:, 1] / 127.0
        return codes16.astype(np.int8), restored.astype(np.complex64)

    def solve(self, theta_deg: float, phi_deg: float) -> OracleResult:
        phase = self.target_power_phase_oracle(theta_deg, phi_deg)
        continuous, success, message = self.continuous_disk_oracle(theta_deg, phi_deg, phase)
        reference_target_power = self.data.target_power(phase, theta_deg, phi_deg)
        codes, integer = self.int8_coordinate_oracle(continuous, theta_deg, phi_deg, reference_target_power)
        return OracleResult(
            target_theta_deg=float(theta_deg),
            target_phi_deg=float(phi_deg),
            phase_only_weights=phase,
            continuous_weights=continuous,
            int8_codes=codes,
            int8_weights=integer,
            phase_only_metrics=self.data.metrics(phase, theta_deg, phi_deg, self.config.exclusion_deg),
            continuous_metrics=self.data.metrics(continuous, theta_deg, phi_deg, self.config.exclusion_deg),
            int8_metrics=self.data.metrics(integer, theta_deg, phi_deg, self.config.exclusion_deg),
            reference_target_power=reference_target_power,
            optimizer_success=success,
            optimizer_message=message,
        )
