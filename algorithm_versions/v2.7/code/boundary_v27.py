"""Four-level boundary definitions and the public offline query API."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from v27_compat import enable_v26

enable_v26()
from hfss_physics_v26 import angular_distance_deg  # noqa: E402


GAIN_DB = -0.5
POINTING_DEG = 5.0
STRICT_MARGIN_DB = 3.0
ROBUST_FRACTION = 0.95
HPBW_DROP_DB = 3.0
TRACK_PEAK_JUMP_DEG = 3.0
TRACK_WEIGHT_CHANGE = 0.25


@dataclass(frozen=True)
class BoundaryFlags:
    gain_reachable: bool
    pointing_reachable: bool
    strict_unambiguous: bool
    robust_unambiguous: bool


@dataclass(frozen=True)
class BoundaryResult:
    target_hfss_deg: tuple[float, float]
    state_class: str
    flags: BoundaryFlags
    target_to_peak_db: float
    pointing_error_deg: float
    strict_margin_db: float
    hpbw_aware_margin_db: float
    robust_pass_fraction: float
    robust_evaluated: bool
    nearest_reachable_hfss_deg: tuple[float, float] | None
    continuous_tracking: bool
    reason: tuple[str, ...]

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_hfss_deg"] = list(self.target_hfss_deg)
        if self.nearest_reachable_hfss_deg is not None:
            value["nearest_reachable_hfss_deg"] = list(self.nearest_reachable_hfss_deg)
        value["reason"] = list(self.reason)
        return value


def classify_boundary(
    target_to_peak_db: float,
    pointing_error_deg: float,
    strict_margin_db: float,
    robust_pass_fraction: float = 0.0,
) -> BoundaryFlags:
    gain = bool(target_to_peak_db >= GAIN_DB)
    pointing = bool(gain and pointing_error_deg <= POINTING_DEG)
    strict = bool(pointing and strict_margin_db >= STRICT_MARGIN_DB)
    robust = bool(strict and robust_pass_fraction >= ROBUST_FRACTION)
    return BoundaryFlags(gain, pointing, strict, robust)


def boundary_reasons(flags: BoundaryFlags) -> tuple[str, ...]:
    reasons: list[str] = []
    if not flags.gain_reachable:
        reasons.append("TARGET_NOT_GLOBAL_PEAK")
    if flags.gain_reachable and not flags.pointing_reachable:
        reasons.append("POINTING_UNACHIEVABLE")
    if flags.pointing_reachable and not flags.strict_unambiguous:
        reasons.append("AMBIGUOUS_LOBE")
    if flags.strict_unambiguous and not flags.robust_unambiguous:
        reasons.append("NOT_ROBUST_AT_95_PERCENT")
    return tuple(reasons or ("NONE",))


def _connected_half_power_mask(power: np.ndarray) -> np.ndarray:
    """Connected -3 dB region containing the global peak; phi wraps at +/-180."""
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 2 or not np.isfinite(power).all():
        raise ValueError("power must be a finite [phi,theta] grid")
    peak = np.unravel_index(int(np.argmax(power)), power.shape)
    eligible = power >= float(power[peak]) * 10.0 ** (-HPBW_DROP_DB / 10.0)
    visited = np.zeros_like(eligible, dtype=bool)
    queue: deque[tuple[int, int]] = deque([peak])
    visited[peak] = True
    p_count, t_count = power.shape
    while queue:
        pi, ti = queue.popleft()
        for dp in (-1, 0, 1):
            for dt in (-1, 0, 1):
                if dp == 0 and dt == 0:
                    continue
                pj = (pi + dp) % p_count
                tj = ti + dt
                if tj < 0 or tj >= t_count or visited[pj, tj] or not eligible[pj, tj]:
                    continue
                visited[pj, tj] = True
                queue.append((pj, tj))
    return visited


def hpbw_aware_margin_db(power: np.ndarray, target_power: float) -> tuple[float, int]:
    connected = _connected_half_power_mask(power)
    outside = np.asarray(power)[~connected]
    outside_peak = float(np.max(outside)) if outside.size else 0.0
    margin = 10.0 * np.log10((float(target_power) + 1e-30) / (outside_peak + 1e-30))
    return float(margin), int(np.sum(connected))


def global_phase_aligned_weight_change(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized complex change after eliminating one unobservable global phase."""
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    inner = np.vdot(a, b)
    aligned = b * np.exp(-1j * np.angle(inner)) if abs(inner) > 1e-30 else b
    denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(aligned)), 1e-30)
    return float(np.linalg.norm(a - aligned) / denominator)


def tracking_pair(
    peak_a: tuple[float, float],
    peak_b: tuple[float, float],
    weight_a: np.ndarray,
    weight_b: np.ndarray,
) -> tuple[bool, float, float]:
    jump = float(angular_distance_deg(peak_a[0], peak_a[1], peak_b[0], peak_b[1]))
    change = global_phase_aligned_weight_change(weight_a, weight_b)
    return bool(jump <= TRACK_PEAK_JUMP_DEG and change <= TRACK_WEIGHT_CHANGE), jump, change


def tracking_grid(
    peak_theta: np.ndarray,
    peak_phi: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Trackability to the next theta/phi grid neighbor (no artificial edge wrap)."""
    peak_theta = np.asarray(peak_theta)
    peak_phi = np.asarray(peak_phi)
    weights = np.asarray(weights)
    if peak_theta.shape != peak_phi.shape or weights.shape[:2] != peak_theta.shape:
        raise ValueError("tracking grid shapes are inconsistent")
    shape = peak_theta.shape
    track_theta = np.zeros(shape, dtype=bool)
    track_phi = np.zeros(shape, dtype=bool)
    jump_theta = np.full(shape, np.nan, dtype=np.float32)
    jump_phi = np.full(shape, np.nan, dtype=np.float32)
    change_theta = np.full(shape, np.nan, dtype=np.float32)
    change_phi = np.full(shape, np.nan, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i + 1 < shape[0]:
                ok, jump, change = tracking_pair(
                    (peak_theta[i, j], peak_phi[i, j]),
                    (peak_theta[i + 1, j], peak_phi[i + 1, j]),
                    weights[i, j], weights[i + 1, j],
                )
                track_theta[i, j], jump_theta[i, j], change_theta[i, j] = ok, jump, change
            if j + 1 < shape[1]:
                ok, jump, change = tracking_pair(
                    (peak_theta[i, j], peak_phi[i, j]),
                    (peak_theta[i, j + 1], peak_phi[i, j + 1]),
                    weights[i, j], weights[i, j + 1],
                )
                track_phi[i, j], jump_phi[i, j], change_phi[i, j] = ok, jump, change
    continuous = track_theta | track_phi
    return {
        "continuous_tracking": continuous,
        "track_theta": track_theta,
        "track_phi": track_phi,
        "peak_jump_theta_deg": jump_theta,
        "peak_jump_phi_deg": jump_phi,
        "weight_change_theta": change_theta,
        "weight_change_phi": change_phi,
    }


class BoundaryIndex:
    """Read-only lookup over a released boundary map."""

    REQUIRED = (
        "theta_deg",
        "phi_deg",
        "state_classes",
        "target_to_peak_db",
        "pointing_error_deg",
        "strict_margin_db",
        "hpbw_aware_margin_db",
        "robust_pass_fraction",
        "continuous_tracking",
    )

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self.arrays = {key: np.asarray(value) for key, value in arrays.items()}
        missing = [key for key in self.REQUIRED if key not in self.arrays]
        if missing:
            raise ValueError(f"boundary map is missing {missing}")
        self.theta = self.arrays["theta_deg"].astype(float)
        self.phi = self.arrays["phi_deg"].astype(float)
        self.state_classes = tuple(str(value) for value in self.arrays["state_classes"].tolist())

    @classmethod
    def from_npz(cls, path: Path | str) -> "BoundaryIndex":
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({key: archive[key] for key in archive.files})

    def query_boundary(self, target_hfss_deg: tuple[float, float], state_class: str) -> BoundaryResult:
        theta, phi = map(float, target_hfss_deg)
        if state_class not in self.state_classes:
            raise KeyError(f"unknown state_class {state_class!r}; expected one of {self.state_classes}")
        si = self.state_classes.index(state_class)
        ti = int(np.argmin(np.abs(self.theta - theta)))
        pi = int(np.argmin(np.abs(self.phi - phi)))
        value = lambda name: float(self.arrays[name][si, ti, pi])
        robust = value("robust_pass_fraction")
        robust_evaluated = bool(self.arrays.get("robust_evaluated", np.ones_like(self.arrays["robust_pass_fraction"], dtype=bool))[si, ti, pi])
        in_domain = bool(self.theta[0] <= theta <= self.theta[-1] and self.phi[0] <= phi <= self.phi[-1])
        flags = classify_boundary(
            value("target_to_peak_db"), value("pointing_error_deg"), value("strict_margin_db"), robust
        )
        if not in_domain:
            flags = BoundaryFlags(False, False, False, False)
        pointing = (
            (self.arrays["target_to_peak_db"][si] >= GAIN_DB)
            & (self.arrays["pointing_error_deg"][si] <= POINTING_DEG)
        )
        nearest = None
        if np.any(pointing):
            ii, jj = np.nonzero(pointing)
            distance = angular_distance_deg(self.theta[ii], self.phi[jj], theta, phi)
            best = int(np.argmin(distance))
            nearest = (float(self.theta[ii[best]]), float(self.phi[jj[best]]))
        reasons = list(boundary_reasons(flags))
        if not in_domain:
            reasons = ["OUT_OF_DOMAIN"]
        if not robust_evaluated:
            reasons = [reason for reason in reasons if reason != "NOT_ROBUST_AT_95_PERCENT"]
            reasons.append("ROBUSTNESS_NOT_EVALUATED")
        return BoundaryResult(
            target_hfss_deg=(theta, phi),
            state_class=state_class,
            flags=flags,
            target_to_peak_db=value("target_to_peak_db"),
            pointing_error_deg=value("pointing_error_deg"),
            strict_margin_db=value("strict_margin_db"),
            hpbw_aware_margin_db=value("hpbw_aware_margin_db"),
            robust_pass_fraction=robust,
            robust_evaluated=robust_evaluated,
            nearest_reachable_hfss_deg=nearest,
            continuous_tracking=bool(self.arrays["continuous_tracking"][si, ti, pi]),
            reason=tuple(reasons),
        )


def query_boundary(
    target_hfss_deg: tuple[float, float],
    state_class: str,
    map_path: Path | str,
) -> BoundaryResult:
    return BoundaryIndex.from_npz(map_path).query_boundary(target_hfss_deg, state_class)
