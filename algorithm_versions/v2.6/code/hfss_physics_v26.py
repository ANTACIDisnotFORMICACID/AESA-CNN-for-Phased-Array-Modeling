"""Canonical HFSS physical observer and geometry contract for V2.6."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NPZ = VERSION_ROOT / "data" / "hfss_reference_1x4" / "hfss_active_element_patterns_47ghz.npz"
CANONICAL_META = CANONICAL_NPZ.with_suffix(".json")
GEOMETRY_CONTRACT = CANONICAL_NPZ.parent / "geometry_contract.json"


def iq_to_complex(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-3] != 2:
        raise ValueError(f"expected IQ axis at -3, got {value.shape}")
    return value[..., 0, :, :] + 1j * value[..., 1, :, :]


def complex_to_iq(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=-2).astype(np.float32)


def unit_vector_hfss(theta_deg: np.ndarray | float, phi_deg: np.ndarray | float) -> np.ndarray:
    """HFSS spherical convention: theta from +Z, phi from +X toward +Y."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    )


def angular_distance_deg(theta_a, phi_a, theta_b, phi_b) -> np.ndarray:
    a = unit_vector_hfss(theta_a, phi_a)
    b = unit_vector_hfss(theta_b, phi_b)
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


@dataclass(frozen=True)
class PatternMetrics:
    target_power: float
    peak_power: float
    target_to_peak_db: float
    peak_theta_deg: float
    peak_phi_deg: float
    pointing_error_deg: float
    outside_peak_power: float
    strong_peak_margin_db: float
    etheta_power: float
    ephi_power: float
    dominant_polarization_ratio_db: float

    @property
    def feasible(self) -> bool:
        return (
            self.target_to_peak_db >= -0.5
            and self.pointing_error_deg <= 5.0
            and self.strong_peak_margin_db >= 3.0
        )


@dataclass
class HfssPatternData:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    etheta: np.ndarray
    ephi: np.ndarray
    s: np.ndarray
    port_names: tuple[str, ...]
    y_mm: np.ndarray

    @classmethod
    def load(cls, npz_path: Path | str = CANONICAL_NPZ) -> "HfssPatternData":
        npz_path = Path(npz_path)
        with np.load(npz_path) as data:
            theta = np.asarray(data["theta_deg"], dtype=np.float64)
            phi = np.asarray(data["phi_deg"], dtype=np.float64)
            etheta = np.asarray(data["etheta"], dtype=np.complex64)
            ephi = np.asarray(data["ephi"], dtype=np.complex64)
            s = np.asarray(data["s"], dtype=np.complex64)
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        geometry = json.loads(GEOMETRY_CONTRACT.read_text(encoding="utf-8"))
        port_names = tuple(meta["port_names"])
        y_mm = np.asarray([item["y_mm"] for item in geometry["ports"]], dtype=np.float64)
        obj = cls(theta, phi, etheta, ephi, s, port_names, y_mm)
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.etheta.shape != (4, len(self.phi_deg), len(self.theta_deg)):
            raise ValueError(f"invalid ETheta shape {self.etheta.shape}")
        if self.ephi.shape != self.etheta.shape or self.s.shape != (4, 4):
            raise ValueError("inconsistent HFSS tensor shapes")
        if tuple(self.port_names) != ("1:1", "2:1", "3:1", "4:1"):
            raise ValueError(f"unexpected port order {self.port_names}")
        if not np.allclose(np.diff(self.y_mm), 6.8, atol=1e-9):
            raise ValueError(f"unexpected port geometry {self.y_mm}")
        for name, value in (("etheta", self.etheta), ("ephi", self.ephi), ("s", self.s)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")

    @property
    def flat_theta_phi(self) -> tuple[np.ndarray, np.ndarray]:
        phi, theta = np.meshgrid(self.phi_deg, self.theta_deg, indexing="ij")
        return theta.reshape(-1), phi.reshape(-1)

    def nearest_indices(self, theta_deg: float, phi_deg: float) -> tuple[int, int]:
        ti = int(np.argmin(np.abs(self.theta_deg - theta_deg)))
        wrapped = ((self.phi_deg - phi_deg + 180.0) % 360.0) - 180.0
        pi = int(np.argmin(np.abs(wrapped)))
        return pi, ti

    def response_at(self, theta_deg: float, phi_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Bilinearly interpolate the four embedded complex responses."""
        theta = float(np.clip(theta_deg, self.theta_deg[0], self.theta_deg[-1]))
        phi = float(((phi_deg + 180.0) % 360.0) - 180.0)
        ti1 = int(np.searchsorted(self.theta_deg, theta, side="right"))
        ti1 = min(max(ti1, 1), len(self.theta_deg) - 1)
        ti0 = ti1 - 1
        pi1 = int(np.searchsorted(self.phi_deg, phi, side="right"))
        pi1 = min(max(pi1, 1), len(self.phi_deg) - 1)
        pi0 = pi1 - 1
        at = (theta - self.theta_deg[ti0]) / (self.theta_deg[ti1] - self.theta_deg[ti0])
        ap = (phi - self.phi_deg[pi0]) / (self.phi_deg[pi1] - self.phi_deg[pi0])

        def interp(field: np.ndarray) -> np.ndarray:
            return (
                field[:, pi0, ti0] * (1 - ap) * (1 - at)
                + field[:, pi1, ti0] * ap * (1 - at)
                + field[:, pi0, ti1] * (1 - ap) * at
                + field[:, pi1, ti1] * ap * at
            )

        return interp(self.etheta), interp(self.ephi)

    def fields(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(weights, dtype=np.complex128)
        if weights.shape[-1] != 4:
            raise ValueError(f"expected four complex weights, got {weights.shape}")
        etheta = np.einsum("...n,npt->...pt", weights, self.etheta, optimize=True)
        ephi = np.einsum("...n,npt->...pt", weights, self.ephi, optimize=True)
        return etheta, ephi

    def power(self, weights: np.ndarray) -> np.ndarray:
        etheta, ephi = self.fields(weights)
        return np.abs(etheta) ** 2 + np.abs(ephi) ** 2

    def target_power(self, weights: np.ndarray, theta_deg: float, phi_deg: float) -> float:
        ht, hp = self.response_at(theta_deg, phi_deg)
        et = np.dot(ht, weights)
        ep = np.dot(hp, weights)
        return float(abs(et) ** 2 + abs(ep) ** 2)

    def metrics(
        self,
        weights: np.ndarray,
        target_theta_deg: float,
        target_phi_deg: float,
        exclusion_deg: float = 8.0,
    ) -> PatternMetrics:
        weights = np.asarray(weights, dtype=np.complex128)
        power = self.power(weights)
        target_power = self.target_power(weights, target_theta_deg, target_phi_deg)
        peak_flat = int(np.argmax(power))
        pi, ti = np.unravel_index(peak_flat, power.shape)
        peak_power = float(power[pi, ti])
        peak_theta = float(self.theta_deg[ti])
        peak_phi = float(self.phi_deg[pi])
        grid_theta, grid_phi = self.flat_theta_phi
        separation = angular_distance_deg(grid_theta, grid_phi, target_theta_deg, target_phi_deg)
        outside = separation > exclusion_deg
        outside_peak = float(np.max(power.reshape(-1)[outside]))
        ht, hp = self.response_at(target_theta_deg, target_phi_deg)
        et_power = float(abs(np.dot(ht, weights)) ** 2)
        ep_power = float(abs(np.dot(hp, weights)) ** 2)
        dominant = 10.0 * np.log10((max(et_power, ep_power) + 1e-30) / (min(et_power, ep_power) + 1e-30))
        return PatternMetrics(
            target_power=target_power,
            peak_power=peak_power,
            target_to_peak_db=10.0 * np.log10((target_power + 1e-30) / (peak_power + 1e-30)),
            peak_theta_deg=peak_theta,
            peak_phi_deg=peak_phi,
            pointing_error_deg=float(angular_distance_deg(peak_theta, peak_phi, target_theta_deg, target_phi_deg)),
            outside_peak_power=outside_peak,
            strong_peak_margin_db=10.0 * np.log10((target_power + 1e-30) / (outside_peak + 1e-30)),
            etheta_power=et_power,
            ephi_power=ep_power,
            dominant_polarization_ratio_db=float(dominant),
        )


def project_complex_disk(weights: np.ndarray, radius: float = 1.0) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.complex128)
    magnitude = np.abs(weights)
    scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-30))
    return weights * scale


def quantize_int8_disk(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest practical Gaussian integer followed by exact integer disk projection."""
    w = project_complex_disk(weights)
    codes = np.stack((np.rint(w.real * 127.0), np.rint(w.imag * 127.0)), axis=-1).astype(np.int16)
    flat = codes.reshape(-1, 2)
    for row in flat:
        if int(row[0]) ** 2 + int(row[1]) ** 2 <= 127 ** 2:
            continue
        angle = np.arctan2(float(row[1]), float(row[0]))
        seed = np.array([127.0 * np.cos(angle), 127.0 * np.sin(angle)])
        base = np.rint(seed).astype(np.int16)
        best = None
        best_error = np.inf
        original = row.astype(np.float64)
        for di in range(-2, 3):
            for dq in range(-2, 3):
                candidate = base + np.array([di, dq], dtype=np.int16)
                if int(candidate[0]) ** 2 + int(candidate[1]) ** 2 > 127 ** 2:
                    continue
                error = float(np.sum((candidate - original) ** 2))
                if error < best_error:
                    best, best_error = candidate, error
        if best is None:
            raise RuntimeError("failed to project an INT8 code to the integer disk")
        row[:] = best
    codes8 = codes.astype(np.int8)
    restored = codes8[..., 0].astype(np.float32) / 127.0 + 1j * codes8[..., 1].astype(np.float32) / 127.0
    return codes8, restored.astype(np.complex64)

