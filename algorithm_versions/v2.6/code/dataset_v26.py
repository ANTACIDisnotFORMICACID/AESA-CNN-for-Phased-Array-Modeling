"""Leakage-controlled state profiles and resumable HFSS-oracle teacher data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from calibration_v26 import build_d_optimal_design, simulate_measurements
from controller_v26 import BeamformingControllerV26
from hfss_physics_v26 import HfssPatternData
from oracle_v26 import OracleConfig, PhysicalOracle


V26_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = V26_ROOT / "output"


@dataclass(frozen=True)
class StateProfile:
    split: str
    state_index: int
    seed: int
    calibration_iq: list[list[float]]
    health_mask: list[float]
    amplitude_rms_fraction: float
    phase_rms_deg: float
    measurement_snr_db: float
    failed_port: int | None

    @property
    def calibration(self) -> np.ndarray:
        array = np.asarray(self.calibration_iq, dtype=np.float64)
        return array[:, 0] + 1j * array[:, 1]


SPLIT_SPEC = {
    "train": {"count": 96, "seed": 26001},
    "validation": {"count": 24, "seed": 26002},
    "test": {"count": 48, "seed": 26003},
}


def target_block_owner(theta_block: int, phi_block: int) -> str:
    """Assign an entire 7.5-degree block to one split; never split adjacent samples randomly."""
    code = (theta_block + 2 * phi_block) % 4
    if code in (0, 1):
        return "train"
    return "validation" if code == 2 else "test"


def _zero_mean_rms(rng: np.random.Generator, rms: float, count: int = 4) -> np.ndarray:
    value = rng.normal(size=count)
    value -= np.mean(value)
    actual = float(np.sqrt(np.mean(value**2)))
    return value * (rms / max(actual, 1e-12))


def generate_state_profiles(split: str) -> list[StateProfile]:
    spec = SPLIT_SPEC[split]
    root_rng = np.random.default_rng(spec["seed"])
    profiles = []
    for index in range(spec["count"]):
        state_seed = int(root_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(state_seed)
        amplitude_rms = float(rng.uniform(0.0, 0.13))
        phase_rms = float(rng.uniform(0.0, 22.0))
        amplitude = np.maximum(0.05, 1.0 + _zero_mean_rms(rng, amplitude_rms))
        phase = np.deg2rad(_zero_mean_rms(rng, phase_rms))
        calibration = amplitude * np.exp(1j * phase)
        health = np.ones(4, dtype=np.float64)
        failed_port = None
        if rng.random() < 0.10:
            failed_port = int(rng.integers(0, 4))
            health[failed_port] = 0.0
            calibration[failed_port] = 0.0
        profiles.append(
            StateProfile(
                split=split,
                state_index=index,
                seed=state_seed,
                calibration_iq=np.stack((calibration.real, calibration.imag), axis=-1).tolist(),
                health_mask=health.tolist(),
                amplitude_rms_fraction=amplitude_rms,
                phase_rms_deg=phase_rms,
                measurement_snr_db=float(rng.uniform(25.0, 40.0)),
                failed_port=failed_port,
            )
        )
    return profiles


def target_angles_for_state(profile: StateProfile, count: int = 64) -> np.ndarray:
    """Stratified targets inside split-owned 2-D blocks with internal guard margins."""
    if count != 64:
        raise ValueError("the release contract uses exactly 64 targets per state")
    rng = np.random.default_rng(profile.seed ^ 0x2606)
    theta_edges = np.linspace(60.0, 120.0, 9)
    phi_edges = np.linspace(-30.0, 30.0, 9)
    owned = [(ti, pi) for ti in range(8) for pi in range(8) if target_block_owner(ti, pi) == profile.split]
    repeats = count // len(owned)
    remainder = count % len(owned)
    targets = []
    for block_index, (ti, pi) in enumerate(owned):
        samples = repeats + int(block_index < remainder)
        theta_guard = 0.10 * (theta_edges[ti + 1] - theta_edges[ti])
        phi_guard = 0.10 * (phi_edges[pi + 1] - phi_edges[pi])
        for _ in range(samples):
            theta = rng.uniform(theta_edges[ti] + theta_guard, theta_edges[ti + 1] - theta_guard)
            phi = rng.uniform(phi_edges[pi] + phi_guard, phi_edges[pi + 1] - phi_guard)
            targets.append((theta, phi))
    rng.shuffle(targets)
    return np.asarray(targets, dtype=np.float32)


def write_profile_manifest(path: Path = OUTPUT_ROOT / "data" / "state_profiles.json") -> dict:
    payload = {
        "schema_version": 1,
        "split_policy": {
            "method": "whole 8x8 angular macroblocks assigned to one split; targets use a 10 percent internal guard margin",
            "block_owner_formula": "code=(theta_block+2*phi_block)%4; train=0/1, validation=2, test=3",
            "domain_theta_deg": [60.0, 120.0],
            "domain_phi_deg": [-30.0, 30.0],
            "warning": "validation and test blocks are not used for loss weights, checkpoint selection outside validation, or feasibility thresholds",
        },
        "profiles": {split: [asdict(p) for p in generate_state_profiles(split)] for split in SPLIT_SPEC},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def state_pattern(data: HfssPatternData, profile: StateProfile) -> HfssPatternData:
    transfer = profile.calibration * np.asarray(profile.health_mask)
    return HfssPatternData(
        theta_deg=data.theta_deg,
        phi_deg=data.phi_deg,
        etheta=(data.etheta * transfer[:, None, None]).astype(np.complex64),
        ephi=(data.ephi * transfer[:, None, None]).astype(np.complex64),
        s=data.s,
        port_names=data.port_names,
        y_mm=data.y_mm,
    )


def build_teacher_state(
    profile: StateProfile,
    data: HfssPatternData,
    oracle_config: OracleConfig,
    destination: Path,
) -> Path:
    """Build one resumable 64-sample state shard. Existing complete shards are immutable."""
    if destination.exists():
        with np.load(destination) as existing:
            if existing["deployment_x"].shape == (64, 15, 1, 4):
                return destination
    design = build_d_optimal_design(data)
    measurement = simulate_measurements(
        design,
        profile.calibration,
        snr_db=profile.measurement_snr_db,
        global_phase_rad=np.random.default_rng(profile.seed).uniform(-np.pi, np.pi),
        seed=profile.seed,
    )
    controller = BeamformingControllerV26(data, calibration_design=design)
    estimated_iq, confidence = controller.calibrate(measurement, np.asarray(profile.health_mask))
    transformed = state_pattern(data, profile)
    oracle = PhysicalOracle(transformed, oracle_config)
    targets = target_angles_for_state(profile)
    inputs, initials, continuous, integer, codes, oracle_metrics = [], [], [], [], [], []
    for theta, phi in targets:
        x, initial = controller.deployment_input(
            (float(theta), float(phi)), estimated_iq[0], confidence[0], np.asarray(profile.health_mask)
        )
        result = oracle.solve(float(theta), float(phi))
        inputs.append(x)
        initials.append(initial)
        continuous.append(result.continuous_weights)
        integer.append(result.int8_weights)
        codes.append(result.int8_codes)
        oracle_metrics.append(
            [
                result.continuous_metrics.target_to_peak_db,
                result.continuous_metrics.pointing_error_deg,
                result.continuous_metrics.strong_peak_margin_db,
                float(result.continuous_metrics.feasible),
            ]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        deployment_x=np.asarray(inputs, dtype=np.float32),
        initial_weights=np.asarray(initials, dtype=np.complex64),
        teacher_weights=np.asarray(continuous, dtype=np.complex64),
        teacher_int8_weights=np.asarray(integer, dtype=np.complex64),
        teacher_int8_codes=np.asarray(codes, dtype=np.int8),
        target_hfss_deg=targets,
        calibration=np.repeat(profile.calibration[None, :], 64, axis=0).astype(np.complex64),
        health_mask=np.repeat(np.asarray(profile.health_mask, dtype=np.float32)[None, :], 64, axis=0),
        calibration_confidence=np.repeat(confidence.astype(np.float32), 64, axis=0),
        oracle_metrics=np.asarray(oracle_metrics, dtype=np.float32),
    )
    return destination
