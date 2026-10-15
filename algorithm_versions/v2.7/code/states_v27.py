"""Deterministic 19 one-factor and 64 Latin-hypercube audit states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from v27_compat import V27_ROOT


MIXED_SEED = 27001
TARGET_SEED = 27002


def _zero_mean_rms(rng: np.random.Generator, rms: float, count: int = 4) -> np.ndarray:
    if rms == 0.0:
        return np.zeros(count, dtype=np.float64)
    value = rng.normal(size=count)
    value -= np.mean(value)
    actual = float(np.sqrt(np.mean(value**2)))
    return value * (float(rms) / max(actual, 1e-15))


@dataclass(frozen=True)
class AuditState:
    state_id: str
    state_class: str
    state_index: int
    seed: int
    amplitude_rms_fraction: float
    phase_rms_deg: float
    measurement_snr_db: float
    failed_port: int | None
    global_phase_deg: float
    calibration_iq: list[list[float]]
    health_mask: list[float]

    @property
    def calibration(self) -> np.ndarray:
        iq = np.asarray(self.calibration_iq, dtype=np.float64)
        return iq[:, 0] + 1j * iq[:, 1]

    @property
    def health(self) -> np.ndarray:
        return np.asarray(self.health_mask, dtype=np.float64)


def _make_state(
    state_id: str,
    state_class: str,
    index: int,
    seed: int,
    amplitude_rms: float = 0.0,
    phase_rms_deg: float = 0.0,
    snr_db: float = 40.0,
    failed_port: int | None = None,
    global_phase_deg: float = 0.0,
) -> AuditState:
    rng = np.random.default_rng(seed)
    amplitude = np.maximum(0.05, 1.0 + _zero_mean_rms(rng, amplitude_rms))
    phase = np.deg2rad(_zero_mean_rms(rng, phase_rms_deg))
    calibration = amplitude * np.exp(1j * phase)
    health = np.ones(4, dtype=np.float64)
    if failed_port is not None:
        if failed_port not in range(4):
            raise ValueError("failed_port must be a zero-based port index")
        health[failed_port] = 0.0
        calibration[failed_port] = 0.0
    return AuditState(
        state_id=state_id,
        state_class=state_class,
        state_index=index,
        seed=seed,
        amplitude_rms_fraction=float(amplitude_rms),
        phase_rms_deg=float(phase_rms_deg),
        measurement_snr_db=float(snr_db),
        failed_port=failed_port,
        global_phase_deg=float(global_phase_deg),
        calibration_iq=np.stack((calibration.real, calibration.imag), axis=-1).tolist(),
        health_mask=health.tolist(),
    )


def one_factor_states() -> list[AuditState]:
    specs: list[tuple[str, str, float, float, float, int | None, float]] = [
        ("baseline", "baseline", 0.0, 0.0, 40.0, None, 0.0)
    ]
    specs += [(f"amp_rms_{int(v*100):02d}pct", "amplitude", v, 0.0, 40.0, None, 0.0) for v in (0.03, 0.06, 0.09, 0.13)]
    specs += [(f"phase_rms_{int(v):02d}deg", "phase", 0.0, v, 40.0, None, 0.0) for v in (5.0, 10.0, 15.0, 22.0)]
    specs += [(f"snr_{int(v)}db", "snr", 0.0, 0.0, v, None, 0.0) for v in (25.0, 30.0, 35.0)]
    specs += [(f"failed_port_{port+1}", "failed_port", 0.0, 0.0, 40.0, port, 0.0) for port in range(4)]
    specs += [(f"global_phase_{int(v):+d}deg", "global_phase", 0.0, 0.0, 40.0, None, v) for v in (90.0, 180.0, -90.0)]
    result = []
    for index, (state_id, cls, amp, phase, snr, failed, global_phase) in enumerate(specs):
        result.append(_make_state(state_id, cls, index, 27010 + index, amp, phase, snr, failed, global_phase))
    if len(result) != 19:
        raise RuntimeError(f"one-factor contract requires 19 states, built {len(result)}")
    return result


def mixed_latin_hypercube_states() -> list[AuditState]:
    # Columns: amplitude, phase, SNR, global phase, failure selector, failed-port selector.
    sample = qmc.LatinHypercube(d=6, seed=MIXED_SEED, scramble=True).random(64)
    result = []
    for index, row in enumerate(sample):
        failed = int(np.floor(row[5] * 4.0)) if row[4] < 0.10 else None
        result.append(
            _make_state(
                state_id=f"mixed_{index:02d}",
                state_class="mixed",
                index=19 + index,
                seed=MIXED_SEED * 100 + index,
                amplitude_rms=float(row[0] * 0.13),
                phase_rms_deg=float(row[1] * 22.0),
                snr_db=float(25.0 + row[2] * 15.0),
                failed_port=failed,
                global_phase_deg=float(-180.0 + row[3] * 360.0),
            )
        )
    return result


def all_audit_states() -> list[AuditState]:
    states = one_factor_states() + mixed_latin_hypercube_states()
    if len(states) != 83 or len({state.state_id for state in states}) != 83:
        raise RuntimeError("audit-state contract requires 83 unique states")
    return states


def stratified_targets(state_index: int, count: int = 256) -> np.ndarray:
    """A jittered 16x16 sample spanning the complete target domain."""
    if count != 256:
        raise ValueError("the V2.7 audit contract uses exactly 256 targets per state")
    rng = np.random.default_rng(TARGET_SEED + int(state_index) * 1009)
    theta_edges = np.linspace(60.0, 120.0, 17)
    phi_edges = np.linspace(-30.0, 30.0, 17)
    targets = []
    for ti in range(16):
        for pi in range(16):
            theta = rng.uniform(theta_edges[ti], theta_edges[ti + 1])
            phi = rng.uniform(phi_edges[pi], phi_edges[pi + 1])
            targets.append((theta, phi))
    targets = np.asarray(targets, dtype=np.float32)
    rng.shuffle(targets)
    return targets


def write_state_manifest(path: Path | None = None) -> dict:
    path = path or (V27_ROOT / "output" / "state_manifest.json")
    states = all_audit_states()
    payload = {
        "schema_version": 1,
        "seeds": {"mixed_latin_hypercube": MIXED_SEED, "stratified_targets": TARGET_SEED},
        "counts": {"one_factor": 19, "mixed": 64, "total": 83, "targets_per_state": 256, "cases": 21248},
        "state_policy": "fixed HFSS embedded patterns; only feed-chain calibration, measurement noise, global measurement phase and health change",
        "states": [
            {
                **asdict(state),
                "targets_sha256": hashlib.sha256(stratified_targets(state.state_index).tobytes()).hexdigest().upper(),
            }
            for state in states
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(write_state_manifest()["counts"], ensure_ascii=False, indent=2))
