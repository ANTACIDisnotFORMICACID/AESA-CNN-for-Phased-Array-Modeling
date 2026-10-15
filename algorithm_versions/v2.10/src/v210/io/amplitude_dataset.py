"""Amplitude-only measurement contract shared by HFSS feasibility and hardware."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


AMPLITUDE_ONLY_TRAINING_POLICY = "v210-amplitude-only-positive-hemisphere-1"
_AMPLITUDE_ARRAY_FILES = {
    "direction.npy",
    "command_iq.npy",
    "amplitude.npy",
    "command_family.npy",
    "theta_deg.npy",
    "phi_deg.npy",
}


@dataclass(frozen=True)
class AmplitudeManifest:
    """Physical semantics required before amplitude observations are accepted."""

    schema_version: str
    source_type: str
    frequency_hz: float
    receiver_channel: str
    amplitude_unit: str
    reference_distance_m: float
    excitation_reference: str
    inactive_port_termination: str
    calibration_reference_plane: str
    array_id: str
    session_id: str
    noise_floor_amplitude: float = 0.0
    phase_available: bool = False
    rows: int = 4
    cols: int = 4
    command_axes: tuple[str, ...] = ("command", "iq", "row", "col")
    amplitude_axes: tuple[str, ...] = ("command", "query")
    direction_axes: tuple[str, ...] = ("query", "xyz")
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "v2.8-amplitude-1":
            raise ValueError("unsupported amplitude dataset schema")
        if self.source_type not in {"hfss_amplitude", "anechoic_chamber_amplitude", "synthetic_fixture"}:
            raise ValueError("invalid amplitude dataset source_type")
        if self.phase_available:
            raise ValueError("amplitude-only training data must not expose measured phase")
        if self.frequency_hz <= 0 or self.reference_distance_m <= 0:
            raise ValueError("frequency and reference distance must be positive")
        if self.noise_floor_amplitude < 0:
            raise ValueError("noise_floor_amplitude must be non-negative")
        required_text = {
            "receiver_channel": self.receiver_channel,
            "amplitude_unit": self.amplitude_unit,
            "excitation_reference": self.excitation_reference,
            "inactive_port_termination": self.inactive_port_termination,
            "calibration_reference_plane": self.calibration_reference_plane,
            "array_id": self.array_id,
            "session_id": self.session_id,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if missing:
            raise ValueError(f"amplitude manifest fields must be explicit; missing={missing}")
        if (self.rows, self.cols) != (4, 4):
            raise ValueError("the V2.8-compatible amplitude schema requires a 4x4 array")


@dataclass(frozen=True)
class AmplitudeDataset:
    """Known 4x4 commands and scalar amplitudes measured over a spherical grid."""

    direction: np.ndarray
    command_iq: np.ndarray
    amplitude: np.ndarray
    command_family: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    manifest: AmplitudeManifest

    def __post_init__(self) -> None:
        query_count = self.direction.shape[0]
        command_count = self.command_iq.shape[0]
        if self.direction.shape != (query_count, 3):
            raise ValueError("direction must have shape [Q,3]")
        if self.command_iq.shape != (command_count, 2, 4, 4):
            raise ValueError("command_iq must have shape [N,2,4,4]")
        if self.amplitude.shape != (command_count, query_count):
            raise ValueError("amplitude must have shape [N,Q]")
        if self.command_family.shape != (command_count,):
            raise ValueError("command_family must have shape [N]")
        if self.theta_deg.shape != (query_count,) or self.phi_deg.shape != (query_count,):
            raise ValueError("theta_deg and phi_deg must have shape [Q]")
        if not np.all(np.isfinite(self.amplitude)) or np.any(self.amplitude < 0):
            raise ValueError("amplitude observations must be finite and non-negative")
        if np.iscomplexobj(self.amplitude):
            raise ValueError("training labels must be real scalar amplitudes, not complex fields")
        if np.iscomplexobj(self.command_iq):
            raise ValueError("known commands must use two real IQ channels, not a complex-valued array")
        if not np.all(np.isfinite(self.command_iq)):
            raise ValueError("command values must be finite")
        if not np.allclose(np.linalg.norm(self.direction, axis=-1), 1.0, atol=1e-6):
            raise ValueError("direction vectors must be normalized")
        command_power = np.sum(self.command_iq**2, axis=(1, 2, 3))
        if np.any(command_power <= 0):
            raise ValueError("every command must have non-zero power")


def save_amplitude_dataset(dataset: AmplitudeDataset, destination: str | Path) -> None:
    """Save no measured phase, complex FFD operator or S-parameter array."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "direction.npy", dataset.direction)
    np.save(destination / "command_iq.npy", dataset.command_iq)
    np.save(destination / "amplitude.npy", dataset.amplitude)
    np.save(destination / "command_family.npy", dataset.command_family.astype("U32"))
    np.save(destination / "theta_deg.npy", dataset.theta_deg)
    np.save(destination / "phi_deg.npy", dataset.phi_deg)
    (destination / "manifest.json").write_text(
        json.dumps(asdict(dataset.manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_amplitude_dataset(source: str | Path) -> AmplitudeDataset:
    source = Path(source)
    unexpected_arrays = sorted(
        path.name
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".npy", ".npz"}
        and path.name not in _AMPLITUDE_ARRAY_FILES
    )
    forbidden = unexpected_arrays
    if forbidden:
        raise ValueError(
            "amplitude-only training directory contains forbidden or unexpected array files: "
            f"{forbidden}"
        )
    manifest = AmplitudeManifest(**json.loads((source / "manifest.json").read_text(encoding="utf-8")))
    return AmplitudeDataset(
        direction=np.load(source / "direction.npy", allow_pickle=False),
        command_iq=np.load(source / "command_iq.npy", allow_pickle=False),
        amplitude=np.load(source / "amplitude.npy", allow_pickle=False),
        command_family=np.load(source / "command_family.npy", allow_pickle=False),
        theta_deg=np.load(source / "theta_deg.npy", allow_pickle=False),
        phi_deg=np.load(source / "phi_deg.npy", allow_pickle=False),
        manifest=manifest,
    )


def assert_amplitude_only_training_contract(dataset: AmplitudeDataset) -> None:
    """Reject non-amplitude supervision and directions outside the physical front hemisphere."""

    if dataset.manifest.phase_available:
        raise ValueError("measured phase is forbidden by the amplitude-only training policy")
    provenance = dataset.manifest.provenance or {}
    forbidden_flags = (
        "measured_phase_available",
        "measured_phase_used",
        "complex_field_used",
        "operator_iq_used",
        "s_parameters_used",
    )
    enabled = sorted(name for name in forbidden_flags if provenance.get(name) is True)
    if enabled:
        raise ValueError(f"amplitude-only training provenance enables forbidden inputs: {enabled}")
    if not np.all(dataset.direction[:, 0] > 0.0):
        count = int(np.count_nonzero(dataset.direction[:, 0] <= 0.0))
        raise ValueError(
            "training directions must satisfy the strict front-hemisphere contract q_x > 0; "
            f"found {count} horizon/reverse directions"
        )
