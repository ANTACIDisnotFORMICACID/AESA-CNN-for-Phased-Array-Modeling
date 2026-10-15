"""Versioned, inspectable storage shared by HFSS and future chamber measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ARRAY_COLS, ARRAY_ROWS, ArrayGeometry


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    source_type: str
    frequency_hz: float
    rows: int
    cols: int
    row_pitch_m: float
    col_pitch_m: float
    field_basis: str = "cartesian_xyz"
    complex_layout: str = "IQ"
    operator_axes: tuple[str, ...] = ("query", "xyz", "iq", "row", "col")
    direction_axes: tuple[str, ...] = ("query", "xyz")
    network_axes: tuple[str, ...] = (
        "frequency",
        "iq",
        "receive_row",
        "receive_col",
        "source_row",
        "source_col",
    )
    units: dict[str, str] | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def for_geometry(
        cls,
        geometry: ArrayGeometry,
        source_type: str,
        provenance: dict[str, Any] | None = None,
    ) -> "DatasetManifest":
        return cls(
            schema_version="v2.8.0",
            source_type=source_type,
            frequency_hz=geometry.frequency_hz,
            rows=geometry.rows,
            cols=geometry.cols,
            row_pitch_m=geometry.row_pitch_m,
            col_pitch_m=geometry.col_pitch_m,
            units={"frequency": "Hz", "distance": "m", "angle": "degree", "field": "V/m"},
            provenance=provenance,
        )


@dataclass(frozen=True)
class BasisDataset:
    """Per-source response dataset usable for HFSS or chamber measurements."""

    direction: np.ndarray
    operator_iq: np.ndarray
    theta_deg: np.ndarray | None
    phi_deg: np.ndarray | None
    manifest: DatasetManifest
    s_frequency_hz: np.ndarray | None = None
    s_parameters_iq: np.ndarray | None = None

    def __post_init__(self) -> None:
        query_count = self.direction.shape[0]
        if self.direction.shape != (query_count, 3):
            raise ValueError("direction must have shape [Q,3]")
        if self.operator_iq.shape != (query_count, 3, 2, ARRAY_ROWS, ARRAY_COLS):
            raise ValueError("operator_iq must have shape [Q,3,2,4,4]")
        if not np.allclose(np.linalg.norm(self.direction, axis=-1), 1.0, atol=1e-6):
            raise ValueError("direction vectors must be normalized")
        for name, value in (("theta_deg", self.theta_deg), ("phi_deg", self.phi_deg)):
            if value is not None and value.shape != (query_count,):
                raise ValueError(f"{name} must have shape [Q]")
        if (self.s_frequency_hz is None) != (self.s_parameters_iq is None):
            raise ValueError("S-parameter frequency and values must be supplied together")
        if self.s_frequency_hz is not None and self.s_parameters_iq is not None:
            expected = (self.s_frequency_hz.size, 2, 4, 4, 4, 4)
            if self.s_parameters_iq.shape != expected:
                raise ValueError("s_parameters_iq must have shape [F,2,4,4,4,4]")


def save_basis_dataset(dataset: BasisDataset, destination: str | Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "direction.npy", dataset.direction)
    np.save(destination / "operator_iq.npy", dataset.operator_iq)
    if dataset.theta_deg is not None:
        np.save(destination / "theta_deg.npy", dataset.theta_deg)
    if dataset.phi_deg is not None:
        np.save(destination / "phi_deg.npy", dataset.phi_deg)
    if dataset.s_frequency_hz is not None and dataset.s_parameters_iq is not None:
        np.save(destination / "s_frequency_hz.npy", dataset.s_frequency_hz)
        np.save(destination / "s_parameters_iq.npy", dataset.s_parameters_iq)
    (destination / "manifest.json").write_text(
        json.dumps(asdict(dataset.manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_basis_dataset(source: str | Path) -> BasisDataset:
    source = Path(source)
    manifest = DatasetManifest(**json.loads((source / "manifest.json").read_text(encoding="utf-8")))
    theta_path = source / "theta_deg.npy"
    phi_path = source / "phi_deg.npy"
    s_frequency_path = source / "s_frequency_hz.npy"
    s_parameters_path = source / "s_parameters_iq.npy"
    return BasisDataset(
        direction=np.load(source / "direction.npy", allow_pickle=False),
        operator_iq=np.load(source / "operator_iq.npy", allow_pickle=False),
        theta_deg=np.load(theta_path, allow_pickle=False) if theta_path.exists() else None,
        phi_deg=np.load(phi_path, allow_pickle=False) if phi_path.exists() else None,
        manifest=manifest,
        s_frequency_hz=np.load(s_frequency_path, allow_pickle=False) if s_frequency_path.exists() else None,
        s_parameters_iq=np.load(s_parameters_path, allow_pickle=False) if s_parameters_path.exists() else None,
    )
