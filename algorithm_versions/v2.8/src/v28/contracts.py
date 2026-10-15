"""Validated tensor contracts shared by data, physics, models and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np
import torch

ARRAY_ROWS = 4
ARRAY_COLS = 4
FIELD_COMPONENTS = 3
IQ_COMPONENTS = 2


class CoordinateType(str, Enum):
    DIRECTION = "direction"
    CARTESIAN_TARGET = "cartesian_target"


@dataclass(frozen=True)
class ArrayGeometry:
    """Physical geometry; row and column remain distinct axes everywhere."""

    rows: int = ARRAY_ROWS
    cols: int = ARRAY_COLS
    frequency_hz: float = 47.0e9
    row_pitch_m: float = 0.015
    col_pitch_m: float = 0.0068
    rotation_algorithm_to_hfss: np.ndarray = field(default_factory=lambda: np.eye(3))
    port_names: tuple[str, ...] = tuple(f"{index}:1" for index in range(1, 17))

    def __post_init__(self) -> None:
        if (self.rows, self.cols) != (ARRAY_ROWS, ARRAY_COLS):
            raise ValueError("v2.8 requires an explicit 4x4 array grid")
        if self.frequency_hz <= 0 or self.row_pitch_m <= 0 or self.col_pitch_m <= 0:
            raise ValueError("frequency and pitches must be positive")
        rotation = np.asarray(self.rotation_algorithm_to_hfss, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_algorithm_to_hfss must have shape [3,3]")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("coordinate rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("coordinate rotation must be a proper rotation")
        if len(self.port_names) != ARRAY_ROWS * ARRAY_COLS:
            raise ValueError("exactly 16 row-major port names are required")
        object.__setattr__(self, "rotation_algorithm_to_hfss", rotation)

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / self.frequency_hz

    @property
    def positions_m(self) -> np.ndarray:
        """Return positions as [row, col, xyz], never as a 16-element axis."""
        row_z = (np.arange(self.rows) - (self.rows - 1) / 2.0) * self.row_pitch_m
        col_y = (np.arange(self.cols) - (self.cols - 1) / 2.0) * self.col_pitch_m
        positions = np.zeros((self.rows, self.cols, 3), dtype=np.float64)
        positions[:, :, 1] = col_y[None, :]
        positions[:, :, 2] = row_z[:, None]
        return positions

    @property
    def aperture_diameter_m(self) -> float:
        positions = self.positions_m
        pairwise = positions[:, :, None, None, :] - positions[None, None, :, :, :]
        return float(np.linalg.norm(pairwise, axis=-1).max())

    @property
    def fraunhofer_distance_m(self) -> float:
        return 2.0 * self.aperture_diameter_m**2 / self.wavelength_m


@dataclass(frozen=True)
class QueryBatch:
    """Far-field queries in algorithm coordinates, shaped [batch, query, xyz]."""

    unit_direction: torch.Tensor
    range_m: torch.Tensor | None = None
    target_xyz_m: torch.Tensor | None = None
    coordinate_type: CoordinateType = CoordinateType.DIRECTION

    def __post_init__(self) -> None:
        if self.unit_direction.ndim != 3 or self.unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        norms = torch.linalg.vector_norm(self.unit_direction, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("unit_direction vectors must be normalized")
        expected = self.unit_direction.shape[:2]
        if self.range_m is not None and self.range_m.shape != expected:
            raise ValueError("range_m must have shape [B,Q]")
        if self.target_xyz_m is not None and self.target_xyz_m.shape != (*expected, 3):
            raise ValueError("target_xyz_m must have shape [B,Q,3]")


@dataclass(frozen=True)
class OperatorPrediction:
    """Effective per-element field operator [B,Q,xyz,IQ,row,col]."""

    effective_iq: torch.Tensor
    terms_iq: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_tail = (FIELD_COMPONENTS, IQ_COMPONENTS, ARRAY_ROWS, ARRAY_COLS)
        if self.effective_iq.ndim != 6 or self.effective_iq.shape[-4:] != expected_tail:
            raise ValueError("effective_iq must have shape [B,Q,3,2,4,4]")
        for name, value in self.terms_iq.items():
            if value.shape != self.effective_iq.shape:
                raise ValueError(f"operator term {name!r} has an incompatible shape")


@dataclass(frozen=True)
class FieldPrediction:
    """Predicted Cartesian complex field [B,Q,xyz,IQ] and received power."""

    field_iq: torch.Tensor
    power: torch.Tensor
    operator: OperatorPrediction | None = None

    def __post_init__(self) -> None:
        if self.field_iq.ndim != 4 or self.field_iq.shape[-2:] != (3, 2):
            raise ValueError("field_iq must have shape [B,Q,3,2]")
        if self.power.shape != self.field_iq.shape[:2]:
            raise ValueError("power must have shape [B,Q]")

