"""Parse 16 embedded-element HFSS FFD files into a canonical operator tensor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from ..contracts import ARRAY_COLS, ARRAY_ROWS, ArrayGeometry
from ..coordinates import hfss_angles_to_direction, spherical_basis


@dataclass(frozen=True)
class HfssFfdExport:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    direction_algorithm: np.ndarray
    operator_iq: np.ndarray
    source_files: tuple[str, ...]

    def __post_init__(self) -> None:
        query_count = self.direction_algorithm.shape[0]
        if self.theta_deg.shape != (query_count,) or self.phi_deg.shape != (query_count,):
            raise ValueError("angle arrays must share the query dimension")
        if self.operator_iq.shape != (query_count, 3, 2, ARRAY_ROWS, ARRAY_COLS):
            raise ValueError("operator_iq must have shape [Q,3,2,4,4]")


def _numeric_tokens(line: str) -> list[float]:
    return [
        float(value.replace("D", "E").replace("d", "e"))
        for value in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?", line)
    ]


def _axis_from_header(values: list[float], name: str) -> np.ndarray:
    if len(values) < 3:
        raise ValueError(f"{name} header requires start, stop and sample count")
    start, stop, raw_count = values[:3]
    count = int(round(raw_count))
    if count <= 0 or not np.isclose(raw_count, count):
        raise ValueError(f"invalid {name} sample count {raw_count}")
    return np.linspace(start, stop, count, dtype=np.float64)


def parse_ffd(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return theta grid, phi grid, Etheta and Ephi complex samples.

    HFSS FFD rows are expected as Re(Etheta), Im(Etheta), Re(Ephi), Im(Ephi).
    HFSS holds theta constant while phi is swept, then advances theta.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    numeric = [(index, _numeric_tokens(line)) for index, line in enumerate(lines)]
    headers = [(index, values) for index, values in numeric if len(values) == 3]
    if len(headers) < 2:
        raise ValueError(f"{source}: could not find theta/phi grid headers")
    theta = _axis_from_header(headers[0][1], "theta")
    phi = _axis_from_header(headers[1][1], "phi")
    data_start = headers[1][0] + 1
    rows = [values[:4] for index, values in numeric if index >= data_start and len(values) >= 4]
    expected = theta.size * phi.size
    if len(rows) != expected:
        raise ValueError(f"{source}: expected {expected} field rows, found {len(rows)}")
    data = np.asarray(rows, dtype=np.float64)
    e_theta = data[:, 0] + 1j * data[:, 1]
    e_phi = data[:, 2] + 1j * data[:, 3]
    theta_grid = np.repeat(theta, phi.size)
    phi_grid = np.tile(phi, theta.size)
    return theta_grid, phi_grid, e_theta, e_phi


def load_hfss_ffd_basis(
    port_to_file: Mapping[str, str | Path],
    geometry: ArrayGeometry | None = None,
) -> HfssFfdExport:
    """Load all 16 row-major source responses without inferring filenames."""
    geometry = geometry or ArrayGeometry()
    missing = [name for name in geometry.port_names if name not in port_to_file]
    extra = [name for name in port_to_file if name not in geometry.port_names]
    if missing or extra:
        raise ValueError(f"port mapping mismatch; missing={missing}, extra={extra}")

    reference_angles: tuple[np.ndarray, np.ndarray] | None = None
    operator_complex: np.ndarray | None = None
    files: list[str] = []
    rotation = geometry.rotation_algorithm_to_hfss

    for port_index, port_name in enumerate(geometry.port_names):
        source = Path(port_to_file[port_name]).resolve()
        theta, phi, e_theta, e_phi = parse_ffd(source)
        if reference_angles is None:
            reference_angles = (theta, phi)
            operator_complex = np.zeros(
                (theta.size, 3, ARRAY_ROWS, ARRAY_COLS),
                dtype=np.complex128,
            )
        elif not (np.array_equal(theta, reference_angles[0]) and np.array_equal(phi, reference_angles[1])):
            raise ValueError(f"{source}: angular grid differs from the first port export")

        e_theta_basis, e_phi_basis = spherical_basis(theta, phi)
        field_hfss = e_theta[..., None] * e_theta_basis + e_phi[..., None] * e_phi_basis
        field_algorithm = field_hfss @ rotation
        row, col = divmod(port_index, ARRAY_COLS)
        operator_complex[:, :, row, col] = field_algorithm
        files.append(str(source))

    if reference_angles is None or operator_complex is None:
        raise ValueError("no FFD files were supplied")
    theta, phi = reference_angles
    direction_hfss = hfss_angles_to_direction(theta, phi)
    direction_algorithm = direction_hfss @ rotation
    operator_iq = np.stack((operator_complex.real, operator_complex.imag), axis=2)
    return HfssFfdExport(theta, phi, direction_algorithm, operator_iq, tuple(files))
