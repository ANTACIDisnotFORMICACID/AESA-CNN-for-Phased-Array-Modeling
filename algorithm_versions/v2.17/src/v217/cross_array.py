"""Canonical contracts and parsers for reusable cross-array HFSS data.

The canonical layer is deliberately independent of the V2.12/V2.16 4x4
contracts.  It stores absolute complex fields in float64/complex128 and only
derives lower precision training views later.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np


LIGHT_SPEED_M_S = 299_792_458.0


@dataclass(frozen=True)
class ArraySpec:
    array_id: str
    rows: int
    cols: int
    row_pitch_m: float
    col_pitch_m: float
    frequency_hz: float = 47.0e9
    coordinate_rotation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.array_id or self.rows <= 0 or self.cols <= 0:
            raise ValueError("array_id and positive rows/cols are required")
        if self.row_pitch_m <= 0 or self.col_pitch_m <= 0 or self.frequency_hz <= 0:
            raise ValueError("pitches and frequency must be positive")
        rotation = (
            np.eye(3, dtype=np.float64)
            if self.coordinate_rotation is None
            else np.asarray(self.coordinate_rotation, dtype=np.float64)
        )
        if rotation.shape != (3, 3):
            raise ValueError("coordinate_rotation must have shape [3,3]")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("coordinate_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("coordinate_rotation must be a proper rotation")
        object.__setattr__(self, "coordinate_rotation", rotation)

    @property
    def element_count(self) -> int:
        return self.rows * self.cols

    @property
    def wavelength_m(self) -> float:
        return LIGHT_SPEED_M_S / self.frequency_hz

    @property
    def positions_m(self) -> np.ndarray:
        row_z = (np.arange(self.rows) - (self.rows - 1) / 2.0) * self.row_pitch_m
        col_y = (np.arange(self.cols) - (self.cols - 1) / 2.0) * self.col_pitch_m
        positions = np.zeros((self.rows, self.cols, 3), dtype=np.float64)
        positions[:, :, 1] = col_y[None, :]
        positions[:, :, 2] = row_z[:, None]
        return positions.reshape(-1, 3)

    @property
    def port_names(self) -> tuple[str, ...]:
        return tuple(f"{index}:1" for index in range(1, self.element_count + 1))

    def to_json(self) -> dict[str, object]:
        return {
            "array_id": self.array_id,
            "rows": self.rows,
            "cols": self.cols,
            "row_pitch_m": self.row_pitch_m,
            "col_pitch_m": self.col_pitch_m,
            "frequency_hz": self.frequency_hz,
            "coordinate_rotation": self.coordinate_rotation.tolist(),
        }


@dataclass(frozen=True)
class FfdGrid:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    frequency_hz: float


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _numbers(line: str) -> list[float]:
    return [
        float(token.replace("D", "E").replace("d", "e"))
        for token in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?", line)
    ]


def _axis(header: list[float], name: str) -> np.ndarray:
    if len(header) != 3:
        raise ValueError(f"{name} header must contain start, stop and count")
    start, stop, raw_count = header
    count = int(round(raw_count))
    if count <= 0 or not np.isclose(raw_count, count):
        raise ValueError(f"invalid {name} count: {raw_count}")
    return np.linspace(start, stop, count, dtype=np.float64)


def parse_ffd(path: str | Path) -> FfdGrid:
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    numeric = [(index, _numbers(line)) for index, line in enumerate(lines)]
    headers = [(index, values) for index, values in numeric if len(values) == 3]
    if len(headers) < 2:
        raise ValueError(f"{source}: missing theta/phi headers")
    theta_axis = _axis(headers[0][1], "theta")
    phi_axis = _axis(headers[1][1], "phi")
    start = headers[1][0] + 1
    rows = [values[:4] for index, values in numeric if index >= start and len(values) >= 4]
    expected = len(theta_axis) * len(phi_axis)
    if len(rows) != expected:
        raise ValueError(f"{source}: expected {expected} rows, found {len(rows)}")
    values = np.asarray(rows, dtype=np.float64)
    frequency_candidates = [
        numbers[0]
        for line, numbers in ((line.strip().lower(), _numbers(line)) for line in lines)
        if line.startswith("frequency ") and numbers
    ]
    if len(frequency_candidates) != 1 or frequency_candidates[0] <= 0:
        raise ValueError(f"{source}: expected one positive Frequency record")
    return FfdGrid(
        theta_deg=np.repeat(theta_axis, len(phi_axis)),
        phi_deg=np.tile(phi_axis, len(theta_axis)),
        e_theta=values[:, 0] + 1j * values[:, 1],
        e_phi=values[:, 2] + 1j * values[:, 3],
        frequency_hz=float(frequency_candidates[0]),
    )


def _parse_ffd_mapping_records(path: str | Path) -> dict[str, tuple[str, np.ndarray]]:
    """Read explicit FFD paths and HFSS phase-reference positions in metres."""
    source = Path(path)
    records: dict[str, tuple[str, np.ndarray]] = {}
    for raw in source.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("Source Name"):
            continue
        parts = line.split()
        if len(parts) < 2 or not re.fullmatch(r"\d+:\d+", parts[0]):
            continue
        if len(parts) < 5:
            raise ValueError(f"{source}: missing phase-reference XYZ for {parts[0]}")
        filename = parts[1]
        candidate = source.parent / filename
        if candidate.suffix.lower() != ".ffd":
            candidate = candidate.with_suffix(".ffd")
        phase_reference = np.asarray([float(value) for value in parts[2:5]], dtype=np.float64)
        if not np.isfinite(phase_reference).all():
            raise ValueError(f"{source}: non-finite phase-reference XYZ for {parts[0]}")
        records[parts[0]] = (str(candidate.resolve()), phase_reference)
    if not records:
        raise ValueError(f"{source}: no source/file records found")
    return records


def parse_ffd_mapping(path: str | Path) -> dict[str, str]:
    """Read the HFSS element-pattern mapping without inferring file order."""
    return {name: record[0] for name, record in _parse_ffd_mapping_records(path).items()}


def hfss_basis(theta_deg: np.ndarray, phi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    direction = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    )
    e_theta = np.stack(
        (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)),
        axis=-1,
    )
    e_phi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    return direction, e_theta, e_phi


def load_ffd_operator(mapping_path: str | Path, spec: ArraySpec) -> dict[str, np.ndarray]:
    records = _parse_ffd_mapping_records(mapping_path)
    mapping = {name: record[0] for name, record in records.items()}
    missing = [name for name in spec.port_names if name not in mapping]
    extra = [name for name in mapping if name not in spec.port_names]
    if missing or extra:
        raise ValueError(f"port mapping mismatch; missing={missing}, extra={extra}")
    theta_ref: np.ndarray | None = None
    phi_ref: np.ndarray | None = None
    g_theta: np.ndarray | None = None
    g_phi: np.ndarray | None = None
    for port, name in enumerate(spec.port_names):
        grid = parse_ffd(mapping[name])
        if not np.isclose(grid.frequency_hz, spec.frequency_hz, rtol=1e-9, atol=1.0):
            raise ValueError(
                f"{mapping[name]} frequency {grid.frequency_hz} differs from {spec.frequency_hz}"
            )
        if theta_ref is None:
            theta_ref, phi_ref = grid.theta_deg, grid.phi_deg
            g_theta = np.empty((len(theta_ref), spec.element_count), dtype=np.complex128)
            g_phi = np.empty_like(g_theta)
        elif not (np.array_equal(theta_ref, grid.theta_deg) and np.array_equal(phi_ref, grid.phi_deg)):
            raise ValueError(f"{mapping[name]} uses a different angular grid")
        g_theta[:, port] = grid.e_theta
        g_phi[:, port] = grid.e_phi
    assert theta_ref is not None and phi_ref is not None and g_theta is not None and g_phi is not None
    direction_hfss, theta_basis, phi_basis = hfss_basis(theta_ref, phi_ref)
    phase_reference_m = np.stack([records[name][1] for name in spec.port_names])
    # HFSS exports each FFD relative to the per-port phase reference recorded
    # in the mapping.  Convert all ports back to the common global reference
    # before direct combinations or canonicalization.  Legacy exports use zero
    # references, so this is exactly identity for the frozen 4x4 baseline.
    wave_number = 2.0 * math.pi * spec.frequency_hz / LIGHT_SPEED_M_S
    phase_to_global = np.exp(1j * wave_number * (direction_hfss @ phase_reference_m.T))
    g_theta *= phase_to_global
    g_phi *= phase_to_global
    rotation = np.asarray(spec.coordinate_rotation)
    direction = direction_hfss @ rotation
    field_hfss = (
        g_theta[:, None, :] * theta_basis[:, :, None]
        + g_phi[:, None, :] * phi_basis[:, :, None]
    )
    field_algorithm = np.einsum("qkn,kl->qln", field_hfss, rotation)
    return {
        "theta_deg": theta_ref,
        "phi_deg": phi_ref,
        "direction": direction,
        "g_etheta": g_theta,
        "g_ephi": g_phi,
        "h_cartesian": field_algorithm,
        "source_files": np.asarray([mapping[name] for name in spec.port_names], dtype=np.str_),
        "source_phase_reference_m": phase_reference_m,
    }


def load_touchstone(path: str | Path, nports: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse Touchstone 1.x RI/MA/DB data in standard column-major S order."""
    source = Path(path)
    option: list[str] | None = None
    tokens: list[float] = []
    for raw in source.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            option = line[1:].upper().split()
            continue
        tokens.extend(_numbers(line))
    if option is None or len(option) < 3 or option[1] != "S":
        raise ValueError(f"{source}: unsupported or missing Touchstone option line")
    unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}.get(option[0])
    if unit_scale is None:
        raise ValueError(f"{source}: unsupported frequency unit {option[0]}")
    representation = option[2]
    reference = 50.0
    if "R" in option:
        reference = float(option[option.index("R") + 1])
    record_width = 1 + 2 * nports * nports
    if len(tokens) % record_width:
        raise ValueError(
            f"{source}: {len(tokens)} numeric values do not form {record_width}-value records"
        )
    records = np.asarray(tokens, dtype=np.float64).reshape(-1, record_width)
    frequency = records[:, 0] * unit_scale
    pair = records[:, 1:].reshape(len(records), nports * nports, 2)
    if representation == "RI":
        complex_flat = pair[..., 0] + 1j * pair[..., 1]
    elif representation == "MA":
        complex_flat = pair[..., 0] * np.exp(1j * np.deg2rad(pair[..., 1]))
    elif representation == "DB":
        complex_flat = 10.0 ** (pair[..., 0] / 20.0) * np.exp(1j * np.deg2rad(pair[..., 1]))
    else:
        raise ValueError(f"{source}: unsupported S representation {representation}")
    matrix = np.empty((len(records), nports, nports), dtype=np.complex128)
    for source_port in range(nports):
        for receive_port in range(nports):
            matrix[:, receive_port, source_port] = complex_flat[
                :, source_port * nports + receive_port
            ]
    return frequency, matrix, reference


def _front_hemisphere_unique(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    direction = np.asarray(data["direction"], dtype=np.float64)
    keep = direction[:, 0] > 1e-12
    indices = np.flatnonzero(keep)
    keys = np.round(direction[indices], decimals=12)
    _, first = np.unique(keys, axis=0, return_index=True)
    selected = indices[np.sort(first)]
    output: dict[str, np.ndarray] = {}
    for name, value in data.items():
        array = np.asarray(value)
        output[name] = array[selected] if array.ndim and array.shape[0] == len(direction) else array
    return output


def combine_grids(integer: Mapping[str, np.ndarray], offset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    left = _front_hemisphere_unique(integer)
    right = _front_hemisphere_unique(offset)
    combined: dict[str, np.ndarray] = {}
    query_fields = {"theta_deg", "phi_deg", "direction", "g_etheta", "g_ephi", "h_cartesian"}
    for name in query_fields:
        combined[name] = np.concatenate((left[name], right[name]), axis=0)
    combined["source_grid"] = np.concatenate(
        (
            np.full(len(left["direction"]), "integer", dtype="U7"),
            np.full(len(right["direction"]), "offset", dtype="U7"),
        )
    )
    rounded = np.round(combined["direction"], decimals=12)
    if len(np.unique(rounded, axis=0)) != len(rounded):
        raise ValueError("integer/offset union contains duplicate directions")
    order = np.lexsort((combined["phi_deg"], combined["theta_deg"]))
    for name in list(combined):
        combined[name] = combined[name][order]
    theta_rad = np.deg2rad(combined["theta_deg"])
    weights = np.maximum(np.sin(theta_rad), 0.0)
    combined["solid_angle_weights"] = weights / weights.sum()
    return combined


def select_frequency(
    frequency_hz: np.ndarray, s_parameters: np.ndarray, target_hz: float
) -> tuple[float, np.ndarray]:
    index = int(np.argmin(np.abs(frequency_hz - target_hz)))
    if abs(float(frequency_hz[index]) - target_hz) > max(1.0, 1e-6 * target_hz):
        raise ValueError("Touchstone does not contain the requested frequency")
    return float(frequency_hz[index]), np.asarray(s_parameters[index], dtype=np.complex128)


def canonicalize_array(
    spec: ArraySpec,
    *,
    integer_mapping: str | Path,
    offset_mapping: str | Path,
    touchstone: str | Path,
    output: str | Path,
) -> dict[str, object]:
    integer = load_ffd_operator(integer_mapping, spec)
    offset = load_ffd_operator(offset_mapping, spec)
    combined = combine_grids(integer, offset)
    frequency, s_all, reference = load_touchstone(touchstone, spec.element_count)
    selected_frequency, s_matrix = select_frequency(frequency, s_all, spec.frequency_hz)
    positions = spec.positions_m
    k = 2.0 * math.pi / spec.wavelength_m
    phase = k * np.einsum("qk,nk->qn", combined["direction"], positions)
    envelope = combined["g_ephi"] * np.exp(-1j * phase)
    transverse = np.einsum("qk,qkn->qn", combined["direction"], combined["h_cartesian"])
    field_norm = np.linalg.norm(combined["h_cartesian"], axis=1)
    transverse_relative = np.abs(transverse) / np.maximum(field_norm, 1e-30)
    quality_mask = np.isfinite(combined["g_ephi"]) & np.isfinite(combined["g_etheta"])
    payload = {
        "array_id": np.asarray(spec.array_id),
        "logical_shape": np.asarray([spec.rows, spec.cols], dtype=np.int64),
        "positions_m": positions,
        "active_mask": np.ones((spec.rows, spec.cols), dtype=np.bool_),
        "port_names": np.asarray(spec.port_names, dtype=np.str_),
        "direction": combined["direction"].astype(np.float64),
        "theta_deg": combined["theta_deg"].astype(np.float64),
        "phi_deg": combined["phi_deg"].astype(np.float64),
        "g_ephi": combined["g_ephi"].astype(np.complex128),
        "g_etheta": combined["g_etheta"].astype(np.complex128),
        "h_cartesian": combined["h_cartesian"].astype(np.complex128),
        "envelope_ephi": envelope.astype(np.complex128),
        "s_parameters": s_matrix,
        "solid_angle_weights": combined["solid_angle_weights"].astype(np.float64),
        "frequency_hz": np.asarray(selected_frequency, dtype=np.float64),
        "reference_impedance_ohm": np.asarray(reference, dtype=np.float64),
        "coordinate_rotation": np.asarray(spec.coordinate_rotation, dtype=np.float64),
        "source_grid": combined["source_grid"],
        "quality_mask": quality_mask,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "array_id": spec.array_id,
        "output": str(destination.resolve()),
        "queries": int(len(combined["direction"])),
        "elements": spec.element_count,
        "integer_front_queries": int(np.sum(integer["direction"][:, 0] > 1e-12)),
        "offset_front_queries": int(np.sum(offset["direction"][:, 0] > 1e-12)),
        "duplicate_directions": 0,
        "finite": bool(all(np.isfinite(value).all() for value in (payload["direction"], payload["g_ephi"], payload["g_etheta"], payload["s_parameters"]))),
        "transverse_relative_p95": float(np.percentile(transverse_relative, 95)),
        "transverse_relative_max": float(np.max(transverse_relative)),
        "s_reciprocity_relative": float(
            np.linalg.norm(s_matrix - s_matrix.T) / max(np.linalg.norm(s_matrix), 1e-30)
        ),
        "canonical_sha256": file_sha256(destination),
        "sources": {
            "integer_mapping": str(Path(integer_mapping).resolve()),
            "offset_mapping": str(Path(offset_mapping).resolve()),
            "touchstone": str(Path(touchstone).resolve()),
        },
    }


def load_canonical(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def spherical_block_split(
    direction: np.ndarray, *, seed: int, block_deg: float = 10.0
) -> dict[str, np.ndarray]:
    values = np.asarray(direction, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("direction must have shape [Q,3]")
    theta = np.rad2deg(np.arccos(np.clip(values[:, 2], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(values[:, 1], values[:, 0]))
    block = np.stack(
        (np.floor(theta / block_deg).astype(np.int64), np.floor((phi + 180.0) / block_deg).astype(np.int64)),
        axis=1,
    )
    unique, inverse = np.unique(block, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    train_end = int(round(0.70 * len(unique)))
    validation_end = train_end + int(round(0.15 * len(unique)))
    assignment = np.empty(len(unique), dtype=np.int8)
    assignment[order[:train_end]] = 0
    assignment[order[train_end:validation_end]] = 1
    assignment[order[validation_end:]] = 2
    query_assignment = assignment[inverse]
    result = {
        "train_indices": np.flatnonzero(query_assignment == 0),
        "validation_indices": np.flatnonzero(query_assignment == 1),
        "blind_test_indices": np.flatnonzero(query_assignment == 2),
        "block_id": inverse,
        "block_assignment": assignment,
        "unique_blocks": unique,
    }
    if sum(len(result[name]) for name in ("train_indices", "validation_indices", "blind_test_indices")) != len(values):
        raise RuntimeError("split lost directions")
    return result


def _normalise_command(command: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(command)
    if norm <= 1e-12:
        raise ValueError("command has zero norm")
    return command / norm


def generate_commands(spec: ArraySpec, *, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    fractions = (0.35, 0.20, 0.15, 0.15)
    counts = [int(round(count * value)) for value in fractions]
    counts.append(count - sum(counts))
    families = ("random_constant_modulus", "scan", "taper", "sparse", "hard_null")
    commands: list[np.ndarray] = []
    labels: list[str] = []
    positions = spec.positions_m
    k = 2.0 * math.pi / spec.wavelength_m
    row_window = np.hanning(spec.rows + 2)[1:-1] if spec.rows > 1 else np.ones(1)
    col_window = np.hanning(spec.cols + 2)[1:-1] if spec.cols > 1 else np.ones(1)
    taper = np.outer(row_window, col_window).reshape(-1)
    for family, family_count in zip(families, counts):
        for _ in range(family_count):
            if family == "random_constant_modulus":
                command = np.exp(1j * rng.uniform(-math.pi, math.pi, spec.element_count))
            elif family in {"scan", "taper"}:
                q = rng.normal(size=3)
                q[0] = abs(q[0])
                q /= np.linalg.norm(q)
                command = np.exp(-1j * k * (positions @ q))
                if family == "taper":
                    command = command * taper
            elif family == "sparse":
                command = np.zeros(spec.element_count, dtype=np.complex128)
                active = rng.choice(spec.element_count, size=max(1, spec.element_count // 4), replace=False)
                command[active] = np.exp(1j * rng.uniform(-math.pi, math.pi, len(active)))
            else:
                q = rng.normal(size=3)
                q[0] = abs(q[0])
                q /= np.linalg.norm(q)
                steering = np.exp(-1j * k * (positions @ q))
                random_vector = rng.normal(size=spec.element_count) + 1j * rng.normal(size=spec.element_count)
                command = random_vector - steering * np.vdot(steering, random_vector) / np.vdot(steering, steering)
            commands.append(_normalise_command(command).astype(np.complex128))
            labels.append(family)
    output = np.stack(commands)
    if not np.allclose(np.sum(np.abs(output) ** 2, axis=1), 1.0, atol=1e-12):
        raise RuntimeError("command normalisation failed")
    return output, np.asarray(labels, dtype=np.str_)


def generate_direct_audit_commands(spec: ArraySpec, *, seed: int = 21740) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen 4+4+4+4 direct-HFSS audit command set."""
    rng = np.random.default_rng(seed)
    positions = spec.positions_m
    k = 2.0 * math.pi / spec.wavelength_m
    commands: list[np.ndarray] = []
    families: list[str] = []
    scans = np.asarray(((1, 0, 0), (1, 0.45, 0), (1, -0.45, 0), (1, 0, 0.45)), dtype=float)
    scans /= np.linalg.norm(scans, axis=1, keepdims=True)
    for direction in scans:
        commands.append(_normalise_command(np.exp(-1j * k * (positions @ direction))))
        families.append("scan")
    for _ in range(4):
        commands.append(_normalise_command(np.exp(1j * rng.uniform(-math.pi, math.pi, spec.element_count))))
        families.append("random_constant_modulus")
    sparse_size = max(1, spec.element_count // 2)
    for index in range(4):
        command = np.zeros(spec.element_count, dtype=np.complex128)
        active = (np.arange(sparse_size) + index * max(1, sparse_size // 2)) % spec.element_count
        command[active] = np.exp(1j * rng.uniform(-math.pi, math.pi, sparse_size))
        commands.append(_normalise_command(command))
        families.append("sparse_subarray")
    for _ in range(4):
        direction = rng.normal(size=3)
        direction[0] = abs(direction[0])
        direction /= np.linalg.norm(direction)
        steering = np.exp(-1j * k * (positions @ direction))
        vector = rng.normal(size=spec.element_count) + 1j * rng.normal(size=spec.element_count)
        command = vector - steering * np.vdot(steering, vector) / np.vdot(steering, steering)
        commands.append(_normalise_command(command))
        families.append("cancellation_null")
    output = np.stack(commands)
    if output.shape != (16, spec.element_count):
        raise RuntimeError("direct audit command count is not 16")
    if not np.allclose(np.sum(np.abs(output) ** 2, axis=1), 1.0, atol=1e-12):
        raise RuntimeError("direct audit command power is not one")
    return output, np.asarray(families, dtype=np.str_)


def write_split_and_commands(
    canonical_path: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int,
    validation_command_seed: int,
    test_command_seed: int,
) -> dict[str, object]:
    data = load_canonical(canonical_path)
    rows, cols = (int(value) for value in data["logical_shape"])
    spec = ArraySpec(
        array_id=str(data["array_id"]),
        rows=rows,
        cols=cols,
        row_pitch_m=float(np.ptp(data["positions_m"][:, 2]) / max(rows - 1, 1)) if rows > 1 else 0.015,
        col_pitch_m=float(np.ptp(data["positions_m"][:, 1]) / max(cols - 1, 1)) if cols > 1 else 0.0068,
        frequency_hz=float(data["frequency_hz"]),
        coordinate_rotation=data["coordinate_rotation"],
    )
    split = spherical_block_split(data["direction"], seed=split_seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "direction_split.npz", **split)
    for name, seed in (("validation", validation_command_seed), ("blind_test", test_command_seed)):
        command, family = generate_commands(spec, count=512, seed=seed)
        np.savez_compressed(destination / f"{name}_commands.npz", weights=command, family=family)
    manifest = {
        "array_id": spec.array_id,
        "canonical": str(Path(canonical_path).resolve()),
        "split_seed": split_seed,
        "block_deg": 10.0,
        "counts": {name: int(len(split[f"{name}_indices"])) for name in ("train", "validation", "blind_test")},
        "validation_command_seed": validation_command_seed,
        "blind_test_command_seed": test_command_seed,
        "commands_per_evaluation_split": 512,
        "normalization_source": "train-only-single-port-amplitude",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_training_views(
    canonical_path: str | Path, split_dir: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Write lower-precision train/validation views without a blind-test cache."""
    data = load_canonical(canonical_path)
    with np.load(Path(split_dir) / "direction_split.npz", allow_pickle=False) as archive:
        indices = {
            "train": np.asarray(archive["train_indices"], dtype=np.int64),
            "validation": np.asarray(archive["validation_indices"], dtype=np.int64),
        }
    train_g = data["g_ephi"][indices["train"]]
    scale = float(np.sqrt(np.mean(np.abs(train_g) ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("train-only single-port RMS scale is not finite and positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, selected in indices.items():
        np.savez_compressed(
            destination / f"{split_name}_operator_cache.npz",
            canonical_indices=selected,
            direction=data["direction"][selected].astype(np.float32),
            theta_deg=data["theta_deg"][selected].astype(np.float32),
            phi_deg=data["phi_deg"][selected].astype(np.float32),
            g_ephi=(data["g_ephi"][selected] / scale).astype(np.complex64),
            g_etheta=(data["g_etheta"][selected] / scale).astype(np.complex64),
            h_cartesian=(data["h_cartesian"][selected] / scale).astype(np.complex64),
            envelope_ephi=(data["envelope_ephi"][selected] / scale).astype(np.complex64),
            solid_angle_weights=data["solid_angle_weights"][selected].astype(np.float32),
            quality_mask=data["quality_mask"][selected],
            normalization_scale=np.asarray(scale, dtype=np.float64),
            positions_m=data["positions_m"].astype(np.float64),
            active_mask=data["active_mask"],
            frequency_hz=data["frequency_hz"],
        )
    receipt = {
        "normalization": "train-only RMS of absolute single-port Ephi",
        "scale": scale,
        "train_queries": int(len(indices["train"])),
        "validation_queries": int(len(indices["validation"])),
        "blind_test_cache_written": False,
        "canonical_precision": "float64/complex128",
        "derived_precision": "float32/complex64",
    }
    (destination / "normalization.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
