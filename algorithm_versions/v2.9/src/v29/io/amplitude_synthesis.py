"""Create amplitude-only feasibility observations from raw complex HFSS FFD."""

from __future__ import annotations

import numpy as np

from ..coordinates import spherical_basis
from .amplitude_dataset import AmplitudeDataset, AmplitudeManifest
from .dataset_store import BasisDataset


RECEIVER_CHANNELS = ("etheta", "ephi", "x", "y", "z")


def canonical_spherical_query_mask(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Remove coordinate-singular poles and one duplicated full-sphere seam.

    Local Etheta/Ephi polarization is undefined at theta=0/180, while phi=-180
    and phi=+180 describe the same physical direction. Keeping those rows would
    either create conflicting labels or let an identical direction cross a data
    split. Partial scans that do not contain both seam endpoints are untouched.
    """
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if theta.shape != phi.shape or theta.ndim != 1:
        raise ValueError("theta_deg and phi_deg must be matching one-dimensional arrays")
    keep = ~(np.isclose(theta, 0.0, atol=1e-9) | np.isclose(theta, 180.0, atol=1e-9))
    has_negative_seam = np.any(np.isclose(phi, -180.0, atol=1e-9))
    has_positive_seam = np.any(np.isclose(phi, 180.0, atol=1e-9))
    if has_negative_seam and has_positive_seam:
        keep &= ~np.isclose(phi, 180.0, atol=1e-9)
    if not np.any(keep):
        raise ValueError("spherical canonicalization removed every query")
    return keep


def generate_command_codebook(
    *,
    random_commands: int = 96,
    seed: int = 28001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return impulse, 2D-DFT and random phase commands, each with unit power."""
    if random_commands < 0:
        raise ValueError("random_commands must be non-negative")
    commands: list[np.ndarray] = []
    families: list[str] = []

    for row in range(4):
        for col in range(4):
            command = np.zeros((4, 4), dtype=np.complex128)
            command[row, col] = 1.0
            commands.append(command)
            families.append("single_element")

    row = np.arange(4)[:, None]
    col = np.arange(4)[None, :]
    for row_mode in range(4):
        for col_mode in range(4):
            phase = 2.0 * np.pi * (row_mode * row / 4.0 + col_mode * col / 4.0)
            commands.append(np.exp(1j * phase) / 4.0)
            families.append("dft_phase")

    generator = np.random.default_rng(seed)
    for _ in range(random_commands):
        phase = generator.uniform(-np.pi, np.pi, size=(4, 4))
        commands.append(np.exp(1j * phase) / 4.0)
        families.append("random_phase")

    complex_commands = np.stack(commands, axis=0)
    command_iq = np.stack((complex_commands.real, complex_commands.imag), axis=1).astype(np.float32)
    return command_iq, np.asarray(families, dtype="U32")


def _receiver_vectors(basis: BasisDataset, receiver_channel: str) -> np.ndarray:
    if receiver_channel not in RECEIVER_CHANNELS:
        raise ValueError(f"receiver_channel must be one of {RECEIVER_CHANNELS}")
    if receiver_channel in {"etheta", "ephi"}:
        if basis.theta_deg is None or basis.phi_deg is None:
            raise ValueError("spherical receiver channels require theta/phi coordinates")
        e_theta, e_phi = spherical_basis(basis.theta_deg, basis.phi_deg)
        return e_theta if receiver_channel == "etheta" else e_phi
    axis = np.zeros(3, dtype=np.float64)
    axis[{"x": 0, "y": 1, "z": 2}[receiver_channel]] = 1.0
    return np.broadcast_to(axis, basis.direction.shape)


def synthesize_amplitude_observations(
    basis: BasisDataset,
    command_iq: np.ndarray,
    command_family: np.ndarray,
    manifest: AmplitudeManifest,
) -> AmplitudeDataset:
    """Project HFSS onto one receiver channel, contract known commands, keep |S21| only."""
    if manifest.source_type != "hfss_amplitude":
        raise ValueError("HFSS synthesis requires source_type='hfss_amplitude'")
    # HFSS FFD stores r*E in volts. Convert it to field amplitude at the
    # declared reference distance so the model's later 1/r scaling is physical.
    amplitude = amplitude_for_commands_from_basis(
        basis, command_iq, manifest.receiver_channel
    ) / manifest.reference_distance_m
    return AmplitudeDataset(
        direction=basis.direction.astype(np.float64),
        command_iq=np.asarray(command_iq, dtype=np.float32),
        amplitude=amplitude,
        command_family=np.asarray(command_family, dtype="U32"),
        theta_deg=np.asarray(basis.theta_deg, dtype=np.float64),
        phi_deg=np.asarray(basis.phi_deg, dtype=np.float64),
        manifest=manifest,
    )


def amplitude_for_commands_from_basis(
    basis: BasisDataset,
    command_iq: np.ndarray,
    receiver_channel: str,
) -> np.ndarray:
    """Reference-only calculation; returns scalar amplitude [command,query]."""
    command_iq = np.asarray(command_iq)
    if command_iq.ndim != 4 or command_iq.shape[1:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [N,2,4,4]")
    receiver = _receiver_vectors(basis, receiver_channel)
    operator = basis.operator_iq[:, :, 0] + 1j * basis.operator_iq[:, :, 1]
    scalar_operator = np.einsum("qx,qxrc->qrc", receiver, operator)
    commands = command_iq[:, 0] + 1j * command_iq[:, 1]
    scalar_field = np.einsum("qrc,nrc->nq", scalar_operator, commands)
    return np.abs(scalar_field).astype(np.float32)
