"""Offline VSH order audit for the frozen 47 GHz HFSS full-sphere fields.

The script is intentionally isolated from the amplitude-only training path. It
reads the four frozen complex-field partitions, reconstructs the 180x360 sphere,
projects the transverse field onto orthonormal vector spherical harmonics, and
compares the complete field H with the per-port dephased envelope A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y


C0_M_S = 299_792_458.0
FREQUENCY_HZ = 47.0e9
ROW_PITCH_M = 0.015
COL_PITCH_M = 0.0068
PORT_Z_OFFSET_M = 0.000127
STRUCTURE_BBOX_MIN_M = np.array([-0.5, -3.4, -0.797], dtype=np.float64) * 1e-3
STRUCTURE_BBOX_MAX_M = np.array([5.0, 23.8, 45.289], dtype=np.float64) * 1e-3
PARTITIONS = (
    "dev/train.npz",
    "dev/validation.npz",
    "reserved/calibration.npz",
    "reserved/holdout.npz",
)


@dataclass(frozen=True)
class SphereData:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    direction: np.ndarray
    field_cartesian: np.ndarray
    source_manifest: list[dict[str, object]]


@dataclass(frozen=True)
class DegreeBasis:
    l: int
    m: np.ndarray
    x_theta: np.ndarray
    x_phi: np.ndarray
    rx_theta: np.ndarray
    rx_phi: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _db10(value: np.ndarray | float, floor_db: float = -300.0) -> np.ndarray:
    value_array = np.asarray(value, dtype=np.float64)
    floor = 10.0 ** (floor_db / 10.0)
    return 10.0 * np.log10(np.maximum(value_array, floor))


def _weighted_square(field: np.ndarray, sin_theta: np.ndarray) -> float:
    return float(np.sum(np.abs(field) ** 2 * sin_theta[:, None, None, None]))


def _weighted_scalar_square(field: np.ndarray, sin_theta: np.ndarray) -> float:
    return float(np.sum(np.abs(field) ** 2 * sin_theta[:, None, None]))


def _port_positions_absolute() -> np.ndarray:
    positions = np.zeros((16, 3), dtype=np.float64)
    for row in range(4):
        for col in range(4):
            index = 4 * row + col
            positions[index] = (0.0, col * COL_PITCH_M, PORT_Z_OFFSET_M + row * ROW_PITCH_M)
    return positions


def load_full_sphere(input_root: Path) -> SphereData:
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in ("ids", "q", "h", "theta", "phi")}
    manifest: list[dict[str, object]] = []
    for relative in PARTITIONS:
        path = input_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            for name in arrays:
                arrays[name].append(np.asarray(archive[name]))
        manifest.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})

    ids = np.concatenate(arrays["ids"]).astype(np.int64, copy=False)
    q = np.concatenate(arrays["q"]).astype(np.float64, copy=False)
    h = np.concatenate(arrays["h"]).astype(np.complex128, copy=False)
    theta = np.concatenate(arrays["theta"]).astype(np.float64, copy=False)
    phi = np.concatenate(arrays["phi"]).astype(np.float64, copy=False)
    order = np.argsort(ids)
    ids, q, h, theta, phi = ids[order], q[order], h[order], theta[order], phi[order]

    expected_ids = np.arange(180 * 360, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("the four partitions do not form exactly one 180x360 full sphere")
    theta_unique = np.unique(theta)
    phi_unique = np.unique(phi)
    expected_theta = np.arange(0.5, 180.0, 1.0)
    expected_phi = np.arange(-179.5, 180.0, 1.0)
    if not np.array_equal(theta_unique, expected_theta) or not np.array_equal(phi_unique, expected_phi):
        raise ValueError("unexpected HFSS angular grid")
    if not np.allclose(theta.reshape(180, 360), theta_unique[:, None]):
        raise ValueError("theta ordering is not row-major")
    if not np.allclose(phi.reshape(180, 360), phi_unique[None, :]):
        raise ValueError("phi ordering is not row-major")
    if h.shape != (64_800, 3, 16):
        raise ValueError(f"unexpected field shape {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("field contains non-finite values")
    norm_error = float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)))
    if norm_error > 1e-6:
        raise ValueError(f"direction normalization error {norm_error}")
    transverse = np.abs(np.einsum("qi,qip->qp", q, h))
    relative = transverse / np.maximum(np.linalg.norm(h, axis=1), 1e-15)
    if float(np.max(relative)) > 1e-5:
        raise ValueError("field is not transverse")

    return SphereData(
        theta_deg=theta_unique,
        phi_deg=phi_unique,
        direction=q.reshape(180, 360, 3),
        field_cartesian=h.reshape(180, 360, 3, 16),
        source_manifest=manifest,
    )


def cartesian_to_spherical_components(
    field_cartesian: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
) -> np.ndarray:
    theta = theta_rad[:, None]
    phi = phi_rad[None, :]
    e_theta = np.stack(
        (
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta) * np.ones_like(phi),
        ),
        axis=2,
    )
    e_phi = np.stack(
        (
            -np.ones_like(theta) * np.sin(phi),
            np.ones_like(theta) * np.cos(phi),
            np.zeros((theta_rad.size, phi_rad.size)),
        ),
        axis=2,
    )
    e_theta_component = np.einsum("tjxp,tjx->tjp", field_cartesian, e_theta, optimize=True)
    e_phi_component = np.einsum("tjxp,tjx->tjp", field_cartesian, e_phi, optimize=True)
    return np.stack((e_theta_component, e_phi_component), axis=2)


def build_degree_basis(theta_rad: np.ndarray, l: int) -> DegreeBasis:
    if l < 1:
        raise ValueError("VSH degree starts at l=1")
    sin_theta = np.sin(theta_rad)
    if np.any(sin_theta <= 0.0):
        raise ValueError("cell-centred theta grid must exclude the poles")
    m_values = np.arange(-l, l + 1, dtype=np.int64)
    x_theta = np.empty((theta_rad.size, m_values.size), dtype=np.complex128)
    x_phi = np.empty_like(x_theta)
    normalization = math.sqrt(l * (l + 1.0))
    zero_phi = np.zeros_like(theta_rad)
    for column, m in enumerate(m_values):
        y_lm, derivatives = sph_harm_y(l, int(m), theta_rad, zero_phi, diff_n=1)
        dtheta = derivatives[:, 0]
        x_theta[:, column] = -float(m) * y_lm / (normalization * sin_theta)
        x_phi[:, column] = -1j * dtheta / normalization
    return DegreeBasis(
        l=l,
        m=m_values,
        x_theta=x_theta,
        x_phi=x_phi,
        rx_theta=-x_phi,
        rx_phi=x_theta,
    )


def basis_quadrature_diagnostics(theta_rad: np.ndarray, degrees: Iterable[int]) -> dict[str, float]:
    dtheta = float(np.deg2rad(1.0))
    dphi = dtheta
    weights = np.sin(theta_rad) * dtheta * dphi * 360.0
    maximum_norm_error = 0.0
    maximum_cross = 0.0
    for l in degrees:
        basis = build_degree_basis(theta_rad, l)
        for column in range(basis.m.size):
            norm = np.sum(
                weights
                * (np.abs(basis.x_theta[:, column]) ** 2 + np.abs(basis.x_phi[:, column]) ** 2)
            )
            cross = np.sum(
                weights
                * (
                    np.conj(basis.x_theta[:, column]) * basis.rx_theta[:, column]
                    + np.conj(basis.x_phi[:, column]) * basis.rx_phi[:, column]
                )
            )
            maximum_norm_error = max(maximum_norm_error, abs(float(norm) - 1.0))
            maximum_cross = max(maximum_cross, float(abs(cross)))
    sphere_weight = float(np.sum(np.sin(theta_rad) * dtheta) * 360.0 * dphi)
    return {
        "sphere_weight": sphere_weight,
        "sphere_weight_error": abs(sphere_weight - 4.0 * math.pi),
        "selected_mode_max_norm_error": maximum_norm_error,
        "selected_mode_max_x_rx_inner_product": maximum_cross,
    }


def project_single_degree(
    fourier_field: np.ndarray,
    basis: DegreeBasis,
    theta_weight: np.ndarray,
    m_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(m_all, basis.m)
    selected = fourier_field[:, indices]
    f_theta = selected[:, :, 0, :]
    f_phi = selected[:, :, 1, :]
    weight = theta_weight[:, None, None]
    c_m = np.sum(
        weight
        * (
            np.conj(basis.x_theta)[:, :, None] * f_theta
            + np.conj(basis.x_phi)[:, :, None] * f_phi
        ),
        axis=0,
    )
    c_e = np.sum(
        weight
        * (
            np.conj(basis.rx_theta)[:, :, None] * f_theta
            + np.conj(basis.rx_phi)[:, :, None] * f_phi
        ),
        axis=0,
    )
    return c_m, c_e


def synthesize_single_degree(
    basis: DegreeBasis,
    c_m: np.ndarray,
    c_e: np.ndarray,
    phase_positive: np.ndarray,
    m_all: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(m_all, basis.m)
    spectral = np.empty((basis.x_theta.shape[0], basis.m.size, 2, c_m.shape[1]), dtype=np.complex128)
    spectral[:, :, 0, :] = (
        basis.x_theta[:, :, None] * c_m[None, :, :]
        + basis.rx_theta[:, :, None] * c_e[None, :, :]
    )
    spectral[:, :, 1, :] = (
        basis.x_phi[:, :, None] * c_m[None, :, :]
        + basis.rx_phi[:, :, None] * c_e[None, :, :]
    )
    return np.einsum("tmcp,jm->tjcp", spectral, phase_positive[:, indices], optimize=True)


def run_basis_self_tests() -> dict[str, float]:
    theta = np.deg2rad(np.arange(0.5, 180.0, 1.0))
    phi = np.deg2rad(np.arange(-179.5, 180.0, 1.0))
    diagnostics = basis_quadrature_diagnostics(theta, (1, 2, 8, 20, 40))
    m_all = np.arange(-8, 9)
    phase_positive = np.exp(1j * np.outer(phi, m_all))
    phase_negative = np.conj(phase_positive)
    basis = build_degree_basis(theta, 3)
    selected_index = int(np.where(basis.m == 2)[0][0])
    c_m_true = np.zeros((basis.m.size, 1), dtype=np.complex128)
    c_e_true = np.zeros_like(c_m_true)
    c_m_true[selected_index, 0] = 0.7 - 0.2j
    c_e_true[selected_index, 0] = -0.3 + 0.4j
    field = synthesize_single_degree(basis, c_m_true, c_e_true, phase_positive, m_all)
    fourier = np.einsum("tjcp,jm->tmcp", field, phase_negative, optimize=True) * np.deg2rad(1.0)
    c_m, c_e = project_single_degree(
        fourier,
        basis,
        np.sin(theta) * np.deg2rad(1.0),
        m_all,
    )
    recovery_error = max(float(np.max(np.abs(c_m - c_m_true))), float(np.max(np.abs(c_e - c_e_true))))
    diagnostics["synthetic_mode_max_coefficient_error"] = recovery_error
    if diagnostics["sphere_weight_error"] > 5e-4:
        raise AssertionError(diagnostics)
    if diagnostics["selected_mode_max_norm_error"] > 2e-3:
        raise AssertionError(diagnostics)
    if diagnostics["selected_mode_max_x_rx_inner_product"] > 2e-10:
        raise AssertionError(diagnostics)
    if recovery_error > 2e-3:
        raise AssertionError(diagnostics)
    return diagnostics


def compute_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    sin_theta: np.ndarray,
) -> dict[str, object]:
    error = reconstruction - truth
    total_energy = _weighted_square(truth, sin_theta)
    field_nmse = _weighted_square(error, sin_theta) / total_energy
    true_amplitude = np.linalg.norm(truth, axis=2)
    predicted_amplitude = np.linalg.norm(reconstruction, axis=2)
    amplitude_nmse = _weighted_scalar_square(predicted_amplitude - true_amplitude, sin_theta) / _weighted_scalar_square(
        true_amplitude, sin_theta
    )
    per_port_error = np.sum(
        np.abs(error) ** 2 * sin_theta[:, None, None, None], axis=(0, 1, 2)
    )
    per_port_total = np.sum(
        np.abs(truth) ** 2 * sin_theta[:, None, None, None], axis=(0, 1, 2)
    )
    per_port_nmse_db = _db10(per_port_error / per_port_total)
    peak = np.max(true_amplitude, axis=(0, 1), keepdims=True)
    strong = true_amplitude >= peak * (10.0 ** (-10.0 / 20.0))
    true_db = 20.0 * np.log10(np.maximum(true_amplitude, peak * 1e-12))
    predicted_db = 20.0 * np.log10(np.maximum(predicted_amplitude, peak * 1e-12))
    strong_weights = sin_theta[:, None, None] * strong
    strong_abs_error = np.abs(predicted_db - true_db)
    strong_mae = float(np.sum(strong_abs_error * strong_weights) / np.sum(strong_weights))
    return {
        "field_nmse_db": float(_db10(field_nmse)),
        "amplitude_nmse_db": float(_db10(amplitude_nmse)),
        "strong_field_mae_db": strong_mae,
        "per_port_nmse_db": per_port_nmse_db.tolist(),
        "worst_port_nmse_db": float(np.max(per_port_nmse_db)),
    }


def audit_vsh(
    fields: dict[str, np.ndarray],
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    l_max: int,
    progress: bool = True,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    dict[str, np.ndarray],
    list[dict[str, object]],
]:
    m_all = np.arange(-l_max, l_max + 1, dtype=np.int64)
    phase_positive = np.exp(1j * np.outer(phi_rad, m_all))
    phase_negative = np.conj(phase_positive)
    dtheta = float(np.deg2rad(1.0))
    dphi = dtheta
    theta_weight = np.sin(theta_rad) * dtheta
    sin_theta = np.sin(theta_rad)
    fourier = {
        name: np.einsum("tjcp,jm->tmcp", field, phase_negative, optimize=True) * dphi
        for name, field in fields.items()
    }
    reconstructions = {name: np.zeros_like(field) for name, field in fields.items()}
    coefficients_m: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    coefficients_e: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    order_power = {name: np.zeros((l_max, 16), dtype=np.float64) for name in fields}
    total_energy = {name: _weighted_square(field, sin_theta) * dtheta * dphi for name, field in fields.items()}
    cumulative_coefficient_energy = {name: 0.0 for name in fields}
    rows: list[dict[str, object]] = []

    for l in range(1, l_max + 1):
        basis = build_degree_basis(theta_rad, l)
        for name in fields:
            c_m, c_e = project_single_degree(fourier[name], basis, theta_weight, m_all)
            coefficients_m[name].append(c_m)
            coefficients_e[name].append(c_e)
            per_port_power = np.sum(np.abs(c_m) ** 2 + np.abs(c_e) ** 2, axis=0)
            order_power[name][l - 1] = per_port_power
            cumulative_coefficient_energy[name] += float(np.sum(per_port_power))
            reconstructions[name] += synthesize_single_degree(basis, c_m, c_e, phase_positive, m_all)
            metrics = compute_metrics(fields[name], reconstructions[name], sin_theta)
            coefficient_fraction = cumulative_coefficient_energy[name] / total_energy[name]
            coefficient_closure_error = max(abs(1.0 - coefficient_fraction), 10.0 ** -30)
            rows.append(
                {
                    "field": name,
                    "l_max": l,
                    "order_power": float(np.sum(per_port_power)),
                    "order_power_db_relative_total": float(_db10(np.sum(per_port_power) / total_energy[name])),
                    "cumulative_coefficient_energy_fraction": coefficient_fraction,
                    "coefficient_energy_closure_error_db": float(_db10(coefficient_closure_error)),
                    **metrics,
                }
            )
        if progress:
            h_row = rows[-2]
            a_row = rows[-1]
            print(
                f"l={l:02d}/{l_max}: H {h_row['field_nmse_db']:.2f} dB, "
                f"A {a_row['field_nmse_db']:.2f} dB",
                flush=True,
            )
    return coefficients_m, coefficients_e, order_power, rows


def _threshold_orders(rows: list[dict[str, object]], field: str) -> dict[str, int | None]:
    selected = [row for row in rows if row["field"] == field]
    result: dict[str, int | None] = {}
    for threshold in (-20, -30, -40):
        result[f"field_nmse_{threshold}_db"] = next(
            (int(row["l_max"]) for row in selected if float(row["field_nmse_db"]) <= threshold),
            None,
        )
    return result


def _rows_by_field(rows: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    return [row for row in rows if row["field"] == name]


def _format_order(value: int | None, l_max: int) -> str:
    return str(value) if value is not None else f"未在 Λ≤{l_max} 达到"


def save_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    scalar_keys = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})


def save_port_spectrum_csv(path: Path, order_power: dict[str, np.ndarray], total_energy_port: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "port", "l", "power", "power_db_relative_port_total"))
        for name, values in order_power.items():
            for port in range(16):
                for l in range(1, values.shape[0] + 1):
                    writer.writerow(
                        (
                            name,
                            port + 1,
                            l,
                            values[l - 1, port],
                            float(_db10(values[l - 1, port] / total_energy_port[name][port])),
                        )
                    )


def flatten_coefficients(coefficients: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(coefficients, axis=0)


def mode_index(l_max: int) -> tuple[np.ndarray, np.ndarray]:
    degrees: list[int] = []
    orders: list[int] = []
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            degrees.append(l)
            orders.append(m)
    return np.asarray(degrees, dtype=np.int16), np.asarray(orders, dtype=np.int16)


def plot_order_spectrum(
    path: Path,
    order_power: dict[str, np.ndarray],
    total_energy_port: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = {"H_centered": "#1f77b4", "A_dephased": "#d62728"}
    labels = {"H_centered": "H: centered complete field", "A_dephased": "A: per-port phase removed"}
    l_axis = np.arange(1, next(iter(order_power.values())).shape[0] + 1)
    for name, power in order_power.items():
        per_port_db = _db10(power / total_energy_port[name][None, :])
        median = np.median(per_port_db, axis=1)
        low = np.percentile(per_port_db, 10, axis=1)
        high = np.percentile(per_port_db, 90, axis=1)
        ax.plot(l_axis, median, linewidth=2.1, color=colors[name], label=labels[name])
        ax.fill_between(l_axis, low, high, color=colors[name], alpha=0.16, label=f"{name} port P10-P90")
    ax.axvline(24.335, color="#666666", linestyle="--", linewidth=1.2, label="center-only kR=24.34")
    ax.axvline(26.496, color="#111111", linestyle=":", linewidth=1.5, label="structure kR=26.50")
    ax.set_xlabel("VSH degree l")
    ax.set_ylabel("Per-degree power / per-port total (dB)")
    ax.set_title("HFSS vector-spherical-harmonic order spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tail_energy(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = {"H_centered": "#1f77b4", "A_dephased": "#d62728"}
    for name in colors:
        selected = _rows_by_field(rows, name)
        x = [int(row["l_max"]) for row in selected]
        y = [float(row["field_nmse_db"]) for row in selected]
        ax.plot(x, y, marker="o", markersize=3, color=colors[name], label=name)
    for threshold in (-20, -30, -40):
        ax.axhline(threshold, color="#888888", linestyle="--", linewidth=0.8)
        ax.text(40.1, threshold, f"{threshold} dB", va="center", fontsize=8)
    ax.axvline(24.335, color="#666666", linestyle="--", linewidth=1.0)
    ax.axvline(26.496, color="#111111", linestyle=":", linewidth=1.3)
    ax.set_xlim(1, 42)
    ax.set_xlabel("Truncation degree Lambda")
    ax.set_ylabel("Weighted residual vector energy / field energy (dB)")
    ax.set_title("Cumulative VSH reconstruction residual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reconstruction_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), constrained_layout=True)
    colors = {"H_centered": "#1f77b4", "A_dephased": "#d62728"}
    for name in colors:
        selected = _rows_by_field(rows, name)
        x = [int(row["l_max"]) for row in selected]
        axes[0].plot(x, [float(row["field_nmse_db"]) for row in selected], color=colors[name], linewidth=2, label=f"{name} vector")
        axes[0].plot(x, [float(row["amplitude_nmse_db"]) for row in selected], color=colors[name], linewidth=1.5, linestyle="--", label=f"{name} magnitude")
        axes[1].plot(x, [float(row["strong_field_mae_db"]) for row in selected], color=colors[name], linewidth=2, label=name)
    for threshold in (-20, -30, -40):
        axes[0].axhline(threshold, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("Truncation degree Lambda")
    axes[0].set_ylabel("Weighted reconstruction NMSE (dB)")
    axes[0].set_title("Complex vector and magnitude reconstruction")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Truncation degree Lambda")
    axes[1].set_ylabel("Top-10 dB region magnitude MAE (dB)")
    axes[1].set_title("Strong-field reconstruction error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_port_heatmap(
    path: Path,
    order_power: dict[str, np.ndarray],
    total_energy_port: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.8), constrained_layout=True)
    for ax, name in zip(axes, ("H_centered", "A_dephased")):
        values = _db10(order_power[name].T / total_energy_port[name][:, None])
        image = ax.imshow(values, origin="lower", aspect="auto", extent=(0.5, values.shape[1] + 0.5, 0.5, 16.5), cmap="viridis", vmin=-70, vmax=-5)
        ax.set_title(name)
        ax.set_xlabel("VSH degree l")
        ax.set_ylabel("HFSS port")
        ax.set_yticks(np.arange(1, 17))
        fig.colorbar(image, ax=ax, label="Degree power / port total (dB)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lm_spectrum(
    path: Path,
    coefficients_m: dict[str, list[np.ndarray]],
    coefficients_e: dict[str, list[np.ndarray]],
    total_energy: dict[str, float],
    l_max: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
    for ax, name in zip(axes, ("H_centered", "A_dephased")):
        matrix = np.full((l_max, 2 * l_max + 1), np.nan)
        for l, (c_m, c_e) in enumerate(zip(coefficients_m[name], coefficients_e[name]), start=1):
            power = np.sum(np.abs(c_m) ** 2 + np.abs(c_e) ** 2, axis=1)
            matrix[l - 1, l_max - l : l_max + l + 1] = _db10(power / total_energy[name])
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, origin="lower", aspect="auto", extent=(-l_max - 0.5, l_max + 0.5, 0.5, l_max + 0.5), cmap="magma", vmin=-85, vmax=-8)
        ax.set_xlabel("Order m")
        ax.set_ylabel("Degree l")
        ax.set_title(name)
        fig.colorbar(image, ax=ax, label="Mode power / field total (dB)")
    fig.suptitle("VSH (l,m) power spectrum, electric and magnetic families combined")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconstruct_to_order(
    coefficients_m: list[np.ndarray],
    coefficients_e: list[np.ndarray],
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    order: int,
) -> np.ndarray:
    m_all = np.arange(-order, order + 1)
    phase_positive = np.exp(1j * np.outer(phi_rad, m_all))
    output = np.zeros((theta_rad.size, phi_rad.size, 2, 16), dtype=np.complex128)
    for l in range(1, order + 1):
        basis = build_degree_basis(theta_rad, l)
        output += synthesize_single_degree(
            basis,
            coefficients_m[l - 1],
            coefficients_e[l - 1],
            phase_positive,
            m_all,
        )
    return output


def plot_reconstruction_examples(
    path: Path,
    fields: dict[str, np.ndarray],
    reconstructions: dict[str, np.ndarray],
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    order: int,
) -> None:
    ports = (0, 5, 15)
    fig, axes = plt.subplots(6, 3, figsize=(15.5, 19.0), constrained_layout=True)
    row = 0
    extent = (float(phi_deg.min()), float(phi_deg.max()), float(theta_deg.min()), float(theta_deg.max()))
    for name in ("H_centered", "A_dephased"):
        truth = np.linalg.norm(fields[name], axis=2)
        predicted = np.linalg.norm(reconstructions[name], axis=2)
        for port in ports:
            reference = max(float(np.max(truth[:, :, port])), 1e-15)
            original_db = 20.0 * np.log10(np.maximum(truth[:, :, port] / reference, 1e-6))
            reconstructed_db = 20.0 * np.log10(np.maximum(predicted[:, :, port] / reference, 1e-6))
            error_db = 20.0 * np.log10(
                np.maximum(
                    np.linalg.norm(reconstructions[name][:, :, :, port] - fields[name][:, :, :, port], axis=2) / reference,
                    1e-6,
                )
            )
            images = (original_db, reconstructed_db, error_db)
            titles = ("Original magnitude", f"VSH reconstruction L={order}", "Vector error magnitude")
            for column, (values, title) in enumerate(zip(images, titles)):
                image = axes[row, column].imshow(values, origin="lower", aspect="auto", extent=extent, cmap="turbo", vmin=-60, vmax=0)
                axes[row, column].set_title(f"{name}, port {port + 1}: {title}", fontsize=9)
                axes[row, column].set_xlabel("phi (deg)")
                axes[row, column].set_ylabel("theta (deg)")
                fig.colorbar(image, ax=axes[row, column], label="dB relative to port peak")
            row += 1
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    path: Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "# V2.11 HFSS-VSH 阶数测试报告",
        "",
        f"> 生成日期：{summary['date']}  ",
        "> 数据：47 GHz、180×360 完整球面、16 个 HFSS 嵌入端口复矢量场  ",
        "> 说明：这是离线复场物理审计，不改变 V2.10 单极化幅度训练合同。",
        "",
        "## 1. 结论",
        "",
    ]
    for name in ("H_centered", "A_dephased"):
        thresholds = summary["threshold_orders"][name]
        selected = _rows_by_field(rows, name)
        last = selected[-1]
        best = min(selected, key=lambda row: float(row["field_nmse_db"]))
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 达到复矢量 NMSE -20/-30/-40 dB 的最小阶数："
                f"`{_format_order(thresholds['field_nmse_-20_db'], summary['l_max'])}` / "
                f"`{_format_order(thresholds['field_nmse_-30_db'], summary['l_max'])}` / "
                f"`{_format_order(thresholds['field_nmse_-40_db'], summary['l_max'])}`；",
                f"- 在 Λ={summary['l_max']} 时：复矢量 NMSE {last['field_nmse_db']:.3f} dB，"
                f"幅度 NMSE {last['amplitude_nmse_db']:.3f} dB，强场幅度 MAE {last['strong_field_mae_db']:.3f} dB；",
                f"- 扫描内最佳复矢量 NMSE 为 {best['field_nmse_db']:.3f} dB（Λ={best['l_max']}）；",
                f"- 最坏端口复矢量 NMSE：{last['worst_port_nmse_db']:.3f} dB。",
                "",
            ]
        )
    lines.extend(
        [
            "物理阶数应以 `H_centered` 为准：若采用 -40 dB 复矢量 NMSE 标准，本数据建议 Λ=30；Λ≥34 后进入约 -69 dB 的积分/离散化底噪，不应继续解释成有效物理带宽。`A_dephased` 只用于判断解析阵元相位移除后的学习复杂度，不能把它的阶数重新解释为天线包围球的 kR。",
            "",
            "`A_dephased` 在宽松阈值下更早收敛（-20 dB 为 Λ=18），但严格尾部由边缘/角端口主导，Λ=40 仍未整体达到 -40 dB。这说明移除解析阵列相位降低了主体复杂度，但嵌入方向图中的互耦、跨孔径感应电流和逐端口参考原点仍保留高阶尾部。",
            "",
            "## 2. 物理先验",
            "",
            f"- 波长：{summary['physics']['wavelength_mm']:.6f} mm；",
            f"- 阵元中心 kR：{summary['physics']['center_kR']:.3f}；",
            f"- HFSS 实体包围盒半径：{summary['physics']['structure_radius_mm']:.3f} mm，对应 kR={summary['physics']['structure_kR']:.3f}；",
            f"- 论文 -40 dB 近似连续阶数：{summary['physics']['paper_minus40_continuous_order']:.3f}；",
            f"- 实际扫描：Λ=1..{summary['l_max']}。",
            "",
            "## 3. 数值验证",
            "",
            f"- 球面权重误差：{summary['basis_validation']['sphere_weight_error']:.3e}；",
            f"- 抽查 VSH 模态最大范数误差：{summary['basis_validation']['selected_mode_max_norm_error']:.3e}；",
            f"- X 与 r×X 最大内积：{summary['basis_validation']['selected_mode_max_x_rx_inner_product']:.3e}；",
            f"- 合成单模态最大系数恢复误差：{summary['basis_validation']['synthetic_mode_max_coefficient_error']:.3e}。",
            "",
            "## 4. 图像",
            "",
            "- `figures/vsh_order_spectrum_H_vs_A.png`：逐阶功率谱及端口 P10-P90；",
            "- `figures/vsh_cumulative_tail_energy.png`：随截断阶数变化的实际复矢量残差能量；",
            "- `figures/vsh_reconstruction_error_vs_order.png`：复矢量、幅度和强场重构误差；",
            "- `figures/vsh_port_order_heatmap.png`：16 端口逐阶热图；",
            "- `figures/vsh_lm_spectrum.png`：完整 (l,m) 频谱；",
            "- `figures/vsh_reconstruction_examples.png`：代表端口原场、重构场与误差。",
            "",
            "## 5. 数据和解释边界",
            "",
            "- `H_centered` 使用结构包围盒中心作为展开相位中心；",
            "- `A_dephased` 进一步移除每个端口绝对位置的解析 `exp(+jk q·r_n)`；",
            "- 频谱系数采用正交 VSH 分析系数，不宣称已完成绝对辐射功率标定；",
            "- 由于 1° 网格数值正交性有限，系数累计能量在高阶处存在微小闭合误差；阶数判据使用直接重构残差，不使用系数差值外推；",
            "- 阶谱功率使用 10log10，场幅度图使用 20log10；",
            "- 本结果不能直接决定 CNN 的 SSH 输入阶数，后者仍需单独做幅度模型消融。",
            "",
            "## 6. 机器可读产物",
            "",
            "- `summary.json`：主要结论、阈值阶数、源文件哈希与数值验证；",
            "- `vsh_order_metrics.csv`：每个 Λ 的重构指标；",
            "- `vsh_port_order_spectrum.csv`：每端口逐阶功率；",
            "- `vsh_coefficients.npz`：H/A 的 electric/magnetic 正交展开系数与 (l,m) 索引。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l-max", type=int, default=40)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.l_max < 1 or args.l_max > 80:
        raise ValueError("l-max must be in [1,80]")
    validation = run_basis_self_tests()
    print(json.dumps(validation, indent=2))
    if args.self_test_only:
        return

    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = load_full_sphere(args.input_root.resolve())
    theta_rad = np.deg2rad(data.theta_deg)
    phi_rad = np.deg2rad(data.phi_deg)
    wavelength = C0_M_S / FREQUENCY_HZ
    wavenumber = 2.0 * math.pi / wavelength
    structure_center = 0.5 * (STRUCTURE_BBOX_MIN_M + STRUCTURE_BBOX_MAX_M)
    direction = data.direction
    phase_center = np.exp(-1j * wavenumber * np.einsum("tjx,x->tj", direction, structure_center))
    centered_cartesian = data.field_cartesian * phase_center[:, :, None, None]
    port_positions = _port_positions_absolute()
    port_phase = np.exp(-1j * wavenumber * np.einsum("tjx,px->tjp", direction, port_positions))
    dephased_cartesian = data.field_cartesian * port_phase[:, :, None, :]
    fields = {
        "H_centered": cartesian_to_spherical_components(centered_cartesian, theta_rad, phi_rad),
        "A_dephased": cartesian_to_spherical_components(dephased_cartesian, theta_rad, phi_rad),
    }
    sin_theta = np.sin(theta_rad)
    dtheta = float(np.deg2rad(1.0))
    dphi = dtheta
    total_energy = {name: _weighted_square(value, sin_theta) * dtheta * dphi for name, value in fields.items()}
    total_energy_port = {
        name: np.sum(np.abs(value) ** 2 * sin_theta[:, None, None, None], axis=(0, 1, 2)) * dtheta * dphi
        for name, value in fields.items()
    }

    coefficients_m, coefficients_e, order_power, rows = audit_vsh(
        fields,
        theta_rad,
        phi_rad,
        args.l_max,
    )
    save_metrics_csv(output / "vsh_order_metrics.csv", rows)
    save_port_spectrum_csv(output / "vsh_port_order_spectrum.csv", order_power, total_energy_port)
    degrees, orders = mode_index(args.l_max)
    np.savez_compressed(
        output / "vsh_coefficients.npz",
        l=degrees,
        m=orders,
        h_c_m=flatten_coefficients(coefficients_m["H_centered"]),
        h_c_e=flatten_coefficients(coefficients_e["H_centered"]),
        a_c_m=flatten_coefficients(coefficients_m["A_dephased"]),
        a_c_e=flatten_coefficients(coefficients_e["A_dephased"]),
        basis_family=np.asarray(["X_lm", "r_cross_X_lm"]),
    )

    half_extent = 0.5 * (STRUCTURE_BBOX_MAX_M - STRUCTURE_BBOX_MIN_M)
    structure_radius = float(np.linalg.norm(half_extent))
    center_radius = math.hypot(1.5 * ROW_PITCH_M, 1.5 * COL_PITCH_M)
    structure_kr = wavenumber * structure_radius
    summary: dict[str, object] = {
        "schema": "v211-vsh-order-audit-1",
        "date": "2026-09-10",
        "l_max": args.l_max,
        "input_grid": {
            "theta_count": 180,
            "phi_count": 360,
            "direction_count": 64_800,
            "theta_deg": [0.5, 179.5, 1.0],
            "phi_deg": [-179.5, 179.5, 1.0],
            "port_count": 16,
        },
        "physics": {
            "frequency_hz": FREQUENCY_HZ,
            "wavelength_mm": wavelength * 1e3,
            "center_radius_mm": center_radius * 1e3,
            "center_kR": wavenumber * center_radius,
            "structure_center_mm": (structure_center * 1e3).tolist(),
            "structure_radius_mm": structure_radius * 1e3,
            "structure_kR": structure_kr,
            "paper_minus40_continuous_order": structure_kr + 1.8 * np.cbrt(structure_kr),
            "phase_convention": "project convention exp(+j k q dot r); center/dephase use exp(-j k q dot r)",
        },
        "basis": {
            "definition": "X_lm=-j/sqrt(l(l+1))*r_cross_grad(Y_lm); companion=r_cross_X_lm",
            "normalization": "orthonormal complex VSH with SciPy sph_harm_y Condon-Shortley phase",
            "quadrature": "1-degree cell-centred equiangular grid with sin(theta) dtheta dphi",
        },
        "basis_validation": validation,
        "threshold_orders": {name: _threshold_orders(rows, name) for name in fields},
        "best_field_nmse": {
            name: min(
                (
                    {"l_max": int(row["l_max"]), "field_nmse_db": float(row["field_nmse_db"])}
                    for row in _rows_by_field(rows, name)
                ),
                key=lambda item: item["field_nmse_db"],
            )
            for name in fields
        },
        "source_files": data.source_manifest,
        "final_metrics": {name: _rows_by_field(rows, name)[-1] for name in fields},
        "interpretation_boundary": {
            "physical_order_field": "H_centered",
            "complexity_diagnostic_field": "A_dephased",
            "not_used_for_amplitude_training": True,
            "does_not_select_ssh_encoding_order": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_order_spectrum(figures / "vsh_order_spectrum_H_vs_A.png", order_power, total_energy_port)
    plot_tail_energy(figures / "vsh_cumulative_tail_energy.png", rows)
    plot_reconstruction_metrics(figures / "vsh_reconstruction_error_vs_order.png", rows)
    plot_port_heatmap(figures / "vsh_port_order_heatmap.png", order_power, total_energy_port)
    plot_lm_spectrum(figures / "vsh_lm_spectrum.png", coefficients_m, coefficients_e, total_energy, args.l_max)

    example_order = summary["threshold_orders"]["H_centered"]["field_nmse_-30_db"]
    if example_order is None:
        example_order = args.l_max
    example_order = int(example_order)
    examples = {
        name: reconstruct_to_order(coefficients_m[name], coefficients_e[name], theta_rad, phi_rad, example_order)
        for name in fields
    }
    plot_reconstruction_examples(
        figures / "vsh_reconstruction_examples.png",
        fields,
        examples,
        data.theta_deg,
        data.phi_deg,
        example_order,
    )
    summary["reconstruction_example_order"] = example_order
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output / "REPORT.md", summary, rows)
    print(f"wrote VSH audit to {output}")


if __name__ == "__main__":
    main()
