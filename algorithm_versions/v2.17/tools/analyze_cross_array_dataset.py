from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import sph_harm_y
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.cross_array import LIGHT_SPEED_M_S, load_canonical  # noqa: E402


def nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    error = np.sum(weights[:, None] * np.abs(reference - prediction) ** 2)
    energy = np.sum(weights[:, None] * np.abs(reference) ** 2)
    return float(10.0 * np.log10(max(float(error / max(energy, 1e-300)), 1e-300)))


def q_nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(reference), 2048):
        stop = min(start + 2048, len(reference))
        ref = reference[start:stop]
        pred = prediction[start:stop]
        q_ref = np.einsum("qi,qj->qij", np.conj(ref), ref)
        q_pred = np.einsum("qi,qj->qij", np.conj(pred), pred)
        local = weights[start:stop, None, None]
        numerator += float(np.sum(local * np.abs(q_ref - q_pred) ** 2))
        denominator += float(np.sum(local * np.abs(q_ref) ** 2))
    return float(10.0 * np.log10(max(numerator / max(denominator, 1e-300), 1e-300)))


def shell_spectrum(theta_deg: np.ndarray, phi_deg: np.ndarray, field: np.ndarray, weights: np.ndarray, lmax: int = 12) -> list[float]:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    powers = []
    for degree in range(lmax + 1):
        shell = 0.0
        for order in range(-degree, degree + 1):
            basis = sph_harm_y(degree, order, theta, phi)
            coefficient = np.sum(weights * field * np.conj(basis))
            shell += float(abs(coefficient) ** 2)
        powers.append(shell)
    total = max(sum(powers), 1e-300)
    return [value / total for value in powers]


def analyse(path: Path) -> dict[str, object]:
    data = load_canonical(path)
    direction = data["direction"]
    g = data["g_ephi"]
    envelope = data["envelope_ephi"]
    weights = data["solid_angle_weights"]
    positions = data["positions_m"]
    s_matrix = data["s_parameters"]
    frequency = float(data["frequency_hz"])
    wavelength = LIGHT_SPEED_M_S / frequency
    scalar = np.sum(weights[:, None] * envelope) / envelope.shape[1]
    m0_envelope = np.full_like(envelope, scalar)
    common = np.mean(envelope, axis=1, keepdims=True)
    m1_envelope = np.repeat(common, envelope.shape[1], axis=1)
    k = 2 * math.pi / wavelength
    phase = np.exp(1j * k * np.einsum("qk,nk->qn", direction, positions))
    m0 = m0_envelope * phase
    m1 = m1_envelope * phase
    transverse = np.einsum("qk,qkn->qn", direction, data["h_cartesian"])
    field_norm = np.linalg.norm(data["h_cartesian"], axis=1)
    trans_rel = np.abs(transverse) / np.maximum(field_norm, 1e-30)
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    off = ~np.eye(len(positions), dtype=bool)
    coupling = np.abs(s_matrix)[off]
    distance_flat = distance[off]
    correlation = spearmanr(distance_flat, coupling).statistic
    residual = envelope - common
    covariance = np.abs(residual.conj().T @ (weights[:, None] * residual))
    covariance_off = covariance[off]
    cov_distance_correlation = spearmanr(distance_flat, covariance_off).statistic
    array_span = np.ptp(positions, axis=0)
    unique_count = len(np.unique(np.round(direction, 12), axis=0))
    ephi_power = float(np.sum(weights[:, None] * np.abs(data["g_ephi"]) ** 2))
    etheta_power = float(np.sum(weights[:, None] * np.abs(data["g_etheta"]) ** 2))
    spectrum = shell_spectrum(data["theta_deg"], data["phi_deg"], common[:, 0], weights)
    return {
        "array_id": str(data["array_id"]),
        "canonical": str(path.resolve()),
        "logical_shape": data["logical_shape"].tolist(),
        "directions": len(direction),
        "unique_directions": unique_count,
        "duplicate_directions": len(direction) - unique_count,
        "finite": bool(all(np.isfinite(value).all() for value in (direction, g, data["g_etheta"], s_matrix))),
        "source_grid_counts": {name: int(np.sum(data["source_grid"] == name)) for name in np.unique(data["source_grid"])},
        "solid_angle_weight_sum": float(np.sum(weights)),
        "quality_mask_fraction": float(np.mean(data["quality_mask"])),
        "transverse_relative": {"p50": float(np.percentile(trans_rel, 50)), "p95": float(np.percentile(trans_rel, 95)), "max": float(np.max(trans_rel))},
        "aperture": {"span_m": array_span.tolist(), "max_D_over_lambda": float(np.max(array_span) / wavelength)},
        "s_parameters": {
            "reciprocity_relative": float(np.linalg.norm(s_matrix - s_matrix.T) / max(np.linalg.norm(s_matrix), 1e-30)),
            "diagonal_db_p50": float(np.median(20 * np.log10(np.maximum(np.abs(np.diag(s_matrix)), 1e-30)))),
            "off_diagonal_db_p50": float(np.median(20 * np.log10(np.maximum(coupling, 1e-30)))),
            "off_diagonal_db_p95": float(np.percentile(20 * np.log10(np.maximum(coupling, 1e-30)), 95)),
            "distance_coupling_spearman": float(correlation),
        },
        "m0_m1_m2_ladder": {
            "m0_isotropic_array_factor_operator_nmse_db": nmse_db(g, m0, weights),
            "m0_isotropic_array_factor_Q_nmse_db": q_nmse_db(g, m0, weights),
            "m1_common_eep_operator_nmse_db": nmse_db(g, m1, weights),
            "m1_common_eep_Q_nmse_db": q_nmse_db(g, m1, weights),
            "m2_full_eep_operator_nmse_db": -300.0,
            "m2_definition": "canonical full EEP reference; numerical identity floor",
        },
        "element_nonuniformity": {
            "weighted_residual_fraction": float(np.sum(weights[:, None] * np.abs(residual) ** 2) / max(np.sum(weights[:, None] * np.abs(envelope) ** 2), 1e-300)),
            "residual_covariance_distance_spearman": float(cov_distance_correlation),
        },
        "polarization": {"etheta_to_ephi_power_db": float(10 * np.log10(max(etheta_power / max(ephi_power, 1e-300), 1e-300)))},
        "common_envelope_ssh_shell_fraction_l0_to_l12": spectrum,
        "common_envelope_ssh_high_tail_l9_to_l12": float(sum(spectrum[9:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.canonical)
    output = args.output or args.canonical.with_name("initial_physics_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
