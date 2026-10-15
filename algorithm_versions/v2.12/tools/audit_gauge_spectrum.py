"""Audit the physical SSH spectrum of dephased gauge-invariant Q-tilde."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.contracts import ArrayGeometry  # noqa: E402
from v212.audits import dephase_q_matrix, gauge_invariant_spectrum, rank_one_q_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--degree", type=int, default=12)
    args = parser.parse_args()
    with np.load(args.complex_audit, allow_pickle=False) as archive:
        direction = torch.from_numpy(
            np.asarray(archive["direction"], dtype=np.float64).copy()
        )
        g = torch.from_numpy(np.asarray(archive["g"], dtype=np.complex128).copy())
        solid_angle = torch.from_numpy(
            np.asarray(archive["solid_angle_weights"], dtype=np.float64).copy()
        )
    geometry = ArrayGeometry()
    positions = torch.as_tensor(geometry.positions_m.reshape(16, 3), dtype=torch.float64)
    wavenumber = 2.0 * math.pi / geometry.wavelength_m
    q_matrix = rank_one_q_matrix(g)
    q_tilde = dephase_q_matrix(q_matrix, direction, positions, wavenumber)
    spectrum = gauge_invariant_spectrum(
        q_tilde,
        direction,
        degree=args.degree,
        solid_angle_weights=solid_angle,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "gauge_invariant_q_spectrum.npz",
        degree_energy=spectrum.degree_energy.numpy(),
        coefficient_real=spectrum.coefficients.real.numpy(),
        coefficient_imag=spectrum.coefficients.imag.numpy(),
    )
    report = {
        "schema": "v212-gauge-invariant-q-spectrum-1",
        "quantity": "Q_tilde=D_r Q D_r^H=a^*a^T",
        "gauge_invariant": True,
        "latent_a_spectral_regularization_used": False,
        "audit_degree": args.degree,
        "cumulative_energy_fraction": {
            str(key): value for key, value in spectrum.cumulative_fraction.items()
        },
        "degree_energy": spectrum.degree_energy.tolist(),
        "normalized_gradient_energy": spectrum.normalized_gradient_energy,
    }
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
