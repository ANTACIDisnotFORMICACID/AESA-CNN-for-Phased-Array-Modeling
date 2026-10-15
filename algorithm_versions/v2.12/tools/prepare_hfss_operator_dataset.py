"""Create the strict V2.12 96-command amplitude dataset from HFSS complex audit data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.coordinates import spherical_basis  # noqa: E402
from v210.io.amplitude_dataset import (  # noqa: E402
    AmplitudeDataset,
    AmplitudeManifest,
    save_amplitude_dataset,
)
from v210.io.amplitude_split import (  # noqa: E402
    AmplitudeDataSplit,
    amplitude_dataset_fingerprint,
    save_amplitude_split,
)
from v210.training.datasets import angular_block_split  # noqa: E402
from v212.codebook import build_baseline_codebook  # noqa: E402


def _cell_widths(values: np.ndarray, lower: float, upper: float) -> dict[float, float]:
    unique = np.unique(np.round(values, decimals=12))
    if len(unique) == 1:
        return {float(unique[0]): upper - lower}
    boundaries = np.empty(len(unique) + 1, dtype=np.float64)
    boundaries[1:-1] = 0.5 * (unique[:-1] + unique[1:])
    boundaries[0] = max(lower, unique[0] - 0.5 * (unique[1] - unique[0]))
    boundaries[-1] = min(upper, unique[-1] + 0.5 * (unique[-1] - unique[-2]))
    return {
        float(value): float(boundaries[index + 1] - boundaries[index])
        for index, value in enumerate(unique)
    }


def _grid_solid_angle_weights(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Physical dOmega cell weights, separate from equal empirical loss weights."""
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    theta_key = np.round(theta, decimals=12)
    phi_key = np.round(phi, decimals=12)
    theta_width = _cell_widths(theta, 0.0, np.pi)
    phi_width = _cell_widths(phi, -np.pi, np.pi)
    unique_theta = np.unique(theta_key)
    theta_boundaries: dict[float, tuple[float, float]] = {}
    for value in unique_theta:
        width = theta_width[float(value)]
        theta_boundaries[float(value)] = (
            max(0.0, float(value) - 0.5 * width),
            min(np.pi, float(value) + 0.5 * width),
        )
    area = np.asarray(
        [
            (np.cos(theta_boundaries[float(t)][0]) - np.cos(theta_boundaries[float(t)][1]))
            * phi_width[float(p)]
            for t, p in zip(theta_key, phi_key)
        ],
        dtype=np.float64,
    )
    return area / area.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=21296)
    parser.add_argument("--write-complex-audit", action="store_true")
    args = parser.parse_args()

    q_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    theta_parts: list[np.ndarray] = []
    phi_parts: list[np.ndarray] = []
    for source in args.source:
        with np.load(source, allow_pickle=False) as archive:
            q = np.asarray(archive["q"], dtype=np.float64)
            theta = np.asarray(archive["theta"], dtype=np.float64)
            phi = np.asarray(archive["phi"], dtype=np.float64)
            h = np.asarray(archive["h"], dtype=np.complex128)
        if h.shape != (len(q), 3, 16):
            raise ValueError(f"{source} h must have shape [Q,3,16]")
        _, e_phi = spherical_basis(theta, phi)
        g = np.einsum("qx,qxn->qn", e_phi, h, optimize=True)
        front = q[:, 0] > 0.0
        q_parts.append(q[front])
        g_parts.append(g[front])
        theta_parts.append(theta[front])
        phi_parts.append(phi[front])

    direction = np.concatenate(q_parts)
    g = np.concatenate(g_parts)
    theta = np.concatenate(theta_parts)
    phi = np.concatenate(phi_parts)
    raw_solid_angle = _grid_solid_angle_weights(theta, phi)
    _, unique_index, inverse = np.unique(
        np.round(direction, decimals=10), axis=0, return_index=True, return_inverse=True
    )
    unique_index = np.sort(unique_index)
    group_weight = np.bincount(inverse, weights=raw_solid_angle)
    solid_angle_weights = group_weight[inverse[unique_index]]
    solid_angle_weights = solid_angle_weights / solid_angle_weights.sum()
    direction, g, theta, phi = (
        value[unique_index] for value in (direction, g, theta, phi)
    )
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

    codebook = build_baseline_codebook(args.seed)
    commands = codebook.weights.numpy()
    amplitude_qc = np.abs(np.einsum("qn,cn->qc", g, commands, optimize=True)).astype(np.float32)
    command_iq = np.stack((commands.real, commands.imag), axis=1).reshape(96, 2, 4, 4).astype(np.float32)
    dataset = AmplitudeDataset(
        direction=direction.astype(np.float32),
        command_iq=command_iq,
        amplitude=amplitude_qc.T,
        command_family=np.asarray(codebook.kinds, dtype="U32"),
        theta_deg=theta,
        phi_deg=phi,
        manifest=AmplitudeManifest(
            schema_version="v2.8-amplitude-1",
            source_type="hfss_amplitude",
            frequency_hz=47.0e9,
            receiver_channel="ephi",
            amplitude_unit="V/m",
            reference_distance_m=1.0,
            excitation_reference="known unit-L2 4x4 complex command at HFSS port planes",
            inactive_port_termination="50 ohm matched",
            calibration_reference_plane="HFSS lumped-port reference planes",
            array_id="hfss-47ghz-4x4",
            session_id="v212-hfss-ephi-96-command",
            phase_available=False,
            provenance={
                "conversion": "HFSS complex Ephi was coherently synthesized, then only abs(.) entered training",
                "source_files": [str(path.resolve()) for path in args.source],
                "hemisphere": "strict q_x > 0",
                "raw_complex_data_available_to_training": False,
                "measured_phase_exported_to_training_dataset": False,
                "operator_iq_used": False,
                "s_parameters_used": False,
                "command_protocol": "v212-measurement-codebook-1",
                "repeat_artifact": "three identical synthetic HFSS repeats; zero-noise baseline only",
            },
        ),
    )
    dataset_dir = args.output_root / "amplitude_dataset"
    split_dir = args.output_root / "split"
    audit_dir = args.output_root / "audit"
    save_amplitude_dataset(dataset, dataset_dir)
    query_split = angular_block_split(theta, phi, seed=args.seed)
    split = AmplitudeDataSplit(
        schema_version="v2.8-amplitude-split-1",
        dataset_fingerprint=amplitude_dataset_fingerprint(dataset_dir),
        seed=args.seed,
        query_count=len(direction),
        command_count=96,
        theta_block_deg=10.0,
        phi_block_deg=10.0,
        validation_fraction=0.15,
        test_fraction=0.15,
        query_train=query_split.train,
        query_validation=query_split.validation,
        query_test=query_split.test,
        command_train=np.asarray(codebook.splits["train"], dtype=np.int64),
        command_validation=np.asarray(codebook.splits["validation"], dtype=np.int64),
        command_test=np.asarray(codebook.splits["test"], dtype=np.int64),
        audit={
            "query_indices_disjoint_and_complete": True,
            "command_indices_disjoint_and_complete": True,
            "shared_angular_blocks_train_validation": 0,
            "shared_angular_blocks_train_test": 0,
            "shared_angular_blocks_validation_test": 0,
            "duplicate_physical_directions": 0,
            "split_before_training": True,
        },
    )
    save_amplitude_split(split, split_dir)
    codebook.save(args.output_root / "codebook.json")
    audit_dir.mkdir(parents=True, exist_ok=True)
    np.save(audit_dir / "power_repeats.npy", np.repeat(amplitude_qc[:, :, None] ** 2, 3, axis=2))
    np.save(audit_dir / "solid_angle_weights.npy", solid_angle_weights.astype(np.float64))
    if args.write_complex_audit:
        np.savez(
            audit_dir / "hfss_complex_operator_audit_only.npz",
            direction=direction.astype(np.float32),
            g=g.astype(np.complex64),
            query_train=query_split.train,
            query_validation=query_split.validation,
            query_test=query_split.test,
            solid_angle_weights=solid_angle_weights.astype(np.float64),
        )
    summary = {
        "schema": "v212-prepared-hfss-operator-data-1",
        "query_count": len(direction),
        "command_count": 96,
        "training_labels": "single-polarization amplitude only",
        "complex_audit_written": bool(args.write_complex_audit),
        "repeat_measurements": {
            "count": 3,
            "kind": "synthetic_identical_zero_noise_baseline",
            "represents_hardware_repeatability": False,
        },
        "dataset_manifest": asdict(dataset.manifest),
    }
    (args.output_root / "PREPARATION.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
