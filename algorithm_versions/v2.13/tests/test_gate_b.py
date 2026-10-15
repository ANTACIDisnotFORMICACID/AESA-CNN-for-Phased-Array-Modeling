from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from v212.codebook import build_baseline_codebook
from v213.gate_b import (
    CANONICAL_SCHEMA,
    GateBMeasurementPackage,
    import_long_csv,
    pooled_relative_noise_covariance,
    recover_gate_b_rank1,
    synthetic_power_repeats,
    validate_gate_b_package,
)


def _package(*, source_kind: str, independent: bool, nonidentical: bool) -> GateBMeasurementPackage:
    codebook = build_baseline_codebook()
    command_indices = np.arange(80, dtype=np.int64)
    commands = codebook.weights.numpy()[command_indices]
    command_iq = np.stack((commands.real, commands.imag), axis=1).reshape(80, 2, 4, 4)
    direction = np.asarray([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]], dtype=np.float64)
    powers = np.ones((2, 80, 3), dtype=np.float64)
    if nonidentical:
        powers[:, :, 1] *= 1.001
        powers[:, :, 2] *= 0.999
    return GateBMeasurementPackage(
        manifest={
            "schema": CANONICAL_SCHEMA,
            "dataset_id": "unit-test",
            "source_kind": source_kind,
            "independent_repeats": independent,
            "frequency_hz": 47e9,
            "receiver_channel": "power",
            "power_unit": "W",
            "array_id": "array-1",
            "instrument": "receiver-1",
            "calibration_reference_plane": "ports",
            "calibration_id": "cal-1",
            "source_files": ["unit-test.csv"],
            "generated_from_single_complex_g": source_kind == "synthetic",
            "not_for_training": True,
            "external_test_used": False,
        },
        direction=direction,
        theta_deg=np.asarray([90.0, 90.0]),
        phi_deg=np.asarray([0.0, np.rad2deg(np.arctan2(0.6, 0.8))]),
        command_indices=command_indices,
        command_iq=command_iq,
        power_repeats=powers,
        repeat_id=np.asarray(["r0", "r1", "r2"]),
        acquired_at_utc=np.asarray(
            ["2026-09-12T00:00:00Z", "2026-09-12T00:01:00Z", "2026-09-12T00:02:00Z"]
        ),
        session_id=np.asarray(["s0", "s1", "s2"]),
    )


def test_identical_synthetic_repeats_are_blocked() -> None:
    report = validate_gate_b_package(
        _package(source_kind="synthetic", independent=False, nonidentical=False),
        build_baseline_codebook(),
    )
    assert report["valid"]
    assert report["status"] == "blocked_not_independent_nonidentical_repeats"
    assert report["repeat_evidence"]["bit_identical_repeat_pairs"] == 3
    assert not report["hardware_gate_b_eligible"]


def test_nonidentical_hardware_repeats_are_eligible() -> None:
    report = validate_gate_b_package(
        _package(source_kind="hardware_measurement", independent=True, nonidentical=True),
        build_baseline_codebook(),
    )
    assert report["valid"]
    assert report["status"] == "eligible_hardware_repeats"
    assert report["repeat_evidence"]["all_repeat_sweeps_nonidentical"]


def test_covariance_uses_train_columns_and_is_psd() -> None:
    package = _package(source_kind="hardware_measurement", independent=True, nonidentical=True)
    report = pooled_relative_noise_covariance(package.power_repeats, np.arange(64))
    covariance = report["pooled_relative_covariance"]
    assert covariance.shape == (64, 64)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12
    assert report["per_direction_diagonal_variance"].shape == (2, 64)


def test_long_csv_import_is_rectangular_and_keeps_test_sealed() -> None:
    codebook = build_baseline_codebook()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "import.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "v213-gate-b-import-manifest-1",
                    "dataset_id": "tiny-import",
                    "source_kind": "hardware_measurement",
                    "independent_repeats": True,
                    "frequency_hz": 47e9,
                    "receiver_channel": "power",
                    "power_unit": "W",
                    "array_id": "array-1",
                    "instrument": "receiver-1",
                    "calibration_reference_plane": "ports",
                    "calibration_id": "cal-1",
                    "source_files": ["tiny.csv"],
                    "generated_from_single_complex_g": False,
                    "not_for_training": True,
                    "external_test_used": False,
                }
            ),
            encoding="utf-8",
        )
        table = root / "measurements.csv"
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "direction_id", "theta_deg", "phi_deg", "qx", "qy", "qz",
                    "command_id", "repeat_id", "power", "acquired_at_utc", "session_id",
                ),
            )
            writer.writeheader()
            for repeat in range(3):
                for command in range(80):
                    writer.writerow(
                        {
                            "direction_id": 0,
                            "theta_deg": 90,
                            "phi_deg": 0,
                            "qx": 1,
                            "qy": 0,
                            "qz": 0,
                            "command_id": command,
                            "repeat_id": f"r{repeat}",
                            "power": 1.0 + 1e-3 * repeat,
                            "acquired_at_utc": f"2026-09-12T00:0{repeat}:00Z",
                            "session_id": f"s{repeat}",
                        }
                    )
        package = import_long_csv(table, manifest, root / "canonical", codebook)
        assert package.power_repeats.shape == (1, 80, 3)
        assert np.array_equal(package.command_indices, np.arange(80))
        assert package.manifest["external_test_used"] is False


def test_synthetic_noise_and_rank1_wrapper_are_finite() -> None:
    torch.manual_seed(5)
    codebook = build_baseline_codebook()
    g = torch.complex(torch.randn(2, 16), torch.randn(2, 16))
    commands = codebook.weights[:80]
    clean = torch.einsum("qn,cn->qc", g, commands).abs().square()
    repeats = synthetic_power_repeats(
        clean,
        noise_model="multiplicative_lognormal_power",
        snr_db=40.0,
        repeats=3,
        seed=6,
    )
    recovery = recover_gate_b_rank1(
        repeats,
        codebook,
        truth_g=g,
        clean_validation_power=clean[:, 64:80],
        steps=5,
        batch_size=2,
        seed=7,
    )
    assert recovery.q_hat.shape == (2, 16, 16)
    assert torch.isfinite(recovery.jacobian_condition_noise_weighted).all()
    assert recovery.clean_power_error_numerator is not None
