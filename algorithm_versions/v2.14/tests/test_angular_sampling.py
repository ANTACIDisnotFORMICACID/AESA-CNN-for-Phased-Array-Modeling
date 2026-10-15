import json
from pathlib import Path

import numpy as np

from v214.angular_sampling import (
    angular_block_ids,
    deterministic_spatial_folds,
    load_amplitude_label_package,
    select_per_block_maximin,
)


ROOT = Path(__file__).resolve().parents[1]


def test_spatial_folds_are_deterministic_and_keep_blocks_whole():
    theta = np.asarray([0.5, 1.5, 10.5, 11.5, 20.5, 21.5])
    phi = np.asarray([-179.5, -178.5, -169.5, -168.5, -159.5, -158.5])
    first = deterministic_spatial_folds(theta, phi, 3)
    second = deterministic_spatial_folds(theta, phi, 3)
    np.testing.assert_array_equal(first, second)
    blocks = angular_block_ids(theta, phi)
    for block in np.unique(blocks, axis=0):
        member = np.all(blocks == block, axis=1)
        assert len(np.unique(first[member])) == 1


def test_maximin_is_repeatable_unique_and_geometry_only():
    theta = np.deg2rad(np.arange(1.0, 13.0))
    direction = np.stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=1)
    blocks = np.zeros((len(direction), 2), dtype=np.int64)
    training = np.asarray([[1.0, 0.0, 0.0]])
    first = select_per_block_maximin(direction, blocks, {(0, 0)}, training, per_block=3)
    second = select_per_block_maximin(direction, blocks, {(0, 0)}, training, per_block=3)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == len(np.unique(first[0])) == 3
    assert len(first[1]) == 9


def test_amplitude_loader_rejects_complex_audit_package():
    label_dir = ROOT / "artifacts/angular_labels_integer_grid_20260913"
    acquisition = load_amplitude_label_package(label_dir / "acquisition_train.npz")
    assert acquisition["amplitude"].shape == (64, 490)
    try:
        load_amplitude_label_package(label_dir / "complex_audit_only.npz")
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("complex audit package was accepted as a training label package")


def test_integer_grid_label_packages_are_sealed_and_nonoverlapping():
    label_dir = ROOT / "artifacts/angular_labels_integer_grid_20260913"
    geometry = json.loads((label_dir / "GEOMETRY_SELECTION.json").read_text(encoding="utf-8"))
    manifest = json.loads((label_dir / "LABEL_MANIFEST.json").read_text(encoding="utf-8"))
    acquisition = load_amplitude_label_package(label_dir / "acquisition_train.npz")
    probe = load_amplitude_label_package(label_dir / "coverage_probe.npz")
    assert geometry["candidate_count"] == 32041
    assert geometry["candidate_role_counts"] == {"train": 22301, "validation": 4870, "test": 4870}
    assert geometry["validation_block_count"] == 49
    assert geometry["acquisition_count"] == 490 and geometry["coverage_probe_count"] == 4380
    assert manifest["external_test_blocks_excluded"] is True
    assert manifest["external_test_labels_used"] is False
    assert manifest["test_command_indices_included"] is False
    assert set(acquisition) == {"direction", "theta_deg", "phi_deg", "command_indices", "command_iq", "amplitude"}
    assert set(probe) == {"direction", "theta_deg", "phi_deg", "command_indices", "command_iq", "amplitude"}
    assert not ({tuple(v) for v in np.round(acquisition["direction"], 8)} & {tuple(v) for v in np.round(probe["direction"], 8)})


def test_completed_three_arm_coverage_validation_passes_all_gates():
    path = ROOT / "artifacts/angular_labels_integer_grid_20260913/coverage_validation/NEW_LABEL_COVERAGE_SUMMARY.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "new_angular_labels_pass"
    assert report["external_test_used"] is False
    assert report["cnn_structure_changed"] is False
    assert report["budget"] == {
        "control_training_directions": 22600,
        "exchange_training_directions": 22600,
        "plus_training_directions": 23090,
        "acquisition_directions": 490,
        "coverage_probe_directions": 4380,
        "old_outer_validation_directions": 4900,
    }
    assert len(report["paired_results"]) == 3
    assert all(report["acceptance"].values())

