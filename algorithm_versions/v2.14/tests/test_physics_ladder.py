import json
from pathlib import Path
import tempfile

import numpy as np

from v210.contracts import ArrayGeometry
from v210.io.hfss_ffd import load_hfss_ffd_basis, parse_ffd
from v214.physics_ladder import (
    deembed,
    fit_element_gains,
    fit_low_rank_residual,
    fit_phase_centers,
    fit_s_guided_coupling,
    geometric_phase,
    reembed,
    ssh_basis,
)
from v214.physics_metrics import operator_metrics


ROOT = Path(__file__).resolve().parents[1]


def _directions(count=120):
    generator = np.random.default_rng(21401)
    value = generator.normal(size=(count, 3))
    value[:, 0] = np.abs(value[:, 0]) + 0.05
    return value / np.linalg.norm(value, axis=1, keepdims=True)


def test_ffd_components_units_port_order_and_coordinate_rotation():
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geometry = ArrayGeometry(rotation_algorithm_to_hfss=rotation)
    with tempfile.TemporaryDirectory(prefix="v214_ffd_") as directory:
        root = Path(directory)
        mapping = {}
        for port, name in enumerate(geometry.port_names, start=1):
            # theta=90, phi=0: e_theta=[0,0,-1], e_phi=[0,1,0].
            path = root / f"port_{port}.ffd"
            path.write_text(f"90 90 1\n0 0 1\nFrequencies 1\nFrequency 47e9\n0.5 -0.25 {port}.0 {-port / 10.0}\n", encoding="utf-8")
            mapping[name] = path
        theta, phi, etheta, ephi = parse_ffd(mapping[geometry.port_names[0]])
        np.testing.assert_array_equal(theta, [90.0])
        np.testing.assert_array_equal(phi, [0.0])
        np.testing.assert_allclose(etheta, [0.5 - 0.25j])
        np.testing.assert_allclose(ephi, [1.0 - 0.1j])
        loaded = load_hfss_ffd_basis(mapping, geometry)
        field = loaded.operator_iq[0, :, 0].reshape(3, 16) + 1j * loaded.operator_iq[0, :, 1].reshape(3, 16)
        expected_port = np.arange(1.0, 17.0) * (1.0 - 0.1j)
        np.testing.assert_allclose(field[0], expected_port, atol=1e-14)
        np.testing.assert_allclose(field[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(field[2], -0.5 + 0.25j, atol=1e-14)
        np.testing.assert_allclose(loaded.direction_algorithm[0], [0.0, -1.0, 0.0], atol=1e-14)


def test_geometric_phase_remove_restore_closes():
    generator = np.random.default_rng(21402)
    direction = _directions(50)
    positions = generator.normal(scale=0.003, size=(16, 3))
    field = generator.normal(size=(50, 16)) + 1j * generator.normal(size=(50, 16))
    restored = reembed(deembed(field, direction, positions, 47e9), direction, positions, 47e9)
    np.testing.assert_allclose(restored, field, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(np.abs(geometric_phase(direction, positions, 47e9)), 1.0, atol=1e-15)


def test_rank_one_operator_metric_is_common_phase_gauge_invariant():
    generator = np.random.default_rng(21403)
    target = generator.normal(size=(40, 16)) + 1j * generator.normal(size=(40, 16))
    phase = np.exp(1j * generator.uniform(-np.pi, np.pi, size=40))
    metrics = operator_metrics(target * phase[:, None], target)
    assert metrics["nmse_db"] < -250.0
    assert metrics["per_direction_relative_error_max"] < 1e-28


def test_element_gain_and_s_guided_coupling_are_recoverable():
    generator = np.random.default_rng(21404)
    base = generator.normal(size=(180, 16)) + 1j * generator.normal(size=(180, 16))
    gain = 1.0 + 0.08 * (generator.normal(size=16) + 1j * generator.normal(size=16))
    target_gain = base * gain[None, :]
    prediction, _, _ = fit_element_gains(base, target_gain, np.ones(len(base)))
    np.testing.assert_allclose(prediction, target_gain, atol=1e-10, rtol=1e-10)

    s_matrix = 0.03 * (generator.normal(size=(16, 16)) + 1j * generator.normal(size=(16, 16)))
    np.fill_diagonal(s_matrix, 0.1)
    off = s_matrix.copy()
    np.fill_diagonal(off, 0.0)
    off /= np.linalg.norm(off)
    alpha = np.asarray([0.2 - 0.1j, -0.05 + 0.08j])
    target_s = base + alpha[0] * (base @ off) + alpha[1] * (base @ (off @ off))
    prediction_s, fitted = fit_s_guided_coupling(base, target_s, s_matrix, np.ones(len(base)))
    np.testing.assert_allclose(fitted, alpha, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(prediction_s, target_s, atol=2e-8, rtol=2e-8)


def test_bounded_zero_gauge_phase_centers_are_recoverable():
    generator = np.random.default_rng(21406)
    direction = _directions(240)
    frequency = 47e9
    wavelength = 299_792_458.0 / frequency
    delta = generator.normal(scale=0.025 * wavelength, size=(16, 3))
    delta -= delta.mean(axis=0, keepdims=True)
    base = generator.normal(size=(240, 16)) + 1j * generator.normal(size=(240, 16))
    k = 2.0 * np.pi / wavelength
    target = base * np.exp(1j * k * direction @ delta.T)
    prediction, fitted = fit_phase_centers(base, target, direction, np.ones(len(direction)), frequency, max_iter=80)
    relative = np.sum(np.abs(prediction - target) ** 2) / np.sum(np.abs(target) ** 2)
    assert relative < 1e-8
    np.testing.assert_allclose(fitted.mean(axis=0), 0.0, atol=1e-12)
    assert np.max(np.abs(fitted)) <= wavelength / 4.0 + 1e-12


def test_low_rank_residual_must_be_ssh_predictable():
    generator = np.random.default_rng(21405)
    direction = _directions(180)
    basis = ssh_basis(direction, 3)
    coefficient = generator.normal(size=(basis.shape[1], 2)) + 1j * generator.normal(size=(basis.shape[1], 2))
    aperture = generator.normal(size=(2, 16)) + 1j * generator.normal(size=(2, 16))
    base = np.zeros((len(direction), 16), dtype=np.complex128)
    target = (basis @ coefficient) @ aperture
    prediction, state = fit_low_rank_residual(base, target, direction, np.ones(len(direction)), 2)
    relative = np.sum(np.abs(prediction - target) ** 2) / np.sum(np.abs(target) ** 2)
    # The production fit deliberately uses a small ridge; require practical,
    # rather than bit-exact, recovery of an SSH-predictable rank-two residual.
    assert relative < 1e-8
    assert state["coefficient"].shape[1] == 2


def test_completed_physics_artifacts_close_source_and_gate_contracts():
    source = json.loads((ROOT / "artifacts/physics_sources_20260913/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    ladder = json.loads((ROOT / "artifacts/physics_residual_ladder_20260913/LADDER_SUMMARY.json").read_text(encoding="utf-8"))
    assert source["train_count"] == 22600 and source["validation_count"] == 4900
    assert source["test_values_exported"] is False and source["external_test_used"] is False
    assert source["project_unchanged_on_disk"] is True
    assert source["diagnostics"]["ephi_audit_complex_nmse_db"] < -100.0
    assert source["diagnostics"]["gate_a_direct_complex_nmse_db"] < -120.0
    assert ladder["status"] == "physics_ladder_complete"
    assert ladder["external_test_used"] is False
    assert ladder["complex_labels_used_for_cnn_training"] is False
    assert ladder["selection"]["folds"] == 3
    assert ladder["selection"]["accepted_stages"] == []
