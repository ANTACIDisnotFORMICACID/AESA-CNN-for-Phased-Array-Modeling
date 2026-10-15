import numpy as np

from calibration_v26 import PhaseInvariantCalibrator, build_d_optimal_design, normalize_global_phase, simulate_measurements
from hfss_physics_v26 import HfssPatternData


def test_calibration_design_and_phase_invariance():
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    assert design.scalar_response.shape == (8, 4)
    assert design.commands.shape == (4, 4)
    assert design.condition_number <= 20.0
    truth = np.array([1.05 * np.exp(0.1j), 0.91 * np.exp(-0.2j), 1.08 * np.exp(0.3j), 0.88 * np.exp(-0.1j)])
    measurement = simulate_measurements(design, truth, snr_db=np.inf, global_phase_rad=1.71)
    estimate = PhaseInvariantCalibrator(design).calibrate(measurement)
    recovered = estimate.calibration_iq[0, 0, 0] + 1j * estimate.calibration_iq[0, 1, 0]
    np.testing.assert_allclose(recovered, normalize_global_phase(truth), rtol=2e-6, atol=2e-6)
    assert estimate.confidence[0, 0] > 0.99


def test_calibration_noisy_holdout_nmse_gate():
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    calibrator = PhaseInvariantCalibrator(design)
    truth = np.array([1.0, 0.92 * np.exp(0.12j), 1.07 * np.exp(-0.18j), 0.96 * np.exp(0.08j)])
    measurement = simulate_measurements(design, truth, snr_db=40.0, global_phase_rad=-2.2, seed=91)
    estimate = calibrator.calibrate(measurement)
    recovered = estimate.calibration_iq[0, 0, 0] + 1j * estimate.calibration_iq[0, 1, 0]
    holdout = np.array([[65.0, -25.0], [75.0, 5.0], [95.0, -15.0], [115.0, 25.0]])
    nmse = calibrator.holdout_nmse_db(data, recovered, truth, holdout)
    assert nmse[0] <= -20.0


def test_single_failed_port_remains_finite():
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    truth = np.array([1.0, 0.0, 0.9 * np.exp(0.1j), 1.1 * np.exp(-0.1j)])
    health = np.array([1.0, 0.0, 1.0, 1.0])
    measurement = simulate_measurements(design, truth, snr_db=35.0)
    estimate = PhaseInvariantCalibrator(design).calibrate(measurement, health)
    assert np.isfinite(estimate.calibration_iq).all()
    assert np.isfinite(estimate.confidence).all()

