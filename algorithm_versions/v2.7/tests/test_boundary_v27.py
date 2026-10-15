import numpy as np

from boundary_v27 import (
    BoundaryIndex,
    classify_boundary,
    global_phase_aligned_weight_change,
    hpbw_aware_margin_db,
    tracking_pair,
)


def test_four_levels_are_nested():
    assert classify_boundary(-0.4, 4.0, 3.1, 0.96).robust_unambiguous
    value = classify_boundary(-0.4, 6.0, 4.0, 1.0)
    assert value.gain_reachable and not value.pointing_reachable and not value.strict_unambiguous
    assert not classify_boundary(-0.6, 0.0, 9.0, 1.0).gain_reachable


def test_hpbw_diagnostic_removes_connected_main_lobe():
    power = np.zeros((7, 7), dtype=np.float64)
    power[3, 3] = 10.0
    power[3, 4] = 6.0
    power[0, 0] = 2.0
    margin, count = hpbw_aware_margin_db(power, 10.0)
    assert count == 2
    assert np.isclose(margin, 10.0 * np.log10(5.0))


def test_global_phase_is_not_a_weight_jump():
    a = np.asarray([1.0, 1j, -1.0, -1j])
    b = a * np.exp(1j * 1.234)
    assert global_phase_aligned_weight_change(a, b) < 1e-12
    ok, jump, change = tracking_pair((90.0, 0.0), (91.0, 0.0), a, b)
    assert ok and jump <= 1.01 and change < 1e-12


def test_query_contract_and_nearest_reachable():
    shape = (1, 2, 2)
    arrays = {
        "theta_deg": np.asarray([60.0, 61.0]),
        "phi_deg": np.asarray([-1.0, 0.0]),
        "state_classes": np.asarray(["baseline"]),
        "target_to_peak_db": np.zeros(shape),
        "pointing_error_deg": np.zeros(shape),
        "strict_margin_db": np.full(shape, 4.0),
        "hpbw_aware_margin_db": np.full(shape, 4.5),
        "robust_pass_fraction": np.ones(shape),
        "continuous_tracking": np.ones(shape, dtype=bool),
    }
    result = BoundaryIndex(arrays).query_boundary((60.2, -0.8), "baseline")
    assert result.flags.robust_unambiguous
    assert result.nearest_reachable_hfss_deg is not None
    outside = BoundaryIndex(arrays).query_boundary((10.0, -0.8), "baseline")
    assert not outside.flags.gain_reachable and "OUT_OF_DOMAIN" in outside.reason
