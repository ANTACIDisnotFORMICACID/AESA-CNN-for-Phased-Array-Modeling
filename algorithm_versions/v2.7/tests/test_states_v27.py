import numpy as np

from states_v27 import all_audit_states, one_factor_states, stratified_targets


def test_exact_state_counts_and_ranges():
    states = all_audit_states()
    assert len(states) == 83 and len({state.state_id for state in states}) == 83
    assert len(one_factor_states()) == 19
    mixed = states[19:]
    assert all(0.0 <= state.amplitude_rms_fraction <= 0.13 for state in mixed)
    assert all(0.0 <= state.phase_rms_deg <= 22.0 for state in mixed)
    assert all(25.0 <= state.measurement_snr_db <= 40.0 for state in mixed)
    failures = sum(state.failed_port is not None for state in mixed)
    assert 4 <= failures <= 9


def test_one_factor_contract():
    states = one_factor_states()
    assert states[0].state_id == "baseline"
    assert sum(state.state_class == "amplitude" for state in states) == 4
    assert sum(state.state_class == "phase" for state in states) == 4
    assert sum(state.state_class == "snr" for state in states) == 3
    assert sum(state.state_class == "failed_port" for state in states) == 4
    assert sum(state.state_class == "global_phase" for state in states) == 3


def test_stratified_targets_cover_domain_deterministically():
    first = stratified_targets(0)
    second = stratified_targets(0)
    assert first.shape == (256, 2) and np.array_equal(first, second)
    assert np.all((60.0 <= first[:, 0]) & (first[:, 0] <= 120.0))
    assert np.all((-30.0 <= first[:, 1]) & (first[:, 1] <= 30.0))
