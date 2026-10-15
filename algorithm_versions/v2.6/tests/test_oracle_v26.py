import numpy as np

from hfss_physics_v26 import HfssPatternData
from oracle_v26 import OracleConfig, PhysicalOracle


def test_phase_only_oracle_is_deterministic_and_target_optimal_against_random():
    data = HfssPatternData.load()
    oracle = PhysicalOracle(data, OracleConfig(starts=8, max_steps=2))
    a = oracle.target_power_phase_oracle(100.0, 10.0)
    b = oracle.target_power_phase_oracle(100.0, 10.0)
    np.testing.assert_allclose(a, b, atol=0.0)
    rng = np.random.default_rng(8)
    random_w = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(64, 4)))
    target = np.array([data.target_power(w, 100.0, 10.0) for w in random_w])
    assert data.target_power(a, 100.0, 10.0) >= np.max(target) * 0.999

