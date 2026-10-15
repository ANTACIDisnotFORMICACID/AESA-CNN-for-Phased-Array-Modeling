import numpy as np

from oracle_v27 import target_grid
from state_oracle_v27 import representative_state_indices, representative_target_indices
from verify_oracle_v27 import verification_indices
from v25_adapter_v27 import FrozenV25Adapter, hfss_to_v25_angles_rad
from v27_compat import enable_v26

enable_v26()
from controller_v26 import BeamformingControllerV26
from hfss_physics_v26 import HfssPatternData


def test_nominal_grid_has_4096_unique_targets():
    grid = target_grid()
    assert grid.shape == (4096, 2)
    assert len(np.unique(grid, axis=0)) == 4096


def test_hfss_geometry_contract_is_fixed():
    data = HfssPatternData.load()
    assert data.port_names == ("1:1", "2:1", "3:1", "4:1")
    assert np.allclose(np.diff(data.y_mm), 6.8)
    assert data.etheta.shape == (4, 361, 181)


def test_v25_rotation_maps_its_z_axis_to_hfss_y():
    # HFSS +Y is theta=90, phi=90; the inverse rotation is V2.5 +Z, theta=0.
    angle = hfss_to_v25_angles_rad(np.asarray([[90.0, 90.0]]))[0]
    assert abs(float(angle[0])) < 1e-6


def test_v26_public_input_is_unchanged():
    controller = BeamformingControllerV26()
    calibration = np.zeros((2, 1, 4), dtype=np.float32)
    calibration[0] = 1.0
    x, initial = controller.deployment_input((90.0, 0.0), calibration, np.asarray([1.0]), np.ones(4))
    assert x.shape == (15, 1, 4) and initial.shape == (4,)


def test_frozen_v25_checkpoint_loads_strictly():
    adapter = FrozenV25Adapter()
    value = adapter.commands(np.asarray([[90.0, 0.0]], dtype=np.float32), np.ones(4), np.ones(4))
    assert value.shape == (1, 4) and np.isfinite(value).all()


def test_state_oracle_selection_is_exact_and_disjoint():
    states = representative_state_indices()
    targets, strata = representative_target_indices()
    assert states.shape == (16,) and len(np.unique(states)) == 16
    assert targets.shape == (64,) and len(np.unique(targets)) == 64
    labels, counts = np.unique(strata, return_counts=True)
    assert dict(zip(labels.tolist(), counts.tolist())) == {"boundary": 21, "outside": 21, "reachable_core": 22}


def test_repeatability_sample_is_exactly_five_percent_rounded_up():
    indices = verification_indices()
    assert indices.shape == (205,) and len(np.unique(indices)) == 205
    assert indices.min() >= 0 and indices.max() < 4096
