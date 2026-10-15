import numpy as np
import torch

from controller_v26 import BeamformingControllerV26, Reason, VitisBeamformingCNNV26
from hfss_physics_v26 import HfssPatternData


def test_model_fixed_shapes_and_parameter_budget():
    model = VitisBeamformingCNNV26()
    x = torch.zeros(3, 15, 1, 4)
    y = model(x)
    assert tuple(y.shape) == (3, 2, 1, 4)
    assert sum(p.numel() for p in model.parameters()) < 5000


def test_untrained_zero_head_is_exact_analytic_baseline_and_reports_ambiguity():
    controller = BeamformingControllerV26(HfssPatternData.load())
    calibration = np.zeros((2, 1, 4), dtype=np.float32)
    calibration[0] = 1.0
    x, initial = controller.deployment_input((113.0, -8.0), calibration, 1.0, np.ones(4))
    assert x.shape == (15, 1, 4)
    result = controller.beamform((113.0, -8.0), calibration, 1.0)
    np.testing.assert_allclose(result.weights_fp32, initial, atol=1e-6)
    assert np.isfinite(result.weights_fp32).all()
    radius2 = result.weights_int8_codes[:, 0].astype(np.int16) ** 2 + result.weights_int8_codes[:, 1].astype(np.int16) ** 2
    assert np.all(radius2 <= 127**2)
    assert result.reason_mask & int(Reason.AMBIGUOUS_LOBE)


def test_public_result_reason_bits_for_domain_confidence_and_health():
    controller = BeamformingControllerV26(HfssPatternData.load())
    calibration = np.zeros((2, 1, 4), dtype=np.float32)
    calibration[0] = 1.0
    result = controller.beamform((130.0, 0.0), calibration, 0.1, np.array([1, 1, 0, 0]))
    assert result.reason_mask & int(Reason.OUT_OF_DOMAIN)
    assert result.reason_mask & int(Reason.CALIBRATION_UNCERTAIN)
    assert result.reason_mask & int(Reason.INSUFFICIENT_ELEMENTS)
    assert not result.feasible

