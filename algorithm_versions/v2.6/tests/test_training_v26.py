import numpy as np
import torch

from controller_v26 import BeamformingControllerV26, VitisBeamformingCNNV26
from hfss_physics_v26 import HfssPatternData
from train_v26 import ExactHfssLoss, LossWeights


def test_exact_hfss_loss_is_finite_and_backpropagates():
    data = HfssPatternData.load()
    controller = BeamformingControllerV26(data)
    calibration_iq = np.zeros((2, 1, 4), dtype=np.float32)
    calibration_iq[0] = 1.0
    targets = [(113.0, -8.0), (100.0, 10.0)]
    inputs, initial, ht, hp = [], [], [], []
    for target in targets:
        x, w = controller.deployment_input(target, calibration_iq, 1.0, np.ones(4))
        a, b = data.response_at(*target)
        inputs.append(x)
        initial.append(w)
        ht.append(a)
        hp.append(b)
    batch = {
        "deployment_x": torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
        "teacher_weights": torch.from_numpy(np.asarray(initial, dtype=np.complex64)),
        "calibration": torch.ones(2, 4, dtype=torch.complex64),
        "health_mask": torch.ones(2, 4),
        "target_ht": torch.from_numpy(np.asarray(ht, dtype=np.complex64)),
        "target_hp": torch.from_numpy(np.asarray(hp, dtype=np.complex64)),
        "target_hfss_deg": torch.tensor(targets, dtype=torch.float32),
        "oracle_metrics": torch.zeros(2, 4),
    }
    model = VitisBeamformingCNNV26()
    criterion = ExactHfssLoss(data, LossWeights(1.0, 0.25, 0.5, 0.05))
    loss, terms = criterion(batch["deployment_x"], model(batch["deployment_x"]), batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()

