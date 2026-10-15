"""Frozen V2.5 low-update controller adapter for the mismatched HFSS geometry audit."""

from __future__ import annotations

import numpy as np
import torch

from v27_compat import V25_ROOT, enable_v25

enable_v25()
from beamforming_pre_fpga import (  # noqa: E402
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    analytic_far_field_weights,
)


ROTATION_HFSS_FROM_V25 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
CHECKPOINT = V25_ROOT / "output" / "controller_training" / "v24_1x4_low_update_controller.pt"


def hfss_to_v25_angles_rad(target_hfss_deg: np.ndarray) -> np.ndarray:
    target = np.asarray(target_hfss_deg, dtype=np.float64)
    theta = np.deg2rad(target[:, 0])
    phi = np.deg2rad(target[:, 1])
    u_hfss = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1
    )
    u_v25 = u_hfss @ ROTATION_HFSS_FROM_V25
    theta_v25 = np.arccos(np.clip(u_v25[:, 2], -1.0, 1.0))
    phi_v25 = np.arctan2(u_v25[:, 1], u_v25[:, 0])
    return np.stack((theta_v25, phi_v25), axis=-1).astype(np.float32)


class FrozenV25Adapter:
    """Runs the original 15 mm interface; output is interpreted as four port commands."""

    def __init__(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        config = ArrayGeometryConfig(**checkpoint["array_config"])
        model_config = dict(checkpoint["model_config"])
        self.model = TopologyAdaptiveBeamformingCNN(config, **model_config).eval()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.config = config

    def commands(
        self,
        target_hfss_deg: np.ndarray,
        calibration: np.ndarray,
        health: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        targets_v25 = hfss_to_v25_angles_rad(target_hfss_deg)
        count = len(targets_v25)
        calibration = np.asarray(calibration, dtype=np.complex64).reshape(4)
        health = np.asarray(health, dtype=np.float32).reshape(4)
        outputs = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                angles = torch.from_numpy(targets_v25[start:stop])
                initial = analytic_far_field_weights(self.config, angles)
                cal = np.broadcast_to(calibration, (stop - start, 4))
                cal_iq = np.stack((cal.real, cal.imag), axis=1).reshape(-1, 2, 1, 4).astype(np.float32)
                mask = np.broadcast_to(health, (stop - start, 4)).reshape(-1, 1, 1, 4).copy()
                result = self.model(
                    initial,
                    angles,
                    calibration_iq=torch.from_numpy(cal_iq),
                    health_mask=torch.from_numpy(mask),
                    position_errors=torch.zeros((stop - start, 3, 1, 4)),
                    state=torch.zeros((stop - start, 4)),
                    quantize_output=False,
                )["weights_iq"]
                outputs.append(result[:, 0, 0].numpy() + 1j * result[:, 1, 0].numpy())
        return np.concatenate(outputs).astype(np.complex64)
