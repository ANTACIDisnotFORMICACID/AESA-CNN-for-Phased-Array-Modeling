"""Vitis-friendly CNN core and host-side public V2.6 beamforming API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from calibration_v26 import CalibrationDesign, PhaseInvariantCalibrator, build_d_optimal_design
from hfss_physics_v26 import HfssPatternData, PatternMetrics, complex_to_iq, iq_to_complex, project_complex_disk, quantize_int8_disk, unit_vector_hfss


class Reason(IntFlag):
    OUT_OF_DOMAIN = 1
    AMBIGUOUS_LOBE = 2
    POINTING_UNACHIEVABLE = 4
    CALIBRATION_UNCERTAIN = 8
    INSUFFICIENT_ELEMENTS = 16


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1), groups=width, bias=False)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1, bias=True)
        self.activation = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.pointwise(self.depthwise(x)))


class VitisBeamformingCNNV26(nn.Module):
    """Fixed graph: Conv2d, depthwise Conv2d, ReLU6, and residual addition only."""

    def __init__(self, input_channels: int = 15, width: int = 16, blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(input_channels, width, kernel_size=1), nn.ReLU6(inplace=False))
        self.blocks = nn.Sequential(*[DepthwiseResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 2, kernel_size=1)
        # A new/untrained controller is exactly the analytic physical baseline.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, deployment_x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(deployment_x)))


@dataclass
class BeamformingResult:
    target_theta_deg: float
    target_phi_deg: float
    weights_fp32: np.ndarray
    weights_int8_codes: np.ndarray
    weights_int8: np.ndarray
    feasible: bool
    reason_mask: int
    metrics_fp32: PatternMetrics
    metrics_int8: PatternMetrics
    calibration_confidence: float

    def serializable(self) -> dict[str, Any]:
        def complex_pairs(value):
            value = np.asarray(value)
            return np.stack((value.real, value.imag), axis=-1).tolist()

        return {
            "target_hfss_deg": [self.target_theta_deg, self.target_phi_deg],
            "weights_fp32_iq": complex_pairs(self.weights_fp32),
            "weights_int8_codes_iq": self.weights_int8_codes.astype(int).tolist(),
            "weights_int8_iq": complex_pairs(self.weights_int8),
            "feasible": self.feasible,
            "reason_mask": self.reason_mask,
            "reason_names": [reason.name for reason in Reason if reason & self.reason_mask],
            "metrics_fp32": asdict(self.metrics_fp32),
            "metrics_int8": asdict(self.metrics_int8),
            "calibration_confidence": self.calibration_confidence,
        }


class BeamformingControllerV26:
    def __init__(
        self,
        data: HfssPatternData | None = None,
        model: nn.Module | None = None,
        calibration_design: CalibrationDesign | None = None,
        residual_scale: float = 0.5,
        device: str = "cpu",
    ):
        self.data = data or HfssPatternData.load()
        self.model = model or VitisBeamformingCNNV26()
        self.model.to(device).eval()
        self.device = device
        self.residual_scale = float(residual_scale)
        self.calibration_design = calibration_design or build_d_optimal_design(self.data)
        self.calibrator = PhaseInvariantCalibrator(self.calibration_design)
        self.y_normalized = (2.0 * (self.data.y_mm - self.data.y_mm.min()) / np.ptp(self.data.y_mm) - 1.0).astype(np.float32)
        domain_t = np.arange(60.0, 121.0)
        domain_p = np.arange(-30.0, 31.0)
        envelope = []
        for theta in domain_t:
            for phi in domain_p:
                ht, hp = self.data.response_at(theta, phi)
                envelope.append(np.sqrt(np.abs(ht) ** 2 + np.abs(hp) ** 2))
        self.envelope_reference = float(np.max(envelope))

    def calibrate(self, measurement_iq: np.ndarray, health_mask: np.ndarray | None = None):
        estimate = self.calibrator.calibrate(measurement_iq, health_mask)
        return estimate.calibration_iq, estimate.confidence

    @staticmethod
    def _analytic_initial(ht: np.ndarray, hp: np.ndarray, calibration: np.ndarray, health: np.ndarray) -> np.ndarray:
        htc = ht * calibration * health
        hpc = hp * calibration * health
        gram = np.outer(htc.conj(), htc) + np.outer(hpc.conj(), hpc)
        _, vectors = np.linalg.eigh(gram)
        command = np.exp(1j * np.angle(vectors[:, -1])) * health
        for _ in range(64):
            previous = command.copy()
            for n in range(4):
                if health[n] <= 0:
                    command[n] = 0.0
                    continue
                cross = gram[n] @ command - gram[n, n] * command[n]
                if abs(cross) > 1e-15:
                    command[n] = np.exp(1j * np.angle(cross))
            if np.max(np.abs(command - previous)) < 1e-9:
                break
        return command

    def deployment_input(
        self,
        target_hfss_deg: tuple[float, float],
        calibration_iq: np.ndarray,
        confidence: float | np.ndarray,
        health_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta, phi = map(float, target_hfss_deg)
        calibration_iq = np.asarray(calibration_iq)
        if calibration_iq.shape == (2, 1, 4):
            calibration = calibration_iq[0, 0] + 1j * calibration_iq[1, 0]
        elif calibration_iq.shape == (2, 4):
            calibration = calibration_iq[0] + 1j * calibration_iq[1]
        else:
            raise ValueError(f"expected calibration [2,1,4] or [2,4], got {calibration_iq.shape}")
        health = np.asarray(health_mask, dtype=np.float32).reshape(4)
        ht, hp = self.data.response_at(theta, phi)
        initial = self._analytic_initial(ht, hp, calibration, health)
        response_scale = max(float(np.max(np.abs(np.concatenate((ht, hp))))), 1e-12)
        direction = unit_vector_hfss(theta, phi).astype(np.float32)
        envelope = np.sqrt(np.abs(ht) ** 2 + np.abs(hp) ** 2) / max(self.envelope_reference, 1e-12)
        x = np.zeros((15, 1, 4), dtype=np.float32)
        x[0, 0], x[1, 0] = initial.real, initial.imag
        x[2, 0], x[3, 0] = calibration.real, calibration.imag
        x[4, 0] = health
        x[5, 0], x[6, 0] = ht.real / response_scale, ht.imag / response_scale
        x[7, 0], x[8, 0] = hp.real / response_scale, hp.imag / response_scale
        x[9, 0] = self.y_normalized
        x[10, 0], x[11, 0], x[12, 0] = direction
        x[13, 0] = envelope.astype(np.float32)
        x[14, 0] = float(np.asarray(confidence).reshape(-1)[0])
        return x, initial

    def beamform(
        self,
        target_hfss_deg: tuple[float, float],
        calibration_iq: np.ndarray,
        confidence: float | np.ndarray,
        health_mask: np.ndarray | None = None,
    ) -> BeamformingResult:
        theta, phi = map(float, target_hfss_deg)
        health = np.ones(4, dtype=np.float32) if health_mask is None else np.asarray(health_mask, dtype=np.float32).reshape(4)
        x, initial = self.deployment_input((theta, phi), calibration_iq, confidence, health)
        with torch.no_grad():
            delta_iq = self.model(torch.from_numpy(x[None]).to(self.device)).cpu().numpy()[0]
        delta = delta_iq[0, 0] + 1j * delta_iq[1, 0]
        command = project_complex_disk(initial * (1.0 + self.residual_scale * delta)) * health
        codes, command_int8 = quantize_int8_disk(command)
        calibration_iq = np.asarray(calibration_iq)
        if calibration_iq.shape == (2, 1, 4):
            calibration = calibration_iq[0, 0] + 1j * calibration_iq[1, 0]
        else:
            calibration = calibration_iq[0] + 1j * calibration_iq[1]
        metrics_fp32 = self.data.metrics(command * calibration * health, theta, phi)
        metrics_int8 = self.data.metrics(command_int8 * calibration * health, theta, phi)
        reason = Reason(0)
        if not (60.0 <= theta <= 120.0 and -30.0 <= phi <= 30.0):
            reason |= Reason.OUT_OF_DOMAIN
        if metrics_int8.strong_peak_margin_db < 3.0:
            reason |= Reason.AMBIGUOUS_LOBE
        if metrics_int8.target_to_peak_db < -0.5 or metrics_int8.pointing_error_deg > 5.0:
            reason |= Reason.POINTING_UNACHIEVABLE
        confidence_value = float(np.asarray(confidence).reshape(-1)[0])
        if confidence_value < 0.5:
            reason |= Reason.CALIBRATION_UNCERTAIN
        if int(np.sum(health > 0.5)) < 3:
            reason |= Reason.INSUFFICIENT_ELEMENTS
        feasible = int(reason) == 0
        return BeamformingResult(
            theta,
            phi,
            command.astype(np.complex64),
            codes,
            command_int8,
            feasible,
            int(reason),
            metrics_fp32,
            metrics_int8,
            confidence_value,
        )
