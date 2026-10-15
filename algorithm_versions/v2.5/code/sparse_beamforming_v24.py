"""V2.4 稀疏方向测量监督与 FPGA 前整数输出参考。

本模块不包含 HLS/RTL/bitstream。它定义算法移植前必须冻结的两项行为：

1. 训练目标只使用 M 个稀疏方向上的相干 I/Q 或功率，不读取完整方向图；
2. 输出 I/Q 量化到严格位于整数单位圆盘内的 INT8 码字。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from beamforming_pre_fpga import ArrayGeometryConfig


ALGORITHM_VERSION = "2.4.0"


class ComplexDiskInt8Quantizer(nn.Module):
    """把逐阵元 I/Q 映射为严格满足 ``qI²+qQ²≤127²`` 的码字。"""

    qmax: int = 127

    def forward(self, iq: torch.Tensor) -> Dict[str, torch.Tensor]:
        if iq.ndim < 3 or iq.shape[1] != 2:
            raise ValueError("iq 的复数维必须位于 dim=1 且长度为2")
        if not bool(torch.isfinite(iq).all()):
            raise ValueError("iq 包含 NaN 或 Inf")
        target_real = (iq[:, 0] * self.qmax).reshape(-1).to(torch.float64)
        target_imag = (iq[:, 1] * self.qmax).reshape(-1).to(torch.float64)
        unconstrained_real = torch.round(target_real).to(torch.int64)
        unconstrained_imag = torch.round(target_imag).to(torch.int64)
        unconstrained_overflow = (
            unconstrained_real.square() + unconstrained_imag.square()
            > self.qmax * self.qmax
        )

        # 精确求欧氏距离最近的整数圆盘格点。对每个qI枚举[-127,127]，
        # qQ在该列可行区间内取距目标最近的整数，因而是全局最优而非径向近似。
        real_candidates = torch.arange(
            -self.qmax, self.qmax + 1, device=iq.device, dtype=torch.float64
        )
        imag_limit = torch.floor(
            torch.sqrt((self.qmax * self.qmax - real_candidates.square()).clamp_min(0.0))
        )
        candidate_imag = torch.round(target_imag[:, None]).expand(
            -1, real_candidates.numel()
        )
        candidate_imag = torch.maximum(
            torch.minimum(candidate_imag, imag_limit[None]), -imag_limit[None]
        )
        distance_sq = (
            (real_candidates[None] - target_real[:, None]).square()
            + (candidate_imag - target_imag[:, None]).square()
        )
        best = distance_sq.argmin(dim=1)
        real = real_candidates[best].to(torch.int32).reshape(iq[:, 0].shape)
        imag = candidate_imag.gather(1, best[:, None])[:, 0].to(torch.int32).reshape(
            iq[:, 1].shape
        )
        codes = torch.stack((real, imag), dim=1)
        final_radius_sq = (
            codes[:, 0].to(torch.int64).square()
            + codes[:, 1].to(torch.int64).square()
        )
        if torch.any(final_radius_sq > self.qmax * self.qmax):
            raise RuntimeError("整数圆盘投影失败")
        codes_int8 = codes.to(torch.int8)
        dequantized = codes.to(dtype=iq.dtype) / float(self.qmax)
        input_projection = (
            iq[:, 0].square() + iq[:, 1].square() > (1.0 + 1.0e-6) ** 2
        ).to(dtype=iq.dtype).mean()
        return {
            "codes_iq": codes_int8,
            "dequantized_iq": dequantized,
            "input_disk_projection_rate": input_projection,
            "lattice_constraint_adjustment_rate": unconstrained_overflow.to(
                dtype=iq.dtype
            ).mean(),
            "max_integer_radius_sq": final_radius_sq.amax(),
        }


class SparseFieldObjective(nn.Module):
    """仅使用稀疏复场或功率标签的归一化损失。"""

    def __init__(
        self,
        *,
        mode: str = "coherent_iq",
        target_point_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"coherent_iq", "phase_invariant_iq", "power_only"}:
            raise ValueError("mode 必须为 coherent_iq、phase_invariant_iq 或 power_only")
        if target_point_weight <= 0.0:
            raise ValueError("target_point_weight 必须为正数")
        self.mode = mode
        self.target_point_weight = float(target_point_weight)

    def forward(
        self,
        predicted: torch.Tensor,
        measured: torch.Tensor,
    ) -> torch.Tensor:
        if predicted.shape != measured.shape or predicted.ndim != 2:
            raise ValueError("predicted 和 measured 必须同为 [B,M] 张量")
        if not torch.is_complex(predicted):
            raise ValueError("predicted 必须是复张量")
        if self.mode != "power_only" and not torch.is_complex(measured):
            raise ValueError("I/Q 模式下 measured 必须是复张量")
        if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(measured).all()):
            raise ValueError("predicted 或 measured 包含 NaN/Inf")
        if self.mode == "phase_invariant_iq":
            phase_weights = predicted.real.new_ones(predicted.shape)
            phase_weights[:, 0] = self.target_point_weight
            correlation = (phase_weights * torch.conj(predicted) * measured).sum(dim=1)
            phase = correlation / correlation.abs().clamp_min(1.0e-12)
            predicted = predicted * phase[:, None]
        if self.mode == "power_only":
            prediction_value = predicted.abs().square()
            measured_value = measured.abs().square() if torch.is_complex(measured) else measured
            error = (prediction_value - measured_value).square()
            reference_energy = measured_value.square()
        else:
            error = (predicted - measured).abs().square()
            reference_energy = measured.abs().square()
        weights = error.new_ones(error.shape)
        weights[:, 0] = self.target_point_weight
        weighted_error = (error * weights).sum(dim=1) / weights.sum(dim=1)
        denominator = (
            (reference_energy * weights).sum(dim=1) / weights.sum(dim=1)
        ).clamp_min(1.0e-12)
        return (weighted_error / denominator).mean()


@dataclass(frozen=True)
class PreFpgaAlgorithmContract:
    algorithm_version: str
    topology: str
    x_num: int
    y_num: int
    sparse_measurement_count: int
    measurement_mode: str
    input_iq_bits: int = 16
    output_iq_bits: int = 8
    accumulator_bits: int = 32
    complex_amplitude_limit: float = 1.0
    target_feature_count: int = 7
    state_feature_count: int = 4

    @classmethod
    def create(
        cls,
        config: ArrayGeometryConfig,
        measurement_count: int,
        measurement_mode: str,
    ) -> "PreFpgaAlgorithmContract":
        if measurement_count <= 0:
            raise ValueError("measurement_count 必须为正数")
        if measurement_mode not in {
            "coherent_iq",
            "phase_invariant_iq",
            "power_only",
        }:
            raise ValueError("未知 measurement_mode")
        return cls(
            algorithm_version=ALGORITHM_VERSION,
            topology=config.topology_name,
            x_num=config.x_num,
            y_num=config.y_num,
            sparse_measurement_count=measurement_count,
            measurement_mode=measurement_mode,
        )

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "controller_input_shape": [
                    "B",
                    10,
                    self.x_num,
                    self.y_num,
                ],
                "controller_output_shape": [
                    "B",
                    2,
                    self.x_num,
                    self.y_num,
                ],
                "sparse_label_shape": [
                    "B",
                    2 if self.measurement_mode != "power_only" else 1,
                    self.sparse_measurement_count,
                ],
                "total_measurement_count": self.sparse_measurement_count,
                "dynamic_target_measurement_count": 1,
                "fixed_measurement_count": self.sparse_measurement_count - 1,
                "measurement_order": "dynamic_target_then_fixed_angles",
                "integer_output_constraint": "qI^2+qQ^2<=127^2",
            }
        )
        return result


__all__ = [
    "ALGORITHM_VERSION",
    "ComplexDiskInt8Quantizer",
    "PreFpgaAlgorithmContract",
    "SparseFieldObjective",
]
