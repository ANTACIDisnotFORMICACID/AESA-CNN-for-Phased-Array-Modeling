"""V2.3 近场测量到孔径/远场的物理初始化前端。

固定物理基线为：

``a_hat = (G^H G + lambda I)^-1 G^H y``

其中 G 是近场 Green 矩阵，a 是有效复孔径系数。复矩阵通过实块矩阵
``[[Re,-Im],[Im,Re]]`` 写入单层 Linear，随后可用零初始化低秩适配器
学习位置失配、噪声和模型误差。该前端不替代 CNN；它为后续 CNN 提供
稀疏测量得到的逐阵元校准 I/Q。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from beamforming_pre_fpga import ArrayGeometryConfig, array_positions, direction_vectors


ALGORITHM_VERSION = "2.3.0"


def design_probe_grid(
    config: ArrayGeometryConfig,
    *,
    rows: Optional[int] = None,
    columns: Optional[int] = None,
    range_factor: float = 2.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """生成确定性的近场平面探针网格。

    默认 4×4 使用 4×8=32 点，1×4 使用 3×4=12 点。该网格只用于
    算法可辨识性实验；实测阶段应以实际探针坐标替换。
    """

    if range_factor <= 0.0:
        raise ValueError("range_factor 必须为正数")
    if rows is None:
        rows = 4 if config.x_num > 1 else 3
    if columns is None:
        columns = 8 if config.x_num > 1 else 4
    if rows <= 0 or columns <= 0:
        raise ValueError("rows 和 columns 必须为正数")
    positions = array_positions(config, dtype=dtype)
    centre = positions.mean(dim=0)
    aperture = torch.cdist(positions, positions).amax().clamp_min(config.wavelength)
    y_half_span = torch.maximum(aperture, aperture.new_tensor(2.0 * config.wavelength))
    y_axis = torch.linspace(
        float(centre[1] - y_half_span), float(centre[1] + y_half_span), rows, dtype=dtype
    )
    z_axis = torch.linspace(
        float(centre[2] - aperture), float(centre[2] + aperture), columns, dtype=dtype
    )
    yy, zz = torch.meshgrid(y_axis, z_axis, indexing="ij")
    xx = torch.full_like(yy, float(centre[0] + range_factor * aperture))
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def _normalise_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("probe_points 必须为 [M,3]")
    return points


def near_field_green_matrix(
    config: ArrayGeometryConfig,
    probe_points: torch.Tensor,
    *,
    position_errors: Optional[torch.Tensor] = None,
    include_path_loss: bool = False,
    min_distance: float = 1.0e-6,
) -> torch.Tensor:
    """返回复 Green 矩阵 G。

    无位置误差时形状为 ``[M,N]``；有 ``[B,3,H,W]`` 位置误差时为
    ``[B,M,N]``。传播约定为 ``exp(-j*k*d)``，与项目远场正号一致。
    """

    points = _normalise_points(probe_points)
    if min_distance <= 0.0:
        raise ValueError("min_distance 必须为正数")
    nominal = array_positions(config, device=points.device, dtype=points.dtype)
    if position_errors is None:
        displacement = points[:, None, :] - nominal[None, :, :]
    else:
        expected = (position_errors.shape[0], 3, config.x_num, config.y_num)
        if tuple(position_errors.shape) != expected:
            raise ValueError(f"position_errors 应为 {expected}")
        actual = nominal.unsqueeze(0) + position_errors.flatten(2).transpose(1, 2)
        displacement = points[None, :, None, :] - actual[:, None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(min_distance)
    magnitude = distance.reciprocal() if include_path_loss else torch.ones_like(distance)
    phase = -(2.0 * math.pi / config.wavelength) * distance
    return torch.polar(magnitude, phase)


def far_field_matrix(
    config: ArrayGeometryConfig,
    theta_rad: torch.Tensor,
    phi_rad: torch.Tensor,
    *,
    position_errors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """返回复远场矩阵 S，满足 ``E=S a``。"""

    theta_rad, phi_rad = torch.broadcast_tensors(theta_rad, phi_rad)
    directions = direction_vectors(theta_rad, phi_rad).reshape(-1, 3)
    nominal = array_positions(config, device=theta_rad.device, dtype=theta_rad.dtype)
    k = 2.0 * math.pi / config.wavelength
    if position_errors is None:
        phase = k * (directions @ nominal.T)
    else:
        expected = (position_errors.shape[0], 3, config.x_num, config.y_num)
        if tuple(position_errors.shape) != expected:
            raise ValueError(f"position_errors 应为 {expected}")
        actual = nominal.unsqueeze(0) + position_errors.flatten(2).transpose(1, 2)
        phase = k * torch.einsum("pc,bnc->bpn", directions, actual)
    return torch.polar(torch.ones_like(phase), phase)


def regularized_pseudoinverse(matrix: torch.Tensor, ridge: float = 1.0e-6) -> torch.Tensor:
    """复 Tikhonov 伪逆 ``(GᴴG+λI)⁻¹Gᴴ``。"""

    if matrix.ndim != 2 or not torch.is_complex(matrix):
        raise ValueError("matrix 必须为二维复张量")
    if ridge < 0.0:
        raise ValueError("ridge 不得为负")
    _, columns = matrix.shape
    hermitian = matrix.mH @ matrix
    identity = torch.eye(columns, device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.solve(hermitian + ridge * identity, matrix.mH)


def near_to_far_transform(
    near_matrix: torch.Tensor,
    far_matrix: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> torch.Tensor:
    """物理近远场变换 ``T=S(GᴴG+λI)⁻¹Gᴴ``。"""

    if far_matrix.ndim != 2 or far_matrix.shape[1] != near_matrix.shape[1]:
        raise ValueError("far_matrix 与 near_matrix 的阵元维必须一致")
    return far_matrix @ regularized_pseudoinverse(near_matrix, ridge)


def complex_matrix_to_real_block(matrix: torch.Tensor) -> torch.Tensor:
    """将复线性映射写为显式 I/Q 实块矩阵。"""

    if matrix.ndim != 2 or not torch.is_complex(matrix):
        raise ValueError("matrix 必须为二维复张量")
    real, imag = matrix.real, matrix.imag
    return torch.cat(
        (torch.cat((real, -imag), dim=1), torch.cat((imag, real), dim=1)), dim=0
    )


def iq_to_complex(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("iq 必须为 [B,2,L]")
    return torch.complex(iq[:, 0], iq[:, 1])


def complex_to_iq(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or not torch.is_complex(value):
        raise ValueError("value 必须为 [B,L] 复张量")
    return torch.stack((value.real, value.imag), dim=1)


class PhysicsInitializedApertureEstimator(nn.Module):
    """物理伪逆初始化的单层复 FC 与零初始化低秩适配器。"""

    def __init__(
        self,
        pseudoinverse: torch.Tensor,
        *,
        init_mode: str = "physics",
        adapter_rank: int = 4,
        adapter_scale: float = 0.25,
        train_base: bool = True,
    ) -> None:
        super().__init__()
        if pseudoinverse.ndim != 2 or not torch.is_complex(pseudoinverse):
            raise ValueError("pseudoinverse 必须为 [N,M] 复张量")
        if init_mode not in {"physics", "random"}:
            raise ValueError("init_mode 必须为 physics 或 random")
        if adapter_rank < 0 or adapter_scale <= 0.0:
            raise ValueError("adapter_rank 不得为负且 adapter_scale 必须为正")
        self.num_elements, self.num_measurements = pseudoinverse.shape
        self.init_mode = init_mode
        self.adapter_rank = int(adapter_rank)
        self.adapter_scale = float(adapter_scale)
        self.base = nn.Linear(2 * self.num_measurements, 2 * self.num_elements, bias=False)
        if init_mode == "physics":
            with torch.no_grad():
                self.base.weight.copy_(complex_matrix_to_real_block(pseudoinverse).float())
        else:
            nn.init.xavier_uniform_(self.base.weight)
        self.base.weight.requires_grad_(train_base)
        if adapter_rank > 0:
            self.adapter_down = nn.Linear(2 * self.num_measurements, adapter_rank, bias=False)
            self.adapter_up = nn.Linear(adapter_rank, 2 * self.num_elements, bias=True)
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)
        else:
            self.adapter_down = None
            self.adapter_up = None

    def forward(self, measurements_iq: torch.Tensor) -> torch.Tensor:
        if measurements_iq.ndim != 3 or measurements_iq.shape[1:] != (
            2,
            self.num_measurements,
        ):
            raise ValueError(
                f"measurements_iq 必须为 [B,2,{self.num_measurements}]"
            )
        vector = torch.cat((measurements_iq[:, 0], measurements_iq[:, 1]), dim=1)
        estimate = self.base(vector)
        if self.adapter_down is not None and self.adapter_up is not None:
            estimate = estimate + self.adapter_scale * self.adapter_up(
                torch.tanh(self.adapter_down(vector))
            )
        real, imag = estimate.split(self.num_elements, dim=1)
        return torch.stack((real, imag), dim=1)

    def profile(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        base_macs = 4 * self.num_elements * self.num_measurements
        adapter_macs = (
            self.adapter_rank * (2 * self.num_measurements + 2 * self.num_elements)
            if self.adapter_rank > 0
            else 0
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "macs_per_sample": base_macs + adapter_macs,
            "base_macs": base_macs,
            "adapter_macs": adapter_macs,
        }


def analytic_near_field_weights(
    config: ArrayGeometryConfig,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """理想相位型近场聚焦权值，输出 ``[B,2,H,W]``。"""

    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points 必须为 [B,3]")
    positions = array_positions(
        config, device=target_points.device, dtype=target_points.dtype
    )
    distance = torch.linalg.vector_norm(
        target_points[:, None, :] - positions[None, :, :], dim=-1
    )
    phase = (2.0 * math.pi / config.wavelength) * distance
    value = torch.polar(torch.ones_like(phase), phase)
    return complex_to_iq(value).reshape(-1, 2, config.x_num, config.y_num)


def align_global_phase(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """按样本移除不影响功率方向图的公共复相位。"""

    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError("reference 与 estimate 必须同为 [B,N] 复张量")
    correlation = (torch.conj(estimate) * reference).sum(dim=1)
    phase = correlation / correlation.abs().clamp_min(1.0e-12)
    return estimate * phase[:, None]


__all__ = [
    "ALGORITHM_VERSION",
    "PhysicsInitializedApertureEstimator",
    "align_global_phase",
    "analytic_near_field_weights",
    "complex_matrix_to_real_block",
    "complex_to_iq",
    "design_probe_grid",
    "far_field_matrix",
    "iq_to_complex",
    "near_field_green_matrix",
    "near_to_far_transform",
    "regularized_pseudoinverse",
]
