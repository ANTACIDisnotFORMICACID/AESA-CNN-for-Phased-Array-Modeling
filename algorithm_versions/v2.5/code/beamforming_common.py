"""跨版本共享的阵列方向、复权值约束与 QAT 基础算子。

本模块只放置 V2.2 及后续控制器真实共用的无状态数学内核，不包含
任何旧版网络、实验入口、随机种子、结果路径或版本号。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def direction_vectors(theta_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
    """球坐标转单位方向向量 ``[..., 3]``。"""

    theta_rad, phi_rad = torch.broadcast_tensors(theta_rad, phi_rad)
    return torch.stack(
        (
            torch.sin(theta_rad) * torch.cos(phi_rad),
            torch.sin(theta_rad) * torch.sin(phi_rad),
            torch.cos(theta_rad),
        ),
        dim=-1,
    )


def project_complex_unit_disk_iq(iq: torch.Tensor) -> torch.Tensor:
    """逐复数径向投影到单位圆盘，保证阵元复幅度不超过 1。"""

    if iq.ndim < 2 or iq.shape[1] != 2:
        raise ValueError("iq 的 dim=1 必须为 I/Q 两通道")
    magnitude_sq = iq[:, 0].square() + iq[:, 1].square()
    scale = torch.rsqrt(magnitude_sq.clamp_min(1.0))
    return iq * scale.unsqueeze(1)


class SymmetricFakeQuantizer(nn.Module):
    """带直通估计器的对称定点假量化。"""

    def __init__(
        self,
        num_bits: int = 8,
        clip_value: Optional[float] = None,
        *,
        per_channel_dim: Optional[int] = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_bits < 2 or num_bits > 16:
            raise ValueError("num_bits 必须位于 [2, 16]")
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError("clip_value 必须为正数")
        self.num_bits = int(num_bits)
        self.clip_value = clip_value
        self.per_channel_dim = per_channel_dim
        self.eps = float(eps)

    @property
    def qmax(self) -> int:
        return (1 << (self.num_bits - 1)) - 1

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_value is not None:
            return x.new_tensor(self.clip_value / self.qmax)
        if self.per_channel_dim is None:
            max_abs = x.detach().abs().amax()
        else:
            dim = self.per_channel_dim % x.ndim
            reduce_dims = tuple(i for i in range(x.ndim) if i != dim)
            max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
        return (max_abs / self.qmax).clamp_min(self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._scale(x)
        q = torch.round(x / scale).clamp(-self.qmax, self.qmax)
        dequantized = q * scale
        return x + (dequantized - x).detach()


class QConv2d(nn.Conv2d):
    """Conv2d 的 QAT 包装；权重逐输出通道对称假量化。"""

    def __init__(self, *args, weight_bits: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight_quantizer = (
            SymmetricFakeQuantizer(weight_bits, per_channel_dim=0)
            if weight_bits is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
