"""FPGA 移植前的拓扑自适应智能波束成形算法。

本模块继承 clean P0 的物理约定，并从 ``beamforming_common.py`` 导入
跨版本基础算子。它只覆盖 FPGA 综合以前的算法研究，核心约束是：

* 同一套物理约定支持 4x4 理论阵列和 1x4 实际阵列；
* 4x4 使用二维 3x3 空间卷积，1x4 自动退化为 1x3 卷积；
* 不用全局平均池化替代逐阵元特征，GAP 只能作为广播回空间分支的上下文；
* 以解析共轭导向权值作为运行时物理初值，CNN 只学习小残差；
* 输出逐复数投影到单位圆盘，严格满足每阵元幅度不超过 1；
* 张量使用显式 I/Q 表示，便于后续定点等价和 FPGA 数据接口定义。

V2.2 首次引入本模块；后续小版本只以向后兼容方式扩展。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from beamforming_common import (
    QConv2d,
    SymmetricFakeQuantizer,
    direction_vectors,
    project_complex_unit_disk_iq,
)


ALGORITHM_VERSION = "2.2.0"
_EPS = 1.0e-8


@dataclass(frozen=True)
class ArrayGeometryConfig:
    """阵列几何；当前认证拓扑为 4x4 和 1x4。

    位置编号与旧仿真一致：``xi`` 为外层、``yi`` 为内层，阵元位于
    y-z 平面，坐标为 ``(0, x_spacing*xi, y_spacing*yi)``。
    """

    wavelength: float = 6.36502
    x_spacing: float = 6.83
    y_spacing: float = 15.0
    x_num: int = 4
    y_num: int = 4

    def __post_init__(self) -> None:
        if self.wavelength <= 0.0:
            raise ValueError("wavelength 必须为正数")
        if self.x_spacing <= 0.0 or self.y_spacing <= 0.0:
            raise ValueError("阵元间距必须为正数")
        if (self.x_num, self.y_num) not in {(4, 4), (1, 4)}:
            raise ValueError("当前仅认证 4x4 理论阵列和 1x4 实际阵列")

    @property
    def num_elements(self) -> int:
        return self.x_num * self.y_num

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self.x_num, self.y_num

    @property
    def topology_name(self) -> str:
        return f"{self.x_num}x{self.y_num}"

    @classmethod
    def theoretical_4x4(cls, **overrides: float) -> "ArrayGeometryConfig":
        return cls(x_num=4, y_num=4, **overrides)

    @classmethod
    def hardware_1x4(cls, **overrides: float) -> "ArrayGeometryConfig":
        return cls(x_num=1, y_num=4, **overrides)


def array_positions(
    config: ArrayGeometryConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """返回 ``[N,3]`` 名义阵元坐标。"""

    positions = [
        (0.0, config.x_spacing * xi, config.y_spacing * yi)
        for xi in range(config.x_num)
        for yi in range(config.y_num)
    ]
    return torch.tensor(positions, device=device, dtype=dtype)


def _normalise_targets(target_angles_rad: torch.Tensor) -> torch.Tensor:
    if target_angles_rad.ndim == 1 and target_angles_rad.shape[0] == 2:
        return target_angles_rad.reshape(1, 1, 2)
    if target_angles_rad.ndim == 2 and target_angles_rad.shape[-1] == 2:
        return target_angles_rad.unsqueeze(1)
    if target_angles_rad.ndim == 3 and target_angles_rad.shape[-1] == 2:
        return target_angles_rad
    raise ValueError("target_angles_rad 必须为 [2]、[B,2] 或 [B,T,2]")


def _normalise_mask(angles: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    batch, targets, _ = angles.shape
    if mask is None:
        return angles.new_ones(batch, targets)
    if mask.shape != (batch, targets):
        raise ValueError(f"target_mask 应为 {(batch, targets)}")
    mask = mask.to(device=angles.device, dtype=angles.dtype)
    if torch.any(mask < 0.0) or torch.any(mask > 1.0):
        raise ValueError("target_mask 必须位于 [0,1]")
    if torch.any(mask.sum(dim=1) <= 0.0):
        raise ValueError("每个样本至少需要一个有效目标")
    return mask


def topology_target_features(
    target_angles_rad: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    *,
    max_targets: int = 4,
) -> torch.Tensor:
    """与阵元数量解耦的 7 维目标集合摘要。"""

    if max_targets <= 0:
        raise ValueError("max_targets 必须为正数")
    angles = _normalise_targets(target_angles_rad)
    mask = _normalise_mask(angles, target_mask)
    if angles.shape[1] > max_targets:
        raise ValueError("目标数超过 max_targets")
    directions = direction_vectors(angles[..., 0], angles[..., 1])
    count = mask.sum(dim=1, keepdim=True)
    mean = (directions * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1.0)
    second = (directions.square() * mask.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1.0)
    return torch.cat((mean, second, count / float(max_targets)), dim=1)


def analytic_far_field_weights(
    config: ArrayGeometryConfig,
    target_angles_rad: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    target_weights: Optional[torch.Tensor] = None,
    *,
    max_element_amplitude: float = 1.0,
) -> torch.Tensor:
    """解析远场共轭导向初值，输出 ``[B,2,H,W]``。

    单目标时 ``w_n=exp(-j k r_n dot u)``，目标场幅度严格为 N。
    多目标使用加权导向叠加并按样本缩放到复幅度可行域。
    """

    if max_element_amplitude <= 0.0:
        raise ValueError("max_element_amplitude 必须为正数")
    angles = _normalise_targets(target_angles_rad)
    mask = _normalise_mask(angles, target_mask)
    batch, targets, _ = angles.shape
    if target_weights is None:
        weights = mask
    else:
        if target_weights.shape != (batch, targets):
            raise ValueError(f"target_weights 应为 {(batch, targets)}")
        weights = target_weights.to(device=angles.device, dtype=angles.dtype) * mask
        if torch.any(weights < 0.0):
            raise ValueError("target_weights 不得为负")
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(_EPS)

    positions = array_positions(config, device=angles.device, dtype=angles.dtype)
    directions = direction_vectors(angles[..., 0], angles[..., 1])
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "btc,nc->btn", directions, positions
    )
    real = (torch.cos(phase) * weights.unsqueeze(-1)).sum(dim=1)
    imag = (-torch.sin(phase) * weights.unsqueeze(-1)).sum(dim=1)
    magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(_EPS)
    scale = max_element_amplitude / magnitude.amax(dim=1, keepdim=True)
    return torch.stack((real * scale, imag * scale), dim=1).reshape(
        batch, 2, config.x_num, config.y_num
    )


def steering_iq(
    config: ArrayGeometryConfig,
    theta_rad: torch.Tensor,
    phi_rad: torch.Tensor,
    *,
    position_errors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """构造导向张量。

    无位置误差时输出 ``[2,...,N]``；逐样本位置误差 ``[B,3,H,W]``
    存在时输出 ``[B,2,...,N]``。
    """

    theta_rad, phi_rad = torch.broadcast_tensors(theta_rad, phi_rad)
    directions = direction_vectors(theta_rad, phi_rad)
    nominal = array_positions(config, device=theta_rad.device, dtype=theta_rad.dtype)
    k = 2.0 * math.pi / config.wavelength
    if position_errors is None:
        phase = k * torch.einsum("...c,nc->...n", directions, nominal)
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=0)
    expected = (position_errors.shape[0], 3, config.x_num, config.y_num)
    if tuple(position_errors.shape) != expected:
        raise ValueError(f"position_errors 应为 {expected}")
    actual = nominal.unsqueeze(0) + position_errors.flatten(2).transpose(1, 2)
    phase = k * torch.einsum("...c,bnc->b...n", directions, actual)
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)


def steering_at_directions_iq(
    config: ArrayGeometryConfig,
    target_angles_rad: torch.Tensor,
    *,
    position_errors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """构造逐样本配对目标的导向张量 ``[B,2,N]``。"""

    if target_angles_rad.ndim != 2 or target_angles_rad.shape[1] != 2:
        raise ValueError("target_angles_rad 必须为 [B,2]")
    batch = target_angles_rad.shape[0]
    positions = array_positions(
        config, device=target_angles_rad.device, dtype=target_angles_rad.dtype
    )
    if position_errors is None:
        actual = positions.unsqueeze(0).expand(batch, -1, -1)
    else:
        expected = (batch, 3, config.x_num, config.y_num)
        if tuple(position_errors.shape) != expected:
            raise ValueError(f"position_errors 应为 {expected}")
        actual = positions.unsqueeze(0) + position_errors.flatten(2).transpose(1, 2)
    directions = direction_vectors(target_angles_rad[:, 0], target_angles_rad[:, 1])
    phase = (2.0 * math.pi / config.wavelength) * torch.einsum(
        "bnc,bc->bn", actual, directions
    )
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)


def field_from_weights_iq(
    weights_iq: torch.Tensor,
    steering: torch.Tensor,
    *,
    element_response_iq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """按显式 I/Q 计算远场；支持共享或逐样本导向张量。"""

    if weights_iq.ndim != 4 or weights_iq.shape[1] != 2:
        raise ValueError("weights_iq 必须为 [B,2,H,W]")
    batch = weights_iq.shape[0]
    weights = weights_iq.flatten(2)
    if element_response_iq is not None:
        if element_response_iq.shape != weights_iq.shape:
            raise ValueError("element_response_iq 必须与 weights_iq 同形")
        response = element_response_iq.flatten(2)
        wr = weights[:, 0] * response[:, 0] - weights[:, 1] * response[:, 1]
        wi = weights[:, 0] * response[:, 1] + weights[:, 1] * response[:, 0]
    else:
        wr, wi = weights[:, 0], weights[:, 1]
    if steering.ndim >= 3 and steering.shape[0] == batch and steering.shape[1] == 2:
        sr, si = steering[:, 0], steering[:, 1]
        real = torch.einsum("bn,b...n->b...", wr, sr) - torch.einsum(
            "bn,b...n->b...", wi, si
        )
        imag = torch.einsum("bn,b...n->b...", wr, si) + torch.einsum(
            "bn,b...n->b...", wi, sr
        )
    elif steering.ndim >= 3 and steering.shape[0] == 2:
        sr, si = steering[0], steering[1]
        real = torch.einsum("bn,...n->b...", wr, sr) - torch.einsum(
            "bn,...n->b...", wi, si
        )
        imag = torch.einsum("bn,...n->b...", wr, si) + torch.einsum(
            "bn,...n->b...", wi, sr
        )
    else:
        raise ValueError("steering 形状与 batch 或共享导向约定不符")
    return torch.stack((real, imag), dim=1)


def element_response_from_amp_phase(
    amplitude: torch.Tensor, phase_rad: torch.Tensor
) -> torch.Tensor:
    if amplitude.shape != phase_rad.shape or amplitude.ndim != 3:
        raise ValueError("amplitude 和 phase_rad 必须同为 [B,H,W]")
    if torch.any(amplitude < 0.0):
        raise ValueError("amplitude 不得为负")
    return torch.stack(
        (amplitude * torch.cos(phase_rad), amplitude * torch.sin(phase_rad)), dim=1
    )


class _FiLM(nn.Module):
    def __init__(self, context_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(context_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(context).chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma)
        beta = 0.25 * torch.tanh(beta)
        return x * gamma[:, :, None, None] + beta[:, :, None, None]


class _TopologyResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        kernel_size: Tuple[int, int],
        *,
        expansion: int = 2,
        weight_bits: Optional[int] = None,
        activation_bits: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.expand = QConv2d(
            channels, hidden, 1, bias=True, weight_bits=weight_bits
        )
        self.depthwise = QConv2d(
            hidden,
            hidden,
            kernel_size,
            padding=padding,
            groups=hidden,
            bias=True,
            weight_bits=weight_bits,
        )
        self.project = QConv2d(
            hidden, channels, 1, bias=True, weight_bits=weight_bits
        )
        self.film = _FiLM(context_dim, channels)
        self.activation = nn.ReLU6()
        self.activation_quantizer = (
            SymmetricFakeQuantizer(activation_bits, clip_value=6.0)
            if activation_bits is not None
            else None
        )

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        if self.activation_quantizer is not None:
            x = self.activation_quantizer(x)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._activate(self.expand(x))
        x = self._activate(self.depthwise(x))
        x = self.film(self.project(x), context)
        return self._activate(x + residual)


class TopologyAdaptiveBeamformingCNN(nn.Module):
    """4x4/1x4 拓扑自适应的逐阵元残差 CNN。

    ``global_context='none'`` 完全不做空间池化；``'broadcast'`` 只把
    空间均值经 1x1 变换后广播相加，原空间特征始终保留。纯 GAP 模型
    只作为外部消融，不属于本生产候选。
    """

    STATE_SCHEMA: Sequence[str] = (
        "horizontal_coupling_real",
        "horizontal_coupling_imag",
        "vertical_coupling_real",
        "vertical_coupling_imag",
    )

    def __init__(
        self,
        config: ArrayGeometryConfig,
        *,
        width: int = 16,
        num_blocks: int = 3,
        residual_scale: float = 0.75,
        position_error_scale: Optional[float] = None,
        state_dim: int = 4,
        max_targets: int = 4,
        global_context: str = "none",
        weight_bits: Optional[int] = None,
        activation_bits: Optional[int] = None,
        output_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        if width <= 0 or num_blocks <= 0:
            raise ValueError("width 和 num_blocks 必须为正数")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale 必须为正数")
        if state_dim < 0:
            raise ValueError("state_dim 不得为负")
        if max_targets <= 0:
            raise ValueError("max_targets 必须为正数")
        if global_context not in {"none", "broadcast"}:
            raise ValueError("global_context 必须为 none 或 broadcast")
        self.config = config
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.residual_scale = float(residual_scale)
        self.state_dim = int(state_dim)
        self.max_targets = int(max_targets)
        self.global_context = global_context
        self.position_error_scale = float(
            position_error_scale
            if position_error_scale is not None
            else 0.04 * config.wavelength
        )
        if self.position_error_scale <= 0.0:
            raise ValueError("position_error_scale 必须为正数")

        self.kernel_size = (3, 3) if config.x_num > 1 else (1, 3)
        context_dim = 7 + self.state_dim
        # V2.1 的 8 通道再增加两个归一化阵元坐标，明确标识拓扑位置。
        self.stem = QConv2d(10, width, 1, bias=True, weight_bits=weight_bits)
        self.blocks = nn.ModuleList(
            [
                _TopologyResidualBlock(
                    width,
                    context_dim,
                    self.kernel_size,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                )
                for _ in range(num_blocks)
            ]
        )
        self.global_projection = (
            QConv2d(width, width, 1, bias=True, weight_bits=weight_bits)
            if global_context == "broadcast"
            else None
        )
        if self.global_projection is not None:
            nn.init.zeros_(self.global_projection.weight)
            nn.init.zeros_(self.global_projection.bias)
        self.head = QConv2d(width, 2, 1, bias=True, weight_bits=weight_bits)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.input_activation = nn.ReLU6()
        self.input_quantizer = (
            SymmetricFakeQuantizer(activation_bits, clip_value=6.0)
            if activation_bits is not None
            else None
        )
        self.output_quantizer = (
            SymmetricFakeQuantizer(output_bits, clip_value=1.0)
            if output_bits is not None
            else None
        )

        x_axis = (
            torch.linspace(-1.0, 1.0, config.x_num)
            if config.x_num > 1
            else torch.zeros(1)
        )
        y_axis = (
            torch.linspace(-1.0, 1.0, config.y_num)
            if config.y_num > 1
            else torch.zeros(1)
        )
        xx, yy = torch.meshgrid(x_axis, y_axis, indexing="ij")
        self.register_buffer(
            "normalised_coordinates", torch.stack((xx, yy), dim=0).unsqueeze(0)
        )

    def _validate_spatial(
        self, tensor: torch.Tensor, channels: int, name: str
    ) -> None:
        expected = (channels, self.config.x_num, self.config.y_num)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected:
            raise ValueError(f"{name} 必须为 [B,{channels},{self.config.x_num},{self.config.y_num}]")

    def forward(
        self,
        initial_weights_iq: torch.Tensor,
        target_angles_rad: torch.Tensor,
        *,
        target_mask: Optional[torch.Tensor] = None,
        calibration_iq: Optional[torch.Tensor] = None,
        health_mask: Optional[torch.Tensor] = None,
        position_errors: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        precomputed_target_features: Optional[torch.Tensor] = None,
        quantize_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self._validate_spatial(initial_weights_iq, 2, "initial_weights_iq")
        batch = initial_weights_iq.shape[0]
        device, dtype = initial_weights_iq.device, initial_weights_iq.dtype
        target_angles_rad = target_angles_rad.to(device=device, dtype=dtype)

        if calibration_iq is None:
            calibration_iq = torch.zeros_like(initial_weights_iq)
            calibration_iq[:, 0] = 1.0
        else:
            self._validate_spatial(calibration_iq, 2, "calibration_iq")
            calibration_iq = calibration_iq.to(device=device, dtype=dtype)
        if health_mask is None:
            health_mask = initial_weights_iq.new_ones(
                batch, 1, self.config.x_num, self.config.y_num
            )
        else:
            self._validate_spatial(health_mask, 1, "health_mask")
            health_mask = health_mask.to(device=device, dtype=dtype)
        if torch.any(health_mask < 0.0) or torch.any(health_mask > 1.0):
            raise ValueError("health_mask 必须位于 [0,1]")
        if position_errors is None:
            position_errors = initial_weights_iq.new_zeros(
                batch, 3, self.config.x_num, self.config.y_num
            )
        else:
            self._validate_spatial(position_errors, 3, "position_errors")
            position_errors = position_errors.to(device=device, dtype=dtype)
        if state is None:
            state = initial_weights_iq.new_zeros(batch, self.state_dim)
        elif state.shape != (batch, self.state_dim):
            raise ValueError(f"state 必须为 [B,{self.state_dim}]")
        else:
            state = state.to(device=device, dtype=dtype)

        if precomputed_target_features is None:
            target_features = topology_target_features(
                target_angles_rad, target_mask, max_targets=self.max_targets
            )
        else:
            if precomputed_target_features.shape != (batch, 7):
                raise ValueError("precomputed_target_features 必须为 [B,7]")
            target_features = precomputed_target_features.to(device=device, dtype=dtype)
        context = torch.cat((target_features, state), dim=1)
        coordinates = self.normalised_coordinates.to(dtype=dtype).expand(batch, -1, -1, -1)
        position_features = torch.clamp(
            position_errors / self.position_error_scale, -1.0, 1.0
        )
        x = torch.cat(
            (
                initial_weights_iq,
                calibration_iq,
                health_mask,
                position_features,
                coordinates,
            ),
            dim=1,
        )
        x = self.input_activation(self.stem(x))
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        for block in self.blocks:
            x = block(x, context)
        if self.global_projection is not None:
            # GAP 仅形成全局上下文并广播；逐阵元张量 x 从未被替换。
            x = x + self.global_projection(x.mean(dim=(2, 3), keepdim=True))

        delta = self.residual_scale * torch.tanh(self.head(x))
        cr, ci = 1.0 + delta[:, 0], delta[:, 1]
        wr, wi = initial_weights_iq[:, 0], initial_weights_iq[:, 1]
        corrected = torch.stack((wr * cr - wi * ci, wr * ci + wi * cr), dim=1)
        prequant = project_complex_unit_disk_iq(corrected) * health_mask
        if quantize_output and self.output_quantizer is not None:
            weights = project_complex_unit_disk_iq(self.output_quantizer(prequant))
        else:
            weights = prequant
        return {
            "weights_iq": weights,
            "prequant_weights_iq": prequant,
            "delta_iq": delta,
        }

    def export_contract(self) -> Dict[str, object]:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "array": asdict(self.config),
            "topology": self.config.topology_name,
            "input_shape": ["B", 10, self.config.x_num, self.config.y_num],
            "output_shape": ["B", 2, self.config.x_num, self.config.y_num],
            "kernel_size": list(self.kernel_size),
            "state_schema": list(self.STATE_SCHEMA[: self.state_dim]),
            "max_targets": self.max_targets,
            "target_feature_schema": [
                "mean_ux",
                "mean_uy",
                "mean_uz",
                "mean_ux2",
                "mean_uy2",
                "mean_uz2",
                "target_count_over_max_targets",
            ],
            "complex_amplitude_limit": 1.0,
            "global_context": self.global_context,
        }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def profile_topology_model(
    model: TopologyAdaptiveBeamformingCNN,
    *,
    batch_size: int = 1,
) -> Dict[str, int]:
    """统计 Conv/Linear 核心 MAC；明确不含三角函数、激活和圆盘投影。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal macs
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        macs += int(output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal macs
        macs += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    parameter = next(model.parameters())
    initial = torch.zeros(
        batch_size,
        2,
        model.config.x_num,
        model.config.y_num,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    targets = torch.zeros(batch_size, 2, device=parameter.device, dtype=parameter.dtype)
    targets[:, 0] = math.pi / 2.0
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(initial, targets, quantize_output=False)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    counts = count_parameters(model)
    return {
        **counts,
        "macs_per_sample": int(macs // batch_size),
        "fp32_parameter_bytes": counts["total_parameters"] * 4,
        "int8_parameter_bytes_approx": counts["total_parameters"],
        "mac_scope": "Conv2d and Linear only",
    }


__all__ = [
    "ALGORITHM_VERSION",
    "ArrayGeometryConfig",
    "TopologyAdaptiveBeamformingCNN",
    "analytic_far_field_weights",
    "array_positions",
    "count_parameters",
    "element_response_from_amp_phase",
    "field_from_weights_iq",
    "profile_topology_model",
    "steering_iq",
    "steering_at_directions_iq",
    "topology_target_features",
]
