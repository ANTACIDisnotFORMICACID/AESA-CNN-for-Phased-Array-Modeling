"""V2.5 physics-consistent full-pattern compatibility layer.

The legacy CNN learned the deterministic mapping from element excitations to a
64 x 64 complex far-field pattern.  V2.5 keeps the V2.4 beamforming controller
unchanged and exposes the same *function* through the array manifold:

    E(theta, phi) = S(theta, phi; r + delta_r) diag(g) C w

The renderer has no trainable parameters.  A learned residual may be added in a
future version only for measured model mismatch; the old 2.35 M parameter image
decoder is deliberately not reintroduced.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from beamforming_pre_fpga import (
    ArrayGeometryConfig,
    TopologyAdaptiveBeamformingCNN,
    field_from_weights_iq,
    steering_iq,
)


ALGORITHM_VERSION = "2.5.0"
_LEGACY_MAGNITUDE_THRESHOLD = 1.0e-12
_LEGACY_ZERO_DB = -80.0
_NORMALIZED_DB_FLOOR = -80.0


def legacy_forward_1x4_config() -> ArrayGeometryConfig:
    """Return the geometry recorded by the historical 1x4 CNN dataset."""

    return ArrayGeometryConfig(
        wavelength=6.83,
        x_spacing=6.83,
        y_spacing=15.0,
        x_num=1,
        y_num=4,
    )


def weights_iq_from_amplitude_phase(
    amplitude: torch.Tensor,
    phase_rad: torch.Tensor,
    config: ArrayGeometryConfig,
) -> torch.Tensor:
    """Adapt the old amplitude/phase call convention to ``[B,2,H,W]`` I/Q."""

    if amplitude.ndim == 2:
        amplitude = amplitude.unsqueeze(0)
    if phase_rad.ndim == 2:
        phase_rad = phase_rad.unsqueeze(0)
    expected = (amplitude.shape[0], config.x_num, config.y_num)
    if tuple(amplitude.shape) != expected or tuple(phase_rad.shape) != expected:
        raise ValueError(f"amplitude/phase_rad 必须为 [B,{config.x_num},{config.y_num}]")
    if not torch.is_floating_point(amplitude) or not torch.is_floating_point(phase_rad):
        raise ValueError("amplitude/phase_rad 必须是浮点张量")
    if amplitude.dtype != phase_rad.dtype:
        raise ValueError("amplitude/phase_rad dtype 必须一致")
    if not bool(torch.isfinite(amplitude).all()) or not bool(torch.isfinite(phase_rad).all()):
        raise ValueError("amplitude/phase_rad 包含 NaN 或 Inf")
    if bool(torch.any(amplitude < 0.0)):
        raise ValueError("amplitude 不得为负")
    phase_rad = phase_rad.to(device=amplitude.device)
    return torch.stack(
        (amplitude * torch.cos(phase_rad), amplitude * torch.sin(phase_rad)), dim=1
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True)
class PatternGridConfig:
    """Angular grid contract; rows are phi and columns are theta."""

    theta_min_deg: float = 60.0
    theta_max_deg: float = 120.0
    theta_points: int = 64
    phi_min_deg: float = -30.0
    phi_max_deg: float = 30.0
    phi_points: int = 64

    def __post_init__(self) -> None:
        values = (
            self.theta_min_deg,
            self.theta_max_deg,
            self.phi_min_deg,
            self.phi_max_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("角度范围必须为有限数")
        if self.theta_max_deg <= self.theta_min_deg:
            raise ValueError("theta_max_deg 必须大于 theta_min_deg")
        if self.phi_max_deg <= self.phi_min_deg:
            raise ValueError("phi_max_deg 必须大于 phi_min_deg")
        if self.theta_points < 2 or self.phi_points < 2:
            raise ValueError("每个角度轴至少需要两个采样点")

    @classmethod
    def legacy_64(cls) -> "PatternGridConfig":
        """Return the exact angular-grid convention used by the old dataset."""

        return cls()

    @property
    def shape(self) -> tuple[int, int]:
        return self.phi_points, self.theta_points

    def axes(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.linspace(
            self.theta_min_deg,
            self.theta_max_deg,
            self.theta_points,
            device=device,
            dtype=dtype,
        )
        phi = torch.linspace(
            self.phi_min_deg,
            self.phi_max_deg,
            self.phi_points,
            device=device,
            dtype=dtype,
        )
        return torch.deg2rad(theta), torch.deg2rad(phi)


def _validate_weights(weights_iq: torch.Tensor, config: ArrayGeometryConfig) -> None:
    expected = (2, config.x_num, config.y_num)
    if weights_iq.ndim != 4 or tuple(weights_iq.shape[1:]) != expected:
        raise ValueError(f"weights_iq 必须为 [B,2,{config.x_num},{config.y_num}]")
    if not torch.is_floating_point(weights_iq):
        raise ValueError("weights_iq 必须是浮点 I/Q 张量")
    if not bool(torch.isfinite(weights_iq).all()):
        raise ValueError("weights_iq 包含 NaN 或 Inf")


def apply_coupling_iq(
    weights_iq: torch.Tensor,
    coupling_iq: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply a per-sample complex coupling matrix ``C`` to commands ``w``.

    ``coupling_iq`` may be complex ``[B,N,N]`` or explicit-I/Q
    ``[B,2,N,N]``.  ``None`` is the identity mapping.
    """

    if coupling_iq is None:
        return weights_iq
    batch, _, height, width = weights_iq.shape
    elements = height * width
    weights = weights_iq.flatten(start_dim=2)
    if torch.is_complex(coupling_iq):
        if tuple(coupling_iq.shape) != (batch, elements, elements):
            raise ValueError(f"复 coupling_iq 必须为 [B,{elements},{elements}]")
        if coupling_iq.real.dtype != weights_iq.dtype:
            raise ValueError("复 coupling_iq 精度必须与 weights_iq 一致")
        coupling_iq = coupling_iq.to(device=weights_iq.device)
        if not bool(torch.isfinite(coupling_iq).all()):
            raise ValueError("coupling_iq 包含 NaN 或 Inf")
        real, imag = coupling_iq.real.to(weights_iq.dtype), coupling_iq.imag.to(
            weights_iq.dtype
        )
    else:
        expected = (batch, 2, elements, elements)
        if tuple(coupling_iq.shape) != expected:
            raise ValueError(f"I/Q coupling_iq 必须为 {expected}")
        if not torch.is_floating_point(coupling_iq) or coupling_iq.dtype != weights_iq.dtype:
            raise ValueError("I/Q coupling_iq dtype 必须与 weights_iq 一致")
        coupling_iq = coupling_iq.to(device=weights_iq.device)
        if not bool(torch.isfinite(coupling_iq).all()):
            raise ValueError("coupling_iq 包含 NaN 或 Inf")
        real, imag = coupling_iq[:, 0], coupling_iq[:, 1]

    wr, wi = weights[:, 0], weights[:, 1]
    effective_real = torch.einsum("bnm,bm->bn", real, wr) - torch.einsum(
        "bnm,bm->bn", imag, wi
    )
    effective_imag = torch.einsum("bnm,bm->bn", real, wi) + torch.einsum(
        "bnm,bm->bn", imag, wr
    )
    return torch.stack((effective_real, effective_imag), dim=1).reshape_as(weights_iq)


class FullPatternPhysics(nn.Module):
    """Zero-parameter 1x4/4x4 full-pattern renderer with legacy 64x64 mode."""

    def __init__(
        self,
        array_config: ArrayGeometryConfig,
        grid_config: Optional[PatternGridConfig] = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("renderer dtype 必须是浮点类型")
        self.array_config = array_config
        self.grid_config = grid_config or PatternGridConfig.legacy_64()
        self._construction_dtype = dtype
        theta_axis, phi_axis = self.grid_config.axes(dtype=dtype)
        theta_grid, phi_grid = torch.meshgrid(theta_axis, phi_axis, indexing="xy")
        cached_steering = steering_iq(array_config, theta_grid, phi_grid)
        # These buffers are deterministic from the exported configs; excluding them
        # from state_dict avoids silently adding 128/512 KiB to controller checkpoints.
        self.register_buffer("theta_axis_rad", theta_axis, persistent=False)
        self.register_buffer("phi_axis_rad", phi_axis, persistent=False)
        self.register_buffer("theta_grid_rad", theta_grid, persistent=False)
        self.register_buffer("phi_grid_rad", phi_grid, persistent=False)
        self.register_buffer("cached_steering_iq", cached_steering, persistent=False)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.grid_config.shape

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        weights_iq: torch.Tensor,
        *,
        position_errors: Optional[torch.Tensor] = None,
        element_response_iq: Optional[torch.Tensor] = None,
        coupling_iq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _validate_weights(weights_iq, self.array_config)
        if (
            weights_iq.dtype != self._construction_dtype
            or self.theta_axis_rad.dtype != self._construction_dtype
        ):
            raise ValueError(
                "weights_iq dtype 必须与 renderer 构造 dtype 一致；"
                "请显式构造对应精度的 FullPatternPhysics，不要对低精度缓存直接升精度"
            )
        effective_weights = apply_coupling_iq(weights_iq, coupling_iq)
        if element_response_iq is not None:
            if element_response_iq.shape != weights_iq.shape:
                raise ValueError("element_response_iq 必须与 weights_iq 同形")
            if (
                not torch.is_floating_point(element_response_iq)
                or element_response_iq.dtype != weights_iq.dtype
            ):
                raise ValueError("element_response_iq dtype 必须与 weights_iq 一致")
            element_response_iq = element_response_iq.to(device=weights_iq.device)
            if not bool(torch.isfinite(element_response_iq).all()):
                raise ValueError("element_response_iq 包含 NaN 或 Inf")
        if position_errors is None:
            steering = self.cached_steering_iq.to(
                device=weights_iq.device, dtype=weights_iq.dtype
            )
        else:
            expected = (
                weights_iq.shape[0],
                3,
                self.array_config.x_num,
                self.array_config.y_num,
            )
            if tuple(position_errors.shape) != expected:
                raise ValueError(f"position_errors 必须为 {expected}")
            if (
                not torch.is_floating_point(position_errors)
                or position_errors.dtype != weights_iq.dtype
            ):
                raise ValueError("position_errors dtype 必须与 weights_iq 一致")
            position_errors = position_errors.to(device=weights_iq.device)
            if not bool(torch.isfinite(position_errors).all()):
                raise ValueError("position_errors 包含 NaN 或 Inf")
            steering = steering_iq(
                self.array_config,
                self.theta_grid_rad.to(
                    device=weights_iq.device, dtype=weights_iq.dtype
                ),
                self.phi_grid_rad.to(
                    device=weights_iq.device, dtype=weights_iq.dtype
                ),
                position_errors=position_errors,
            )

        field = field_from_weights_iq(
            effective_weights,
            steering,
            element_response_iq=element_response_iq,
        )
        real, imag = field[:, 0], field[:, 1]
        magnitude = torch.sqrt(real.square() + imag.square())
        safe_magnitude = magnitude.clamp_min(_LEGACY_MAGNITUDE_THRESHOLD)
        magnitude_db = 20.0 * torch.log10(safe_magnitude)
        magnitude_db = torch.where(
            magnitude < _LEGACY_MAGNITUDE_THRESHOLD,
            magnitude_db.new_full((), _LEGACY_ZERO_DB),
            magnitude_db,
        )
        peak_magnitude = magnitude.amax(dim=(-2, -1), keepdim=True)
        relative_magnitude = magnitude / peak_magnitude.clamp_min(
            _LEGACY_MAGNITUDE_THRESHOLD
        )
        magnitude_db_normalized = 20.0 * torch.log10(
            relative_magnitude.clamp_min(10.0 ** (_NORMALIZED_DB_FLOOR / 20.0))
        )
        zero_pattern = peak_magnitude < _LEGACY_MAGNITUDE_THRESHOLD
        magnitude_db_normalized = torch.where(
            zero_pattern,
            magnitude_db_normalized.new_full((), _NORMALIZED_DB_FLOOR),
            magnitude_db_normalized,
        )
        phase = torch.remainder(torch.atan2(imag, real), 2.0 * math.pi)
        return {
            "field_iq": field,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
            "magnitude_db_normalized": magnitude_db_normalized,
            "phase_rad": phase,
            "theta_axis_rad": self.theta_axis_rad.to(
                device=weights_iq.device, dtype=weights_iq.dtype
            ),
            "phi_axis_rad": self.phi_axis_rad.to(
                device=weights_iq.device, dtype=weights_iq.dtype
            ),
        }

    def from_amplitude_phase(
        self,
        amplitude: torch.Tensor,
        phase_rad: torch.Tensor,
        **nonideal_inputs: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Render through the historical amplitude/phase input convention."""

        weights_iq = weights_iq_from_amplitude_phase(
            amplitude, phase_rad, self.array_config
        )
        return self(weights_iq, **nonideal_inputs)

    def export_contract(self) -> Dict[str, object]:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "implementation": "deterministic_physics_renderer",
            "renderer_dtype": str(self._construction_dtype).replace("torch.", ""),
            "array": asdict(self.array_config),
            "grid": asdict(self.grid_config),
            "input_shape": [
                "B",
                2,
                self.array_config.x_num,
                self.array_config.y_num,
            ],
            "pattern_output_shape": [
                "B",
                2,
                self.grid_config.phi_points,
                self.grid_config.theta_points,
            ],
            "row_axis": "phi",
            "column_axis": "theta",
            "phase_range_rad": {
                "lower": 0.0,
                "upper": 2.0 * math.pi,
                "upper_inclusive": False,
            },
            "magnitude_db": "absolute_20log10_with_legacy_zero_floor",
            "trainable_parameters": 0,
            "supports_position_errors": True,
            "supports_element_response": True,
            "supports_complex_coupling": True,
        }

    def to_legacy_payload(
        self,
        result: Mapping[str, torch.Tensor],
        *,
        weights_iq: torch.Tensor,
        sample_index: int = 0,
        run_id: Optional[str] = None,
        targets: Optional[Mapping[str, object]] = None,
        saved_at: Optional[str] = None,
    ) -> Dict[str, object]:
        """Create the old JSON-compatible in-memory payload without writing a file."""

        magnitude_db = result["magnitude_db"]
        phase_rad = result["phase_rad"]
        if magnitude_db.ndim != 3 or phase_rad.shape != magnitude_db.shape:
            raise ValueError("result 必须包含 [B,phi,theta] 的 magnitude_db/phase_rad")
        expected_result_shape = (magnitude_db.shape[0], *self.grid_shape)
        if tuple(magnitude_db.shape) != expected_result_shape:
            raise ValueError(f"result 空间形状必须为 {self.grid_shape}")
        if not 0 <= sample_index < magnitude_db.shape[0]:
            raise IndexError("sample_index 超出batch范围")
        _validate_weights(weights_iq, self.array_config)
        if weights_iq.shape[0] != magnitude_db.shape[0]:
            raise ValueError("weights_iq 与 result 的 batch 必须一致")
        magnitude_rows = magnitude_db[sample_index].detach().cpu().to(torch.float64)
        phase_rows = phase_rad[sample_index].detach().cpu().to(torch.float64)
        run_id = run_id or str(uuid.uuid4())[:8]
        payload: Dict[str, object] = {
            "header": {
                "description": "二维波束方向图（dB 与相位）",
                "rows": self.grid_config.phi_points,
                "cols": self.grid_config.theta_points,
                "array_rows": self.array_config.x_num,
                "array_cols": self.array_config.y_num,
                "x_dis": self.array_config.x_spacing,
                "y_dis": self.array_config.y_spacing,
                "wavlen": self.array_config.wavelength,
                "theta_span_deg": [
                    self.grid_config.theta_min_deg,
                    self.grid_config.theta_max_deg,
                ],
                "phi_span_deg": [
                    self.grid_config.phi_min_deg,
                    self.grid_config.phi_max_deg,
                ],
                "phase_unit": "rad",
                "phase_unit_cn": "弧度",
                "row_axis": "phi",
                "column_axis": "theta",
                "producer": "V2.5 deterministic full-pattern physics",
                "saved_at": saved_at or time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime()
                ),
            },
            "targets": _json_compatible(dict(targets or {})),
            "data_file": None,
            "data_shape": list(self.grid_shape),
            "dtype": "float64",
            "mag_db_rows": magnitude_rows.tolist(),
            "phase_rows": phase_rows.tolist(),
            "run_id": run_id,
            "notes": "绝对幅度dB；相位范围[0,2pi)；行phi、列theta。",
        }
        sample = weights_iq[sample_index].detach().cpu().to(torch.float64)
        amplitude = torch.sqrt(sample[0].square() + sample[1].square())
        phase = torch.remainder(torch.atan2(sample[1], sample[0]), 2.0 * math.pi)
        # 沿用旧文件的历史字段名；它们实际保存的是最终阵元幅度和相位。
        payload["applied_amp_errors"] = amplitude.flatten().tolist()
        payload["applied_phase_errors_rad"] = phase.flatten().tolist()
        return payload


class V25BeamformingSystem(nn.Module):
    """Compose an unchanged V2.4 controller with the V2.5 full-pattern observer."""

    def __init__(
        self,
        controller: TopologyAdaptiveBeamformingCNN,
        renderer: FullPatternPhysics,
    ) -> None:
        super().__init__()
        if controller.config != renderer.array_config:
            raise ValueError("controller 与 renderer 的阵列配置必须一致")
        self.controller = controller
        self.renderer = renderer

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
        element_response_iq: Optional[torch.Tensor] = None,
        coupling_iq: Optional[torch.Tensor] = None,
        quantize_output: bool = True,
        render_pattern: bool = True,
    ) -> Dict[str, object]:
        controller_output = self.controller(
            initial_weights_iq,
            target_angles_rad,
            target_mask=target_mask,
            calibration_iq=calibration_iq,
            health_mask=health_mask,
            position_errors=position_errors,
            state=state,
            precomputed_target_features=precomputed_target_features,
            quantize_output=quantize_output,
        )
        output: Dict[str, object] = dict(controller_output)
        if render_pattern:
            output["full_pattern"] = self.renderer(
                controller_output["weights_iq"],
                position_errors=position_errors,
                element_response_iq=element_response_iq,
                coupling_iq=coupling_iq,
            )
        return output


def profile_full_pattern(
    array_config: ArrayGeometryConfig,
    grid_config: Optional[PatternGridConfig] = None,
    *,
    controller_macs: int = 0,
    dtype_bytes: int = 4,
) -> Dict[str, int]:
    """Static arithmetic/storage profile for the cached-grid renderer."""

    grid = grid_config or PatternGridConfig.legacy_64()
    complex_macs = grid.theta_points * grid.phi_points * array_config.num_elements
    renderer_real_macs = 4 * complex_macs
    coupling_real_macs = 4 * array_config.num_elements**2
    element_response_real_macs = 4 * array_config.num_elements
    nonideal_real_macs = (
        renderer_real_macs + coupling_real_macs + element_response_real_macs
    )
    steering_bytes = 2 * complex_macs * dtype_bytes
    return {
        "trainable_parameters": 0,
        "grid_points": grid.theta_points * grid.phi_points,
        "complex_macs": complex_macs,
        "renderer_real_macs": renderer_real_macs,
        "optional_coupling_real_macs": coupling_real_macs,
        "optional_element_response_real_macs": element_response_real_macs,
        "nonideal_contraction_real_macs": nonideal_real_macs,
        "controller_macs": int(controller_macs),
        "end_to_end_real_macs": renderer_real_macs + int(controller_macs),
        "end_to_end_nonideal_contraction_real_macs": nonideal_real_macs
        + int(controller_macs),
        "cached_steering_bytes": steering_bytes,
    }


__all__ = [
    "ALGORITHM_VERSION",
    "FullPatternPhysics",
    "PatternGridConfig",
    "V25BeamformingSystem",
    "apply_coupling_iq",
    "legacy_forward_1x4_config",
    "profile_full_pattern",
    "weights_iq_from_amplitude_phase",
]
