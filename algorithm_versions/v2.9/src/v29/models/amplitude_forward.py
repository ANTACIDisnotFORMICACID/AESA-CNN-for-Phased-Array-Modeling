"""V2.9 physics-layered scalar far-field model trained from amplitude only."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..contracts import ArrayGeometry, QueryBatch
from ..physics.basis_policy import BasisType, validate_basis_policy
from ..physics.ideal_far_field import IdealScalarFarFieldOperator
from .angular_residual import AngularOperatorResidual
from .contraction import contract_scalar_operator_grid_iq, contract_scalar_operator_iq
from .device import IdentityDeviceModel
from .nominal_pattern import NominalElementPattern
from .port_coupling import StructuredPortCoupling
from .prediction import AmplitudePrediction


class V29AmplitudeForwardModel(nn.Module):
    """Compose named physical stages; trainable layers never consume commands."""

    def __init__(
        self,
        geometry: ArrayGeometry | None = None,
        *,
        reference_distance_m: float = 1.0,
        basis_type: BasisType | str = BasisType.IDEAL_PHASE,
        source_type: str = "hfss_amplitude",
        fourier_bands: int = 4,
        nominal_width: int = 24,
        nominal_depth: int = 2,
        nominal_max_log_amplitude: float = 3.0,
        enable_coupling: bool = True,
        coupling_low_rank: int = 1,
        enable_residual: bool = True,
        residual_width: int = 12,
        residual_depth: int = 1,
        residual_maximum_magnitude: float = 2.0,
        device_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if reference_distance_m <= 0:
            raise ValueError("reference_distance_m must be positive")
        self.geometry = geometry or ArrayGeometry()
        self.reference_distance_m = float(reference_distance_m)
        self.basis_type = validate_basis_policy(
            basis_type,
            enable_coupling=enable_coupling,
        )
        if self.basis_type is not BasisType.IDEAL_PHASE:
            raise NotImplementedError(
                "the trainable V2.9 model currently supports ideal_phase only; "
                "isolated/common/embedded bases remain explicit audit policies"
            )
        self.source_type = str(source_type)
        self.enable_coupling = bool(enable_coupling)
        self.enable_residual = bool(enable_residual)
        self.device_model = device_model or IdentityDeviceModel()
        if self.source_type == "hfss_amplitude" and not isinstance(
            self.device_model, IdentityDeviceModel
        ):
            raise ValueError("HFSS amplitude data requires IdentityDeviceModel")

        self.ideal_operator = IdealScalarFarFieldOperator(self.geometry)
        self.nominal_pattern = NominalElementPattern(
            fourier_bands=fourier_bands,
            width=nominal_width,
            depth=nominal_depth,
            max_log_amplitude=nominal_max_log_amplitude,
        )
        self.port_coupling = StructuredPortCoupling(low_rank=coupling_low_rank)
        self.angular_residual = AngularOperatorResidual(
            fourier_bands=fourier_bands,
            width=residual_width,
            depth=residual_depth,
            maximum_magnitude=residual_maximum_magnitude,
        )

    def operator(
        self,
        query: QueryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ideal = self.ideal_operator(query)
        nominal, _ = self.nominal_pattern(ideal, query)
        nominal_delta = nominal - ideal
        coupled = self.port_coupling(nominal) if self.enable_coupling else nominal
        coupling_delta = coupled - nominal
        residual_delta = (
            self.angular_residual(query) if self.enable_residual else torch.zeros_like(coupled)
        )
        effective = coupled + residual_delta
        terms = {
            "ideal": ideal,
            "nominal_pattern_delta": nominal_delta,
            "port_coupling_delta": coupling_delta,
            "angular_residual_delta": residual_delta,
        }
        return effective, terms

    def _apply_range(
        self,
        field_iq: torch.Tensor,
        query: QueryBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude = torch.linalg.vector_norm(field_iq, dim=-1)
        if query.range_m is not None:
            factor = self.reference_distance_m / query.range_m.clamp_min(1e-8)
            field_iq = field_iq * factor[..., None]
            amplitude = amplitude * factor
        return field_iq, amplitude

    def forward(
        self,
        command_iq: torch.Tensor,
        query: QueryBatch,
        *,
        state: torch.Tensor | None = None,
    ) -> AmplitudePrediction:
        operator, terms = self.operator(query)
        effective_command = self.device_model(command_iq, state)
        field = contract_scalar_operator_iq(operator, effective_command)
        field, amplitude = self._apply_range(field, query)
        return AmplitudePrediction(
            amplitude=amplitude,
            power=amplitude.square(),
            field_iq=field,
            operator_iq=operator,
            terms_iq=terms,
        )

    def forward_command_grid(
        self,
        command_iq: torch.Tensor,
        unit_direction: torch.Tensor,
    ) -> torch.Tensor:
        amplitude, _ = self.forward_command_grid_with_terms(command_iq, unit_direction)
        return amplitude

    def forward_command_grid_with_terms(
        self,
        command_iq: torch.Tensor,
        unit_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if unit_direction.ndim != 2 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [Q,3]")
        operator, terms = self.operator(QueryBatch(unit_direction[None]))
        effective_command = self.device_model(command_iq, None)
        field = contract_scalar_operator_grid_iq(operator[0], effective_command)
        return torch.linalg.vector_norm(field, dim=-1), terms

    def structured_regularization(
        self,
        terms_iq: Mapping[str, torch.Tensor],
        *,
        residual_energy_weight: float,
        coupling_l2_weight: float,
        reciprocity_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual_energy = terms_iq["angular_residual_delta"].square().mean()
        coupling = self.port_coupling.regularization_terms()
        total = (
            residual_energy_weight * residual_energy
            + coupling_l2_weight * coupling["coupling_l2"]
            + reciprocity_weight * coupling["reciprocity"]
        )
        return total, {"residual_energy": residual_energy, **coupling}


def model_from_config(config: Mapping[str, object]) -> V29AmplitudeForwardModel:
    return V29AmplitudeForwardModel(
        reference_distance_m=float(config["reference_distance_m"]),
        basis_type=str(config.get("basis_type", "ideal_phase")),
        source_type=str(config.get("source_type", "hfss_amplitude")),
        fourier_bands=int(config.get("direction_fourier_bands", 4)),
        nominal_width=int(config.get("nominal_width", 24)),
        nominal_depth=int(config.get("nominal_depth", 2)),
        nominal_max_log_amplitude=float(config.get("nominal_max_log_amplitude", 3.0)),
        enable_coupling=bool(config.get("enable_coupling", True)),
        coupling_low_rank=int(config.get("coupling_low_rank", 1)),
        enable_residual=bool(config.get("enable_residual", True)),
        residual_width=int(config.get("residual_width", 12)),
        residual_depth=int(config.get("residual_depth", 1)),
        residual_maximum_magnitude=float(config.get("residual_maximum_magnitude", 2.0)),
    )


# Stable name retained for the V2.8-compatible training and optimization trunk.
AmplitudeOnlyForwardModel = V29AmplitudeForwardModel
