"""Receiver observation contracts and free-space claim guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class ObservationKind(str, Enum):
    HFSS_EPHI = "hfss_ephi"
    CALIBRATED_PROBE = "calibrated_probe"
    ENVIRONMENT_EFFECTIVE = "environment_effective"


@dataclass(frozen=True)
class ReceiverObservation:
    kind: ObservationKind | str
    frequency_hz: float
    probe_response_id: str | None = None
    environment_calibrated: bool = False

    def __post_init__(self) -> None:
        kind = ObservationKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if kind is ObservationKind.CALIBRATED_PROBE and not self.probe_response_id:
            raise ValueError("calibrated_probe requires probe_response_id")
        if kind is ObservationKind.ENVIRONMENT_EFFECTIVE and self.environment_calibrated:
            raise ValueError("environment_effective cannot claim environment calibration")

    @property
    def permits_free_space_label(self) -> bool:
        if self.kind is ObservationKind.HFSS_EPHI:
            return True
        return self.kind is ObservationKind.CALIBRATED_PROBE and self.environment_calibrated

    def validate_output_label(self, output_label: str) -> None:
        if output_label == "free_space_operator" and not self.permits_free_space_label:
            raise ValueError(
                "free_space_operator requires HFSS free-space data or a calibrated probe and environment"
            )

    def project(
        self,
        field: torch.Tensor,
        *,
        probe_response: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project complex fields to one coherent scalar receive channel per port."""
        if not torch.is_complex(field) or field.ndim < 1:
            raise ValueError("field must be complex")
        if self.kind is ObservationKind.ENVIRONMENT_EFFECTIVE and probe_response is None:
            if field.shape[-1] != 16:
                raise ValueError("environment_effective scalar field must end with 16 ports")
            return field
        if field.ndim < 2:
            raise ValueError("polarized field must have polarization and port axes")
        if self.kind is ObservationKind.HFSS_EPHI:
            if field.shape[-2] < 2:
                raise ValueError("hfss_ephi requires [Etheta,Ephi] polarization components")
            return field[..., 1, :]
        if probe_response is None:
            raise ValueError("probe_response is required for calibrated probe projection")
        response = torch.as_tensor(probe_response, device=field.device)
        if not torch.is_complex(response) or response.shape[-1] != field.shape[-2]:
            raise ValueError("probe_response must be complex and match polarization channels")
        return torch.einsum("...p,...pn->...n", response.conj(), field)
