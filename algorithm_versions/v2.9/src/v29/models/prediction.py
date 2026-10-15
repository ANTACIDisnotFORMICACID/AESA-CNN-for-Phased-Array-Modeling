"""Validated scalar far-field prediction returned by the V2.9 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class AmplitudePrediction:
    amplitude: torch.Tensor
    power: torch.Tensor
    field_iq: torch.Tensor
    operator_iq: torch.Tensor
    terms_iq: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amplitude.ndim != 2:
            raise ValueError("amplitude must have shape [B,Q]")
        if self.power.shape != self.amplitude.shape:
            raise ValueError("power must have shape [B,Q]")
        if self.field_iq.shape != (*self.amplitude.shape, 2):
            raise ValueError("field_iq must have shape [B,Q,2]")
        if self.operator_iq.shape != (*self.amplitude.shape, 2, 4, 4):
            raise ValueError("operator_iq must have shape [B,Q,2,4,4]")
        for name, value in self.terms_iq.items():
            if value.shape != self.operator_iq.shape:
                raise ValueError(f"operator term {name!r} has an incompatible shape")
