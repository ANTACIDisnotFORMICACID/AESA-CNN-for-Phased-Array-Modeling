"""Optimization contracts with an explicit four-query requirement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import QueryBatch


@dataclass(frozen=True)
class FourTargetProblem:
    query: QueryBatch
    total_command_power: float = 1.0
    per_element_amplitude_max: float = 1.0

    def __post_init__(self) -> None:
        if self.query.unit_direction.shape[:2] != (1, 4):
            raise ValueError("four-target optimization requires query shape [1,4,3]")
        if self.total_command_power <= 0 or self.per_element_amplitude_max <= 0:
            raise ValueError("command constraints must be positive")
        if 16 * self.per_element_amplitude_max**2 < self.total_command_power:
            raise ValueError("power and per-element amplitude constraints are infeasible")


@dataclass(frozen=True)
class FourTargetResult:
    command_iq: torch.Tensor
    target_power: torch.Tensor
    min_target_power: float
    history: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.command_iq.shape != (2, 4, 4):
            raise ValueError("command_iq must have shape [2,4,4]")
        if self.target_power.shape != (4,):
            raise ValueError("target_power must have shape [4]")

