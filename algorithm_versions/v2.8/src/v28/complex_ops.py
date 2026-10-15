"""Cartesian-IQ operations with explicit 4x4 element axes."""

from __future__ import annotations

import torch

from .contracts import ARRAY_COLS, ARRAY_ROWS


def validate_command_iq(command_iq: torch.Tensor) -> None:
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, ARRAY_ROWS, ARRAY_COLS):
        raise ValueError("command_iq must have shape [B,2,4,4]")


def complex_multiply_iq(left: torch.Tensor, right: torch.Tensor, iq_dim: int) -> torch.Tensor:
    left_real, left_imag = left.unbind(dim=iq_dim)
    right_real, right_imag = right.unbind(dim=iq_dim)
    real = left_real * right_real - left_imag * right_imag
    imag = left_real * right_imag + left_imag * right_real
    return torch.stack((real, imag), dim=iq_dim)


def contract_operator_iq(operator_iq: torch.Tensor, command_iq: torch.Tensor) -> torch.Tensor:
    """Contract only named row/column axes; returns [B,Q,xyz,IQ]."""
    validate_command_iq(command_iq)
    if operator_iq.ndim != 6 or operator_iq.shape[-4:] != (3, 2, ARRAY_ROWS, ARRAY_COLS):
        raise ValueError("operator_iq must have shape [B,Q,3,2,4,4]")
    if operator_iq.shape[0] != command_iq.shape[0]:
        raise ValueError("operator and command batch dimensions must agree")
    op_real, op_imag = operator_iq.unbind(dim=3)
    cmd_real, cmd_imag = command_iq.unbind(dim=1)
    real = torch.einsum("bqxrc,brc->bqx", op_real, cmd_real) - torch.einsum(
        "bqxrc,brc->bqx", op_imag, cmd_imag
    )
    imag = torch.einsum("bqxrc,brc->bqx", op_real, cmd_imag) + torch.einsum(
        "bqxrc,brc->bqx", op_imag, cmd_real
    )
    return torch.stack((real, imag), dim=-1)


def field_power(field_iq: torch.Tensor, range_m: torch.Tensor | None = None) -> torch.Tensor:
    if field_iq.ndim != 4 or field_iq.shape[-2:] != (3, 2):
        raise ValueError("field_iq must have shape [B,Q,3,2]")
    power = field_iq.square().sum(dim=(-1, -2))
    if range_m is not None:
        if range_m.shape != power.shape:
            raise ValueError("range_m must have shape [B,Q]")
        power = power / range_m.square().clamp_min(torch.finfo(power.dtype).eps)
    return power


def project_transverse_operator_iq(operator_iq: torch.Tensor, unit_direction: torch.Tensor) -> torch.Tensor:
    """Remove the radial Cartesian component from every element response."""
    if operator_iq.ndim != 6 or operator_iq.shape[-4:] != (3, 2, 4, 4):
        raise ValueError("operator_iq must have shape [B,Q,3,2,4,4]")
    if unit_direction.shape != (*operator_iq.shape[:2], 3):
        raise ValueError("unit_direction must have shape [B,Q,3]")
    radial = torch.einsum("bqx,bqxirc->bqirc", unit_direction, operator_iq)
    return operator_iq - torch.einsum("bqx,bqirc->bqxirc", unit_direction, radial)

