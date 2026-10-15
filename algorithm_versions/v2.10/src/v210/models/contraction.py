"""Exact transmit-convention complex contractions for scalar far fields."""

from __future__ import annotations

import torch


def contract_scalar_operator_iq(
    operator_iq: torch.Tensor,
    command_iq: torch.Tensor,
) -> torch.Tensor:
    """Return h^T w using ordinary complex products, never conjugating w."""
    if operator_iq.ndim != 5 or operator_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("operator_iq must have shape [B,Q,2,4,4]")
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [B,2,4,4]")
    if operator_iq.shape[0] != command_iq.shape[0]:
        raise ValueError("operator and command batch sizes must match")
    op_real, op_imag = operator_iq.unbind(dim=2)
    cmd_real, cmd_imag = command_iq.unbind(dim=1)
    real = torch.einsum("bqrc,brc->bq", op_real, cmd_real) - torch.einsum(
        "bqrc,brc->bq", op_imag, cmd_imag
    )
    imag = torch.einsum("bqrc,brc->bq", op_real, cmd_imag) + torch.einsum(
        "bqrc,brc->bq", op_imag, cmd_real
    )
    return torch.stack((real, imag), dim=-1)


def contract_scalar_operator_grid_iq(
    operator_iq: torch.Tensor,
    command_iq: torch.Tensor,
) -> torch.Tensor:
    """Contract one [Q] operator grid with a shared set of [C] commands."""
    if operator_iq.ndim != 4 or operator_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("operator_iq must have shape [Q,2,4,4]")
    if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [C,2,4,4]")
    op_real, op_imag = operator_iq.unbind(dim=1)
    cmd_real, cmd_imag = command_iq.unbind(dim=1)
    real = torch.einsum("qij,kij->kq", op_real, cmd_real) - torch.einsum(
        "qij,kij->kq", op_imag, cmd_imag
    )
    imag = torch.einsum("qij,kij->kq", op_real, cmd_imag) + torch.einsum(
        "qij,kij->kq", op_imag, cmd_real
    )
    return torch.stack((real, imag), dim=-1)
