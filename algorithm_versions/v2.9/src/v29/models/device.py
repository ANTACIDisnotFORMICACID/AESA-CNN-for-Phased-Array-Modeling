"""Command-side device models. HFSS V2.9 intentionally uses identity only."""

from __future__ import annotations

import torch
from torch import nn


class IdentityDeviceModel(nn.Module):
    def forward(
        self,
        command_iq: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
            raise ValueError("command_iq must have shape [B,2,4,4]")
        if state is not None:
            raise ValueError("IdentityDeviceModel does not accept device state")
        return command_iq
