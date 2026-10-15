"""Attach V2.11 direction encoders to the frozen V2.10 physical CNN model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from v210.models import model_from_config as v210_model_from_config

from .models import DirectionEncodingConfig, build_direction_encoder, load_ssh_statistics


def _encoding_config(config: Mapping[str, object]) -> DirectionEncodingConfig:
    return DirectionEncodingConfig(
        kind=str(config.get("kind", "fourier")),
        fourier_bands=int(config.get("fourier_bands", 4)),
        ssh_degree=int(config.get("ssh_degree", 4)),
        output_channels=int(config.get("output_channels", 27)),
        normalization_epsilon=float(config.get("normalization_epsilon", 1e-6)),
    )


def build_v211_model(
    base_model_config: Mapping[str, object],
    encoding_config: Mapping[str, object],
    *,
    statistics_path: str | Path | None = None,
):
    """Build V2.10 physics layers and replace only both direction encoders."""
    model = v210_model_from_config(base_model_config)
    encoder_config = _encoding_config(encoding_config)
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    if encoder_config.kind != "fourier":
        if statistics_path is None:
            channel_count = (encoder_config.ssh_degree + 1) ** 2
            mean = torch.zeros(channel_count)
            std = torch.ones(channel_count)
        else:
            mean, std = load_ssh_statistics(statistics_path, encoder_config.ssh_degree)
    model.nominal_pattern.direction_encoder = build_direction_encoder(
        encoder_config,
        ssh_mean=mean,
        ssh_std=std,
    )
    model.angular_residual.direction_encoder = build_direction_encoder(
        encoder_config,
        ssh_mean=mean,
        ssh_std=std,
    )
    if model.nominal_pattern.head.stem.in_channels != 31:
        raise AssertionError("nominal CNN input contract changed")
    if model.angular_residual.branch.stem.in_channels != 31:
        raise AssertionError("residual CNN input contract changed")
    return model


def load_v211_checkpoint_model(checkpoint: Mapping[str, object], device: str | torch.device):
    model = build_v211_model(
        checkpoint["base_model_config"],
        checkpoint["encoding_config"],
        statistics_path=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model
