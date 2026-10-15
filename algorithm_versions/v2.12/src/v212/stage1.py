"""Build the frozen V2.12 Stage-1 controlled encoding experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from v210.models import model_from_config as v210_model_from_config

from .basis import FiltrationSSHTransform
from .encoding import (
    DirectionFourierEncoder,
    FiltrationSSHEncoder,
    MatchedFourierEncoder,
    RawSSHEncoder,
    ResidualizedFourierSSHEncoder,
    raw_ssh_rms,
)


@dataclass(frozen=True)
class Stage1EncodingConfig:
    recipe: str
    ssh_degree: int = 4
    truncation_ratio: float = 1e-4
    seed: int = 21200


def build_stage1_model(
    base_model_config: Mapping[str, object],
    encoding_config: Stage1EncodingConfig,
    training_direction: torch.Tensor,
    *,
    loss_weights: torch.Tensor | None = None,
    transform: FiltrationSSHTransform | None = None,
):
    """Replace only the two V2.10 direction encoders; all other layers stay fixed."""
    recipe = encoding_config.recipe.upper()
    if transform is None and recipe != "E0":
        transform = FiltrationSSHTransform.fit(
            training_direction,
            encoding_config.ssh_degree,
            loss_weights=loss_weights,
            truncation_ratio=encoding_config.truncation_ratio,
        )
    model = v210_model_from_config(base_model_config)

    def encoder(seed_offset: int):
        seed = encoding_config.seed + seed_offset
        if recipe == "E0":
            return DirectionFourierEncoder(4)
        if transform is None:
            raise AssertionError("non-Fourier recipes require a fitted transform")
        if recipe in {"E0-M4", "E0-M5"}:
            expected_degree = 4 if recipe.endswith("4") else 5
            if transform.degree != expected_degree:
                raise ValueError(f"{recipe} requires an L{expected_degree} transform")
            return MatchedFourierEncoder(
                training_direction,
                transform.ranks,
                extended_bands=expected_degree + 1,
                loss_weights=loss_weights,
                seed=seed,
            )
        if recipe == "E-S4":
            if transform.degree != 4:
                raise ValueError("E-S4 requires degree 4")
            return RawSSHEncoder(
                4,
                raw_ssh_rms(training_direction, 4, loss_weights),
                include_fourier=False,
                gate_initial=1.0,
                seed=seed,
            )
        if recipe == "E1-R4":
            if transform.degree != 4:
                raise ValueError("E1-R4 requires degree 4")
            return RawSSHEncoder(
                4,
                raw_ssh_rms(training_direction, 4, loss_weights),
                include_fourier=True,
                seed=seed,
            )
        if recipe in {"E2-F4", "E2-F5"}:
            expected_degree = 4 if recipe.endswith("4") else 5
            if transform.degree != expected_degree:
                raise ValueError(f"{recipe} requires an L{expected_degree} transform")
            return FiltrationSSHEncoder(transform, seed=seed)
        if recipe == "E3-I":
            return ResidualizedFourierSSHEncoder(
                transform,
                training_direction,
                loss_weights=loss_weights,
                seed=seed,
            )
        raise ValueError(f"unsupported Stage-1 recipe: {encoding_config.recipe}")

    model.nominal_pattern.direction_encoder = encoder(0)
    model.angular_residual.direction_encoder = encoder(1000)
    if model.nominal_pattern.direction_encoder.output_channels != 27:
        raise AssertionError("nominal direction contract changed")
    if model.angular_residual.direction_encoder.output_channels != 27:
        raise AssertionError("residual direction contract changed")
    return model, transform
