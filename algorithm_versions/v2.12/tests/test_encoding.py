from __future__ import annotations

import torch

from v212.basis import FiltrationSSHTransform
from v212.encoding import (
    DirectionFourierEncoder,
    FiltrationSSHEncoder,
    MatchedFourierEncoder,
    MatchedFourierBasis,
    ResidualizedDirectionFourierEncoder,
    ResidualizedFourierSSHEncoder,
    fit_weighted_projection,
)


def directions(count: int = 320) -> torch.Tensor:
    generator = torch.Generator().manual_seed(213)
    value = torch.randn(count, 3, generator=generator)
    value[:, 0] = value[:, 0].abs().clamp_min(1e-3)
    return value / value.norm(dim=-1, keepdim=True)


def auxiliary_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for name, parameter in module.named_parameters() if "projection" in name or "alpha" in name)


def test_filtration_and_matched_fourier_are_exact_controls_at_zero_gate() -> None:
    q = directions()
    transform = FiltrationSSHTransform.fit(q.double(), 4)
    hybrid = FiltrationSSHEncoder(transform)
    matched = MatchedFourierEncoder(q, transform.ranks, extended_bands=5)
    expected = DirectionFourierEncoder(4)(q[None])
    torch.testing.assert_close(hybrid(q[None]), expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(matched(q[None]), expected, atol=0.0, rtol=0.0)
    assert auxiliary_parameter_count(hybrid) == auxiliary_parameter_count(matched)


def test_residualized_fourier_is_orthogonal_to_fitted_filtration() -> None:
    q = directions()
    transform = FiltrationSSHTransform.fit(q.double(), 4)
    encoder = ResidualizedFourierSSHEncoder(transform, q)
    residual = encoder.residualized_fourier(q[None])[0]
    phi = encoder.filtration_basis(q[None])[0]
    cross = phi.T @ residual / len(q)
    assert float(cross.abs().max()) < 1e-4
    assert encoder(q[None]).shape == (1, len(q), 27)


def test_general_weighted_projection_uses_gram_solve_and_is_orthogonal() -> None:
    q = directions()
    transform = FiltrationSSHTransform.fit(q.double(), 4)
    phi = transform.evaluate(q.double())
    mixing = torch.eye(phi.shape[1], dtype=torch.float64)
    mixing[0, 0] = 2.0
    nonorthogonal = phi @ mixing
    target = DirectionFourierEncoder(4)(q.double()[None])[0]
    weights = torch.linspace(1.0, 2.0, len(q), dtype=torch.float64)
    weights = weights / weights.sum()
    fit = fit_weighted_projection(nonorthogonal, target, weights)
    residual = target - nonorthogonal @ fit.coefficients
    assert not fit.used_fast_path
    assert fit.identity_error > 1e-3
    assert float((nonorthogonal.T @ (weights[:, None] * residual)).abs().max()) < 1e-9


def test_operator_direction_bases_have_controlled_dimensions() -> None:
    q = directions()
    transform = FiltrationSSHTransform.fit(q.double(), 5)
    residualized = ResidualizedDirectionFourierEncoder(transform, q)
    matched = MatchedFourierBasis(q, transform.ranks)
    assert residualized(q[None]).shape[-1] == 27
    assert matched(q[None]).shape[-1] == sum(transform.ranks[1:])
    assert residualized.orthogonality_residual < 1e-6
