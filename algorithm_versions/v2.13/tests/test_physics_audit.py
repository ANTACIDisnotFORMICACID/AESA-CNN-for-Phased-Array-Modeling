from __future__ import annotations

import math

import torch

from v213.physics_audit import (
    compare_operators,
    deembed_geometric_phase,
    gradient_alignment,
    nested_vector_spectrum,
    operator_contract_metrics,
    phase_only_maximin,
    rank_one_q,
)


def _complex_randn(*shape: int) -> torch.Tensor:
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def test_rank_one_operator_contract_is_exact() -> None:
    torch.manual_seed(1)
    g = _complex_randn(12, 16).to(torch.complex128)
    command = _complex_randn(7, 16).to(torch.complex128)
    command = command / command.norm(dim=1, keepdim=True)
    report = operator_contract_metrics(g, command)
    assert report["hermitian_relative_max"] < 1e-12
    assert report["lambda2_over_lambda1_p95"] < 1e-12
    assert report["quadratic_identity_relative_l2"] < 1e-12


def test_deembedding_removes_nominal_phase() -> None:
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]], dtype=torch.float64)
    positions = torch.zeros((16, 3), dtype=torch.float64)
    positions[:, 1] = torch.arange(16, dtype=torch.float64) * 0.01
    envelope = _complex_randn(2, 16).to(torch.complex128)
    phase = 12.0 * torch.einsum("qk,nk->qn", direction, positions)
    g = envelope * torch.exp(1j * phase)
    recovered = deembed_geometric_phase(g, direction, positions, 12.0)
    torch.testing.assert_close(recovered, envelope)


def test_nested_spectrum_identifies_constant_field() -> None:
    generator = torch.Generator().manual_seed(2)
    direction = torch.randn((400, 3), generator=generator, dtype=torch.float64)
    direction[:, 0] = direction[:, 0].abs()
    direction = direction / direction.norm(dim=1, keepdim=True)
    value = torch.ones((400, 3), dtype=torch.complex128)
    report = nested_vector_spectrum(value, direction, degree=5)
    assert report["cumulative_fraction"]["0"] > 1.0 - 1e-10
    assert report["normalized_filtration_gradient"] < 1e-8


def test_operator_comparison_and_gradient_alignment_are_gauge_invariant() -> None:
    torch.manual_seed(3)
    g = _complex_randn(8, 16)
    phase = torch.exp(1j * torch.linspace(0, math.pi, 8))[:, None]
    truth = rank_one_q(g)
    same = rank_one_q(g * phase)
    command = _complex_randn(5, 16)
    assert compare_operators(same, truth)["aggregate_nmse"] < 1e-12
    assert gradient_alignment(same, truth, command)["cosine_p05"] > 1.0 - 1e-6


def test_phase_only_maximin_improves_over_seeded_random_floor() -> None:
    torch.manual_seed(4)
    g = _complex_randn(3, 4, 16)
    q_matrix = g.conj()[..., :, None] * g[..., None, :]
    weight, objective = phase_only_maximin(q_matrix, starts=4, steps=20, seed=5)
    assert weight.shape == (3, 16)
    torch.testing.assert_close(weight.abs(), torch.full_like(weight.abs(), 0.25), atol=1e-6, rtol=0)
    assert torch.all(objective >= 0)
