from __future__ import annotations

import torch
from pathlib import Path
import tempfile

from v212.audits import (
    gauge_invariant_spectrum,
    nested_filtration_spectrum,
    nested_operator_comparison,
    operator_graph_roughness,
    rank_one_q_matrix,
)
from v212.codebook import build_baseline_codebook
from v212.codebook_optimization import optimize_supplemental_commands
from v212.distillation import OperatorTarget, confidence_from_fisher, operator_distillation_loss
from v212.inversion import (
    graph_synchronization_initialization,
    jacobian_diagnostics,
    measurement_jacobian,
    recover_rank_one,
)


def test_rank_one_inversion_and_gauge_removed_jacobian() -> None:
    torch.manual_seed(215)
    codebook = build_baseline_codebook()
    train = codebook.indices("train")
    commands = codebook.weights[train]
    g = torch.randn(16, dtype=torch.complex64)
    powers = (commands @ g).abs().square()
    result = recover_rank_one(commands, powers, steps=600, learning_rate=2e-2)
    predicted = result.predicted_power
    nmse = (predicted - powers).square().sum() / powers.square().sum()
    assert float(nmse) < 2e-4
    jacobian, anchor = measurement_jacobian(g, commands)
    assert jacobian.shape == (64, 31)
    assert 0 <= anchor < 16
    diagnostics = jacobian_diagnostics(g, commands)
    assert diagnostics["fisher"].shape == (31, 31)


def test_distillation_and_gauge_invariant_spectrum() -> None:
    torch.manual_seed(216)
    g = torch.randn(48, 16, dtype=torch.complex64)
    q_matrix = rank_one_q_matrix(g)
    fisher = torch.eye(31)[None].repeat(48, 1, 1)
    confidence = confidence_from_fisher(fisher)
    torch.testing.assert_close(confidence, torch.ones_like(confidence))
    torch.testing.assert_close(operator_distillation_loss(q_matrix, q_matrix, confidence), torch.tensor(0.0))
    direction = torch.randn(48, 3, dtype=torch.float64)
    direction[:, 0] = direction[:, 0].abs().clamp_min(1e-3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    spectrum = gauge_invariant_spectrum(q_matrix, direction, degree=4)
    assert spectrum.degree_energy.shape == (5,)
    assert 0.0 <= spectrum.cumulative_fraction[4] <= 1.000001


def test_o1_optimizer_keeps_singles_pairs_and_constant_modulus() -> None:
    torch.manual_seed(217)
    baseline = build_baseline_codebook()
    train_g = torch.randn(4, 16, dtype=torch.complex64)
    validation_g = torch.randn(3, 16, dtype=torch.complex64)
    result = optimize_supplemental_commands(
        baseline,
        train_g,
        validation_g,
        restarts=1,
        steps=2,
        learning_rate=1e-2,
    )
    random_indices = [index for index, kind in enumerate(baseline.kinds) if kind == "random"]
    fixed_indices = [index for index in range(96) if index not in random_indices]
    torch.testing.assert_close(result.codebook.weights[fixed_indices], baseline.weights[fixed_indices])
    torch.testing.assert_close(
        result.codebook.weights[random_indices].abs(),
        torch.full((32, 16), 0.25),
        atol=1e-6,
        rtol=1e-6,
    )


def test_graph_synchronization_uses_pair_sign_convention() -> None:
    torch.manual_seed(219)
    codebook = build_baseline_codebook()
    g = torch.randn(16, dtype=torch.complex64) + (0.5 + 0.2j)
    power = (codebook.weights @ g).abs().square()
    estimate = graph_synchronization_initialization(power, codebook)
    torch.testing.assert_close(
        rank_one_q_matrix(estimate), rank_one_q_matrix(g), atol=2e-4, rtol=2e-4
    )


def test_operator_target_round_trip_keeps_leakage_contract() -> None:
    q_hat = rank_one_q_matrix(torch.randn(3, 16, dtype=torch.complex64))
    target = OperatorTarget(
        q_hat=q_hat,
        confidence=torch.ones(3),
        inversion_residual=torch.zeros(3),
        minimum_jacobian_singular_value=torch.ones(3),
        query_indices=torch.tensor([1, 4, 9]),
        command_indices=torch.arange(64),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "target.npz"
        target.save(path)
        loaded = OperatorTarget.load(path)
    torch.testing.assert_close(loaded.q_hat, target.q_hat)
    assert torch.equal(loaded.query_indices, target.query_indices)
    assert torch.equal(loaded.command_indices, target.command_indices)


def test_nested_filtration_is_monotone_and_identical_operator_has_zero_error() -> None:
    torch.manual_seed(223)
    direction = torch.randn(96, 3, dtype=torch.float64)
    direction[:, 0] = direction[:, 0].abs().clamp_min(1e-3)
    direction /= direction.norm(dim=1, keepdim=True)
    envelope = torch.complex(
        1.0 + 0.2 * direction[:, :1].float(),
        0.1 * direction[:, 1:2].float(),
    ).expand(-1, 16)
    q_tilde = rank_one_q_matrix(envelope)
    spectrum = nested_filtration_spectrum(q_tilde, direction, degree=3)
    assert torch.all(spectrum.cumulative_fraction[1:] >= spectrum.cumulative_fraction[:-1] - 1e-10)
    comparison = nested_operator_comparison(q_tilde, q_tilde, direction, degree=3)
    assert comparison.spectral_distance < 1e-10
    assert float(comparison.shell_error_fraction.max()) < 1e-10
    assert comparison.tail_error_fraction < 1e-10
    assert operator_graph_roughness(q_tilde, direction, neighbors=6) >= 0
