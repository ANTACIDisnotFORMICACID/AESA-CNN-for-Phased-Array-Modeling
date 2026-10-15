from __future__ import annotations

import torch

from v210.contracts import QueryBatch
from v212.audits import dephase_q_matrix, rank_one_q_matrix
from v212.basis import FiltrationSSHTransform
from v212.encoding import MatchedFourierBasis
from v212.operator import BranchAblation, CoherentRank1Operator


def directions(count: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(214)
    value = torch.randn(count, 3, generator=generator)
    value[:, 0] = value[:, 0].abs().clamp_min(1e-3)
    return value / value.norm(dim=-1, keepdim=True)


def test_g1_g2_zero_start_and_rank_one_physics() -> None:
    q_train = directions(256)
    transform = FiltrationSSHTransform.fit(q_train.double(), 4)
    q = directions(11)[None]
    query = QueryBatch(q)
    g1 = CoherentRank1Operator()
    g2 = CoherentRank1Operator(ssh_transform=transform)
    envelope, terms = g2.envelope(query)
    torch.testing.assert_close(terms["fourier_residual_iq"], torch.zeros_like(terms["fourier_residual_iq"]))
    torch.testing.assert_close(terms["ssh_iq"], torch.zeros_like(terms["ssh_iq"]))
    torch.testing.assert_close(envelope, torch.ones_like(envelope))
    operator1, _ = g1.operator(query)
    operator2, _ = g2.operator(query)
    torch.testing.assert_close(operator2, operator1, atol=2e-6, rtol=2e-6)

    complex_g = torch.complex(operator2[:, :, 0], operator2[:, :, 1]).reshape(1, 11, 16)
    matrix = g2.q_matrix(query)
    torch.testing.assert_close(matrix, rank_one_q_matrix(complex_g))
    torch.testing.assert_close(matrix, matrix.mH, atol=1e-6, rtol=1e-6)
    eigenvalues = torch.linalg.eigvalsh(matrix).real
    assert float(eigenvalues[..., :-1].abs().max().detach()) < 2e-5
    assert float(eigenvalues[..., -1].min().detach()) > 0


def test_quadratic_measurement_and_dephasing_are_exact() -> None:
    q = directions(7)[None]
    query = QueryBatch(q)
    model = CoherentRank1Operator()
    command = torch.randn(1, 2, 4, 4)
    command = command / command.square().sum().sqrt()
    prediction = model(command, query)
    torch.testing.assert_close(prediction.g_iq, prediction.operator_iq)
    matrix = model.q_matrix(query)
    w = torch.complex(command[:, 0], command[:, 1]).reshape(1, 16)
    quadratic = torch.einsum("bn,bqnm,bm->bq", w.conj(), matrix, w).real
    torch.testing.assert_close(quadratic, prediction.power, atol=2e-5, rtol=2e-5)

    direct = model.dephased_q_matrix(query)
    computed = dephase_q_matrix(
        matrix,
        q,
        model.positions_m,
        model.wavenumber_rad_m,
    )
    torch.testing.assert_close(computed, direct, atol=3e-5, rtol=3e-5)
    phase = torch.exp(1j * torch.linspace(0.0, 1.0, q.shape[1]))[None, :, None]
    operator, _ = model.operator(query)
    g = torch.complex(operator[:, :, 0], operator[:, :, 1]).reshape(1, 7, 16)
    torch.testing.assert_close(rank_one_q_matrix(g * phase), rank_one_q_matrix(g))


def test_matched_fourier_has_exact_ssh_parameter_capacity_and_branch_ablation() -> None:
    q_train = directions(320)
    transform = FiltrationSSHTransform.fit(q_train.double(), 5)
    ssh = CoherentRank1Operator(ssh_transform=transform, ssh_seed=91)
    matched = CoherentRank1Operator(
        matched_fourier_basis=MatchedFourierBasis(q_train, transform.ranks, seed=91),
        ssh_seed=91,
    )
    ssh_aux = sum(value.numel() for name, value in ssh.named_parameters() if name.startswith("ssh_"))
    matched_aux = sum(value.numel() for name, value in matched.named_parameters() if name.startswith("ssh_"))
    assert ssh_aux == matched_aux
    with torch.no_grad():
        ssh.ssh_alpha.fill_(0.2)
    query = QueryBatch(directions(12)[None])
    full, terms = ssh.envelope(query)
    removed, _ = ssh.envelope(query, BranchAblation(disable_ssh=True))
    assert all(f"ssh_l{degree}_iq" in terms for degree in range(1, 6))
    assert not torch.allclose(full, removed)


def test_residualized_operator_input_is_explicit_and_zero_start_compatible() -> None:
    q_train = directions(320)
    transform = FiltrationSSHTransform.fit(q_train.double(), 5)
    model = CoherentRank1Operator(
        ssh_transform=transform,
        training_direction=q_train,
        fourier_input_mode="filtration_residualized",
    )
    assert model.fourier_input_mode == "filtration_residualized"
    envelope, terms = model.envelope(QueryBatch(directions(8)[None]))
    torch.testing.assert_close(terms["ssh_iq"], torch.zeros_like(terms["ssh_iq"]))
    torch.testing.assert_close(terms["fourier_residual_iq"], torch.zeros_like(terms["fourier_residual_iq"]))
    torch.testing.assert_close(envelope, torch.ones_like(envelope))
