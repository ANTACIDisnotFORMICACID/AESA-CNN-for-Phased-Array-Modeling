from __future__ import annotations

import torch

from v212.audits import rank_one_q_matrix
from v212.basis import FiltrationSSHTransform
from v212.evaluation import (
    amplitude_metrics,
    angular_distance_quartile_report,
    cross_generalization_report,
    spatial_gap_partition,
    stratified_separation_subset,
)
from v212.codebook import build_baseline_codebook
from v212.operator import CoherentRank1Operator
from v212.training import OperatorTrainConfig, train_operator_model
from v212.training import amplitude_objective


def test_cross_generalization_has_four_disjoint_semantic_cells() -> None:
    target = torch.linspace(0.1, 2.0, 8 * 10).reshape(8, 10)
    prediction = target * 1.01
    report = cross_generalization_report(
        prediction,
        target,
        train_direction=torch.tensor([0, 1, 2, 3]),
        test_direction=torch.tensor([6, 7]),
        train_command=torch.tensor([0, 1, 2, 3, 4]),
        test_command=torch.tensor([8, 9]),
    )
    assert set(report) == {
        "A_seen_q_seen_w",
        "B_seen_q_unseen_w",
        "C_unseen_q_seen_w",
        "D_unseen_q_unseen_w",
    }
    assert report["D_unseen_q_unseen_w"]["nmse"] > 0
    assert len(amplitude_metrics(prediction, target)["field_strength_bins"]) == 4


def test_amplitude_objective_adds_only_gauge_invariant_q_supervision() -> None:
    torch.manual_seed(218)
    target = torch.rand(6, 4) + 0.1
    prediction = target * 1.02
    g = torch.randn(6, 16, dtype=torch.complex64)
    target_q = rank_one_q_matrix(g)
    predicted_q = target_q * 1.01
    without = amplitude_objective(prediction, target)
    with_q = amplitude_objective(
        prediction,
        target,
        predicted_q=predicted_q,
        target_q=target_q,
        confidence=torch.ones(6),
        lambda_q=0.1,
    )
    assert float(with_q.total) > float(without.total)
    assert float(without.operator_distillation) == 0.0


def test_cross_generalization_rejects_split_leakage() -> None:
    target = torch.ones(4, 4)
    try:
        cross_generalization_report(
            target,
            target,
            train_direction=torch.tensor([0, 1]),
            test_direction=torch.tensor([1, 2]),
            train_command=torch.tensor([0, 1]),
            test_command=torch.tensor([2, 3]),
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("direction leakage was accepted")


def test_rank1_training_path_runs_and_selects_validation_d() -> None:
    torch.manual_seed(220)
    codebook = build_baseline_codebook()
    train_commands = codebook.weights[codebook.indices("train")[:4]]
    validation_commands = codebook.weights[codebook.indices("validation")[:4]]
    train_direction = torch.randn(8, 3)
    validation_direction = torch.randn(5, 3)
    train_direction[:, 0] = train_direction[:, 0].abs().clamp_min(1e-3)
    validation_direction[:, 0] = validation_direction[:, 0].abs().clamp_min(1e-3)
    train_direction = train_direction / train_direction.norm(dim=-1, keepdim=True)
    validation_direction = validation_direction / validation_direction.norm(dim=-1, keepdim=True)
    teacher = CoherentRank1Operator()
    train_target = teacher.forward_command_grid(train_commands, train_direction).T.detach()
    validation_target = teacher.forward_command_grid(
        validation_commands, validation_direction
    ).T.detach()
    student = CoherentRank1Operator()
    result = train_operator_model(
        student,
        train_direction,
        train_target,
        train_commands,
        validation_direction,
        validation_target,
        validation_commands,
        config=OperatorTrainConfig(epochs=1, batch_size=4),
    )
    assert result.best_epoch == 1
    assert len(result.history) == 1


def test_staged_schedule_and_late_learning_rate_are_recorded() -> None:
    torch.manual_seed(221)
    codebook = build_baseline_codebook()
    train_commands = codebook.weights[codebook.indices("train")[:2]]
    validation_commands = codebook.weights[codebook.indices("validation")[:2]]
    train_direction = torch.randn(8, 3)
    validation_direction = torch.randn(5, 3)
    for value in (train_direction, validation_direction):
        value[:, 0] = value[:, 0].abs().clamp_min(1e-3)
        value /= value.norm(dim=-1, keepdim=True)
    teacher = CoherentRank1Operator()
    transform = FiltrationSSHTransform.fit(train_direction.double(), 2)
    student = CoherentRank1Operator(ssh_transform=transform)
    result = train_operator_model(
        student,
        train_direction,
        teacher.forward_command_grid(train_commands, train_direction).T.detach(),
        train_commands,
        validation_direction,
        teacher.forward_command_grid(validation_commands, validation_direction).T.detach(),
        validation_commands,
        config=OperatorTrainConfig(
            epochs=3,
            batch_size=8,
            training_schedule="ssh_fourier_joint",
            ssh_phase_end_epoch=1,
            fourier_phase_end_epoch=2,
            learning_rate_drop_epoch=3,
            late_learning_rate=3e-4,
        ),
    )
    assert [row["training_phase"] for row in result.history] == ["ssh", "fourier", "joint"]
    assert result.history[-1]["learning_rate"] == 3e-4


def test_stratified_sep_spatial_gap_and_distance_quartiles() -> None:
    generator = torch.Generator().manual_seed(222)
    direction = torch.randn(5000, 3, generator=generator)
    direction[:, 0] = direction[:, 0].abs().clamp_min(1e-3)
    direction /= direction.norm(dim=1, keepdim=True)
    index, weights = stratified_separation_subset(direction)
    assert len(index) == 960
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    train, gap = spatial_gap_partition(direction)
    assert not torch.isin(train, gap).any()
    gap_sep, gap_weights = stratified_separation_subset(
        direction[train], require_all_cells=False
    )
    assert len(gap_sep) > 0
    assert int(gap_sep.max()) < len(train)
    torch.testing.assert_close(
        gap_weights.sum(), torch.tensor(1.0, dtype=gap_weights.dtype)
    )
    prediction = torch.ones(len(gap), 3)
    target = prediction * 1.01
    report = angular_distance_quartile_report(prediction, target, direction[gap], direction[train])
    assert set(report) == {"Q1", "Q2", "Q3", "Q4"}


def test_fixed_q_sep_loss_uses_precomputed_global_basis() -> None:
    torch.manual_seed(224)
    codebook = build_baseline_codebook()
    train_commands = codebook.weights[codebook.indices("train")[:2]]
    validation_commands = codebook.weights[codebook.indices("validation")[:2]]
    train_direction = torch.randn(8, 3)
    validation_direction = torch.randn(4, 3)
    for value in (train_direction, validation_direction):
        value[:, 0] = value[:, 0].abs().clamp_min(1e-3)
        value /= value.norm(dim=1, keepdim=True)
    transform = FiltrationSSHTransform.fit(train_direction.double(), 2)
    teacher = CoherentRank1Operator()
    student = CoherentRank1Operator(ssh_transform=transform)
    result = train_operator_model(
        student,
        train_direction,
        teacher.forward_command_grid(train_commands, train_direction).T.detach(),
        train_commands,
        validation_direction,
        teacher.forward_command_grid(validation_commands, validation_direction).T.detach(),
        validation_commands,
        separation_direction=train_direction,
        separation_weights=torch.full((8,), 1 / 8),
        separation_basis=transform.evaluate(train_direction.double()).float(),
        config=OperatorTrainConfig(
            epochs=1,
            batch_size=4,
            lambda_separation=0.001,
            separation_start_epoch=1,
            separation_interval_steps=1,
        ),
    )
    assert result.history[0]["separation_loss"] >= 0
