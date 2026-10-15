from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from v210.contracts import QueryBatch
from v212.basis import FiltrationSSHTransform
from v216.axial_operator import (
    AxialEnvelopeResidual,
    ControlledDCAREnvelopeResidual,
    IsotropicEnvelopeResidual,
)
from v216.experiment import ARMS, STAGE1_ARMS, build_model, build_stage1_model
from v216.statistics import paired_three_seed_summary


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    generator = np.random.default_rng(21601)
    direction = generator.normal(size=(240, 3))
    direction[:, 0] = np.abs(direction[:, 0]) + 0.1
    direction = torch.as_tensor(
        direction / np.linalg.norm(direction, axis=1, keepdims=True), dtype=torch.float32
    )
    transform = FiltrationSSHTransform.fit(direction.double(), degree=4)
    return direction, transform


def test_dcar_has_only_two_coordinate_maps_and_axial_spatial_kernels():
    static = AxialEnvelopeResidual(mode="axial_static")
    gated = AxialEnvelopeResidual(mode="axial_gated")
    assert static.coordinate_maps.shape == (2, 4, 4)
    assert static.stem.kernel_size == (1, 1) and static.head.kernel_size == (1, 1)
    assert static.row_operator.kernel_size == (3, 1)
    assert static.col_operator.kernel_size == (1, 3)
    assert static.direction_gate is None
    assert gated.direction_gate.in_features == 27 and gated.direction_gate.out_features == 2
    spatial = [module.kernel_size for module in gated.modules() if isinstance(module, nn.Conv2d)]
    assert (3, 3) not in spatial


def test_isotropic_control_has_one_3x3_and_pointwise_stem_head():
    model = IsotropicEnvelopeResidual()
    assert model.coordinate_maps.shape == (2, 4, 4)
    assert model.stem.kernel_size == (1, 1)
    assert model.spatial_operator.kernel_size == (3, 3)
    assert model.head.kernel_size == (1, 1)


def test_new_arms_are_capacity_matched_and_counts_are_frozen():
    _, transform = _fixture()
    counts = {
        arm: sum(parameter.numel() for parameter in build_model(
            arm, transform=transform, envelope_scale=1.0, seed=21601
        ).parameters())
        for arm in ARMS
    }
    assert counts == {
        "control_v212": 28789,
        "axial_static": 30861,
        "axial_gated": 30917,
        "iso_control": 30877,
    }
    assert abs(counts["axial_gated"] / counts["iso_control"] - 1.0) <= 0.005


def test_all_arms_are_zero_start_finite_and_shape_compatible():
    direction, transform = _fixture()
    command = torch.complex(torch.randn(5, 16), torch.randn(5, 16))
    for arm in ARMS:
        model = build_model(arm, transform=transform, envelope_scale=1.0, seed=21601)
        residual = model.residual(direction[:12][None])
        assert torch.count_nonzero(residual) == 0
        prediction = model.forward_command_grid(command, direction[:12])
        assert prediction.shape == (5, 12)
        assert torch.isfinite(prediction).all() and torch.all(prediction >= 0)


def test_analytic_phase_and_rank_one_contract_are_unchanged():
    direction, transform = _fixture()
    model = build_model("axial_gated", transform=transform, envelope_scale=1.0, seed=21602)
    query = QueryBatch(direction[:20][None])
    envelope, _ = model.envelope(query)
    operator_iq, _ = model.operator(query)
    operator = torch.complex(operator_iq[0, :, 0], operator_iq[0, :, 1]).reshape(20, 16)
    phase_argument = model.wavenumber_rad_m * (direction[:20] @ model.positions_m.T)
    phase = torch.exp(torch.complex(torch.zeros_like(phase_argument), phase_argument))
    torch.testing.assert_close(operator, envelope[0] * phase, atol=1e-6, rtol=1e-6)
    q_matrix = model.q_matrix(query)[0]
    torch.testing.assert_close(q_matrix, operator.conj()[:, :, None] * operator[:, None, :])


def test_prepared_packages_exclude_all_external_test_fields():
    prepared = ROOT / "artifacts/prepared_20260914"
    if not prepared.exists():
        return
    manifest = json.loads((prepared / "PREPARATION.json").read_text(encoding="utf-8"))
    train = np.load(prepared / "train_exchange.npz", allow_pickle=False)
    validation = np.load(prepared / "validation.npz", allow_pickle=False)
    assert train["direction"].shape == (22600, 3) and train["amplitude"].shape == (22600, 64)
    assert validation["direction"].shape == (4900, 3) and validation["amplitude"].shape == (4900, 16)
    for archive in (train, validation):
        assert not any("test" in name.lower() or name.startswith("g_") for name in archive.files)
    assert manifest["old_v215_test_used_for_preparation_decisions"] is False
    assert manifest["new_confirmatory_test_defined"] is False


def test_three_seed_statistics_use_paired_student_t_interval():
    summary = paired_three_seed_summary([1.0, 2.0, 3.0])
    assert summary["mean_db"] == 2.0
    assert summary["sample_standard_deviation_db"] == 1.0
    assert summary["positive_count"] == 3
    np.testing.assert_allclose(
        summary["ci95_student_t_db"],
        [2.0 - 4.302652729696142 / np.sqrt(3.0), 2.0 + 4.302652729696142 / np.sqrt(3.0)],
    )


def test_completed_training_contains_all_twelve_sealed_runs():
    training = ROOT / "artifacts/training_20260914"
    if not training.exists():
        return
    config = json.loads((training / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["arms"] == list(ARMS)
    assert config["seeds"] == [21601, 21602, 21603]
    assert config["epochs"] == 240 and config["coverage_probe_used"] is False
    assert config["old_v215_test_used"] is False
    assert config["new_confirmatory_test_defined_or_opened"] is False
    for arm in ARMS:
        for seed in (21601, 21602, 21603):
            run = training / f"{arm}_seed{seed}"
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert result["epochs"] == 240 and result["trained_from_scratch"] is True
            assert result["old_v215_test_status"] == "not_used"
            assert result["new_confirmatory_test_status"] == "not_defined_or_opened"
            assert checkpoint["coverage_probe_used_for_selection"] is False
            assert checkpoint["old_v215_test_used"] is False
            assert checkpoint["new_confirmatory_test_opened"] is False


def test_completed_stage0_stops_and_preserves_external_test_seal():
    audit = ROOT / "artifacts/stage0_audit_20260914"
    if not audit.exists():
        return
    report = json.loads((audit / "STAGE0_AUDIT_SUMMARY.json").read_text(encoding="utf-8"))
    statistics = json.loads((audit / "STATISTICAL_ANALYSIS.json").read_text(encoding="utf-8"))
    assert report["status"] == "v216_stage0_stop"
    assert report["mechanism_gate"]["passed"] is False
    assert report["mechanism_gate"]["all_predictions_finite_nonnegative"] is True
    assert report["old_v215_external_test_used"] is False
    assert report["new_confirmatory_test_defined_or_opened"] is False
    assert statistics["status"] == "v216_stage0_stop"
    assert statistics["old_v215_external_test_used"] is False
    assert statistics["new_confirmatory_test_defined_or_opened"] is False


def test_controlled_dcar_restores_v212_maps_stem_head_depth_and_width():
    direction, transform = _fixture()
    control = build_stage1_model(
        "control_v212", transform=transform, envelope_scale=1.0, seed=21601
    )
    dcar = build_stage1_model(
        "dcar_gated_native", transform=transform, envelope_scale=1.0, seed=21601
    )
    torch.testing.assert_close(dcar.residual.geometry_maps, control.residual.geometry_maps)
    assert dcar.residual.stem.in_channels == control.residual.branch.stem.in_channels == 31
    assert dcar.residual.stem.out_channels == control.residual.branch.stem.out_channels == 24
    assert dcar.residual.stem.kernel_size == control.residual.branch.stem.kernel_size == (3, 3)
    assert dcar.residual.head.kernel_size == control.residual.branch.head.kernel_size == (3, 3)
    assert len(dcar.residual.blocks) == len(control.residual.branch.blocks) == 2
    assert dcar.residual.gate_values(direction[:9][None]).shape == (1, 9, 2, 2)


def test_controlled_dcar_changes_only_residual_block_spatial_operators():
    model = ControlledDCAREnvelopeResidual(mode="gated", width=24, depth=2)
    for block in model.blocks:
        assert block.row_conv1.kernel_size == block.row_conv2.kernel_size == (3, 1)
        assert block.col_conv1.kernel_size == block.col_conv2.kernel_size == (1, 3)
        assert torch.count_nonzero(block.row_conv2.weight) == 0
        assert torch.count_nonzero(block.col_conv2.weight) == 0
    spatial = [module.kernel_size for block in model.blocks for module in block.modules() if isinstance(module, nn.Conv2d)]
    assert (3, 3) not in spatial


def test_controlled_dcar_parameter_counts_and_primary_match_are_frozen():
    _, transform = _fixture()
    counts = {
        arm: sum(parameter.numel() for parameter in build_stage1_model(
            arm, transform=transform, envelope_scale=1.0, seed=21601
        ).parameters())
        for arm in STAGE1_ARMS
    }
    assert counts == {
        "control_v212": 28789,
        "dcar_static_native": 21973,
        "dcar_gated_native": 22085,
        "dcar_static_pm": 28643,
        "dcar_gated_pm": 28755,
    }
    assert abs(counts["dcar_gated_pm"] / counts["control_v212"] - 1.0) <= 0.005


def test_controlled_dcar_and_v212_have_identical_zero_start_operator():
    direction, transform = _fixture()
    command = torch.complex(torch.randn(5, 16), torch.randn(5, 16))
    control = build_stage1_model(
        "control_v212", transform=transform, envelope_scale=1.0, seed=21602
    )
    control_prediction = control.forward_command_grid(command, direction[:15])
    for arm in STAGE1_ARMS[1:]:
        candidate = build_stage1_model(
            arm, transform=transform, envelope_scale=1.0, seed=21602
        )
        torch.testing.assert_close(
            candidate.forward_command_grid(command, direction[:15]), control_prediction
        )


def test_completed_stage1_has_all_twelve_sealed_candidate_runs():
    training = ROOT / "artifacts/stage1_controlled_training_20260914"
    if not training.exists():
        return
    config = json.loads((training / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["control_checkpoint_reused_not_retrained"] is True
    assert config["coverage_probe_used"] is False
    assert config["old_v215_test_used"] is False
    assert config["new_confirmatory_test_defined_or_opened"] is False
    for arm in STAGE1_ARMS[1:]:
        for seed in (21601, 21602, 21603):
            run = training / f"{arm}_seed{seed}"
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert result["epochs"] == 240 and result["trained_from_scratch"] is True
            assert result["old_v215_test_status"] == "not_used"
            assert result["new_confirmatory_test_status"] == "not_defined_or_opened"
            assert checkpoint["coverage_probe_used_for_selection"] is False
            assert checkpoint["old_v215_test_used"] is False
            assert checkpoint["new_confirmatory_test_opened"] is False


def test_completed_stage1_passes_controlled_gate_without_external_test():
    audit = ROOT / "artifacts/stage1_controlled_audit_20260914"
    if not audit.exists():
        return
    report = json.loads(
        (audit / "STAGE1_CONTROLLED_AUDIT_SUMMARY.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "v216_stage1_pass_eligible_for_fresh_confirmation"
    assert report["mechanism_gate"]["passed"] is True
    assert all(report["mechanism_gate"].values())
    assert report["old_v215_external_test_used"] is False
    assert report["new_confirmatory_test_defined_or_opened"] is False
