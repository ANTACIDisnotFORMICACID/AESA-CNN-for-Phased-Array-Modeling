from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from v210.contracts import QueryBatch
from v212.basis import FiltrationSSHTransform
from v215.experiment import build_model
from v215.graph_operator import build_coupling_graphs


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    generator = np.random.default_rng(21501)
    q = generator.normal(size=(240, 3))
    q[:, 0] = np.abs(q[:, 0]) + 0.1
    q = torch.as_tensor(q / np.linalg.norm(q, axis=1, keepdims=True), dtype=torch.float32)
    transform = FiltrationSSHTransform.fit(q.double(), degree=4)
    s = generator.normal(size=(16, 16)) + 1j * generator.normal(size=(16, 16))
    np.fill_diagonal(s, 0.1)
    return q, transform, s


def test_graphs_are_local_normalized_and_shuffled_counterfactual_preserves_distribution():
    _, _, s = _fixture()
    row, col, true_s = build_coupling_graphs(s, "graph_s_true")
    _, _, shuffled = build_coupling_graphs(s, "graph_s_shuffled")
    assert np.allclose(row, row.T) and np.allclose(col, col.T) and np.allclose(true_s, true_s.T)
    assert np.count_nonzero(np.diag(row)) == np.count_nonzero(np.diag(col)) == 0
    for left, right in zip(*np.nonzero(row)):
        assert abs(left // 4 - right // 4) == 1 and left % 4 == right % 4
    for left, right in zip(*np.nonzero(col)):
        assert left // 4 == right // 4 and abs(left % 4 - right % 4) == 1
    np.testing.assert_allclose(np.sort(true_s.ravel()), np.sort(shuffled.ravel()))
    assert not np.allclose(true_s, shuffled)
    _, _, geometry_s = build_coupling_graphs(s, "graph_geometry")
    assert np.count_nonzero(geometry_s) == 0


def test_graph_arms_are_parameter_matched_and_within_control_budget():
    _, transform, s = _fixture()
    counts = {
        arm: sum(p.numel() for p in build_model(
            arm, transform=transform, s_matrix=s, envelope_scale=1.0, seed=21501
        ).parameters())
        for arm in ("control", "graph_geometry", "graph_s_true", "graph_s_shuffled")
    }
    assert len({counts[arm] for arm in counts if arm.startswith("graph_")}) == 1
    assert abs(counts["graph_s_true"] / counts["control"] - 1.0) <= 0.05


def test_graph_modes_have_identical_zero_start_forward_and_finite_shapes():
    q, transform, s = _fixture()
    command = torch.complex(torch.randn(5, 16), torch.randn(5, 16))
    predictions = {}
    for arm in ("graph_geometry", "graph_s_true", "graph_s_shuffled"):
        model = build_model(arm, transform=transform, s_matrix=s, envelope_scale=1.0, seed=21501)
        residual = model.residual(q[:12][None])
        assert torch.count_nonzero(residual) == 0
        prediction = model.forward_command_grid(command, q[:12])
        assert prediction.shape == (5, 12)
        assert torch.isfinite(prediction).all() and torch.all(prediction >= 0)
        predictions[arm] = prediction
    torch.testing.assert_close(predictions["graph_geometry"], predictions["graph_s_true"])
    torch.testing.assert_close(predictions["graph_s_true"], predictions["graph_s_shuffled"])


def test_analytic_phase_and_rank_one_operator_contract_are_preserved():
    q, transform, s = _fixture()
    model = build_model("graph_s_true", transform=transform, s_matrix=s, envelope_scale=1.0, seed=21502)
    envelope, _ = model.envelope(QueryBatch(q[:20][None]))
    operator_iq, _ = model.operator(QueryBatch(q[:20][None]))
    operator = torch.complex(operator_iq[0, :, 0], operator_iq[0, :, 1]).reshape(20, 16)
    positions = model.positions_m
    phase = torch.exp(torch.complex(torch.zeros((20, 16)), model.wavenumber_rad_m * (q[:20] @ positions.T)))
    torch.testing.assert_close(operator, envelope[0] * phase, atol=1e-6, rtol=1e-6)
    q_matrix = model.q_matrix(QueryBatch(q[:20][None]))[0]
    torch.testing.assert_close(q_matrix, operator.conj()[:, :, None] * operator[:, None, :])


def test_prepared_packages_contain_no_test_or_complex_fields():
    prepared = ROOT / "artifacts/prepared_20260913"
    train = np.load(prepared / "train_exchange.npz", allow_pickle=False)
    validation = np.load(prepared / "validation.npz", allow_pickle=False)
    assert train["direction"].shape == (22600, 3) and train["amplitude"].shape == (22600, 64)
    assert validation["direction"].shape == (4900, 3) and validation["amplitude"].shape == (4900, 16)
    for archive in (train, validation):
        assert not any("test" in name.lower() or name.startswith("g_") for name in archive.files)
    assert int(np.sum(train["source_role"] == 1)) == 490
    assert len(np.unique(train["removed_original_global_indices"])) == 490


def test_complete_training_has_all_twelve_sealed_runs():
    training = ROOT / "artifacts/training_20260913"
    config = json.loads((training / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["epochs"] == 240
    assert config["coverage_probe_used"] is False and config["test_split_opened"] is False
    counts = {}
    for arm in ("control", "graph_geometry", "graph_s_true", "graph_s_shuffled"):
        counts[arm] = set()
        for seed in (21501, 21502, 21503):
            run = training / f"{arm}_seed{seed}"
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert result["epochs"] == 240 and result["trained_from_scratch"] is True
            assert result["test_status"] == "sealed"
            assert checkpoint["test_split_opened"] is False
            assert checkpoint["training_data_has_test_fields"] is False
            assert checkpoint["coverage_probe_used_for_selection"] is False
            assert np.isfinite(result["validation_metrics"]["nmse_db"])
            counts[arm].add(result["parameter_count"])
    assert len(counts["control"]) == 1
    assert len(counts["graph_geometry"] | counts["graph_s_true"] | counts["graph_s_shuffled"]) == 1


def test_forward_audit_and_test_opening_close_no_leakage_protocol():
    forward_path = ROOT / "artifacts/forward_validation_20260913/FORWARD_VALIDATION_SUMMARY.json"
    forward_bytes = forward_path.read_bytes()
    forward = json.loads(forward_bytes.decode("utf-8"))
    opening = json.loads((ROOT / "artifacts/final_test_20260913/TEST_OPENING_RECORD.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "artifacts/final_test_20260913/FINAL_TEST_SUMMARY.json").read_text(encoding="utf-8"))
    assert forward["status"] == "forward_validation_complete" and forward["test_used"] is False
    assert forward["predeclared_primary_candidate"] == "graph_s_true"
    assert forward["test_open_gate"]["passed"] is True
    assert forward["structure_gate"]["passed"] is False
    assert forward["structure_gate"]["true_s_vs_shuffled_mean_at_least_0p10"] is False
    assert opening["forward_summary_sha256"] == hashlib.sha256(forward_bytes).hexdigest()
    assert opening["selection_and_checkpoints_frozen_before_test_read"] is True
    assert opening["test_will_not_be_used_for_further_v215_tuning"] is True
    assert final["status"] == "v215_candidate_qualified"
    assert final["test_opened_once_after_freeze"] is True
    assert final["test_directions"] == 4900 and final["test_commands"] == 16
    assert final["paired_candidate_test_improvement_db"]["positive_count"] == 3
    assert all(final["qualification"].values())
