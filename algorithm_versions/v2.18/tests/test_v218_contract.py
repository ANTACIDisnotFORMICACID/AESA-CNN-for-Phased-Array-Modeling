from pathlib import Path
import json
import tempfile

import numpy as np
import torch

from v212.basis import FiltrationSSHTransform
from v218.audit import audit_run
from v218.canonical import PreparedData, amplitude_from_operator, assert_unsealed_path_not_used
from v218.experiment import (
    ARMS, BASELINE_ARMS, OPAUX_ARM, PHASE_AUX_ARMS, SEEDS, build_model,
)
from v218.stage2 import amplitude_metrics, direction_overlap, equivalence_metrics
from v218.training import (
    TrainConfig, complex_operator_nmse, phase_aligned_operator_nmse, train_run,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    generator = np.random.default_rng(21801)
    direction = generator.normal(size=(240, 3))
    direction[:, 0] = np.abs(direction[:, 0]) + 0.1
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    tensor = torch.from_numpy(direction).float()
    return tensor, FiltrationSSHTransform.fit(tensor.double(), degree=4)


def test_frozen_arm_seed_and_parameter_contract():
    _, transform = fixture()
    assert SEEDS == (21801, 21802, 21803)
    assert BASELINE_ARMS == (
        "control_v212_canonical", "dcar_static_pm_canonical", "dcar_gated_pm_canonical"
    )
    assert OPAUX_ARM in ARMS
    assert ARMS[-2:] == PHASE_AUX_ARMS
    counts = {
        arm: sum(p.numel() for p in build_model(arm, transform=transform, envelope_scale=1, seed=21801).parameters())
        for arm in ARMS
    }
    assert counts == {
        "control_v212_canonical": 28789,
        "dcar_static_pm_canonical": 28643,
        "dcar_gated_pm_canonical": 28755,
        "dcar_gated_pm_opaux": 28755,
        "control_v212_phase_aux": 28789,
        "dcar_gated_pm_phase_aux": 28755,
    }
    assert abs(counts["dcar_gated_pm_canonical"] / counts["control_v212_canonical"] - 1) <= 0.005


def test_paired_initialization_is_equal_but_parameters_are_independent():
    direction, transform = fixture()
    gated = build_model("dcar_gated_pm_canonical", transform=transform, envelope_scale=1, seed=21802)
    opaux = build_model(OPAUX_ARM, transform=transform, envelope_scale=1, seed=21802)
    for left, right in zip(gated.parameters(), opaux.parameters()):
        torch.testing.assert_close(left, right)
        assert left.data_ptr() != right.data_ptr()
    commands = torch.complex(torch.randn(3, 16), torch.randn(3, 16))
    torch.testing.assert_close(
        gated.forward_command_grid(commands, direction[:8]),
        opaux.forward_command_grid(commands, direction[:8]),
    )
    for baseline_arm, auxiliary_arm in (
        ("control_v212_canonical", "control_v212_phase_aux"),
        ("dcar_gated_pm_canonical", "dcar_gated_pm_phase_aux"),
    ):
        baseline = build_model(baseline_arm, transform=transform, envelope_scale=1, seed=21802)
        auxiliary = build_model(auxiliary_arm, transform=transform, envelope_scale=1, seed=21802)
        for left, right in zip(baseline.parameters(), auxiliary.parameters()):
            torch.testing.assert_close(left, right)
            assert left.data_ptr() != right.data_ptr()


def test_all_models_have_zero_residual_start_and_finite_gradient():
    direction, transform = fixture()
    commands = torch.complex(torch.randn(4, 16), torch.randn(4, 16))
    for arm in ARMS:
        model = build_model(arm, transform=transform, envelope_scale=1, seed=21803)
        assert torch.count_nonzero(model.residual(direction[:9][None])) == 0
        prediction = model.forward_command_grid(commands, direction[:9])
        prediction.square().mean().backward()
        assert prediction.shape == (4, 9)
        assert torch.isfinite(prediction).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_canonical_amplitude_reconstruction_and_complex_loss():
    rng = np.random.default_rng(21804)
    operator = (rng.normal(size=(7, 16)) + 1j * rng.normal(size=(7, 16))).astype(np.complex64)
    commands = (rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))).astype(np.complex64)
    expected = np.abs(np.einsum("qn,cn->qc", operator, commands))
    np.testing.assert_allclose(amplitude_from_operator(operator, commands), expected, rtol=1e-6)
    tensor = torch.from_numpy(operator)
    weights = torch.arange(1, 8).float()
    assert abs(float(complex_operator_nmse(tensor, tensor, weights))) <= 1e-12


def test_phase_aligned_loss_is_gauge_invariant_and_has_finite_gradient():
    generator = torch.Generator().manual_seed(21806)
    real = torch.randn(11, 16, generator=generator)
    imag = torch.randn(11, 16, generator=generator)
    prediction = torch.complex(real, imag).requires_grad_(True)
    phase = torch.exp(1j * torch.linspace(-2.0, 2.0, 11))
    target = prediction.detach() * phase[:, None]
    weights = torch.arange(1, 12).float()
    loss = phase_aligned_operator_nmse(prediction, target, weights)
    assert float(loss.detach()) < 1e-12
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    common_phase = torch.exp(1j * torch.linspace(0.1, 1.1, 11))[:, None]
    shifted = phase_aligned_operator_nmse(
        prediction.detach() * common_phase, target * common_phase, weights
    )
    torch.testing.assert_close(loss.detach(), shifted, atol=1e-12, rtol=0)


def test_stage2_overlap_equivalence_and_fixed_field_bins():
    direction = np.eye(3, dtype=np.float64)
    source = np.vstack((direction, np.asarray([[1, 1, 1]]) / np.sqrt(3)))
    source_index, reference_index, error = direction_overlap(source, direction)
    np.testing.assert_array_equal(source_index, np.arange(3))
    np.testing.assert_array_equal(reference_index, np.arange(3))
    assert float(error.max()) == 0.0
    target = np.asarray([[1.0, 0.1], [0.5, 0.01], [0.01, 0.001]])
    metrics = equivalence_metrics(target, target)
    assert metrics["aggregate_nmse_db"] <= -60
    report = amplitude_metrics(target, target, direction_weights=np.ones(3), command_family=np.asarray(["scan", "hard_null"]))
    assert report["solid_angle_micro_nmse_db"] <= -60
    assert report["fixed_field_bins"]["strong_-3_to_0_db"]["count"] > 0
    assert set(report["command_family"]) == {"hard_null", "scan"}


def test_sealed_paths_are_rejected_and_tools_have_no_confirmation_path_option():
    try:
        assert_unsealed_path_not_used(Path("some_blind_test.npz"))
    except ValueError:
        pass
    else:
        raise AssertionError("sealed path was accepted")
    for name in (
        "train_v218.py", "audit_v218.py", "run_stage2_attribution.py",
        "train_v218_stage2.py", "audit_v218_stage2.py",
        "complete_v218_stage2.py",
    ):
        source = (ROOT / "tools" / name).read_text(encoding="utf-8").lower()
        assert "--blind" not in source
        assert "test_operator" not in source
        assert "test_commands" not in source


def test_prepared_package_contract_when_present():
    prepared = ROOT / "artifacts/prepared_4x4_20261014"
    if not prepared.exists():
        return
    from v218.canonical import load_prepared
    data = load_prepared(prepared)
    assert data.train["direction"].shape == (45071, 3)
    assert data.train["g_ephi"].shape == (45071, 16)
    assert data.train["commands"].shape == (64, 16)
    assert data.train["amplitude"].shape == (45071, 64)
    assert data.validation["direction"].shape == (9780, 3)
    assert data.validation["commands"].shape == (512, 16)
    assert data.validation["amplitude"].shape == (9780, 512)
    assert len(np.unique(data.train["canonical_indices"])) == 45071
    np.testing.assert_allclose(
        np.sum(np.abs(data.train["commands"]) ** 2, axis=1), 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.sum(np.abs(data.validation["commands"]) ** 2, axis=1), 1.0, atol=1e-5
    )
    sample = np.asarray([0, 17, 45070])
    np.testing.assert_allclose(
        data.train["amplitude"][sample],
        amplitude_from_operator(data.train["g_ephi"][sample], data.train["commands"]),
        rtol=1e-6, atol=1e-6,
    )
    assert np.isclose(np.sqrt(np.mean(np.abs(data.train["g_ephi"]) ** 2)), 1.0, atol=2e-6)
    assert data.manifest["blind_test_status"] == "sealed_not_opened"


def test_training_checkpoint_is_resumable_and_recomputable():
    directions = []
    for ring_x in (1 / 6, 3 / 6, 5 / 6):
        radius = np.sqrt(1 - ring_x**2)
        for sector in range(8):
            angle = (sector + 0.5) * 2 * np.pi / 8
            for _ in range(2):
                directions.append([ring_x, radius * np.cos(angle), radius * np.sin(angle)])
    direction = np.asarray(directions, dtype=np.float32)
    transform = FiltrationSSHTransform.fit(torch.from_numpy(direction).double(), degree=4)
    rng = np.random.default_rng(21805)
    operator = (rng.normal(size=(len(direction), 16)) + 1j * rng.normal(size=(len(direction), 16))).astype(np.complex64)
    operator /= np.sqrt(np.mean(np.abs(operator) ** 2))
    commands = (rng.normal(size=(4, 16)) + 1j * rng.normal(size=(4, 16))).astype(np.complex64)
    commands /= np.linalg.norm(commands, axis=1, keepdims=True)
    package = {
        "direction": direction, "g_ephi": operator,
        "solid_angle_weights": np.ones(len(direction), dtype=np.float64),
        "commands": commands, "amplitude": amplitude_from_operator(operator, commands),
        "canonical_indices": np.arange(len(direction)),
    }
    data = PreparedData(
        train=package,
        validation={**package, "command_family": np.asarray(["smoke"] * len(commands))},
        transform=transform,
        manifest={"envelope_scale": 1.0, "source_fingerprints_sha256": {"smoke": "local"}},
    )
    config1 = TrainConfig(
        epochs=1, batch_size=24, validation_query_chunk=24,
        separation_start_epoch=1, separation_interval_steps=1,
    )
    config2 = TrainConfig(
        epochs=2, batch_size=24, validation_query_chunk=24,
        separation_start_epoch=1, separation_interval_steps=1,
    )
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "run"
        first = train_run("dcar_gated_pm_phase_aux", 21801, data, run, device="cpu", config=config1)
        second = train_run("dcar_gated_pm_phase_aux", 21801, data, run, device="cpu", config=config2)
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        audit = audit_run(run, data, "cpu")
        assert first["epochs"] == 1 and second["epochs"] == 2
        assert latest["completed_epoch"] == 2 and len(latest["history"]) == 2
        assert checkpoint["best_validation_amplitude_linear_nmse"] == min(
            row["validation_linear_nmse"] for row in latest["history"]
        )
        assert checkpoint["blind_test_status"] == "sealed_not_opened"
        assert audit["all_outputs_finite"] is True
        assert len(audit["diagnostics"]["effective_messages"]) == 2
        assert audit["blind_test_status"] == "sealed_not_opened"


def test_completed_stage2_receipts_and_checkpoints_are_recomputable():
    final_path = ROOT / "artifacts/stage2_audit_4x4_20261014/STAGE2_FINAL_AUDIT.json"
    if not final_path.exists():
        return
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["migration_attribution_status"] == "MIGRATION_ATTRIBUTION_PASS"
    assert final["phase_aux_experiment_status"] == "PHASE_AUX_EXPERIMENT_FAIL"
    assert final["engineering_candidate_status"] == "ENGINEERING_CANDIDATE_NONE"
    assert final["confirmation_status"] == "CONFIRMATION_NOT_OPENED"
    training = ROOT / "artifacts/stage2_training_4x4_20261014"
    for arm in PHASE_AUX_ARMS:
        for seed in SEEDS:
            run = training / f"{arm}_seed{seed}"
            checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
            audit = json.loads((run / "stage2_validation_audit.json").read_text(encoding="utf-8"))
            assert checkpoint["blind_test_status"] == "sealed_not_opened"
            assert checkpoint["architecture"]["phase_aligned_auxiliary_enabled"] is True
            np.testing.assert_allclose(
                checkpoint["best_validation_amplitude_linear_nmse"],
                audit["amplitude"]["grid_micro_nmse"],
                rtol=2e-6,
            )
