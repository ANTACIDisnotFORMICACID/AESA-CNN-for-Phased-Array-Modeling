from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from v210.contracts import ArrayGeometry, QueryBatch
from v210.coordinates import hfss_angles_to_direction
from v210.io import (
    AMPLITUDE_ONLY_TRAINING_POLICY,
    AmplitudeManifest,
    BasisDataset,
    DatasetManifest,
    assert_amplitude_only_training_contract,
    canonical_spherical_query_mask,
    generate_command_codebook,
    load_amplitude_dataset,
    save_amplitude_dataset,
    synthesize_amplitude_observations,
)
from v210.models import AmplitudeOnlyForwardModel
from v210.optimization import FourTargetProblem, optimize_four_targets
from v210.training import (
    AmplitudeDirectionDataset,
    AmplitudeObservationDataset,
    AmplitudeTrainConfig,
    amplitude_only_loss,
    balanced_command_family_weights,
    evaluate_amplitude_grid_model,
    train_grouped_amplitude_model,
    train_amplitude_model,
)


def _raw_basis() -> BasisDataset:
    theta = np.array([90.0, 90.0, 80.0, 100.0])
    phi = np.array([0.0, 30.0, -30.0, 0.0])
    direction = hfss_angles_to_direction(theta, phi)
    generator = np.random.default_rng(4)
    operator = generator.normal(size=(4, 3, 2, 4, 4)).astype(np.float32)
    return BasisDataset(
        direction=direction,
        operator_iq=operator,
        theta_deg=theta,
        phi_deg=phi,
        manifest=DatasetManifest.for_geometry(ArrayGeometry(), "synthetic_fixture"),
    )


def _manifest() -> AmplitudeManifest:
    return AmplitudeManifest(
        schema_version="v2.8-amplitude-1",
        source_type="hfss_amplitude",
        frequency_hz=47.0e9,
        receiver_channel="ephi",
        amplitude_unit="V/m",
        reference_distance_m=1.0,
        excitation_reference="unit-test-known-command",
        inactive_port_termination="50ohm",
        calibration_reference_plane="test-port-plane",
        array_id="fixture-array",
        session_id="fixture-session",
    )


class AmplitudeOnlyPipelineTests(unittest.TestCase):
    def test_full_sphere_canonicalization_removes_poles_and_positive_seam(self) -> None:
        theta = np.repeat([0.0, 90.0, 180.0], 3)
        phi = np.tile([-180.0, 0.0, 180.0], 3)
        keep = canonical_spherical_query_mask(theta, phi)
        np.testing.assert_array_equal(np.flatnonzero(keep), [3, 4])

    def test_amplitude_model_keeps_4x4_and_uses_only_3x3_convolutions(self) -> None:
        model = AmplitudeOnlyForwardModel()
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertTrue(convolutions)
        self.assertTrue(all(module.kernel_size == (3, 3) for module in convolutions))
        query = QueryBatch(torch.tensor([[[1.0, 0.0, 0.0]]]))
        prediction = model(torch.ones(1, 2, 4, 4), query)
        self.assertEqual(prediction.operator_iq.shape, (1, 1, 2, 4, 4))
        self.assertEqual(prediction.amplitude.shape, (1, 1))

    def test_grouped_command_contraction_matches_regular_forward(self) -> None:
        model = AmplitudeOnlyForwardModel()
        commands = torch.randn(3, 2, 4, 4)
        commands = commands / commands.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()
        directions = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        grouped = model.forward_command_grid(commands, directions)
        regular = model(
            commands,
            QueryBatch(directions[None].expand(commands.shape[0], -1, -1)),
        ).amplitude
        torch.testing.assert_close(grouped, regular)

    def test_hfss_conversion_exposes_only_amplitude_observations(self) -> None:
        commands, families = generate_command_codebook(random_commands=8, seed=2)
        dataset = synthesize_amplitude_observations(_raw_basis(), commands, families, _manifest())
        self.assertEqual(dataset.command_iq.shape, (40, 2, 4, 4))
        self.assertEqual(dataset.amplitude.shape, (40, 4))
        self.assertTrue(np.all(dataset.amplitude >= 0))
        np.testing.assert_allclose(np.sum(commands**2, axis=(1, 2, 3)), 1.0, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_amplitude_dataset(dataset, root)
            self.assertFalse((root / "operator_iq.npy").exists())
            self.assertFalse((root / "s_parameters_iq.npy").exists())
            restored = load_amplitude_dataset(root)
        np.testing.assert_array_equal(restored.amplitude, dataset.amplitude)

    def test_loader_rejects_phase_bearing_files(self) -> None:
        commands, families = generate_command_codebook(random_commands=1)
        dataset = synthesize_amplitude_observations(_raw_basis(), commands, families, _manifest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_amplitude_dataset(dataset, root)
            np.save(root / "operator_iq.npy", np.zeros(1))
            with self.assertRaisesRegex(ValueError, "forbidden or unexpected"):
                load_amplitude_dataset(root)

    def test_loader_rejects_any_unexpected_training_array(self) -> None:
        commands, families = generate_command_codebook(random_commands=1)
        dataset = synthesize_amplitude_observations(_raw_basis(), commands, families, _manifest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_amplitude_dataset(dataset, root)
            np.save(root / "measured_phase.npy", np.zeros(1))
            with self.assertRaisesRegex(ValueError, "measured_phase.npy"):
                load_amplitude_dataset(root)

    def test_training_contract_requires_strict_front_hemisphere(self) -> None:
        commands, families = generate_command_codebook(random_commands=1)
        dataset = synthesize_amplitude_observations(_raw_basis(), commands, families, _manifest())
        self.assertEqual(AMPLITUDE_ONLY_TRAINING_POLICY, "v210-amplitude-only-positive-hemisphere-1")
        mixed_direction = dataset.direction.copy()
        mixed_direction[0] *= -1.0
        mixed = type(dataset)(
            direction=mixed_direction,
            command_iq=dataset.command_iq,
            amplitude=dataset.amplitude,
            command_family=dataset.command_family,
            theta_deg=dataset.theta_deg,
            phi_deg=dataset.phi_deg,
            manifest=dataset.manifest,
        )
        with self.assertRaisesRegex(ValueError, "q_x > 0"):
            assert_amplitude_only_training_contract(mixed)
        assert_amplitude_only_training_contract(dataset)

    def test_amplitude_training_uses_known_commands_and_scalar_labels(self) -> None:
        directions = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        teacher = AmplitudeOnlyForwardModel()
        commands = torch.randn(6, 2, 4, 4)
        commands = commands / commands.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()
        with torch.no_grad():
            query = QueryBatch(directions[None].expand(6, -1, -1))
            amplitude = teacher(commands, query).amplitude.numpy()
        dataset = synthesize_amplitude_observations(
            _raw_basis(),
            generate_command_codebook(random_commands=1)[0],
            generate_command_codebook(random_commands=1)[1],
            _manifest(),
        )
        dataset = type(dataset)(
            direction=directions.numpy(),
            command_iq=commands.numpy(),
            amplitude=amplitude,
            command_family=np.full(6, "fixture", dtype="U32"),
            theta_deg=np.array([90.0, 90.0]),
            phi_deg=np.array([0.0, 36.86989765]),
            manifest=_manifest(),
        )
        loader = DataLoader(
            AmplitudeObservationDataset(dataset, np.arange(6), np.arange(2)),
            batch_size=4,
        )
        student = AmplitudeOnlyForwardModel()
        history = train_amplitude_model(
            student,
            loader,
            noise_floor_amplitude=0.0,
            config=AmplitudeTrainConfig(epochs=1),
        )
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["loss"]))

    def test_grouped_training_balances_command_families_and_selects_validation(self) -> None:
        directions = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, 0.0, 0.6], [0.6, 0.8, 0.0]]
        )
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        commands = torch.randn(4, 2, 4, 4)
        commands = commands / commands.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()
        teacher = AmplitudeOnlyForwardModel()
        with torch.no_grad():
            amplitude = teacher.forward_command_grid(commands, directions).numpy()
        dataset = synthesize_amplitude_observations(
            _raw_basis(),
            generate_command_codebook(random_commands=1)[0],
            generate_command_codebook(random_commands=1)[1],
            _manifest(),
        )
        dataset = type(dataset)(
            direction=directions.numpy(),
            command_iq=commands.numpy(),
            amplitude=amplitude,
            command_family=np.array(["single", "single", "coded", "coded"]),
            theta_deg=np.full(4, 90.0),
            phi_deg=np.array([0.0, 36.86989765, 0.0, 53.13010235]),
            manifest=_manifest(),
        )
        grouped = AmplitudeDirectionDataset(dataset, np.arange(4), np.arange(4))
        loader = DataLoader(grouped, batch_size=2, shuffle=False)
        weights = balanced_command_family_weights(dataset.command_family)
        student = AmplitudeOnlyForwardModel()
        frozen_pattern = {
            name: value.detach().clone()
            for name, value in student.nominal_pattern.state_dict().items()
        }
        for parameter in student.nominal_pattern.parameters():
            parameter.requires_grad_(False)
        result = train_grouped_amplitude_model(
            student,
            loader,
            grouped.command_iq,
            loader,
            grouped.command_iq,
            noise_floor_amplitude=0.0,
            training_command_weights=weights,
            validation_command_weights=weights,
            config=AmplitudeTrainConfig(epochs=1),
            select_initial_state=True,
        )
        metrics = evaluate_amplitude_grid_model(
            student,
            loader,
            grouped.command_iq,
            noise_floor_amplitude=0.0,
        )
        self.assertEqual(result.best_epoch, 0)
        for name, value in student.nominal_pattern.state_dict().items():
            torch.testing.assert_close(value, frozen_pattern[name])
        self.assertTrue(np.isfinite(metrics["power_nmse"]))

    def test_amplitude_loss_contains_linear_power_and_single_polarization_db_terms(self) -> None:
        prediction = torch.tensor([[1.0, 2.0], [1.5, 0.5]])
        target = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        losses = amplitude_only_loss(
            prediction,
            target,
            noise_floor_amplitude=0.0,
            command_weights=torch.tensor([0.25, 0.75]),
        )
        self.assertGreater(float(losses.linear_nmse), 0.0)
        self.assertGreater(float(losses.power_nmse), 0.0)
        self.assertGreater(float(losses.amplitude_db_mae), 0.0)

    def test_four_target_optimizer_accepts_amplitude_only_model(self) -> None:
        directions = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, -0.6, 0.0], [0.8, 0.0, 0.6]]]
        )
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        result = optimize_four_targets(
            AmplitudeOnlyForwardModel(),
            FourTargetProblem(QueryBatch(directions)),
            steps=2,
            random_restarts=2,
        )
        self.assertEqual(result.command_iq.shape, (2, 4, 4))
        self.assertEqual(result.target_power.shape, (4,))


if __name__ == "__main__":
    unittest.main()

    assert_amplitude_only_training_contract,
