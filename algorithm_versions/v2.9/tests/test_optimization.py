import torch
import unittest

from v29.contracts import QueryBatch
from v29.models import PhysicsStructuredForwardModel
from v29.optimization import FourTargetProblem, optimize_four_targets, project_command_iq
from v29.io import BasisDataset, DatasetManifest
from v29.contracts import ArrayGeometry
from v29.optimization import verify_command_on_basis
import numpy as np


class OptimizationTests(unittest.TestCase):
    def test_command_projection_satisfies_both_constraints(self) -> None:
        torch.manual_seed(3)
        command = torch.randn(5, 2, 4, 4)
        projected = project_command_iq(command, total_power=1.0, amplitude_max=0.4)
        power = projected.square().sum(dim=(1, 2, 3))
        amplitude = torch.linalg.vector_norm(projected, dim=1)
        torch.testing.assert_close(power, torch.ones_like(power), atol=2e-5, rtol=2e-5)
        self.assertTrue(bool(torch.all(amplitude <= 0.40001)))

    def test_zero_command_has_defined_projection(self) -> None:
        projected = project_command_iq(torch.zeros(2, 2, 4, 4))
        self.assertTrue(bool(torch.isfinite(projected).all()))
        torch.testing.assert_close(projected.square().sum(dim=(1, 2, 3)), torch.ones(2))

    def test_end_to_end_four_target_optimizer_preserves_model_state(self) -> None:
        directions = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, -0.6, 0.0], [0.8, 0.0, 0.6]]]
        )
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        model = PhysicsStructuredForwardModel()
        model.train()
        result = optimize_four_targets(
            model,
            FourTargetProblem(QueryBatch(directions)),
            steps=2,
            random_restarts=2,
        )
        self.assertEqual(result.command_iq.shape, (2, 4, 4))
        self.assertEqual(result.target_power.shape, (4,))
        self.assertAlmostEqual(float(result.command_iq.square().sum()), 1.0, places=5)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_direct_basis_verification_does_not_call_model(self) -> None:
        directions = np.array(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, -0.6, 0.0], [0.8, 0.0, 0.6]]
        )
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        operator = np.zeros((4, 3, 2, 4, 4), dtype=np.float32)
        operator[:, 1, 0] = 1.0
        basis = BasisDataset(
            direction=directions,
            operator_iq=operator,
            theta_deg=None,
            phi_deg=None,
            manifest=DatasetManifest.for_geometry(ArrayGeometry(), "synthetic_fixture"),
        )
        command = torch.zeros(1, 2, 4, 4)
        command[:, 0] = 1.0 / 4.0
        verification = verify_command_on_basis(command, directions, basis, max_angular_error_deg=1e-5)
        torch.testing.assert_close(verification.target_power, torch.full((1, 4), 16.0))



if __name__ == "__main__":
    unittest.main()
