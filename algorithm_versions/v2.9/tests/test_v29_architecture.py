import unittest

import torch
from torch import nn

from v29.contracts import QueryBatch
from v29.models import (
    IdentityDeviceModel,
    StructuredPortCoupling,
    V29AmplitudeForwardModel,
    contract_scalar_operator_iq,
)
from v29.physics import align_global_phase_iq
from v29.training import configure_trainable_stage, default_stage_plan


def _query() -> QueryBatch:
    directions = torch.tensor([[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]])
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    return QueryBatch(directions)


class _NonIdentityDevice(nn.Module):
    def forward(self, command_iq: torch.Tensor, state=None) -> torch.Tensor:
        return 2.0 * command_iq


class V29ArchitectureTests(unittest.TestCase):
    def test_identity_initialization_matches_ideal_operator(self) -> None:
        model = V29AmplitudeForwardModel()
        operator, terms = model.operator(_query())
        ideal = model.ideal_operator(_query())
        torch.testing.assert_close(operator, ideal)
        torch.testing.assert_close(
            sum(terms.values()),
            operator,
            atol=1e-7,
            rtol=0.0,
        )

    def test_transmit_convention_uses_h_transpose_w(self) -> None:
        operator = torch.zeros(1, 1, 2, 4, 4)
        command = torch.zeros(1, 2, 4, 4)
        operator[0, 0, :, 0, 0] = torch.tensor([1.0, 2.0])
        command[0, :, 0, 0] = torch.tensor([3.0, 4.0])
        field = contract_scalar_operator_iq(operator, command)
        torch.testing.assert_close(field[0, 0], torch.tensor([-5.0, 10.0]))

    def test_basis_policy_rejects_double_counted_coupling(self) -> None:
        with self.assertRaisesRegex(ValueError, "double count"):
            V29AmplitudeForwardModel(
                basis_type="embedded_element_basis",
                enable_coupling=True,
            )
        with self.assertRaisesRegex(NotImplementedError, "ideal_phase only"):
            V29AmplitudeForwardModel(
                basis_type="common_pattern",
                enable_coupling=False,
            )

    def test_hfss_rejects_nonidentity_device_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "IdentityDeviceModel"):
            V29AmplitudeForwardModel(
                source_type="hfss_amplitude",
                device_model=_NonIdentityDevice(),
            )
        self.assertIsInstance(V29AmplitudeForwardModel().device_model, IdentityDeviceModel)

    def test_coupling_is_identity_initialized_and_diagonal_stays_one(self) -> None:
        coupling = StructuredPortCoupling(low_rank=1)
        identity = torch.eye(16)
        torch.testing.assert_close(coupling.matrix_iq()[0], identity)
        coupling.correction_real.data.fill_(0.2)
        coupling.correction_imag.data.fill_(0.1)
        torch.testing.assert_close(torch.diagonal(coupling.matrix_iq()[0]), torch.ones(16))
        torch.testing.assert_close(torch.diagonal(coupling.matrix_iq()[1]), torch.zeros(16))

    def test_global_phase_alignment_is_diagnostic_and_amplitude_invariant(self) -> None:
        model = V29AmplitudeForwardModel()
        operator, _ = model.operator(_query())
        rotated = torch.stack((-operator[:, :, 1], operator[:, :, 0]), dim=2)
        aligned = align_global_phase_iq(rotated, operator)
        torch.testing.assert_close(aligned, operator, atol=1e-6, rtol=1e-6)

    def test_stage_plan_preserves_200_epochs_and_freezes_named_modules(self) -> None:
        self.assertEqual([stage.epochs for stage in default_stage_plan(4)], [1, 1, 1, 1])
        plan = default_stage_plan(200)
        self.assertEqual([stage.epochs for stage in plan], [80, 40, 60, 20])
        model = V29AmplitudeForwardModel()
        names = configure_trainable_stage(model, "coupling")
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("port_coupling.") for name in names))
        self.assertTrue(all(not p.requires_grad for p in model.nominal_pattern.parameters()))

    def test_forward_shapes_and_4x4_topology(self) -> None:
        model = V29AmplitudeForwardModel()
        command = torch.randn(1, 2, 4, 4)
        prediction = model(command, _query())
        self.assertEqual(prediction.operator_iq.shape, (1, 2, 2, 4, 4))
        self.assertEqual(prediction.field_iq.shape, (1, 2, 2))
        self.assertEqual(prediction.amplitude.shape, (1, 2))
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertTrue(convolutions)
        self.assertTrue(all(module.kernel_size == (3, 3) for module in convolutions))


if __name__ == "__main__":
    unittest.main()
