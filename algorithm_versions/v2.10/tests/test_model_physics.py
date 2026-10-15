import torch
from torch import nn
import unittest

from v210.contracts import QueryBatch
from v210.models import PhysicsStructuredForwardModel


def _query() -> QueryBatch:
    direction = torch.tensor([[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]])
    return QueryBatch(direction)


class ModelPhysicsTests(unittest.TestCase):
    def test_untrained_correction_is_exactly_ideal_and_transverse(self) -> None:
        model = PhysicsStructuredForwardModel()
        query = _query()
        ideal = model.ideal_operator(query)
        corrected = model.operator(query).effective_iq
        torch.testing.assert_close(corrected, ideal)
        radial = torch.einsum("bqx,bqxirc->bqirc", query.unit_direction, corrected)
        torch.testing.assert_close(radial, torch.zeros_like(radial), atol=1e-6, rtol=0.0)

    def test_all_spatial_convolutions_are_two_dimensional_three_by_three(self) -> None:
        model = PhysicsStructuredForwardModel()
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertTrue(convolutions)
        self.assertTrue(all(module.kernel_size == (3, 3) for module in convolutions))
        self.assertFalse(
            any(
                isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d))
                for module in model.modules()
            )
        )

    def test_command_mapping_is_linear_and_uses_each_element(self) -> None:
        torch.manual_seed(7)
        model = PhysicsStructuredForwardModel()
        query = _query()
        first = torch.randn(1, 2, 4, 4)
        second = torch.randn(1, 2, 4, 4)
        combined = model(first + second, query).field_iq
        separate = model(first, query).field_iq + model(second, query).field_iq
        torch.testing.assert_close(combined, separate, atol=2e-5, rtol=2e-5)

        individual_norms = torch.zeros(4, 4)
        for row in range(4):
            for col in range(4):
                command = torch.zeros(1, 2, 4, 4)
                command[0, 0, row, col] = 1.0
                individual_norms[row, col] = torch.linalg.vector_norm(model(command, query).field_iq)
        self.assertTrue(bool(torch.all(individual_norms > 0)))


if __name__ == "__main__":
    unittest.main()

