"""共享阵列/QAT 基础算子的独立回归测试。"""

import math
import unittest

import torch

from beamforming_common import (
    QConv2d,
    SymmetricFakeQuantizer,
    direction_vectors,
    project_complex_unit_disk_iq,
)


class TestBeamformingCommon(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260823)
        torch.set_num_threads(1)

    def test_direction_vectors_follow_project_convention(self) -> None:
        theta = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
        phi = torch.tensor([1.234, math.pi / 2], dtype=torch.float64)
        actual = direction_vectors(theta, phi)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-15)

    def test_complex_projection_is_radial_and_bounded(self) -> None:
        iq = torch.tensor([[[3.0, 0.3], [4.0, -0.4]]])
        projected = project_complex_unit_disk_iq(iq)
        torch.testing.assert_close(
            projected[:, :, 0], torch.tensor([[0.6, 0.8]]), rtol=0.0, atol=1.0e-7
        )
        torch.testing.assert_close(
            projected[:, :, 1], iq[:, :, 1], rtol=0.0, atol=0.0
        )
        magnitude = torch.sqrt(projected[:, 0].square() + projected[:, 1].square())
        self.assertLessEqual(float(magnitude.max()), 1.0 + 1.0e-7)

    def test_symmetric_int8_quantizer_has_255_levels(self) -> None:
        quantizer = SymmetricFakeQuantizer(8, clip_value=1.0)
        probe = torch.linspace(-1.2, 1.2, 2401)
        quantized = quantizer(probe)
        self.assertEqual(int(torch.unique(quantized).numel()), 255)
        self.assertLessEqual(float(quantized.abs().max()), 1.0)

    def test_qconv2d_preserves_parameter_keys_and_shape(self) -> None:
        layer = QConv2d(2, 3, 3, padding=1, weight_bits=8)
        self.assertEqual(tuple(layer(torch.randn(4, 2, 5, 7)).shape), (4, 3, 5, 7))
        self.assertEqual(set(layer.state_dict()), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
