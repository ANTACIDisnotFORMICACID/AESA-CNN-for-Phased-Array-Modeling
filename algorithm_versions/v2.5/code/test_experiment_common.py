"""非理想阵列公共采样、训练和评价内核回归。"""

import unittest

import torch

from beamforming_pre_fpga import ArrayGeometryConfig, TopologyAdaptiveBeamformingCNN
from experiment_common import SOURCE, TARGET, evaluate, make_samples, train_model


class TestExperimentCommon(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260823)
        torch.set_num_threads(1)

    def test_samples_are_bit_exact_for_same_local_seed(self) -> None:
        for config in (
            ArrayGeometryConfig.theoretical_4x4(),
            ArrayGeometryConfig.hardware_1x4(),
        ):
            first = make_samples(16, TARGET, config, seed=23001)
            second = make_samples(16, TARGET, config, seed=23001)
            for name in first.__dataclass_fields__:
                self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)

    def test_domain_rms_is_renormalised_after_smoothing(self) -> None:
        config = ArrayGeometryConfig.theoretical_4x4()
        samples = make_samples(64, TARGET, config, seed=23002)
        response = torch.complex(samples.true_response[:, 0], samples.true_response[:, 1])
        amp_rms = torch.sqrt((response.abs() - 1.0).square().mean())
        phase_rms = torch.rad2deg(torch.sqrt(torch.angle(response).square().mean()))
        position_rms = torch.sqrt(
            (samples.true_position / config.wavelength).square().mean()
        )
        self.assertAlmostEqual(float(amp_rms), TARGET.amp_rms, places=6)
        self.assertAlmostEqual(float(phase_rms), TARGET.phase_rms_deg, places=5)
        self.assertAlmostEqual(
            float(position_rms), TARGET.position_rms_wavelength, places=6
        )

    def test_short_training_and_evaluation_are_finite(self) -> None:
        config = ArrayGeometryConfig.hardware_1x4()
        samples = make_samples(16, SOURCE, config, seed=23003)
        model = TopologyAdaptiveBeamformingCNN(
            config, width=8, num_blocks=1, global_context="broadcast", output_bits=None
        )
        log = train_model(config, model, samples, seed=23004, steps=2)
        metrics = evaluate(config, model, samples, name="short", seed=23004)
        self.assertEqual(len(log), 2)
        for value in metrics.values():
            if isinstance(value, float):
                self.assertTrue(torch.isfinite(torch.tensor(value)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
