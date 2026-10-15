import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


V211_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(V211_SRC))

from v211.models import (  # noqa: E402
    DirectionEncodingConfig,
    DirectionFourierEncoder,
    DirectionalApertureCNN,
    FourierPlusSSHEncoder,
    RealSSHEncoder,
    build_direction_encoder,
    load_ssh_statistics,
    real_spherical_harmonics,
)


def _directions(count: int = 128) -> torch.Tensor:
    generator = torch.Generator().manual_seed(211)
    directions = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    directions[:, 0] = directions[:, 0].abs().clamp_min(1e-3)
    return directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)


class DirectionEncodingTests(unittest.TestCase):
    def test_fourier_encoder_retains_v210_formula_and_27_channels(self) -> None:
        direction = _directions(17).to(torch.float32)[None]
        encoder = DirectionFourierEncoder(4)
        frequencies = 2.0 ** torch.arange(4) * math.pi
        angles = direction[..., :, None] * frequencies
        expected = torch.cat(
            (direction, torch.sin(angles).reshape(1, 17, -1), torch.cos(angles).reshape(1, 17, -1)),
            dim=-1,
        )
        torch.testing.assert_close(encoder(direction), expected)
        self.assertEqual(encoder.output_channels, 27)

    def test_real_ssh_has_expected_constant_and_channel_count(self) -> None:
        direction = _directions(31)
        basis = real_spherical_harmonics(direction, 5)
        self.assertEqual(basis.shape, (31, 36))
        torch.testing.assert_close(
            basis[:, 0],
            torch.full((31,), 1.0 / math.sqrt(4.0 * math.pi), dtype=torch.float64),
        )

    def test_training_standardization_and_prefix_loader(self) -> None:
        direction = _directions(512)
        raw_l8 = real_spherical_harmonics(direction, 8)
        mean = raw_l8.mean(dim=0)
        std = raw_l8.std(dim=0, correction=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.npz"
            np.savez(path, degree=np.asarray(8), mean=mean.numpy(), std=std.numpy())
            mean_l4, std_l4 = load_ssh_statistics(path, 4)
        encoder = RealSSHEncoder(4, mean=mean_l4, std=std_l4).to(torch.float64)
        standardized = encoder.standardized_basis(direction[None])[0]
        active = std_l4 > 1e-6
        torch.testing.assert_close(
            standardized[:, active].mean(dim=0),
            torch.zeros(int(active.sum()), dtype=torch.float64),
            atol=2e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            standardized[:, active].std(dim=0, correction=0),
            torch.ones(int(active.sum()), dtype=torch.float64),
            atol=2e-5,
            rtol=0.0,
        )

    def test_hybrid_is_exact_fourier_control_at_initialization(self) -> None:
        direction = _directions(23).to(torch.float32)[None]
        raw = real_spherical_harmonics(direction, 4)
        mean = raw.mean(dim=(0, 1))
        std = raw.std(dim=(0, 1), correction=0)
        hybrid = FourierPlusSSHEncoder(
            fourier_bands=4,
            ssh_degree=4,
            ssh_mean=mean,
            ssh_std=std,
        )
        torch.testing.assert_close(hybrid(direction), hybrid.fourier(direction), atol=0.0, rtol=0.0)
        self.assertEqual(float(hybrid.alpha.detach()), 0.0)

    def test_all_encoders_preserve_31_channel_cnn_contract(self) -> None:
        direction = _directions(29).to(torch.float32)[None]
        raw = real_spherical_harmonics(direction, 5)
        mean = raw.mean(dim=(0, 1))
        std = raw.std(dim=(0, 1), correction=0)
        configs = [
            DirectionEncodingConfig(kind="fourier"),
            DirectionEncodingConfig(kind="ssh", ssh_degree=5),
            DirectionEncodingConfig(kind="fourier_plus_ssh", ssh_degree=5),
        ]
        for config in configs:
            encoder = build_direction_encoder(config, ssh_mean=mean, ssh_std=std)
            cnn = DirectionalApertureCNN(encoder, width=8, depth=1, output_channels=3)
            self.assertEqual(cnn.build_feature_maps(direction).shape, (1, 29, 31, 4, 4))
            self.assertEqual(cnn(direction).shape, (1, 29, 3, 4, 4))
            self.assertTrue(torch.isfinite(cnn(direction)).all())


if __name__ == "__main__":
    unittest.main()
