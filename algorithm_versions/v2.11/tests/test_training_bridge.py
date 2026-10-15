from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
V210 = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210 / "src"))

from v210.contracts import QueryBatch  # noqa: E402
from v211.training_bridge import build_v211_model, load_v211_checkpoint_model  # noqa: E402


BASE = {
    "architecture_version": "v210_nominal_coupling_residual_1",
    "basis_type": "ideal_phase",
    "source_type": "hfss_amplitude",
    "direction_fourier_bands": 4,
    "nominal_width": 24,
    "nominal_depth": 2,
    "nominal_max_log_amplitude": 3.0,
    "enable_coupling": True,
    "coupling_low_rank": 1,
    "enable_residual": True,
    "residual_width": 12,
    "residual_depth": 1,
    "residual_maximum_magnitude": 2.0,
    "reference_distance_m": 1.0,
}


class TrainingBridgeTests(unittest.TestCase):
    def test_all_variants_run_full_operator_and_reload(self) -> None:
        q = torch.tensor([[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]])
        variants = (
            {"kind": "fourier", "fourier_bands": 4, "output_channels": 27},
            {"kind": "ssh", "ssh_degree": 4, "output_channels": 27},
            {"kind": "ssh", "ssh_degree": 5, "output_channels": 27},
            {
                "kind": "fourier_plus_ssh",
                "fourier_bands": 4,
                "ssh_degree": 4,
                "output_channels": 27,
            },
        )
        for encoding in variants:
            model = build_v211_model(BASE, encoding)
            operator, _ = model.operator(QueryBatch(q))
            self.assertEqual(operator.shape, (1, 2, 2, 4, 4))
            checkpoint = {
                "base_model_config": BASE,
                "encoding_config": encoding,
                "model_state": model.state_dict(),
            }
            reloaded = load_v211_checkpoint_model(checkpoint, "cpu")
            torch.testing.assert_close(reloaded.operator(QueryBatch(q))[0], operator)


if __name__ == "__main__":
    unittest.main()
