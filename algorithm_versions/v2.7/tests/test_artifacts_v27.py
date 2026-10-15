from pathlib import Path

import numpy as np


V27 = Path(__file__).resolve().parents[1]


def test_phase_map_has_4096_finite_unique_records():
    with np.load(V27 / "output" / "maps" / "nominal_phase_only_map.npz", allow_pickle=False) as archive:
        targets = archive["targets_hfss_deg"]
        assert targets.shape == (4096, 2)
        assert len(np.unique(targets, axis=0)) == 4096
        assert np.isfinite(archive["metrics"]).all()


def test_multi_parameter_artifact_has_21248_cases_and_integer_disk_codes():
    with np.load(V27 / "output" / "audit" / "control_cases.npz", allow_pickle=False) as archive:
        assert bool(archive["complete"])
        assert archive["metrics"].shape == (83, 6, 256, 9)
        assert np.isfinite(archive["metrics"]).all()
        codes = archive["int8_codes"].astype(np.int32)
        assert np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)
        assert float(np.max(archive["calibration_nmse_db"])) <= -20.0
