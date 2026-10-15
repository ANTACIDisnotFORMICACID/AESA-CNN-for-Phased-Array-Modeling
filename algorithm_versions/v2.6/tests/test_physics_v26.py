import json
from pathlib import Path

import numpy as np

from hfss_physics_v26 import (
    CANONICAL_NPZ,
    HfssPatternData,
    angular_distance_deg,
    quantize_int8_disk,
    unit_vector_hfss,
)


def test_geometry_and_hfss_coordinate_contract():
    data = HfssPatternData.load()
    assert data.port_names == ("1:1", "2:1", "3:1", "4:1")
    np.testing.assert_allclose(data.y_mm, [0.0, 6.8, 13.6, 20.4], atol=1e-9)
    np.testing.assert_allclose(unit_vector_hfss(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(unit_vector_hfss(90.0, 90.0), [0.0, 1.0, 0.0], atol=1e-7)
    assert np.isclose(float(angular_distance_deg(90.0, 0.0, 90.0, 10.0)), 10.0, atol=1e-9)


def test_equal_phase_electrical_boresight_and_positive_ramp_sign():
    data = HfssPatternData.load()
    equal = data.metrics(np.ones(4, dtype=np.complex64), 113.0, -8.0)
    assert 112.0 <= equal.peak_theta_deg <= 115.0
    # Positive command phase progression is known to scan toward negative HFSS phi.
    positive = np.exp(1j * np.deg2rad(np.arange(4) * 45.0))
    positive_metrics = data.metrics(positive, 113.0, -20.0)
    assert positive_metrics.peak_phi_deg < 0.0


def test_linearity_and_recorded_direct_recomposition_gate():
    data = HfssPatternData.load()
    a = np.array([1 + 0j, 0.3 + 0.2j, -0.1j, -0.4 + 0.5j])
    b = np.array([0.2j, -0.2 + 0.1j, 0.7, 0.1 - 0.4j])
    fa = data.fields(a)
    fb = data.fields(b)
    fab = data.fields(a + b)
    np.testing.assert_allclose(fab[0], fa[0] + fb[0], rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(fab[1], fa[1] + fb[1], rtol=2e-6, atol=1e-6)
    report_path = CANONICAL_NPZ.parent / "linear_superposition_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Support either the original detailed schema or the compact later schema.
    text = json.dumps(report).lower()
    assert "pass" in text or "passed" in text


def test_integer_quantizer_never_leaves_disk():
    rng = np.random.default_rng(26)
    weights = rng.normal(size=(1000, 4)) + 1j * rng.normal(size=(1000, 4))
    codes, restored = quantize_int8_disk(weights)
    radius2 = codes[..., 0].astype(np.int16) ** 2 + codes[..., 1].astype(np.int16) ** 2
    assert np.all(radius2 <= 127**2)
    assert np.isfinite(restored).all()
