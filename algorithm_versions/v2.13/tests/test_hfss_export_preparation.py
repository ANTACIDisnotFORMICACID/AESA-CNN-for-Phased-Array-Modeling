from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_hfss_gate_a_export.py"
SPEC = importlib.util.spec_from_file_location("prepare_hfss_gate_a_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_hfss_gate_a_direct.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_hfss_gate_a_direct", EXPORT_SCRIPT)
EXPORT_MODULE = importlib.util.module_from_spec(EXPORT_SPEC)
assert EXPORT_SPEC.loader is not None
EXPORT_SPEC.loader.exec_module(EXPORT_MODULE)


def test_gate_a_commands_are_deterministic_diverse_and_unit_power():
    left = MODULE.build_commands()
    right = MODULE.build_commands()
    assert len(left) == 8
    matrix = np.stack([entry[2] for entry in left])
    assert np.allclose(matrix, np.stack([entry[2] for entry in right]))
    assert np.allclose(np.sum(np.abs(matrix) ** 2, axis=1), 1.0)
    assert np.allclose(matrix[1], 1j * matrix[0])
    assert {entry[1] for entry in left} == {
        "uniform_phase_control_0deg", "uniform_phase_control_90deg",
        "random_continuous_phase_only", "random_amplitude_phase",
    }


def test_direct_report_parser_converts_millivolts_to_volts():
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "tiny.csv"
        report.write_text(
            '"Theta [deg]","re(rEPhi) [mV] - Freq=\'47GHz\' Phi=\'0deg\'",'
            '"im(rEPhi) [mV] - Freq=\'47GHz\' Phi=\'0deg\'"\n'
            "10,1000,-500\n",
            encoding="utf-8",
        )
        theta, phi, field, unit = EXPORT_MODULE.parse_direct_report(report)
    assert np.array_equal(theta, [10.0])
    assert np.array_equal(phi, [0.0])
    assert np.allclose(field, [1.0 - 0.5j])
    assert unit == "mV"
