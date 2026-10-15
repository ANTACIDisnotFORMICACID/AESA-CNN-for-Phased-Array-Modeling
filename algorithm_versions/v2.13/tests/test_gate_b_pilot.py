from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_gate_b_hardware_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_gate_b_hardware_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_repeat_precision_supports_ten_repeat_recommendation() -> None:
    r3 = MODULE.repeat_precision(3, (30.0,))
    r10 = MODULE.repeat_precision(10, (30.0,))
    width3 = np.diff(r3["standard_deviation_95ci_factor_relative_to_sample_std"])[0]
    width10 = np.diff(r10["standard_deviation_95ci_factor_relative_to_sample_std"])[0]
    assert width10 < width3
    assert r10["mean_power_95ci_half_width_percent_by_repeat_snr_db"]["30"] < 3.0


def test_pilot_selection_is_unique_stratified_and_deterministic() -> None:
    generator = np.random.default_rng(9)
    count = 1000
    theta = generator.uniform(5.0, 175.0, size=count)
    phi = generator.uniform(-85.0, 85.0, size=count)
    theta[:80] = generator.uniform(50.1, 59.9, size=80)
    phi[:80] = generator.uniform(30.1, 39.9, size=80)
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    direction = np.stack(
        (
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ),
        axis=1,
    )
    signal = np.linspace(-30.0, 0.0, count)
    condition = 10.0 + np.linspace(0.0, 20.0, count)
    left = MODULE.select_pilot(direction, theta, phi, signal, condition, seed=10)
    right = MODULE.select_pilot(direction, theta, phi, signal, condition, seed=10)
    selected, _, roles, report = left
    assert np.array_equal(selected, right[0])
    assert len(selected) == len(np.unique(selected)) == 240
    assert roles.count("core_stratified") == 200
    assert roles.count("coverage_diagnostic") == 40
    assert set(report["selected_counts"].values()) == {40}


def test_schedule_contains_only_train_and_validation_commands() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schedule.csv"
        MODULE._write_schedule(path, 2, repeats=3, seed=11)
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 80 * 3
    assert {int(row["command_id"]) for row in rows} == set(range(80))
    assert {row["split"] for row in rows} == {"train", "validation"}
