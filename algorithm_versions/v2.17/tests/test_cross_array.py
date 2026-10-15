from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.cross_array import (  # noqa: E402
    ArraySpec,
    generate_direct_audit_commands,
    generate_commands,
    hfss_basis,
    load_touchstone,
    parse_ffd,
    spherical_block_split,
    write_training_views,
)


def test_array_positions_are_centred_and_row_major() -> None:
    spec = ArraySpec("a2x4", 2, 4, 0.015, 0.0068)
    positions = spec.positions_m
    assert positions.shape == (8, 3)
    assert np.allclose(positions.mean(axis=0), 0.0)
    assert np.isclose(positions[1, 1] - positions[0, 1], 0.0068)
    assert np.isclose(positions[4, 2] - positions[0, 2], 0.015)


def test_hfss_basis_is_orthonormal() -> None:
    direction, e_theta, e_phi = hfss_basis(
        np.asarray([30.0, 70.0]), np.asarray([-20.0, 80.0])
    )
    assert np.allclose(np.linalg.norm(direction, axis=1), 1.0)
    assert np.allclose(np.sum(direction * e_theta, axis=1), 0.0, atol=1e-14)
    assert np.allclose(np.sum(direction * e_phi, axis=1), 0.0, atol=1e-14)


def test_ffd_parser(tmp_path: Path) -> None:
    source = tmp_path / "one.ffd"
    source.write_text(
        "0 1 2\n-1 1 3\nFrequencies 1\nFrequency 4.7e10\n"
        + "\n".join(f"{i} {-i} {2*i} {-2*i}" for i in range(6)),
        encoding="utf-8",
    )
    grid = parse_ffd(source)
    assert grid.e_phi.shape == (6,)
    assert grid.frequency_hz == 47e9
    assert grid.theta_deg.tolist() == [0, 0, 0, 1, 1, 1]
    assert grid.phi_deg.tolist() == [-1, 0, 1, -1, 0, 1]


def test_touchstone_ma_column_major(tmp_path: Path) -> None:
    source = tmp_path / "tiny.s2p"
    # S11, S21, S12, S22 in Touchstone 1.x order.
    source.write_text("# GHz S MA R 50\n47 1 0 2 0 3 0 4 0\n", encoding="utf-8")
    frequency, matrix, reference = load_touchstone(source, 2)
    assert frequency.tolist() == [47e9]
    assert reference == 50.0
    assert np.allclose(matrix[0], [[1, 3], [2, 4]])


def test_block_split_is_disjoint_and_reproducible() -> None:
    rng = np.random.default_rng(17)
    direction = rng.normal(size=(2000, 3))
    direction[:, 0] = np.abs(direction[:, 0])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    left = spherical_block_split(direction, seed=21701)
    right = spherical_block_split(direction, seed=21701)
    for name in ("train_indices", "validation_indices", "blind_test_indices"):
        assert np.array_equal(left[name], right[name])
    sets = [set(left[name].tolist()) for name in ("train_indices", "validation_indices", "blind_test_indices")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == len(direction)


def test_command_families_and_power() -> None:
    spec = ArraySpec("a1x8", 1, 8, 0.015, 0.0068)
    command, family = generate_commands(spec, count=512, seed=21711)
    assert command.shape == (512, 8)
    assert set(family.tolist()) == {
        "random_constant_modulus", "scan", "taper", "sparse", "hard_null"
    }
    assert np.allclose(np.sum(np.abs(command) ** 2, axis=1), 1.0, atol=1e-12)


def test_training_views_exclude_blind_test(tmp_path: Path) -> None:
    canonical = tmp_path / "operator.npz"
    split_dir = tmp_path / "split"
    output = tmp_path / "cache"
    split_dir.mkdir()
    query_count = 10
    np.savez_compressed(
        canonical,
        direction=np.zeros((query_count, 3)), theta_deg=np.arange(query_count),
        phi_deg=np.arange(query_count), g_ephi=np.ones((query_count, 2), dtype=np.complex128),
        g_etheta=np.ones((query_count, 2), dtype=np.complex128),
        h_cartesian=np.ones((query_count, 3, 2), dtype=np.complex128),
        envelope_ephi=np.ones((query_count, 2), dtype=np.complex128),
        solid_angle_weights=np.full(query_count, 0.1),
        quality_mask=np.ones((query_count, 2), dtype=bool),
        positions_m=np.zeros((2, 3)), active_mask=np.ones((1, 2), dtype=bool),
        frequency_hz=np.asarray(47e9),
    )
    np.savez_compressed(
        split_dir / "direction_split.npz", train_indices=np.arange(6),
        validation_indices=np.arange(6, 8), blind_test_indices=np.arange(8, 10),
    )
    receipt = write_training_views(canonical, split_dir, output)
    assert receipt["blind_test_cache_written"] is False
    assert (output / "train_operator_cache.npz").is_file()
    assert (output / "validation_operator_cache.npz").is_file()
    assert not (output / "blind_test_operator_cache.npz").exists()


def test_direct_audit_has_four_commands_per_family() -> None:
    command, family = generate_direct_audit_commands(ArraySpec("a2x4", 2, 4, 0.015, 0.0068))
    assert command.shape == (16, 8)
    names, counts = np.unique(family, return_counts=True)
    assert dict(zip(names.tolist(), counts.tolist())) == {
        "cancellation_null": 4,
        "random_constant_modulus": 4,
        "scan": 4,
        "sparse_subarray": 4,
    }
    assert np.allclose(np.sum(np.abs(command) ** 2, axis=1), 1.0)
