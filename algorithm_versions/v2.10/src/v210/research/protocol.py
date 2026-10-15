"""Data preparation and immutable protocol for the September 2026 campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ..contracts import ArrayGeometry
from ..coordinates import hfss_angles_to_direction
from ..io.amplitude_synthesis import canonical_spherical_query_mask

PROTOCOL = {
    "schema": "v210-prospective-campaign-1",
    "split_seed": 29291,
    "command_seed": 29101,
    "command_rng_policy": "independent SeedSequence([29101, family_index, partition_index])",
    "model_seeds": [29001, 29002, 29003],
    "sampling_seeds": [39001, 39002, 39003],
    "rounds": {"screen": 300, "confirm": 800, "final": 2000},
    "batch_size": 1024, "queries_per_epoch": 8192,
    "validation_batch_size": 2048, "checkpoint_every": 25,
    "minimum_stop_epoch": 400, "plateau_window": 150,
    "plateau_db": 0.05, "cap_window": 200,
    "recipes": {
        "R1": {"schedule": "plateau", "learning_rate": 0.001},
        "R2": {"schedule": "cosine", "learning_rate": 0.001},
        "R3": {"schedule": "plateau", "learning_rate": 0.0005},
    },
    "scheduler_patience": 25, "scheduler_factor": 0.5,
    "minimum_learning_rate": 1e-5, "warmup_epochs": 20,
    "loss": {"db_loss_weight": 0.03, "power_loss_weight": 0.10,
             "residual_energy_weight": 1e-4, "coupling_l2_weight": 1e-4,
             "reciprocity_weight": 1e-3},
    "weight_decay": 1e-5, "gradient_clip_norm": 5.0,
    "acceptance": {"paired_improvement_db": 0.3, "seed_std_db": 0.25,
                   "maximum_seed_regression_db": 0.1, "offset_degradation_db": 0.5,
                   "target_median_regret": 0.05, "target_p95_regret": 0.10},
    "selection": "validation global amplitude NMSE, then power NMSE, then dB MAE",
    "amplitude_observable": "Ephi V/m at 1m; complex FFD used by label generator only",
    "angular_blocks": "floor(theta/10), floor(mod(phi+180,360)/10); absolute origin",
    "blind_policy": "no blind labels loaded before all three final checkpoint hashes are frozen",
    "historical_exposure": "old integer grid and finite single/DFT anchors were used in prior versions",
    "target_reference": "multi-start numerical HFSS reference, not a certified global optimum",
}

MODEL_CONFIG = {
    "architecture_version": "v210_nominal_coupling_residual_1",
    "basis_type": "ideal_phase", "source_type": "hfss_amplitude",
    "direction_fourier_bands": 4, "nominal_width": 24, "nominal_depth": 2,
    "nominal_max_log_amplitude": 3.0, "enable_coupling": True,
    "coupling_low_rank": 1, "enable_residual": True, "residual_width": 12,
    "residual_depth": 1, "residual_maximum_magnitude": 2.0,
    "device_model": "identity", "reference_distance_m": 1.0,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_keys(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.floor(theta / 10).astype(np.int64) * 36 + np.floor(
        np.mod(phi + 180, 360) / 10
    ).astype(np.int64)


def command_signature(command: np.ndarray) -> bytes:
    flat = np.asarray(command, dtype=np.complex128).reshape(-1).copy()
    flat /= np.linalg.norm(flat)
    first = flat[np.flatnonzero(np.abs(flat) > 1e-7)[0]]
    flat *= np.exp(-1j * np.angle(first))
    values = np.stack((flat.real, flat.imag))
    values = np.round(values * 1e6).astype(np.int64)
    return values.tobytes()


def make_codebook() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(PROTOCOL["command_seed"])
    geometry = ArrayGeometry()
    families = []
    values = []
    parts = []
    counts = [
        ("single_element", (12, 2, 2)), ("dft_phase", (12, 2, 2)),
        ("random_phase", (90, 19, 19)), ("random_complex", (90, 19, 19)),
        ("sparse_complex", (44, 10, 10)), ("steering", (44, 10, 10)),
    ]
    for family_index, (family, split_counts) in enumerate(counts):
        commands = []
        for i in range(sum(split_counts)):
            partition_index = int(i >= split_counts[0]) + int(i >= sum(split_counts[:2]))
            if i in (0, split_counts[0], sum(split_counts[:2])):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [PROTOCOL["command_seed"], family_index, partition_index]))
            if family == "single_element":
                w = np.eye(16, dtype=complex)[i]
            elif family == "dft_phase":
                r, c = np.indices((4, 4))
                w = np.exp(2j * np.pi * ((i // 4) * r + (i % 4) * c) / 4).ravel()
            elif family == "random_phase":
                w = np.exp(1j * rng.uniform(-np.pi, np.pi, 16))
            elif family in ("random_complex", "sparse_complex"):
                w = rng.normal(size=16) + 1j * rng.normal(size=16)
                if family == "sparse_complex":
                    active = rng.choice(16, size=(4, 8, 12)[i % 3], replace=False)
                    mask = np.zeros(16)
                    mask[active] = 1
                    w *= mask
            else:
                x = rng.uniform(0.5, 1.0)
                az = rng.uniform(-np.pi, np.pi)
                q = np.array([x, np.sqrt(1-x*x)*np.cos(az), np.sqrt(1-x*x)*np.sin(az)])
                w = np.exp(-2j*np.pi/geometry.wavelength_m*(geometry.positions_m @ q)).ravel()
            commands.append(w / np.linalg.norm(w))
        order = (rng.permutation(len(commands)) if family in ("single_element", "dft_phase")
                 else np.arange(len(commands)))
        start = 0
        for part, count in enumerate(split_counts):
            for index in order[start:start + count]:
                values.append(commands[index])
                families.append(family)
                parts.append(part)
            start += count
    command = np.asarray(values).reshape(-1, 4, 4)
    signatures = [command_signature(w) for w in command]
    if len(set(signatures)) != 416:
        raise ValueError("duplicate commands up to global phase")
    return command, np.asarray(families, dtype="U32"), np.asarray(parts)


def header(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [next(f).strip() for _ in range(4)]
    a, b = ([float(v) for v in rows[i].split()] for i in (0, 1))
    if rows[2] != "Frequencies 1" or not np.isclose(float(rows[3].split()[-1]), 47e9):
        raise ValueError(f"unexpected frequency header: {path}")
    theta = np.linspace(a[0], a[1], int(a[2]))
    phi = np.linspace(b[0], b[1], int(b[2]))
    return np.repeat(theta, len(phi)), np.tile(phi, len(theta))


def source_files(folder: Path, shifted: bool) -> list[Path]:
    stem = "exportelement_new" if shifted else "exportelement"
    mapping = (folder / (stem + ".txt")).read_text(encoding="utf-8-sig")
    for i in range(1, 17):
        match = re.search(rf"^{i}:1\s+{stem}_{i}\s+([^\r\n]+)", mapping, re.MULTILINE)
        if not match or not np.allclose([float(x) for x in match[1].split()], 0):
            raise ValueError(f"missing/nonzero phase-reference mapping for port {i}")
    return [folder / f"{stem}_{i}.ffd" for i in range(1, 17)]


def load_scalar_basis(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, phi = header(files[0])
    basis = []
    for file in files:
        t, p = header(file)
        if not (np.array_equal(t, theta) and np.array_equal(p, phi)):
            raise ValueError("port grids differ")
        field = np.loadtxt(file, skiprows=4)
        if field.shape != (theta.size, 4) or not np.isfinite(field).all():
            raise ValueError("invalid FFD rows")
        basis.append(field[:, 2] + 1j * field[:, 3])
    return theta, phi, np.stack(basis, axis=1)


def prepare(folder: Path, root: Path, source_code_root: Path) -> None:
    if (root / "protocol.json").exists():
        raise FileExistsError("campaign already exists; no protocol/data overwrite")
    old = source_files(folder, False)
    shifted = source_files(folder, True)
    old_t, old_p = header(old[0])
    new_t, new_p = header(shifted[0])
    for file in old:
        t, p = header(file)
        assert np.array_equal(t, old_t) and np.array_equal(p, old_p)
    for file in shifted:
        t, p = header(file)
        assert np.array_equal(t, new_t) and np.array_equal(p, new_p)
    assert (old_t.size, new_t.size) == (65341, 64800)
    assert np.allclose(np.unique(new_t), np.arange(0.5, 180, 1))
    assert np.allclose(np.unique(new_p), np.arange(-179.5, 180, 1))
    old_dirs = hfss_angles_to_direction(old_t, old_p)
    new_dirs = hfss_angles_to_direction(new_t, new_p)
    old_set = {tuple(v) for v in np.round(old_dirs, 10)}
    assert not any(tuple(v) in old_set for v in np.round(new_dirs, 10))
    command, family, part = make_codebook()
    all_blocks = np.arange(648)
    order = np.random.default_rng(PROTOCOL["split_seed"]).permutation(all_blocks)
    test_blocks, val_blocks, train_blocks = order[:97], order[97:194], order[194:]
    theta, phi, basis = load_scalar_basis(old)
    keep = canonical_spherical_query_mask(theta, phi)
    theta, phi, basis = theta[keep], phi[keep], basis[keep]
    directions = hfss_angles_to_direction(theta, phi)
    keys = block_keys(theta, phi)
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, pid, blocks in (("train", 0, train_blocks), ("validation", 1, val_blocks)):
        path = root / "dev" / (name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        qi = np.flatnonzero(np.isin(keys, blocks))
        ci = np.flatnonzero(part == pid)
        amplitude = np.abs(command[ci].reshape(-1, 16) @ basis[qi].T).astype(np.float32)
        np.savez(path, direction=directions[qi], theta=theta[qi], phi=phi[qi],
                 command_iq=np.stack((command[ci].real, command[ci].imag), axis=1).astype(np.float32),
                 family=family[ci], amplitude=amplitude, query_indices=qi, command_indices=ci,
                 block_ids=blocks)
        hashes[name] = digest(path)
    locked = root / "blind" / "spec.npz"
    locked.parent.mkdir(parents=True, exist_ok=True)
    np.savez(locked, command=command, family=family, partition=part,
             train_blocks=train_blocks, validation_blocks=val_blocks, test_blocks=test_blocks)
    file_hashes = {str(p.resolve()): digest(p) for p in old + shifted}
    source_hashes = {str(p.relative_to(source_code_root)): digest(p)
                     for p in source_code_root.rglob("*.py")}
    protocol = {**PROTOCOL, "model_config": MODEL_CONFIG, "raw_source_hashes": file_hashes,
                "source_code_hashes": source_hashes, "source_folder": str(folder.resolve()),
                "dev_hashes": hashes, "blind_spec_sha256": digest(locked),
                "command_counts": [292, 62, 62], "old_queries": len(theta),
                "shifted_queries": len(new_t), "exact_grid_overlap": 0,
                "shifted_field_values_read": False,
                "test_label_files_created": False}
    write_json(root / "protocol.json", protocol)
    print("PREPARED: 292/62/62 commands; dev only; zero integer/shifted direction overlap", flush=True)


def assert_protocol(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    for key, value in PROTOCOL.items():
        if protocol.get(key) != value:
            raise ValueError(f"frozen protocol mismatch: {key}")
    for name, expected in protocol["dev_hashes"].items():
        if digest(root / "dev" / (name + ".npz")) != expected:
            raise ValueError(f"development data mutated: {name}")
    return protocol

