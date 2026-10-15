"""Select complementary integer-grid directions before reading their HFSS labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v210.coordinates import hfss_angles_to_direction  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v214.angular_sampling import angular_block_ids, block_set, select_per_block_maximin  # noqa: E402


def _ffd_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    theta_spec = lines[0].split()
    phi_spec = lines[1].split()
    theta_axis = np.linspace(float(theta_spec[0]), float(theta_spec[1]), int(theta_spec[2]))
    phi_axis = np.linspace(float(phi_spec[0]), float(phi_spec[1]), int(phi_spec[2]))
    theta = np.repeat(theta_axis, len(phi_axis))
    phi = np.tile(phi_axis, len(theta_axis))
    return theta, phi, hfss_angles_to_direction(theta, phi), float(lines[3].split()[-1])


def _load_selected_ephi(ffd_dir: Path, full_indices: np.ndarray) -> np.ndarray:
    wanted = {int(value): index for index, value in enumerate(np.asarray(full_indices))}
    result = np.empty((len(full_indices), 16), dtype=np.complex128)
    for port in range(1, 17):
        path = ffd_dir / f"exportelement_{port}.ffd"
        found = 0
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line_number, line in enumerate(stream):
                if line_number < 4:
                    continue
                source_index = line_number - 4
                destination = wanted.get(source_index)
                if destination is None:
                    continue
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"invalid FFD row {source_index}: {path}")
                result[destination, port - 1] = float(values[2]) + 1j * float(values[3])
                found += 1
        if found != len(full_indices):
            raise ValueError(f"selected row count mismatch for {path}: {found}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffd-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-block", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_theta, full_phi, full_direction, frequency = _ffd_grid(args.ffd_dir / "exportelement_1.ffd")
    front = full_direction[:, 0] > 1e-12
    full_front_index = np.flatnonzero(front)
    theta, phi, candidate_direction = full_theta[front], full_phi[front], full_direction[front]
    current_direction = np.load(args.dataset_root / "amplitude_dataset" / "direction.npy")
    current_theta = np.load(args.dataset_root / "amplitude_dataset" / "theta_deg.npy")
    current_phi = np.load(args.dataset_root / "amplitude_dataset" / "phi_deg.npy")
    split = {
        name: np.load(args.dataset_root / "split" / f"query_{name}.npy")
        for name in ("train", "validation", "test")
    }
    current_blocks = angular_block_ids(current_theta, current_phi)
    role_blocks = {name: block_set(current_blocks[index]) for name, index in split.items()}
    if any(role_blocks[a] & role_blocks[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("current split blocks overlap")
    candidate_blocks = angular_block_ids(theta, phi)
    counts = {
        name: int(sum(tuple(int(v) for v in block) in groups for block in candidate_blocks))
        for name, groups in role_blocks.items()
    }
    expected = {"train": 22301, "validation": 4870, "test": 4870}
    if len(candidate_direction) != 32041 or counts != expected:
        raise ValueError(f"complementary-grid contract changed: total={len(candidate_direction)}, counts={counts}")
    selected, probe = select_per_block_maximin(
        candidate_direction, candidate_blocks, role_blocks["validation"],
        current_direction[split["train"]], per_block=args.per_block,
    )
    if len(selected) != 490 or len(probe) != 4380:
        raise ValueError(f"expected 490/4380 selected/probe directions, got {len(selected)}/{len(probe)}")
    if block_set(candidate_blocks[selected]) != role_blocks["validation"]:
        raise ValueError("selected directions do not cover all validation blocks")
    if any(tuple(int(v) for v in block) in role_blocks["test"] for block in candidate_blocks[np.concatenate((selected, probe))]):
        raise ValueError("external-test block entered label selection")
    rounded_current = {tuple(row) for row in np.round(current_direction, 10)}
    if any(tuple(row) in rounded_current for row in np.round(candidate_direction, 10)):
        raise ValueError("integer and half-degree direction grids overlap")

    geometry_selection = {
        "schema": "v214-angular-geometry-selection-1", "selection_uses_labels": False,
        "candidate_count": 32041, "candidate_role_counts": counts,
        "validation_block_count": len(role_blocks["validation"]),
        "acquisition_count": int(len(selected)), "coverage_probe_count": int(len(probe)),
        "per_block": args.per_block, "external_test_blocks_excluded": True,
        "external_test_labels_used": False,
        "acquisition_angles": [
            {"theta_deg": float(theta[index]), "phi_deg": float(phi[index])}
            for index in selected
        ],
    }
    selection_path = args.output_dir / "GEOMETRY_SELECTION.json"
    encoded = json.dumps(geometry_selection, ensure_ascii=False, indent=2)
    if selection_path.exists() and selection_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("existing geometry selection differs; use a new output directory")
    selection_path.write_text(encoded, encoding="utf-8")

    selected_all = np.concatenate((selected, probe))
    g_all = _load_selected_ephi(args.ffd_dir, full_front_index[selected_all])
    g_acquisition = g_all[:len(selected)]
    g_probe = g_all[len(selected):]
    codebook = MeasurementCodebook.load(args.codebook)
    train_command_index = np.asarray(codebook.splits["train"], dtype=np.int64)
    validation_command_index = np.asarray(codebook.splits["validation"], dtype=np.int64)
    train_commands = codebook.weights[train_command_index].numpy()
    validation_commands = codebook.weights[validation_command_index].numpy()
    acquisition_amplitude = np.abs(g_acquisition @ train_commands.T).T.astype(np.float32)
    probe_amplitude = np.abs(g_probe @ validation_commands.T).T.astype(np.float32)
    np.savez_compressed(
        args.output_dir / "acquisition_train.npz",
        direction=candidate_direction[selected].astype(np.float32), theta_deg=theta[selected], phi_deg=phi[selected],
        command_indices=train_command_index, command_iq=np.stack((train_commands.real, train_commands.imag), axis=1).astype(np.float32),
        amplitude=acquisition_amplitude,
    )
    np.savez_compressed(
        args.output_dir / "coverage_probe.npz",
        direction=candidate_direction[probe].astype(np.float32), theta_deg=theta[probe], phi_deg=phi[probe],
        command_indices=validation_command_index,
        command_iq=np.stack((validation_commands.real, validation_commands.imag), axis=1).astype(np.float32),
        amplitude=probe_amplitude,
    )
    np.savez_compressed(
        args.output_dir / "complex_audit_only.npz",
        acquisition_direction=candidate_direction[selected], acquisition_g_ephi=g_acquisition,
        probe_direction=candidate_direction[probe], probe_g_ephi=g_probe,
        not_for_training=np.asarray(True), external_test_values_included=np.asarray(False),
    )
    manifest = {
        "schema": "v214-new-angular-labels-1", "frequency_hz": frequency,
        "source": str((args.ffd_dir / "exportelement_*.ffd").resolve()),
        "source_grid": "integer theta/phi degrees", "current_grid": "half-degree offset theta/phi",
        "acquisition": {"directions": 490, "commands": 64, "file": "acquisition_train.npz"},
        "coverage_probe": {"directions": 4380, "commands": 16, "file": "coverage_probe.npz"},
        "complex_audit": {"file": "complex_audit_only.npz", "training_allowed": False},
        "external_test_blocks_excluded": True, "external_test_labels_used": False,
        "test_command_indices_included": False, "geometry_selection_frozen_before_label_read": True,
    }
    (args.output_dir / "LABEL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "candidate_role_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

