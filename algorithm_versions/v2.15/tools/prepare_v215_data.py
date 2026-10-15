"""Build physically separate train/validation packages with no test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for version in (ROOT, ROOT.parent / "v2.14", ROOT.parent / "v2.12", ROOT.parent / "v2.10"):
    sys.path.insert(0, str(version / "src"))

from v210.io import (  # noqa: E402
    assert_amplitude_only_training_contract,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    load_amplitude_split,
)
from v212.codebook import MeasurementCodebook  # noqa: E402
from v214.angular_sampling import load_amplitude_label_package  # noqa: E402


def _iq(command: torch.Tensor) -> np.ndarray:
    return np.stack((command.real.numpy(), command.imag.numpy()), axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--removal-seed", type=int, default=21470)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_amplitude_dataset(args.dataset)
    assert_amplitude_only_training_contract(dataset)
    split = load_amplitude_split(args.split_dir)
    assert_split_matches_dataset(split, dataset, args.dataset)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = np.asarray(dataset.direction, dtype=np.float32)
    amplitude = np.asarray(dataset.amplitude.T, dtype=np.float32)
    q_train = np.asarray(split.query_train, dtype=np.int64)
    q_validation = np.asarray(split.query_validation, dtype=np.int64)
    w_train = np.asarray(codebook.splits["train"], dtype=np.int64)
    w_validation = np.asarray(codebook.splits["validation"], dtype=np.int64)
    acquisition = load_amplitude_label_package(args.acquisition)
    if acquisition["direction"].shape != (490, 3) or acquisition["amplitude"].shape != (64, 490):
        raise ValueError("acquisition package does not satisfy the frozen 490x64 contract")
    if not np.array_equal(acquisition["command_indices"], w_train):
        raise ValueError("acquisition train-command indices changed")
    acquisition_command = acquisition["command_iq"][:, 0] + 1j * acquisition["command_iq"][:, 1]
    if not np.allclose(acquisition_command, codebook.weights[w_train].numpy(), atol=1e-7, rtol=1e-7):
        raise ValueError("acquisition commands differ from the codebook")

    generator = torch.Generator().manual_seed(args.removal_seed)
    removed_local = torch.randperm(len(q_train), generator=generator)[:490].numpy()
    keep = np.ones(len(q_train), dtype=bool)
    keep[removed_local] = False
    retained_global = q_train[keep]
    train_direction = np.concatenate((direction[retained_global], acquisition["direction"].astype(np.float32)))
    train_amplitude = np.concatenate((
        amplitude[retained_global][:, w_train],
        acquisition["amplitude"].T.astype(np.float32),
    ))
    if train_direction.shape != (22600, 3) or train_amplitude.shape != (22600, 64):
        raise AssertionError("exchange training budget changed")
    if {tuple(row) for row in np.round(train_direction, 9)} & {tuple(row) for row in np.round(direction[q_validation], 9)}:
        raise ValueError("training and validation directions overlap")

    np.savez_compressed(
        args.output_dir / "train_exchange.npz",
        direction=train_direction,
        amplitude=train_amplitude,
        command_indices=w_train,
        command_iq=_iq(codebook.weights[w_train]),
        source_role=np.concatenate((np.zeros(len(retained_global), dtype=np.uint8), np.ones(490, dtype=np.uint8))),
        original_global_indices=np.concatenate((retained_global, np.full(490, -1, dtype=np.int64))),
        removed_original_global_indices=q_train[removed_local],
    )
    np.savez_compressed(
        args.output_dir / "validation.npz",
        direction=direction[q_validation],
        amplitude=amplitude[q_validation][:, w_validation],
        command_indices=w_validation,
        command_iq=_iq(codebook.weights[w_validation]),
        original_global_indices=q_validation,
    )
    manifest = {
        "schema": "v215-no-leakage-data-1",
        "train": {"file": "train_exchange.npz", "directions": 22600, "commands": 64},
        "validation": {"file": "validation.npz", "directions": 4900, "commands": 16},
        "exchange": {
            "removal_seed": args.removal_seed,
            "retained_original": int(len(retained_global)),
            "acquisition_added": 490,
        },
        "source_dataset": str(args.dataset.resolve()),
        "source_split": str(args.split_dir.resolve()),
        "source_acquisition": str(args.acquisition.resolve()),
        "test_indices_saved": False,
        "test_directions_saved": False,
        "test_commands_saved": False,
        "test_amplitudes_saved": False,
        "external_test_used_for_preparation_decisions": False,
    }
    (args.output_dir / "PREPARATION.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
