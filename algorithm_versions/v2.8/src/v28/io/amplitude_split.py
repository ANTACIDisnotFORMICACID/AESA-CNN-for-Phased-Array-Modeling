"""Persist amplitude train/validation/test partitions before model training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .amplitude_dataset import AmplitudeDataset


_DATASET_FILES = (
    "manifest.json",
    "direction.npy",
    "command_iq.npy",
    "amplitude.npy",
    "command_family.npy",
    "theta_deg.npy",
    "phi_deg.npy",
)


def amplitude_dataset_fingerprint(source: str | Path) -> str:
    """Bind a split to the exact amplitude-only files it was made from."""
    source = Path(source)
    digest = hashlib.sha256()
    for name in _DATASET_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AmplitudeDataSplit:
    schema_version: str
    dataset_fingerprint: str
    seed: int
    query_count: int
    command_count: int
    theta_block_deg: float
    phi_block_deg: float
    validation_fraction: float
    test_fraction: float
    query_train: np.ndarray
    query_validation: np.ndarray
    query_test: np.ndarray
    command_train: np.ndarray
    command_validation: np.ndarray
    command_test: np.ndarray
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != "v2.8-amplitude-split-1":
            raise ValueError("unsupported amplitude split schema")
        if len(self.dataset_fingerprint) != 64:
            raise ValueError("dataset_fingerprint must be SHA-256")
        if self.query_count <= 0 or self.command_count <= 0:
            raise ValueError("split dimensions must be positive")
        self._validate_partition(
            (self.query_train, self.query_validation, self.query_test),
            self.query_count,
            "query",
        )
        self._validate_partition(
            (self.command_train, self.command_validation, self.command_test),
            self.command_count,
            "command",
        )

    @staticmethod
    def _validate_partition(parts: tuple[np.ndarray, ...], size: int, name: str) -> None:
        normalized = []
        for part in parts:
            values = np.asarray(part)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} split indices must be one-dimensional integers")
            if values.size == 0:
                raise ValueError(f"every {name} split must be non-empty")
            normalized.append(values.astype(np.int64, copy=False))
        joined = np.concatenate(normalized)
        if not np.array_equal(np.sort(joined), np.arange(size, dtype=np.int64)):
            raise ValueError(f"{name} splits must be disjoint and cover every index exactly once")


def save_amplitude_split(split: AmplitudeDataSplit, destination: str | Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        "query_train": split.query_train,
        "query_validation": split.query_validation,
        "query_test": split.query_test,
        "command_train": split.command_train,
        "command_validation": split.command_validation,
        "command_test": split.command_test,
    }
    for name, values in arrays.items():
        np.save(destination / f"{name}.npy", np.asarray(values, dtype=np.int64))
    metadata = {
        "schema_version": split.schema_version,
        "dataset_fingerprint": split.dataset_fingerprint,
        "seed": split.seed,
        "query_count": split.query_count,
        "command_count": split.command_count,
        "theta_block_deg": split.theta_block_deg,
        "phi_block_deg": split.phi_block_deg,
        "validation_fraction": split.validation_fraction,
        "test_fraction": split.test_fraction,
        "audit": split.audit,
    }
    (destination / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_amplitude_split(source: str | Path) -> AmplitudeDataSplit:
    source = Path(source)
    metadata = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(source / f"{name}.npy", allow_pickle=False)
        for name in (
            "query_train",
            "query_validation",
            "query_test",
            "command_train",
            "command_validation",
            "command_test",
        )
    }
    return AmplitudeDataSplit(**metadata, **arrays)


def assert_split_matches_dataset(
    split: AmplitudeDataSplit,
    dataset: AmplitudeDataset,
    dataset_source: str | Path,
) -> None:
    if split.query_count != dataset.direction.shape[0]:
        raise ValueError("split query count does not match amplitude dataset")
    if split.command_count != dataset.command_iq.shape[0]:
        raise ValueError("split command count does not match amplitude dataset")
    if split.dataset_fingerprint != amplitude_dataset_fingerprint(dataset_source):
        raise ValueError("split fingerprint does not match the amplitude dataset files")

    labels = np.empty(split.query_count, dtype=np.int8)
    labels[split.query_train] = 0
    labels[split.query_validation] = 1
    labels[split.query_test] = 2
    _, inverse = np.unique(np.round(dataset.direction, decimals=12), axis=0, return_inverse=True)
    for duplicate_id in np.flatnonzero(np.bincount(inverse) > 1):
        if np.unique(labels[inverse == duplicate_id]).size > 1:
            raise ValueError("an identical physical direction appears in multiple query splits")

