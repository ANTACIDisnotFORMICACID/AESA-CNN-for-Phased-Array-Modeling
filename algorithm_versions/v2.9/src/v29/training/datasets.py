"""Dataset adapters and angular-block splits that avoid neighbour leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..io.dataset_store import BasisDataset


class OperatorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, basis: BasisDataset, indices: np.ndarray | None = None) -> None:
        self.direction = torch.as_tensor(basis.direction, dtype=torch.float32)
        self.operator_iq = torch.as_tensor(basis.operator_iq, dtype=torch.float32)
        self.indices = np.arange(len(self.direction), dtype=np.int64) if indices is None else np.asarray(indices)
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return self.direction[index], self.operator_iq[index]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def angular_block_split(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    *,
    theta_block_deg: float = 10.0,
    phi_block_deg: float = 10.0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 28001,
) -> SplitIndices:
    """Assign whole angular blocks, never adjacent samples independently."""
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if theta.shape != phi.shape or theta.ndim != 1:
        raise ValueError("theta_deg and phi_deg must be matching one-dimensional arrays")
    if theta_block_deg <= 0 or phi_block_deg <= 0:
        raise ValueError("angular block sizes must be positive")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid validation/test fractions")

    theta_bin = np.floor((theta - theta.min()) / theta_block_deg).astype(np.int64)
    phi_wrapped = np.mod(phi + 180.0, 360.0)
    phi_bin = np.floor(phi_wrapped / phi_block_deg).astype(np.int64)
    groups = np.stack((theta_bin, phi_bin), axis=1)
    unique_groups, inverse = np.unique(groups, axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique_groups))
    test_count = int(round(len(order) * test_fraction))
    validation_count = int(round(len(order) * validation_fraction))
    test_groups = order[:test_count]
    validation_groups = order[test_count : test_count + validation_count]
    train_groups = order[test_count + validation_count :]

    def indices_for(group_ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(inverse, group_ids))

    return SplitIndices(
        train=indices_for(train_groups),
        validation=indices_for(validation_groups),
        test=indices_for(test_groups),
    )

