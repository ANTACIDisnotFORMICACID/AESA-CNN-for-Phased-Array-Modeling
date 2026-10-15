"""Create and audit fixed leakage-aware splits before any amplitude training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import (
    AmplitudeDataSplit,
    amplitude_dataset_fingerprint,
    assert_split_matches_dataset,
    load_amplitude_dataset,
    save_amplitude_split,
)
from ..training import angular_block_split, stratified_command_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--theta-block-deg", type=float, default=10.0)
    parser.add_argument("--phi-block-deg", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=28001)
    return parser


def _shared_block_count(
    theta: np.ndarray,
    phi: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    theta_block_deg: float,
    phi_block_deg: float,
) -> int:
    theta_bin = np.floor((theta - theta.min()) / theta_block_deg).astype(np.int64)
    phi_bin = np.floor(np.mod(phi + 180.0, 360.0) / phi_block_deg).astype(np.int64)
    keys = theta_bin * (phi_bin.max() + 1) + phi_bin
    return int(np.intersect1d(np.unique(keys[indices_a]), np.unique(keys[indices_b])).size)


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_amplitude_dataset(args.dataset)
    query = angular_block_split(
        dataset.theta_deg,
        dataset.phi_deg,
        theta_block_deg=args.theta_block_deg,
        phi_block_deg=args.phi_block_deg,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    command = stratified_command_split(
        dataset.command_family,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    duplicate_directions = int(
        dataset.direction.shape[0]
        - np.unique(np.round(dataset.direction, decimals=12), axis=0).shape[0]
    )
    audit = {
        "query_indices_disjoint_and_complete": True,
        "command_indices_disjoint_and_complete": True,
        "shared_angular_blocks_train_validation": _shared_block_count(
            dataset.theta_deg, dataset.phi_deg, query.train, query.validation,
            args.theta_block_deg, args.phi_block_deg,
        ),
        "shared_angular_blocks_train_test": _shared_block_count(
            dataset.theta_deg, dataset.phi_deg, query.train, query.test,
            args.theta_block_deg, args.phi_block_deg,
        ),
        "shared_angular_blocks_validation_test": _shared_block_count(
            dataset.theta_deg, dataset.phi_deg, query.validation, query.test,
            args.theta_block_deg, args.phi_block_deg,
        ),
        "duplicate_physical_directions": duplicate_directions,
        "split_before_training": True,
    }
    if any(audit[name] != 0 for name in audit if name.startswith("shared_angular_blocks")):
        raise ValueError("angular blocks leaked across partitions")
    if duplicate_directions:
        raise ValueError("dataset contains duplicate physical directions; canonicalize before splitting")
    split = AmplitudeDataSplit(
        schema_version="v2.8-amplitude-split-1",
        dataset_fingerprint=amplitude_dataset_fingerprint(args.dataset),
        seed=args.seed,
        query_count=dataset.direction.shape[0],
        command_count=dataset.command_iq.shape[0],
        theta_block_deg=args.theta_block_deg,
        phi_block_deg=args.phi_block_deg,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        query_train=query.train,
        query_validation=query.validation,
        query_test=query.test,
        command_train=command.train,
        command_validation=command.validation,
        command_test=command.test,
        audit=audit,
    )
    save_amplitude_split(split, args.output)
    assert_split_matches_dataset(split, dataset, args.dataset)
    print(
        "saved fixed pre-training split: "
        f"queries={query.train.size}/{query.validation.size}/{query.test.size}, "
        f"commands={command.train.size}/{command.validation.size}/{command.test.size} "
        f"to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()

