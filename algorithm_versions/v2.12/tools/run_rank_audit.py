"""Run the non-learning rank spectrum audit and optionally create a G3 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.distillation import OperatorTarget, confidence_from_fisher  # noqa: E402
from v212.inversion import rank_spectrum_audit  # noqa: E402


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--power-repeats", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=21290)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    repeats = np.load(args.power_repeats, allow_pickle=False)
    if repeats.shape[:2] != (len(dataset.direction), 96) or repeats.shape[2] < 3:
        raise ValueError("power repeats must have shape [all directions,96,at least 3]")
    query_indices = np.asarray(split.query_train, dtype=np.int64)
    limited = args.query_limit is not None
    if limited:
        if args.query_limit <= 0:
            raise ValueError("query-limit must be positive")
        query_indices = query_indices[: args.query_limit]
    heldout = "validation_test" if args.open_test else "validation"
    audit = rank_spectrum_audit(
        torch.from_numpy(repeats[query_indices]).to(args.device),
        codebook,
        steps=args.steps,
        seed=args.seed,
        heldout_split=heldout,
        batch_size=args.batch_size,
    )
    records = audit["records"]
    residual = torch.tensor(
        [row["rank1_training_normalized_residual"] for row in records], dtype=torch.float32
    )
    minimum = torch.tensor(
        [row["minimum_jacobian_singular_value"] for row in records], dtype=torch.float32
    )
    fisher = audit["fisher"].cpu()
    target = OperatorTarget(
        q_hat=audit["q_hat"].cpu(),
        confidence=confidence_from_fisher(fisher).cpu(),
        inversion_residual=residual,
        minimum_jacobian_singular_value=minimum,
        query_indices=torch.from_numpy(query_indices.copy()),
        command_indices=codebook.indices("train"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not limited:
        target.save(args.output_dir / "operator_target_train_commands_only.npz")
    elif (args.output_dir / "operator_target_train_commands_only.npz").exists():
        raise RuntimeError("a limited audit cannot overwrite a full G3 target")

    summary = {
        "schema": "v212-rank-spectrum-audit-summary-1",
        "status": audit["status"],
        "query_count": len(query_indices),
        "batch_size": args.batch_size,
        "optimization_steps": args.steps,
        "limited_diagnostic": limited,
        "heldout_commands": heldout,
        "test_split_opened": bool(args.open_test),
        "contract_failure_fraction": audit["contract_failure_fraction"],
        "rank1_acceptance_fraction": audit["rank1_acceptance_fraction"],
        "rank2_improvement_db": _percentiles(
            [row["rank2_improvement_db"] for row in records]
        ),
        "minimum_jacobian_singular_value": _percentiles(
            [row["minimum_jacobian_singular_value"] for row in records]
        ),
        "jacobian_condition": _percentiles(
            [row["jacobian_condition"] for row in records]
        ),
        "operator_target_written": not limited,
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "RECORDS.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
