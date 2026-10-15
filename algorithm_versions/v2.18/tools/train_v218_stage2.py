"""Train only the two preregistered phase-aligned auxiliary Stage-2 arms."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.canonical import load_prepared
from v218.experiment import PHASE_AUX_ARMS, SEEDS
from v218.training import train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--arm", choices=PHASE_AUX_ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    receipt_path = VERSION / "artifacts" / "stage2_attribution_4x4_20261014" / "MIGRATION_ATTRIBUTION.json"
    if not receipt_path.exists():
        raise SystemExit("Stage-2 training is locked until migration attribution completes")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "MIGRATION_ATTRIBUTION_PASS" or receipt.get("phase_aux_training_authorized") is not True:
        raise SystemExit("Stage-2 training is locked by MIGRATION_ATTRIBUTION_FAIL")
    data = load_prepared(VERSION / "artifacts" / "prepared_4x4_20261014")
    output = VERSION / "artifacts" / "stage2_training_4x4_20261014"
    arms = PHASE_AUX_ARMS if args.arm is None else (args.arm,)
    seeds = SEEDS if args.seed is None else (args.seed,)
    for arm in arms:
        for seed in seeds:
            result = train_run(
                arm, seed, data, output / f"{arm}_seed{seed}",
                device=args.device, verbose=False,
            )
            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "best_validation_amplitude_linear_nmse_db": result["best_validation_amplitude_linear_nmse_db"],
            }), flush=True)


if __name__ == "__main__":
    main()
