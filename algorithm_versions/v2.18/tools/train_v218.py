"""Run a gated V2.18 training stage; confirmation data is not accepted."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.canonical import load_prepared
from v218.experiment import BASELINE_ARMS, OPAUX_ARM, SEEDS
from v218.training import train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "opaux"), required=True)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--arm", choices=(*BASELINE_ARMS, OPAUX_ARM))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "baseline":
        arms = BASELINE_ARMS if args.arm is None else (args.arm,)
        if any(arm == OPAUX_ARM for arm in arms):
            raise SystemExit("opaux cannot run during baseline")
    else:
        receipt = VERSION / "artifacts/audit_4x4_20261014/BASELINE_AUDIT.json"
        if not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8")).get("status") != "BASELINE_REPRODUCTION_PASS":
            raise SystemExit("opaux is locked until BASELINE_REPRODUCTION_PASS")
        arms = (OPAUX_ARM,)
        if args.arm not in (None, OPAUX_ARM):
            raise SystemExit("only the preregistered opaux arm may run in opaux stage")
    seeds = SEEDS if args.seed is None else (args.seed,)
    data = load_prepared(VERSION / "artifacts/prepared_4x4_20261014")
    training = VERSION / "artifacts/training_4x4_20261014"
    for arm in arms:
        for seed in seeds:
            train_run(arm, seed, data, training / f"{arm}_seed{seed}", device=args.device)


if __name__ == "__main__":
    main()
