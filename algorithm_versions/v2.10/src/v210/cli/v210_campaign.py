"""Prepare, test, train, freeze, and evaluate the V2.10 controlled campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..research.v210_experiment import evaluate, prepare, selftest, train_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "selftest", "train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    if arguments.action in {"prepare", "all"}:
        if arguments.source_root is None:
            parser.error("prepare/all requires --source-root")
        prepare(arguments.root, arguments.source_root)
    if arguments.action in {"selftest", "all"}:
        result = selftest(arguments.root, arguments.device)
        if not result["passed"]:
            raise SystemExit("selftest failed")
        print("SELFTEST PASSED", result, flush=True)
    if arguments.action in {"train", "all"}:
        print("FROZEN", train_campaign(arguments.root, arguments.device), flush=True)
    if arguments.action in {"evaluate", "all"}:
        print("EVALUATED", evaluate(arguments.root, arguments.device), flush=True)


if __name__ == "__main__":
    main()

