"""Prepare, train, freeze, and evaluate the approved fixed-structure V2.9 campaign."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..research.protocol import assert_protocol, digest, prepare, read_json


def verify_source(root: Path) -> None:
    code = Path(__file__).resolve().parents[1]
    protocol = assert_protocol(root)
    actual = {str(p.relative_to(code)): digest(p) for p in code.rglob("*.py")}
    if actual != protocol["source_code_hashes"]:
        raise ValueError("source changed since protocol freeze; refusing experimental drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hfss-folder", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.hfss_folder is None:
            parser.error("prepare requires --hfss-folder")
        prepare(args.hfss_folder, args.root, Path(__file__).resolve().parents[1])
    elif args.action == "status":
        for name in ("progress.json", "screen_selection.json", "recipe_selection.json",
                     "frozen_selection.json", "dual_blind_report.json"):
            if (args.root/name).exists():
                print(name, read_json(args.root/name))
    else:
        verify_source(args.root)
        if args.action == "train":
            from ..research.train import campaign
            campaign(args.root, args.device)
        else:
            from ..research.evaluate import evaluate_blinds
            evaluate_blinds(args.root, args.device)


if __name__ == "__main__":
    main()
