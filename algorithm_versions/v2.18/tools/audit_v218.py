"""Audit V2.18 validation gates without any confirmatory-data interface."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.audit import audit_run, baseline_gate, optimization_gate
from v218.canonical import atomic_json, load_prepared
from v218.experiment import BASELINE_ARMS, OPAUX_ARM, SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "opaux"), required=True)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    data = load_prepared(VERSION / "artifacts/prepared_4x4_20261014")
    training = VERSION / "artifacts/training_4x4_20261014"
    output = VERSION / "artifacts/audit_4x4_20261014"
    arms = BASELINE_ARMS if args.stage == "baseline" else ("dcar_gated_pm_canonical", OPAUX_ARM)
    reports = {}
    for arm in arms:
        reports[arm] = {}
        for seed in SEEDS:
            run = training / f"{arm}_seed{seed}"
            if not (run / "checkpoint.pt").exists():
                raise SystemExit(f"missing completed run: {run.name}")
            reports[arm][seed] = audit_run(run, data, args.device)
    receipt = baseline_gate(reports) if args.stage == "baseline" else optimization_gate(reports)
    output.mkdir(parents=True, exist_ok=True)
    name = "BASELINE_AUDIT.json" if args.stage == "baseline" else "OPTIMIZATION_AUDIT.json"
    atomic_json(output / name, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
