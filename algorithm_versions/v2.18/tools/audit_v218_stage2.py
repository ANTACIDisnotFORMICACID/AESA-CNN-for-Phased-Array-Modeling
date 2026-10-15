"""Audit all four V2.18 Stage-2 arms without a blind-test interface."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.canonical import load_prepared
from v218.stage2_audit import run_final_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    data = load_prepared(VERSION / "artifacts" / "prepared_4x4_20261014")
    result = run_final_audit(
        version_root=VERSION,
        data=data,
        output_dir=VERSION / "artifacts" / "stage2_audit_4x4_20261014",
        device=args.device,
    )
    print(json.dumps({
        "phase_aux_experiment_status": result["phase_aux_experiment_status"],
        "engineering_candidate": result["engineering_candidate"],
        "budget_sensitive_arms": result["budget_sensitive_arms"],
        "confirmation_status": result["confirmation_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
