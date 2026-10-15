"""Wait for the six authorized runs, then execute the validation-only audit."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.canonical import atomic_json, load_prepared
from v218.experiment import PHASE_AUX_ARMS, SEEDS
from v218.stage2_audit import run_final_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    args = parser.parse_args()
    training = VERSION / "artifacts" / "stage2_training_4x4_20261014"
    output = VERSION / "artifacts" / "stage2_audit_4x4_20261014"
    status_path = output / "COORDINATOR_STATUS.json"
    expected = [training / f"{arm}_seed{seed}" / "result.json" for arm in PHASE_AUX_ARMS for seed in SEEDS]
    deadline = time.monotonic() + args.timeout_hours * 3600
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            missing = [path.parent.name for path in expected if not path.exists()]
            atomic_json(status_path, {
                "status": "BLOCKED_TRAINING_TIMEOUT",
                "missing_runs": missing,
                "confirmation_status": "CONFIRMATION_NOT_OPENED",
            })
            raise SystemExit(f"training timeout; missing {missing}")
        time.sleep(30)
    atomic_json(status_path, {
        "status": "TRAINING_COMPLETE_AUDIT_RUNNING",
        "confirmation_status": "CONFIRMATION_NOT_OPENED",
    })
    data = load_prepared(VERSION / "artifacts" / "prepared_4x4_20261014")
    result = run_final_audit(version_root=VERSION, data=data, output_dir=output, device=args.device)
    atomic_json(status_path, {
        "status": "AUDIT_COMPLETE",
        "phase_aux_experiment_status": result["phase_aux_experiment_status"],
        "engineering_candidate_status": result["engineering_candidate_status"],
        "confirmation_status": result["confirmation_status"],
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
