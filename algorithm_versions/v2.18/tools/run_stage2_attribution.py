"""Run the V2.18 Stage-2 P0-P3 attribution with no blind-test interface."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.stage2 import run_attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_attribution(
        version_root=VERSION,
        output_dir=VERSION / "artifacts" / "stage2_attribution_4x4_20261014",
        device=args.device,
    )
    print(json.dumps({
        "status": result["status"],
        "equivalence": result["equivalence"],
        "attribution_signals": result.get("attribution_signals"),
        "phase_aux_training_authorized": result.get("phase_aux_training_authorized", False),
        "confirmation_status": result["confirmation_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
