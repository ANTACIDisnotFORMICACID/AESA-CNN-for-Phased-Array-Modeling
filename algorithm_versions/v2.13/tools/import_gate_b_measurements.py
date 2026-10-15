"""Import a Gate-B long CSV into the canonical no-pickle measurement package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))

from v212.codebook import MeasurementCodebook  # noqa: E402
from v213.gate_b import import_long_csv, validate_gate_b_package  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    codebook = MeasurementCodebook.load(args.codebook)
    package = import_long_csv(args.csv, args.manifest, args.output_dir, codebook)
    report = validate_gate_b_package(package, codebook)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
