"""Run the V2.5 controller-integration and full-pattern release workflow.

The default invocation is a dry run.  Any command that can overwrite frozen
full-pattern results is executed only after ``--confirm-overwrite`` is passed.
The inherited V2.4 controller-training snapshot is validated, never retrained,
by this pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent


def command_plan() -> list[dict[str, Any]]:
    python = sys.executable
    common = [python, "-X", "utf8", "-B"]
    return [
        {
            "step": "validate_inherited_controller_training",
            "writes_frozen_results": False,
            "command": [
                *common,
                str(CODE_DIR / "validate_v24_controller_artifacts.py"),
                "--version",
                "v2.4",
            ],
        },
        {
            "step": "rebuild_full_pattern_numerical_results",
            "writes_frozen_results": True,
            "command": [
                *common,
                str(CODE_DIR / "run_v25_full_pattern_study.py"),
            ],
        },
        {
            "step": "export_full_pattern_input_output_bundle",
            "writes_frozen_results": True,
            "command": [
                *common,
                str(CODE_DIR / "export_v25_comparison_bundle.py"),
            ],
        },
        {
            "step": "run_complete_unit_tests_22_controller_plus_13_full_pattern",
            "writes_frozen_results": False,
            "command": [
                *common,
                "-m",
                "unittest",
                "-v",
                "test_beamforming_common.py",
                "test_beamforming_pre_fpga.py",
                "test_experiment_common.py",
                "test_full_pattern_v25.py",
            ],
        },
        {
            "step": "verify_exported_bundle_without_writes",
            "writes_frozen_results": False,
            "command": [
                *common,
                str(CODE_DIR / "export_v25_comparison_bundle.py"),
                "--check",
            ],
        },
        {
            "step": "validate_complete_v25_release",
            "writes_frozen_results": False,
            "command": [
                *common,
                str(CODE_DIR / "validate_v25_artifacts.py"),
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-overwrite",
        action="store_true",
        help="explicitly allow rebuilding output/full_pattern and its comparison bundle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ordered release commands without executing them",
    )
    args = parser.parse_args()
    plan = command_plan()
    if args.dry_run or not args.confirm_overwrite:
        print(
            json.dumps(
                {
                    "version": "v2.5",
                    "dry_run": True,
                    "controller_policy": "validate frozen V2.4 snapshot; do not retrain",
                    "requires_confirm_overwrite": True,
                    "steps": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    for item in plan:
        subprocess.run(item["command"], cwd=CODE_DIR, check=True)


if __name__ == "__main__":
    main()
