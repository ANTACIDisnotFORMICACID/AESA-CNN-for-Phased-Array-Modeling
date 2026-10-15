"""Run frozen V2.5/V2.6 and current V2.7 tests and record immutable-path status."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from v27_compat import REPO_ROOT, V27_ROOT


def _run(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    return {
        "command": ["<python>", *command[1:]],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def run() -> dict:
    python = sys.executable
    cases = {
        "v2.5_35_tests": _run(
            [python, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "algorithm_versions/v2.5/code", "-p", "test_*.py", "-v"]
        ),
        "v2.6_16_tests": _run([python, "-X", "utf8", "-B", "algorithm_versions/v2.6/tests/run_tests.py"]),
        "v2.7_tests": _run([python, "-X", "utf8", "-B", "algorithm_versions/v2.7/tests/run_tests.py"]),
    }
    frozen = subprocess.run(
        ["git", "diff", "--quiet", "v2.6", "--", "algorithm_versions/v2.5", "algorithm_versions/v2.6"],
        cwd=REPO_ROOT,
    )
    payload = {
        "schema_version": 1,
        "base_commit": "v2.6",
        "frozen_v2.5_v2.6_hfss_unchanged": frozen.returncode == 0,
        "suites": cases,
        "passed": frozen.returncode == 0 and all(case["passed"] for case in cases.values()),
    }
    destination = V27_ROOT / "output" / "reports" / "regression_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"passed": result["passed"], "frozen_unchanged": result["frozen_v2.5_v2.6_hfss_unchanged"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
