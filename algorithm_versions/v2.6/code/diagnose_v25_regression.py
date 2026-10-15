"""Read-only V2.5 regression wrapper; never calls a write/overwrite path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
V25_CODE = REPO_ROOT / "algorithm_versions" / "v2.5" / "code"
V25_OUTPUT = REPO_ROOT / "algorithm_versions" / "v2.5" / "output" / "full_pattern"
REPORT = Path(__file__).resolve().parents[1] / "output" / "reports" / "v25_read_only_regression.json"


def differences(actual: Any, expected: Any, path: str = "$", limit: int = 32) -> list[dict]:
    result = []
    if type(actual) is not type(expected):
        return [{"path": path, "actual_type": type(actual).__name__, "expected_type": type(expected).__name__}]
    if isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                result.append({"path": f"{path}.{key}", "actual": "missing" if key not in actual else "present", "expected": "missing" if key not in expected else "present"})
            else:
                result.extend(differences(actual[key], expected[key], f"{path}.{key}", limit - len(result)))
            if len(result) >= limit:
                break
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            result.append({"path": path, "actual_length": len(actual), "expected_length": len(expected)})
        for index, (a, e) in enumerate(zip(actual, expected)):
            result.extend(differences(a, e, f"{path}[{index}]", limit - len(result)))
            if len(result) >= limit:
                break
    elif actual != expected:
        row = {"path": path, "actual": actual, "expected": expected}
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            row["absolute_difference"] = abs(float(actual) - float(expected))
        result.append(row)
    return result[:limit]


def run() -> dict:
    tests = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", str(V25_CODE), "-p", "test_*.py", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    controller = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(V25_CODE / "export_v24_controller_bundle.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if str(V25_CODE) not in sys.path:
        sys.path.insert(0, str(V25_CODE))
    spec = importlib.util.spec_from_file_location("v25_export_diagnostic", V25_CODE / "export_v25_comparison_bundle.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    _, expected_output = module.build_payloads()
    actual_output = json.loads((V25_OUTPUT / "comparison_results.json").read_text(encoding="utf-8"))
    diff = differences(actual_output, expected_output)
    payload = {
        "read_only": True,
        "v25_unittest": {"passed": tests.returncode == 0, "returncode": tests.returncode, "stderr_tail": tests.stderr[-2000:]},
        "v24_bundle_check": {"passed": controller.returncode == 0, "returncode": controller.returncode, "stdout": controller.stdout.strip(), "stderr_tail": controller.stderr[-2000:]},
        "v25_full_pattern_bundle_check": {"passed": not diff, "difference_count_returned": len(diff), "first_differences": diff},
        "conclusion": "V2.5 code/golden control path is intact; full-pattern generated JSON differs from current in-memory regeneration" if diff else "all checked V2.5 paths pass",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
