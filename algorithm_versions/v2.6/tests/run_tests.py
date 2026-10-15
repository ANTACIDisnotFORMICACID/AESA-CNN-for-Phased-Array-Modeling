"""Dependency-free test runner used because the frozen project venv has no pytest."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time
import traceback


TEST_ROOT = Path(__file__).resolve().parent
CODE_ROOT = TEST_ROOT.parent / "code"
for path in (TEST_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    modules = sorted(path.stem for path in TEST_ROOT.glob("test_*.py"))
    passed = 0
    passed_labels = []
    failures = []
    started = time.perf_counter()
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_") or not callable(getattr(module, name)):
                continue
            label = f"{module_name}.{name}"
            try:
                getattr(module, name)()
                print(f"PASS {label}")
                passed += 1
                passed_labels.append(label)
            except Exception:
                failures.append((label, traceback.format_exc()))
                print(f"FAIL {label}")
    elapsed = time.perf_counter() - started
    print(f"\n{passed} passed, {len(failures)} failed in {elapsed:.2f}s")
    for label, trace in failures:
        print(f"\n--- {label} ---\n{trace}")
    if "--write-report" in sys.argv:
        report_path = TEST_ROOT.parent / "output" / "reports" / "test_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "status": "pass" if not failures else "fail",
                    "passed": passed,
                    "failed": len(failures),
                    "passed_tests": passed_labels,
                    "failures": [{"test": label, "traceback": trace} for label, trace in failures],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
