"""Run V2.18 tests with all required source roots."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

if __name__ == "__main__":
    namespace = runpy.run_path(str(VERSION / "tests/test_v218_contract.py"))
    tests = sorted(
        (name, function) for name, function in namespace.items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"V2.18 tests: {len(tests)}/{len(tests)} passed")
