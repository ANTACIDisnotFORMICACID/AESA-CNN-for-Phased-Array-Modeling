"""Run V2.12 acceptance tests without requiring pytest to be installed."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> None:
    modules = [path.stem for path in sorted((ROOT / "tests").glob("test_*.py"))]
    failures: list[str] = []
    count = 0
    for module_name in modules:
        module = importlib.import_module(module_name)
        tests = [
            value
            for name, value in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and value.__module__ == module.__name__
        ]
        for test in tests:
            count += 1
            try:
                test()
            except Exception:
                failures.append(f"{module_name}.{test.__name__}\n{traceback.format_exc()}")
                print(f"FAIL {module_name}.{test.__name__}")
            else:
                print(f"PASS {module_name}.{test.__name__}")
    if failures:
        print("\n\n".join(failures))
        raise SystemExit(f"{len(failures)} of {count} tests failed")
    print(f"TOTAL {count} PASS")


if __name__ == "__main__":
    main()
