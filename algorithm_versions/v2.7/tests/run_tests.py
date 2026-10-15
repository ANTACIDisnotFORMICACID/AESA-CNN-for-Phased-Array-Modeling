"""Dependency-free V2.7 test runner (pytest is intentionally not required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback


HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
for path in (HERE, CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    tests = []
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if name.startswith("test_") and callable(getattr(module, name)):
                tests.append((f"{path.stem}.{name}", getattr(module, name)))
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"RESULT {len(tests) - len(failures)}/{len(tests)} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
