"""Run the dependency-light V2.13 validation tests without requiring pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))


def main() -> None:
    failures = 0
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(module, name)()
                print(f"PASS {path.stem}.{name}")
            except Exception:
                failures += 1
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    print(f"TOTAL {total - failures}/{total} PASS")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
