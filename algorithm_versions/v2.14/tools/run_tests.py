"""Dependency-light V2.14 test runner."""

from pathlib import Path
import importlib.util
import sys
import traceback


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root.parent / "v2.12" / "src"))
    sys.path.insert(0, str(root.parent / "v2.10" / "src"))
    passed = total = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in sorted(item for item in vars(module) if item.startswith("test_")):
            total += 1
            try:
                getattr(module, name)()
                passed += 1
                print(f"PASS {path.stem}.{name}")
            except Exception:
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    print(f"TOTAL {passed}/{total} PASS")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
