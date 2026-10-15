from __future__ import annotations

from pathlib import Path
import runpy
import shutil


def main() -> None:
    tests = Path(__file__).resolve().parents[1] / "tests" / "test_cross_array.py"
    namespace = runpy.run_path(str(tests))
    functions = sorted((name, value) for name, value in namespace.items() if name.startswith("test_") and callable(value))
    temporary_root = Path(__file__).resolve().parents[1] / ".test_tmp"
    temporary_root.mkdir(exist_ok=True)
    for index, (name, function) in enumerate(functions):
        if "tmp_path" in function.__annotations__:
            directory = temporary_root / f"{index:02d}_{name}"
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir()
            try:
                function(directory)
            finally:
                shutil.rmtree(directory)
        else:
            function()
        print(f"PASS {name}")
    temporary_root.rmdir()
    print(f"V2.17 tests: {len(functions)}/{len(functions)} passed")


if __name__ == "__main__":
    main()
