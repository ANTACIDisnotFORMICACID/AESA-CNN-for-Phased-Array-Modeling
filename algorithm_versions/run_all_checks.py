"""Run self-contained checks for the versions present in this repository."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "algorithm_versions"
RUNTIME = ROOT / ".test_runtime"


def run(arguments: list[str], python_path: list[Path] | None = None) -> None:
    environment = os.environ.copy()
    environment["TEMP"] = str(RUNTIME)
    environment["TMP"] = str(RUNTIME)
    if python_path:
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_path)
    else:
        environment.pop("PYTHONPATH", None)
    command = [sys.executable, *arguments]
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    RUNTIME.mkdir(exist_ok=True)
    completed = 0
    portfolio_regression = 0

    if (VERSIONS / "v2.5").is_dir():
        print("===== v2.5 =====", flush=True)
        run(["-B", "-m", "unittest", "discover", "-s", "algorithm_versions/v2.5/code", "-p", "test_*.py", "-v"])
        run(["-B", "algorithm_versions/v2.5/code/validate_v25_artifacts.py"])
        completed += 35

    if (VERSIONS / "v2.6" / "tests" / "run_tests.py").is_file():
        print("===== v2.6 =====", flush=True)
        run(["-B", "algorithm_versions/v2.6/tests/run_tests.py"])
        completed += 16

    if (VERSIONS / "v2.7" / "tests" / "run_tests.py").is_file():
        print("===== v2.7 =====", flush=True)
        run(["-B", "algorithm_versions/v2.7/tests/run_tests.py"])
        completed += 16

    for version in ("v2.8", "v2.9", "v2.10", "v2.11"):
        tests = VERSIONS / version / "tests"
        if not tests.is_dir():
            continue
        print(f"===== {version} =====", flush=True)
        source_roots = [VERSIONS / version / "src"]
        if version == "v2.11":
            source_roots.append(VERSIONS / "v2.10" / "src")
        run(["-m", "unittest", "discover", "-s", str(tests.relative_to(ROOT)), "-p", "test_*.py", "-v"], source_roots)
        count = {"v2.8": 23, "v2.9": 42, "v2.10": 47, "v2.11": 7}[version]
        completed += count
        portfolio_regression += count

    for version in ("v2.12", "v2.13", "v2.17"):
        runner = VERSIONS / version / "tools" / "run_tests.py"
        if not runner.is_file():
            continue
        print(f"===== {version} =====", flush=True)
        run([str(runner.relative_to(ROOT))])
        count = {"v2.12": 31, "v2.13": 19, "v2.17": 8}[version]
        completed += count
        portfolio_regression += count

    print(
        f"PASS: {portfolio_regression} portfolio regression cases; "
        f"{completed} self-contained tests in total, plus the V2.5 artifact validation"
    )


if __name__ == "__main__":
    main()
