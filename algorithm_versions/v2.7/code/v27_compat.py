"""Explicit access to the frozen V2.6 implementation without modifying it."""

from __future__ import annotations

from pathlib import Path
import sys


V27_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_VERSIONS = V27_ROOT.parent
REPO_ROOT = ALGORITHM_VERSIONS.parent
V26_ROOT = ALGORITHM_VERSIONS / "v2.6"
V26_CODE = V26_ROOT / "code"
V25_ROOT = ALGORITHM_VERSIONS / "v2.5"
V25_CODE = V25_ROOT / "code"


def enable_v26() -> None:
    path = str(V26_CODE)
    if path not in sys.path:
        sys.path.insert(0, path)

def enable_v25() -> None:
    path = str(V25_CODE)
    if path not in sys.path:
        sys.path.insert(0, path)
