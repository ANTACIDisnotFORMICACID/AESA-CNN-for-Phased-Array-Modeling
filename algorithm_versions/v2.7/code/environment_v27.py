"""Record reproducibility-critical software and fixed HFSS provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess

import numpy as np
import scipy
import torch

from v27_compat import REPO_ROOT, V27_ROOT, enable_v26

enable_v26()
from hfss_physics_v26 import CANONICAL_META, CANONICAL_NPZ, GEOMETRY_CONTRACT  # noqa: E402


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run() -> dict:
    metadata = json.loads(CANONICAL_META.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY_CONTRACT.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    payload = {
        "schema_version": 1,
        "base_head_at_run": head,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "hfss_npz": CANONICAL_NPZ.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "hfss_npz_sha256": _sha256(CANONICAL_NPZ),
        "hfss_source_project_sha256": metadata["metadata"]["source_project_sha256"],
        "hfss_expected_project_sha256": geometry["expected_project_sha256"],
        "hfss_project_hash_contract_passed": metadata["metadata"]["source_project_sha256"] == geometry["expected_project_sha256"],
        "frequency_hz": metadata["frequency_hz"],
        "vitis_ptq_fpga_run": False,
        "scope": "host HFSS saved-field characterization only; no new solve, PTQ, FPGA, VART or OTA",
    }
    destination = V27_ROOT / "output" / "reports" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
