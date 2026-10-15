"""Final machine-readable acceptance audit for a V2.7 release."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from report_v27 import training_gate
from v27_compat import V27_ROOT


def run() -> dict:
    checks: dict[str, dict] = {}
    oracle_path = V27_ROOT / "output" / "maps" / "nominal_oracle_maps.npz"
    if oracle_path.is_file():
        with np.load(oracle_path, allow_pickle=False) as archive:
            indices = archive["indices"]
            metrics = archive["metrics"]
            codes = archive["int8_codes"].astype(np.int16)
            checks["nominal_oracle"] = {
                "passed": bool(
                    archive["complete"]
                    and indices.shape == (4096,)
                    and len(np.unique(indices)) == 4096
                    and np.isfinite(metrics).all()
                    and np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)
                ),
                "records": int(len(indices)),
                "unique": int(len(np.unique(indices))),
                "all_finite": bool(np.isfinite(metrics).all()),
                "int8_disk_passed": bool(np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)),
            }
    else:
        checks["nominal_oracle"] = {"passed": False, "blocked": "missing nominal_oracle_maps.npz"}

    audit_path = V27_ROOT / "output" / "audit" / "control_cases.npz"
    if audit_path.is_file():
        with np.load(audit_path, allow_pickle=False) as archive:
            metrics = archive["metrics"]
            codes = archive["int8_codes"].astype(np.int16)
            checks["multi_parameter_audit"] = {
                "passed": bool(
                    archive["complete"]
                    and metrics.shape == (83, 6, 256, 9)
                    and np.isfinite(metrics).all()
                    and np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)
                ),
                "cases": int(metrics.shape[0] * metrics.shape[2]),
                "method_variants": int(metrics.shape[1]),
                "all_finite": bool(np.isfinite(metrics).all()),
                "int8_disk_passed": bool(np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)),
            }
    else:
        checks["multi_parameter_audit"] = {"passed": False, "blocked": "missing control_cases.npz"}

    state_path = V27_ROOT / "output" / "audit" / "state_continuous_reference_1024.npz"
    if state_path.is_file():
        with np.load(state_path, allow_pickle=False) as archive:
            metrics = archive["metrics"]
            codes = archive["int8_codes"].astype(np.int16)
            checks["state_continuous_reference"] = {
                "passed": bool(
                    metrics.shape == (1024, 3, 13)
                    and np.isfinite(metrics).all()
                    and np.all(codes[..., 0] ** 2 + codes[..., 1] ** 2 <= 127**2)
                ),
                "cases": int(len(metrics)),
            }
    else:
        checks["state_continuous_reference"] = {"passed": False, "blocked": "missing 1024-case state reference"}

    repeat_path = V27_ROOT / "output" / "reports" / "oracle_repeatability_5pct.json"
    if repeat_path.is_file():
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        checks["oracle_repeatability"] = {"passed": bool(repeat["passed"]), **repeat}
    else:
        checks["oracle_repeatability"] = {"passed": False, "blocked": "missing independent 5 percent repeatability report"}

    regression_path = V27_ROOT / "output" / "reports" / "regression_report.json"
    if regression_path.is_file():
        regression = json.loads(regression_path.read_text(encoding="utf-8"))
        checks["regressions"] = {
            "passed": bool(regression["passed"]),
            "frozen_unchanged": bool(regression["frozen_v2.5_v2.6_hfss_unchanged"]),
        }
    else:
        checks["regressions"] = {"passed": False, "blocked": "missing regression report"}

    gate = training_gate()
    checkpoint = V27_ROOT / "output" / "training" / "v27_fp32.pt"
    checks["training_policy"] = {
        "passed": bool((gate.get("train_v2.7") and checkpoint.is_file()) or (not gate.get("train_v2.7") and not checkpoint.exists())),
        "gate": gate,
        "checkpoint_exists": checkpoint.exists(),
    }
    payload = {
        "schema_version": 1,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "scope": "host HFSS characterization; no Vitis PTQ, FPGA, VART or OTA acceptance",
    }
    destination = V27_ROOT / "output" / "reports" / "acceptance.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    value = run()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    raise SystemExit(0 if value["passed"] else 1)
