"""Resumable command-line driver for V2.6 physical validation and oracle shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from calibration_v26 import PhaseInvariantCalibrator, build_d_optimal_design, simulate_measurements
from controller_v26 import BeamformingControllerV26, VitisBeamformingCNNV26
from hfss_physics_v26 import CANONICAL_NPZ, REPO_ROOT, HfssPatternData
from oracle_v26 import OracleConfig, PhysicalOracle
from dataset_v26 import write_profile_manifest
from deployment_v26 import write_audit


V26_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = V26_ROOT / "output"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def smoke() -> None:
    data = HfssPatternData.load()
    design = build_d_optimal_design(data)
    write_json(OUTPUT / "calibration_design.json", design.serializable())
    truth = np.array([1.0, 0.94 * np.exp(0.12j), 1.06 * np.exp(-0.17j), 0.90 * np.exp(0.05j)])
    measurement = simulate_measurements(design, truth, snr_db=40.0, global_phase_rad=2.1)
    calibration = PhaseInvariantCalibrator(design).calibrate(measurement)
    controller = BeamformingControllerV26(data, calibration_design=design)
    result = controller.beamform((113.0, -8.0), calibration.calibration_iq[0], calibration.confidence[0])
    phase_oracle = PhysicalOracle(data, OracleConfig(starts=8, max_steps=20)).target_power_phase_oracle(113.0, -8.0)
    report = {
        "status": "passed",
        "scope": "FP32 physical-interface smoke; not trained, PTQ, DPU, FPGA, or OTA",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "canonical_npz": str(CANONICAL_NPZ),
        },
        "calibration": {
            "design_condition_number": design.condition_number,
            "residual_nmse_db": calibration.residual_nmse_db.tolist(),
            "confidence": calibration.confidence.tolist(),
        },
        "controller": result.serializable(),
        "phase_only_oracle": data.metrics(phase_oracle, 113.0, -8.0).__dict__,
    }
    write_json(OUTPUT / "smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def oracle_shard(shard_index: int, shard_count: int, starts: int, steps: int) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    data = HfssPatternData.load()
    oracle = PhysicalOracle(data, OracleConfig(starts=starts, max_steps=steps))
    theta = np.linspace(60.0, 120.0, 64)
    phi = np.linspace(-30.0, 30.0, 64)
    targets = [(float(t), float(p)) for t in theta for p in phi]
    chosen = [item for index, item in enumerate(targets) if index % shard_count == shard_index]
    path = OUTPUT / "oracle_shards" / f"oracle_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed.add((item["target_theta_deg"], item["target_phi_deg"]))
    with path.open("a", encoding="utf-8") as stream:
        for index, target in enumerate(chosen, start=1):
            if target in completed:
                continue
            result = oracle.solve(*target)
            stream.write(json.dumps(result.serializable(), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(f"[{index}/{len(chosen)}] theta={target[0]:.3f} phi={target[1]:.3f} continuous_margin={result.continuous_metrics.strong_peak_margin_db:.3f} dB")


def oracle_spotcheck(starts: int, steps: int) -> None:
    data = HfssPatternData.load()
    oracle = PhysicalOracle(data, OracleConfig(starts=starts, max_steps=steps))
    targets = ((113.0, -8.0), (90.0, 0.0), (75.0, 20.0))
    rows = []
    for target in targets:
        result = oracle.solve(*target)
        rows.append(result.serializable())
        print(
            f"theta={target[0]:.1f} phi={target[1]:.1f}: "
            f"phase={result.phase_only_metrics.strong_peak_margin_db:.4f} dB, "
            f"continuous={result.continuous_metrics.strong_peak_margin_db:.4f} dB, "
            f"int8={result.int8_metrics.strong_peak_margin_db:.4f} dB"
        )
    report = {
        "scope": "three-direction production-setting spot check; not the complete 64x64 map",
        "starts": starts,
        "max_steps": steps,
        "acceptance_margin_db": 3.0,
        "all_continuous_feasible": all(bool(row["continuous_metrics"]["feasible"]) for row in rows),
        "results": rows,
    }
    write_json(OUTPUT / "reports" / "oracle_spotcheck.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "profiles", "audit", "spotcheck", "oracle"), default="smoke")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "profiles":
        payload = write_profile_manifest()
        print(json.dumps({key: len(value) for key, value in payload["profiles"].items()}, indent=2))
    elif args.stage == "audit":
        print(json.dumps(write_audit(), indent=2, sort_keys=True))
    elif args.stage == "spotcheck":
        oracle_spotcheck(args.starts, args.steps)
    else:
        oracle_shard(args.shard_index, args.shard_count, args.starts, args.steps)


if __name__ == "__main__":
    main()
