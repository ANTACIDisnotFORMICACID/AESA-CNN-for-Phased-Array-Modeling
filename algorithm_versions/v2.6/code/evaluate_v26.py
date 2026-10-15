"""Exact full-sphere release evaluation for a trained V2.6 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from controller_v26 import VitisBeamformingCNNV26
from hfss_physics_v26 import CANONICAL_NPZ, HfssPatternData, quantize_int8_disk
from train_v26 import ExactHfssLoss, LossWeights, TeacherShardDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
V26_ROOT = Path(__file__).resolve().parents[1]


def aggregate(checkpoint: Path, shard_paths: list[Path]) -> dict:
    data = HfssPatternData.load()
    dataset = TeacherShardDataset(shard_paths, data)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = VitisBeamformingCNNV26()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    criterion = ExactHfssLoss(data, LossWeights(1.0, 0.0, 0.0, 0.0))
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        x = sample["deployment_x"].unsqueeze(0)
        with torch.no_grad():
            command = criterion.command_from_delta(x, model(x))[0].numpy()
        codes, integer = quantize_int8_disk(command)
        calibration = sample["calibration"].numpy()
        health = sample["health_mask"].numpy()
        theta, phi = map(float, sample["target_hfss_deg"].numpy())
        fp32 = data.metrics(command * calibration * health, theta, phi)
        int8 = data.metrics(integer * calibration * health, theta, phi)
        teacher_command = sample["teacher_weights"].numpy()
        teacher = data.metrics(teacher_command * calibration * health, theta, phi)
        rows.append((fp32, int8, teacher, codes))
    target_regret = np.asarray([10 * np.log10((row[2].target_power + 1e-30) / (row[0].target_power + 1e-30)) for row in rows])
    margin_loss = np.asarray([row[2].strong_peak_margin_db - row[0].strong_peak_margin_db for row in rows])
    oracle_feasible = np.asarray([row[2].feasible for row in rows], dtype=bool)
    predicted_feasible = np.asarray([row[0].feasible for row in rows], dtype=bool)
    ptq_target_delta = np.asarray([10 * np.log10((row[0].target_power + 1e-30) / (row[1].target_power + 1e-30)) for row in rows])
    pointing_delta = np.asarray([abs(row[1].pointing_error_deg - row[0].pointing_error_deg) for row in rows])
    int8_margin_loss = np.asarray([row[0].strong_peak_margin_db - row[1].strong_peak_margin_db for row in rows])
    radius_ok = all(np.all(code[..., 0].astype(np.int16) ** 2 + code[..., 1].astype(np.int16) ** 2 <= 127**2) for *_, code in rows)
    feasible_subset = {
        "samples": int(np.sum(oracle_feasible)),
        "median_target_gain_regret_db": float(np.median(target_regret[oracle_feasible])) if np.any(oracle_feasible) else None,
        "pointing_error_p95_deg": float(np.percentile([rows[i][0].pointing_error_deg for i in np.flatnonzero(oracle_feasible)], 95)) if np.any(oracle_feasible) else None,
        "margin_p05_db": float(np.percentile([rows[i][0].strong_peak_margin_db for i in np.flatnonzero(oracle_feasible)], 5)) if np.any(oracle_feasible) else None,
    }
    fp32_pass = (
        int(np.sum(predicted_feasible & ~oracle_feasible)) == 0
        and float(np.median(target_regret)) <= 0.5
        and float(np.percentile(margin_loss, 95)) <= 0.5
        and radius_ok
    )
    int8_pass = (
        float(np.max(ptq_target_delta)) <= 0.1
        and float(np.max(pointing_delta)) <= 1.0
        and float(np.max(int8_margin_loss)) <= 0.25
        and radius_ok
    )
    old_path = CANONICAL_NPZ.parent / "v25_hfss_evaluation.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))["methods"]
        old_summary = {key: {k: v for k, v in value.items() if k != "sample_metrics"} for key, value in old.items()}
    else:
        old_summary = {"status": "unavailable", "reason": "optional local V2.5 HFSS evaluation artifact is absent"}
    return {
        "schema_version": 1,
        "samples": len(rows),
        "checkpoint": str(checkpoint),
        "fp32": {
            "pass": fp32_pass,
            "false_feasible": int(np.sum(predicted_feasible & ~oracle_feasible)),
            "median_target_power_regret_db": float(np.median(target_regret)),
            "margin_loss_p95_db": float(np.percentile(margin_loss, 95)),
            "all_finite": bool(np.isfinite(target_regret).all() and np.isfinite(margin_loss).all()),
            "integer_disk_ok": radius_ok,
            "oracle_feasible_subset": feasible_subset,
        },
        "int8_output_codes": {
            "pass": int8_pass,
            "max_target_gain_degradation_db": float(np.max(ptq_target_delta)),
            "max_pointing_change_deg": float(np.max(pointing_delta)),
            "max_margin_degradation_db": float(np.max(int8_margin_loss)),
            "note": "final complex output code quantization; not NNDCT internal W8A8 PTQ",
        },
        "historical_hfss_comparison": old_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-directory", type=Path, default=V26_ROOT / "output" / "teacher")
    parser.add_argument("--output", type=Path, default=V26_ROOT / "output" / "reports" / "fp32_int8_evaluation.json")
    args = parser.parse_args()
    shards = sorted(args.test_directory.glob("test_*.npz"))
    if not shards:
        raise SystemExit("no test teacher shards; complete the oracle and test-state generation first")
    report = aggregate(args.checkpoint, shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
