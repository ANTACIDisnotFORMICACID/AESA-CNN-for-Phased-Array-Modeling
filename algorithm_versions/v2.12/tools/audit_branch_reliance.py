"""Post-hoc branch reliance and angular-distance validation audit (never retrains)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.evaluation import amplitude_metrics, angular_distance_quartile_report  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import BranchAblation, operator_from_checkpoint  # noqa: E402


@torch.no_grad()
def _predict(model, commands, direction, ablation, chunk_size):
    parts = []
    for start in range(0, len(direction), chunk_size):
        parts.append(model.forward_command_grid(
            commands, direction[start : start + chunk_size], ablation
        ).T.cpu())
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    q_train = torch.as_tensor(
        checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
    )
    q_validation = torch.as_tensor(
        checkpoint.get("validation_query_indices", split.query_validation), dtype=torch.long
    )
    w_validation = codebook.indices("validation")
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    transform = FiltrationSSHTransform.load(transform_path) if transform_path.exists() else None
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=direction[q_train],
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(args.device)
    model.eval()
    commands = codebook.weights[w_validation].to(args.device)
    validation_direction = direction[q_validation].to(args.device)
    target = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)[q_validation][:, w_validation]
    ablations = {"full": BranchAblation(), "fourier_zero": BranchAblation(disable_fourier=True)}
    if model.has_ssh:
        ablations["ssh_zero"] = BranchAblation(disable_ssh=True)
        for degree in range(1, model.ssh_degree + 1):
            ablations[f"ssh_l{degree}_zero"] = BranchAblation(disabled_ssh_degrees=(degree,))
    reports = {}
    predictions = {}
    for name, ablation in ablations.items():
        prediction = _predict(model, commands, validation_direction, ablation, args.chunk_size)
        predictions[name] = prediction
        reports[name] = amplitude_metrics(prediction, target)
    full_nmse = float(reports["full"]["nmse_db"])
    for report in reports.values():
        report["nmse_degradation_vs_full_db"] = float(report["nmse_db"]) - full_nmse
    payload = {
        "schema": "v212-posthoc-branch-reliance-1",
        "recipe": checkpoint["recipe"],
        "test_status": "sealed",
        "interpretation": "post-hoc reliance, not retrained branch necessity",
        "ablations": reports,
        "angular_distance_quartiles": angular_distance_quartile_report(
            predictions["full"], target, direction[q_validation], direction[q_train]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
