"""Open the frozen test split once and report the mandatory A/B/C/D matrix."""

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
from v212.evaluation import amplitude_metrics, cross_generalization_report  # noqa: E402
from v212.operator import CoherentRank1Operator, operator_from_checkpoint  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402


@torch.no_grad()
def _predict(
    model: CoherentRank1Operator,
    direction: torch.Tensor,
    commands: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(direction), chunk_size):
        parts.append(
            model.forward_command_grid(commands, direction[start : start + chunk_size]).T.cpu()
        )
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--open-test", action="store_true", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("test report already exists; refusing to open and overwrite it")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "v212-rank1-checkpoint-1":
        raise ValueError("unsupported checkpoint schema")
    recipe = checkpoint["recipe"]
    transform = None
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    if transform_path.exists():
        transform = FiltrationSSHTransform.load(
            transform_path
        )
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    observation = ReceiverObservation(**checkpoint["observation"])
    all_direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    checkpoint_q_train = torch.as_tensor(
        checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
    )
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=all_direction[checkpoint_q_train],
        transform=transform,
        observation=observation,
    ).to(args.device)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32, device=args.device)
    target = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    prediction = _predict(
        model,
        direction,
        codebook.weights.to(args.device),
        args.chunk_size,
    )
    matrix = cross_generalization_report(
        prediction,
        target,
        train_direction=torch.as_tensor(split.query_train),
        test_direction=torch.as_tensor(split.query_test),
        train_command=codebook.indices("train"),
        test_command=codebook.indices("test"),
    )
    q_validation = torch.as_tensor(split.query_validation, dtype=torch.long)
    w_validation = codebook.indices("validation")
    validation_d = amplitude_metrics(
        prediction.index_select(0, q_validation).index_select(1, w_validation),
        target.index_select(0, q_validation).index_select(1, w_validation),
    )
    payload = {
        "schema": "v212-rank1-final-cross-generalization-1",
        "recipe": recipe,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_split_opened": True,
        "validation_D": validation_d,
        "cross_generalization": matrix,
        "primary_result": matrix["D_unseen_q_unseen_w"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
