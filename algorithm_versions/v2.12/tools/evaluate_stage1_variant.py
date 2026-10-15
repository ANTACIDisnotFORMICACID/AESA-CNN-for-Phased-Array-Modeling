"""Open one Stage-1 checkpoint's frozen test split and report A/B/C/D metrics."""

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
from v212.evaluation import amplitude_metrics, cross_generalization_report  # noqa: E402
from v212.stage1 import Stage1EncodingConfig, build_stage1_model  # noqa: E402


@torch.no_grad()
def _predict(model, command_iq, direction, chunk_size: int) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(direction), chunk_size):
        parts.append(
            model.forward_command_grid(command_iq, direction[start : start + chunk_size]).T.cpu()
        )
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--open-test", action="store_true", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("test report already exists; refusing to overwrite it")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "v212-stage1-checkpoint-1":
        raise ValueError("unsupported Stage-1 checkpoint schema")
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    encoding = Stage1EncodingConfig(**checkpoint["encoding_config"])
    transform = None
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    if transform_path.exists():
        transform = FiltrationSSHTransform.load(transform_path)
    training_direction = torch.as_tensor(
        dataset.direction[split.query_train], dtype=torch.float64
    )
    model, _ = build_stage1_model(
        checkpoint["base_model_config"],
        encoding,
        training_direction,
        transform=transform,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(args.device)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32, device=args.device)
    commands = torch.as_tensor(dataset.command_iq, dtype=torch.float32, device=args.device)
    target = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    prediction = _predict(model, commands, direction, args.chunk_size)
    train_q = torch.as_tensor(split.query_train)
    validation_q = torch.as_tensor(split.query_validation)
    test_q = torch.as_tensor(split.query_test)
    train_w = torch.as_tensor(split.command_train)
    validation_w = torch.as_tensor(split.command_validation)
    test_w = torch.as_tensor(split.command_test)
    report = cross_generalization_report(
        prediction,
        target,
        train_direction=train_q,
        test_direction=test_q,
        train_command=train_w,
        test_command=test_w,
    )
    validation_d = amplitude_metrics(
        prediction.index_select(0, validation_q).index_select(1, validation_w),
        target.index_select(0, validation_q).index_select(1, validation_w),
    )
    payload = {
        "schema": "v212-stage1-final-cross-generalization-1",
        "recipe": checkpoint["recipe"],
        "checkpoint": str(args.checkpoint.resolve()),
        "test_split_opened": True,
        "validation_D": validation_d,
        "cross_generalization": report,
        "primary_result": report["D_unseen_q_unseen_w"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
