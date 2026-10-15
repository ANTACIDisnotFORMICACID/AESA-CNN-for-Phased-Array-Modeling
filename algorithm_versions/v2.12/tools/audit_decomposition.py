"""Train-measure latent decomposition audit; reports architecture, not physical truth."""

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

from v210.contracts import QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.audits import fourier_low_order_leakage  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import operator_from_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    all_direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    q_train = torch.as_tensor(
        checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
    )
    transform_path = args.checkpoint.parent / "filtration_transform.npz"
    transform = FiltrationSSHTransform.load(transform_path) if transform_path.exists() else None
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=all_direction[q_train],
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(args.device)
    model.eval()
    residual_parts = []
    degree_parts: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(q_train), args.chunk_size):
            direction = all_direction[q_train[start : start + args.chunk_size]].to(args.device)
            _, terms = model.envelope(QueryBatch(direction[None]))
            residual = terms["fourier_residual_iq"][0]
            residual_parts.append(torch.complex(residual[:, 0], residual[:, 1]).reshape(len(direction), 16).cpu())
            for name, value in terms.items():
                if name.startswith("ssh_l"):
                    degree_parts.setdefault(name, []).append(value[0].cpu())
    residual = torch.cat(residual_parts)
    leakage = None
    if transform is not None:
        phi = transform.evaluate(all_direction[q_train].double())
        leakage = float(fourier_low_order_leakage(phi, residual))
    payload = {
        "schema": "v212s-latent-decomposition-audit-1",
        "recipe": checkpoint["recipe"],
        "measure": "empirical_equal_train_loss",
        "query_count": len(q_train),
        "test_status": "sealed",
        "interpretation": "latent architecture decomposition, not identifiable physical truth",
        "fourier_low_filtration_leakage": leakage,
        "fourier_residual_rms": float(residual.abs().square().mean().sqrt()),
        "ssh_degree_output_rms": {
            name: float(torch.cat(parts).square().mean().sqrt())
            for name, parts in degree_parts.items()
        },
        "input_projection": checkpoint.get("input_projection"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
