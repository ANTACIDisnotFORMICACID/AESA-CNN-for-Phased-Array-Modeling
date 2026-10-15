"""Run only inside an existing Vitis AI PyTorch environment; performs no installs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from controller_v26 import VitisBeamformingCNNV26


def load_inputs(paths: list[Path], limit: int) -> torch.Tensor:
    values = []
    for path in paths:
        with np.load(path) as data:
            values.append(np.asarray(data["deployment_x"], dtype=np.float32))
    array = np.concatenate(values, axis=0)[:limit]
    if array.shape[1:] != (15, 1, 4):
        raise ValueError(f"invalid deployment input shape {array.shape}")
    return torch.from_numpy(array)


def load_model(checkpoint: Path) -> VitisBeamformingCNNV26:
    payload = torch.load(checkpoint, map_location="cpu")
    model = VitisBeamformingCNNV26()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--arch", type=Path)
    args = parser.parse_args()
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError as error:
        raise SystemExit("pytorch_nndct is absent; stop here without installing or downloading") from error
    inputs = load_inputs(args.input, args.limit)
    model = load_model(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = torch_quantizer(args.mode, model, (inputs[:1],), output_dir=str(args.output_dir))
    quant_model = quantizer.quant_model
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            output = quant_model(inputs[start : start + 32])
            if not torch.isfinite(output).all():
                raise RuntimeError("PTQ output contains NaN or Inf")
        quant_model(inputs[:1])
    if args.mode == "calib":
        quantizer.export_quant_config()
    else:
        quantizer.export_xmodel(output_dir=str(args.output_dir), deploy_check=True)
        if args.arch is not None:
            compiler = shutil.which("vai_c_xir")
            if compiler is None:
                raise SystemExit("vai_c_xir is absent; XModel exported but KV260 compilation is blocked")
            exported = sorted(args.output_dir.glob("*.xmodel"))
            if len(exported) != 1:
                raise RuntimeError(f"expected one exported XModel, found {[str(path) for path in exported]}")
            compiled_directory = args.output_dir / "compiled_kv260"
            subprocess.run(
                [compiler, "-x", str(exported[0]), "-a", str(args.arch), "-o", str(compiled_directory), "-n", "v26_beamformer"],
                check=True,
            )
            compiled = sorted(compiled_directory.rglob("*.xmodel"))
            partition = {"compiled_xmodels": [str(path) for path in compiled], "authoritative_xir_audit": "unavailable"}
            try:
                import xir  # type: ignore
                if len(compiled) != 1:
                    raise RuntimeError("compiler did not produce exactly one XModel")
                graph = xir.Graph.deserialize(str(compiled[0]))
                children = graph.get_root_subgraph().toposort_child_subgraph()
                subgraphs = []
                for child in children:
                    device = child.get_attr("device") if child.has_attr("device") else "UNKNOWN"
                    subgraphs.append({"name": child.get_name(), "device": device, "ops": sorted(op.get_type() for op in child.get_ops())})
                dpu_count = sum(row["device"] == "DPU" for row in subgraphs)
                partition = {
                    "compiled_xmodels": [str(path) for path in compiled],
                    "subgraphs": subgraphs,
                    "dpu_subgraph_count": dpu_count,
                    "single_primary_dpu_subgraph": dpu_count == 1,
                    "authoritative_xir_audit": "pass" if dpu_count == 1 else "fail",
                }
            except ImportError:
                pass
            (args.output_dir / "operator_partition_report.json").write_text(json.dumps(partition, indent=2), encoding="utf-8")
            if partition.get("authoritative_xir_audit") == "fail":
                raise RuntimeError("compiled CNN core did not form one primary DPU subgraph")
    summary = {"mode": args.mode, "samples": len(inputs), "output_dir": str(args.output_dir), "arch": str(args.arch) if args.arch else None}
    (args.output_dir / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
