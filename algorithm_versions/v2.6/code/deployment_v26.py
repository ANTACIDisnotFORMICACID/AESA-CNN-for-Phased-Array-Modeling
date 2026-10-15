"""DPU-core audit, deployment contract, and conditional Vitis AI PTQ helpers."""

from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.fx import symbolic_trace

from controller_v26 import VitisBeamformingCNNV26


V26_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = V26_ROOT / "output"


def load_checkpoint(path: Path | None) -> VitisBeamformingCNNV26:
    model = VitisBeamformingCNNV26()
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def audit_core(model: nn.Module) -> dict:
    graph = symbolic_trace(model)
    modules = dict(graph.named_modules())
    rows = []
    unsupported = []
    allowed_modules = (nn.Conv2d, nn.ReLU6)
    allowed_functions = (operator.add,)
    for node in graph.graph.nodes:
        item = {"name": node.name, "op": node.op, "target": str(node.target)}
        if node.op == "call_module":
            module = modules[str(node.target)]
            item["module_type"] = type(module).__name__
            if not isinstance(module, allowed_modules):
                unsupported.append(item)
        elif node.op == "call_function" and node.target not in allowed_functions:
            unsupported.append(item)
        elif node.op not in ("placeholder", "output", "call_module", "call_function"):
            unsupported.append(item)
        rows.append(item)
    return {
        "status": "pass" if not unsupported else "fail",
        "scope": "static PyTorch core audit only; vai_c_xir is authoritative for DPU partitioning",
        "input": {"name": "deployment_x", "shape": [1, 15, 1, 4], "dtype": "float32"},
        "output": {"name": "delta_iq", "shape": [1, 2, 1, 4], "dtype": "float32"},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "nodes": rows,
        "unsupported_nodes": unsupported,
    }


def deployment_contract() -> dict:
    return {
        "schema_version": 1,
        "model_version": "2.6.0",
        "dpu_core": {
            "input": {"name": "deployment_x", "layout": "NCHW", "shape": [1, 15, 1, 4], "dtype": "float32/int8"},
            "output": {"name": "delta_iq", "layout": "NCHW", "shape": [1, 2, 1, 4], "dtype": "float32/int8"},
            "channel_order": [
                "initial_I", "initial_Q", "calibration_I", "calibration_Q", "health_mask",
                "target_ETheta_I", "target_ETheta_Q", "target_EPhi_I", "target_EPhi_Q",
                "port_y_normalized", "target_ux", "target_uy", "target_uz",
                "target_element_envelope", "calibration_confidence",
            ],
        },
        "ps_before_dpu": [
            "phase-invariant calibration", "HFSS target-response interpolation", "analytic initial weights",
            "coordinate and direction channel construction",
        ],
        "ps_after_dpu": [
            "complex residual multiplication", "amplitude clipping", "health mask", "unit-disk projection",
            "nearest INT8 integer-disk quantization", "feasibility reason reporting",
        ],
        "reason_bits": {
            "OUT_OF_DOMAIN": 1, "AMBIGUOUS_LOBE": 2, "POINTING_UNACHIEVABLE": 4,
            "CALIBRATION_UNCERTAIN": 8, "INSUFFICIENT_ELEMENTS": 16,
        },
        "not_in_dpu": ["HFSS full-sphere observer", "physical oracle", "measurement acquisition"],
    }


def write_audit(checkpoint: Path | None = None) -> dict:
    model = load_checkpoint(checkpoint)
    report = audit_core(model)
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    report["docker_cli"] = shutil.which("docker")
    report["pytorch_nndct_available_on_host"] = False
    try:
        import pytorch_nndct  # type: ignore  # noqa: F401
        report["pytorch_nndct_available_on_host"] = True
    except ImportError:
        pass
    report["ptq_status"] = "ready" if report["pytorch_nndct_available_on_host"] else "blocked_missing_existing_runtime"
    destination = OUTPUT_ROOT / "deployment"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "core_operator_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (destination / "deployment_contract.json").write_text(json.dumps(deployment_contract(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    report = write_audit(args.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

