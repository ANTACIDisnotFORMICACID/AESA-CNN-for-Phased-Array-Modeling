"""从冻结 V2.4 golden vectors/metrics 导出可直接对照的 JSON 和图片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
INPUT_DIR = VERSION_DIR / "input" / "controller_training"
OUTPUT_DIR = VERSION_DIR / "output" / "controller_training"
PICTURE_DIR = OUTPUT_DIR / "pictures"


def _json_ready(array: np.ndarray) -> list[Any]:
    return array.tolist()


def build_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    input_payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": metrics["algorithm_version"],
        "purpose": "version-scoped deterministic input/output comparison",
        "source": "frozen golden vectors produced from held-out target-test cases",
        "train_seeds": metrics["train_seeds"],
        "protocol": metrics["protocol"],
        "cases": {},
    }
    output_payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": metrics["algorithm_version"],
        "purpose": "outputs paired one-to-one with input/controller_training/comparison_inputs.json",
        "cases": {},
        "performance_curves": {},
    }
    for topology in ("4x4", "1x4"):
        with np.load(
            OUTPUT_DIR / f"golden_vectors_{topology}.npz", allow_pickle=False
        ) as data:
            input_payload["cases"][topology] = {
                "contract": metrics["contracts"][topology],
                "targets_rad": _json_ready(data["targets_rad"]),
                "initial_weights_iq": _json_ready(data["initial_weights_iq"]),
                "calibration_iq": _json_ready(data["calibration_iq"]),
                "position_errors": _json_ready(data["position_errors"]),
                "state": _json_ready(data["state"]),
            }
            output_payload["cases"][topology] = {
                "checkpoint": metrics["checkpoints"][topology][
                    "single_board_low_update"
                ],
                "output_fp32_iq": _json_ready(data["output_fp32_iq"]),
                "output_int8_iq": _json_ready(data["output_int8_iq"]),
                "quantization_summary": metrics["quantization"][topology],
            }
        curves: dict[str, list[dict[str, Any]]] = {}
        for method in (
            "single_board_scratch",
            "single_board_pretrained",
            "single_board_head_only",
        ):
            curves[method] = sorted(
                [
                    {
                        "train_samples": int(row["train_samples"]),
                        "mean_oracle_fraction_mean": float(
                            row["mean_oracle_fraction_mean"]
                        ),
                        "mean_oracle_fraction_ci95": float(
                            row["mean_oracle_fraction_ci95"]
                        ),
                    }
                    for row in metrics["aggregate"]
                    if row["topology"] == topology and row["method"] == method
                ],
                key=lambda row: row["train_samples"],
            )
        output_payload["performance_curves"][topology] = curves
    return input_payload, output_payload


def _plot_golden_case(topology: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    initial = np.asarray(inputs["initial_weights_iq"], dtype=np.float64)[0]
    final = np.asarray(outputs["output_fp32_iq"], dtype=np.float64)[0]
    initial_complex = initial[0] + 1j * initial[1]
    final_complex = final[0] + 1j * final[1]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    panels = (
        (np.abs(initial_complex), "input amplitude", 0.0, 1.0),
        (np.abs(final_complex), "output amplitude", 0.0, 1.0),
        (np.angle(final_complex), "output phase (rad)", -np.pi, np.pi),
    )
    for axis, (value, title, lower, upper) in zip(axes, panels):
        image = axis.imshow(value, aspect="auto", cmap="viridis", vmin=lower, vmax=upper)
        axis.set_title(title)
        axis.set_xlabel("array column")
        axis.set_ylabel("array row")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(f"V2.4 {topology} golden case 0")
    fig.savefig(
        PICTURE_DIR / f"golden_case_{topology}.png",
        dpi=160,
        metadata={"Software": "AESA-CNN V2.4"},
    )
    plt.close(fig)


def _plot_performance(curves: dict[str, Any]) -> None:
    labels = {
        "single_board_scratch": "scratch",
        "single_board_pretrained": "pretrained full",
        "single_board_head_only": "pretrained head-only",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True, constrained_layout=True)
    for axis, topology in zip(axes, ("4x4", "1x4")):
        for method, rows in curves[topology].items():
            x = [row["train_samples"] for row in rows]
            y = [row["mean_oracle_fraction_mean"] for row in rows]
            error = [row["mean_oracle_fraction_ci95"] for row in rows]
            axis.errorbar(x, y, yerr=error, marker="o", capsize=2, label=labels[method])
        axis.set_xscale("log", base=2)
        axis.set_xticks([8, 16, 32, 64, 128])
        axis.set_xticklabels(["8", "16", "32", "64", "128"])
        axis.set_title(topology)
        axis.set_xlabel("single-board commands")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("mean oracle fraction")
    axes[1].legend(fontsize=8)
    fig.suptitle("V2.4 sparse-supervision comparison (mean ± 95% CI)")
    fig.savefig(
        PICTURE_DIR / "performance_comparison.png",
        dpi=160,
        metadata={"Software": "AESA-CNN V2.4"},
    )
    plt.close(fig)


def write_bundle() -> None:
    input_payload, output_payload = build_payloads()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "comparison_inputs.json").write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "comparison_results.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for topology in ("4x4", "1x4"):
        _plot_golden_case(
            topology,
            input_payload["cases"][topology],
            output_payload["cases"][topology],
        )
    _plot_performance(output_payload["performance_curves"])


def check_bundle() -> None:
    expected_input, expected_output = build_payloads()
    actual_input = json.loads(
        (INPUT_DIR / "comparison_inputs.json").read_text(encoding="utf-8")
    )
    actual_output = json.loads(
        (OUTPUT_DIR / "comparison_results.json").read_text(encoding="utf-8")
    )
    if actual_input != expected_input or actual_output != expected_output:
        raise AssertionError("V2.4 对照 JSON 与冻结 metrics/golden vectors 不一致")
    for name in (
        "golden_case_4x4.png",
        "golden_case_1x4.png",
        "performance_comparison.png",
    ):
        path = PICTURE_DIR / name
        if path.stat().st_size < 1_000 or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"V2.4 对照图片无效：{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_bundle()
        print(json.dumps({"version": "v2.4", "comparison_bundle": "pass"}))
    else:
        write_bundle()
        print(json.dumps({"version": "v2.4", "comparison_bundle": "written"}))


if __name__ == "__main__":
    main()
