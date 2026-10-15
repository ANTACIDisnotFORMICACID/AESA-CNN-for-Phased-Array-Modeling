"""Export and verify the deterministic V2.5 full-pattern comparison bundle.

The exporter never trains or re-runs the numerical study.  It projects the
frozen V2.5 metrics, CSV tables, contracts, golden arrays, and inherited
controller metadata into one explicit input JSON, one paired output JSON, and
three presentation images.  ``--check`` rebuilds the expected payloads in
memory and does not write to disk.  JSON values are compared with a tight
floating-point tolerance, while PNG files are checked by signature and canvas
size because renderer metadata and font rasterization vary across platforms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
VERSION_DIR = CODE_DIR.parent
INPUT_DIR = VERSION_DIR / "input" / "full_pattern"
OUTPUT_DIR = VERSION_DIR / "output" / "full_pattern"
PICTURE_DIR = OUTPUT_DIR / "pictures"
CONTROLLER_OUTPUT_DIR = VERSION_DIR / "output" / "controller_training"

INPUT_NAME = "comparison_inputs.json"
OUTPUT_NAME = "comparison_results.json"
PICTURE_NAMES = (
    "golden_full_pattern_4x4.png",
    "golden_full_pattern_1x4.png",
    "performance_comparison.png",
)
TOPOLOGIES = ("4x4", "1x4")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _sha256_binary(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_text_bytes(path: Path) -> bytes:
    """Return platform-independent UTF-8 bytes for versioned text evidence."""

    text = path.read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _sha256_canonical_text(path: Path) -> str:
    return hashlib.sha256(_canonical_text_bytes(path)).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _coerce_csv_cell(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if not any(character in value.lower() for character in (".", "e")):
            return int(value)
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite CSV value: {value}")
        return number
    except ValueError:
        return value


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: _coerce_csv_cell(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise AssertionError(f"empty comparison source: {path.name}")
    return rows


def _json_array(value: np.ndarray) -> list[Any]:
    if not np.isfinite(value).all():
        raise AssertionError("golden array contains non-finite values")
    return value.tolist()


def _golden_error(field_iq: np.ndarray, reference_iq: np.ndarray) -> dict[str, float]:
    error = field_iq.astype(np.float64) - reference_iq.astype(np.float64)
    reference_norm = max(float(np.linalg.norm(reference_iq.astype(np.float64))), 1.0e-300)
    relative_l2 = float(np.linalg.norm(error) / reference_norm)
    nmse = float(np.sum(error * error) / max(float(np.sum(reference_iq**2)), 1.0e-300))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2": relative_l2,
        "nmse_db": 10.0 * math.log10(max(nmse, 1.0e-300)),
    }


def build_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact JSON payloads from frozen, version-scoped evidence."""

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = _load_json(metrics_path)
    if metrics.get("algorithm_version") != "2.5.0":
        raise AssertionError("full-pattern metrics are not V2.5")

    complexity_path = OUTPUT_DIR / "complexity.csv"
    benchmark_path = OUTPUT_DIR / "benchmark.csv"
    numerical_path = OUTPUT_DIR / "numerical_cases.csv"
    complexity = _load_csv(complexity_path)
    benchmark = _load_csv(benchmark_path)
    numerical = _load_csv(numerical_path)

    input_payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "2.5.0",
        "purpose": "version-scoped full-pattern input/output comparison",
        "source": "frozen V2.5 golden arrays and inherited V2.4 low-update controllers",
        "source_artifact_hash_contract": {
            "algorithm": "SHA-256",
            "encoding": "UTF-8 without BOM",
            "line_endings": "LF",
            "scope": "canonical decoded text bytes",
        },
        "source_artifacts": {
            "metrics.json": _sha256_canonical_text(metrics_path),
            "complexity.csv": _sha256_canonical_text(complexity_path),
            "benchmark.csv": _sha256_canonical_text(benchmark_path),
            "numerical_cases.csv": _sha256_canonical_text(numerical_path),
        },
        "protocol": {
            "seed": metrics["seed"],
            "grid": metrics["grid"],
            "numerical_modes": ["ideal", "position_amp_phase_coupling"],
            "numerical_dtypes": ["float64", "float32"],
            "cases_per_topology_mode_dtype": 8,
            "benchmark_warmup": sorted({int(row["warmup"]) for row in benchmark}),
            "benchmark_repeats": sorted({int(row["repeats"]) for row in benchmark}),
            "historical_replay": metrics["untouched_historical_json_replay"],
        },
        "cases": {},
    }
    output_payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "2.5.0",
        "purpose": "outputs paired one-to-one with input/full_pattern/comparison_inputs.json",
        "acceptance_gates": metrics["acceptance_gates"],
        "accuracy_summary": metrics["accuracy_summary"],
        "legacy_visualization_formula_parity": metrics[
            "legacy_visualization_formula_parity"
        ],
        "controller_regression": metrics["v24_controller_regression"],
        "complexity": complexity,
        "benchmark": benchmark,
        "numerical_cases": numerical,
        "cases": {},
    }

    for topology in TOPOLOGIES:
        golden_path = OUTPUT_DIR / f"golden_full_pattern_{topology}.npz"
        contract_path = OUTPUT_DIR / f"contract_{topology}.json"
        checkpoint_path = (
            CONTROLLER_OUTPUT_DIR / f"v24_{topology}_low_update_controller.pt"
        )
        regression = metrics["v24_controller_regression"][topology]
        if regression.get("algorithm_version") != "2.4.0":
            raise AssertionError(f"wrong inherited controller version: {topology}")
        contract = _load_json(contract_path)
        with np.load(golden_path, allow_pickle=False) as archive:
            field_iq = np.asarray(archive["field_iq"])
            reference_iq = np.asarray(archive["reference_iq"])
            input_payload["cases"][topology] = {
                "array_config": contract["array"],
                "grid_contract": contract,
                "controller_checkpoint": {
                    "package_path": f"output/controller_training/{checkpoint_path.name}",
                    "sha256": _sha256_binary(checkpoint_path),
                    "algorithm_version": regression["algorithm_version"],
                    "training_regime": regression["training_regime"],
                    "selected_update": regression["selected_update"],
                    "selection_split": regression["selection_split"],
                    "selected_seed": regression["selected_seed"],
                },
                "golden_source": {
                    "package_path": f"output/full_pattern/{golden_path.name}",
                    "sha256": _sha256_binary(golden_path),
                },
                "weights_iq": _json_array(np.asarray(archive["weights_iq"])),
                "position_errors": _json_array(
                    np.asarray(archive["position_errors"])
                ),
                "element_response_iq": _json_array(
                    np.asarray(archive["element_response_iq"])
                ),
                "coupling_real": _json_array(np.asarray(archive["coupling_real"])),
                "coupling_imag": _json_array(np.asarray(archive["coupling_imag"])),
                "theta_axis_rad": _json_array(
                    np.asarray(archive["theta_axis_rad"])
                ),
                "phi_axis_rad": _json_array(np.asarray(archive["phi_axis_rad"])),
            }
            output_payload["cases"][topology] = {
                "field_iq": _json_array(field_iq),
                "reference_iq": _json_array(reference_iq),
                "golden_error": _golden_error(field_iq, reference_iq),
                "complexity": next(
                    row for row in complexity if row["topology"] == topology
                ),
                "benchmark": [
                    row for row in benchmark if row["topology"] == topology
                ],
            }
    return input_payload, output_payload


def _save_figure_bytes(figure: plt.Figure) -> bytes:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=160,
        metadata={"Software": "AESA-CNN V2.5 deterministic release exporter"},
    )
    plt.close(figure)
    return stream.getvalue()


def _golden_picture_bytes(
    topology: str, output_case: dict[str, Any]
) -> bytes:
    field_iq = np.asarray(output_case["field_iq"], dtype=np.float64)[0]
    reference_iq = np.asarray(output_case["reference_iq"], dtype=np.float64)[0]
    field = field_iq[0] + 1j * field_iq[1]
    reference = reference_iq[0] + 1j * reference_iq[1]
    magnitude = np.abs(field)
    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude / max(float(magnitude.max()), 1.0e-12), 1.0e-4)
    )
    phase = np.angle(field)
    error = np.abs(field - reference)
    error_db = 20.0 * np.log10(np.maximum(error, 1.0e-16))

    with matplotlib.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(
            1, 3, figsize=(10.8, 3.35), constrained_layout=True
        )
        panels = (
            (magnitude_db, "normalized magnitude (dB)", "viridis", -80.0, 0.0),
            (phase, "phase (rad)", "twilight", -math.pi, math.pi),
            (error_db, "formula error (dB)", "magma", -320.0, -80.0),
        )
        for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            axis.set_title(title)
            axis.set_xlabel("theta index")
            axis.set_ylabel("phi index")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(f"V2.5 {topology} full complex pattern golden case")
    return _save_figure_bytes(figure)


def _performance_picture_bytes(output_payload: dict[str, Any]) -> bytes:
    complexity = {
        row["topology"]: row for row in output_payload["complexity"]
    }
    benchmark = output_payload["benchmark"]
    colors = ("#355C7D", "#C06C84", "#6C9A8B")

    with matplotlib.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(
            1, 3, figsize=(13.0, 3.65), constrained_layout=True
        )
        x = np.arange(len(TOPOLOGIES), dtype=np.float64)
        width = 0.24
        mac_series = (
            ("controller", "controller_macs"),
            ("renderer", "renderer_real_macs"),
            ("legacy CNN", "legacy_forward_cnn_macs"),
        )
        for index, ((label, key), color) in enumerate(zip(mac_series, colors)):
            axes[0].bar(
                x + (index - 1) * width,
                [float(complexity[topology][key]) for topology in TOPOLOGIES],
                width,
                label=label,
                color=color,
            )
        axes[0].set_yscale("log")
        axes[0].set_xticks(x, TOPOLOGIES)
        axes[0].set_ylabel("real MACs / sample")
        axes[0].set_title("compute comparison")
        axes[0].legend()
        axes[0].grid(True, axis="y", alpha=0.25)

        operations = (
            "controller_only",
            "full_pattern_only_cached_ideal",
            "controller_plus_full_pattern",
        )
        labels = ("controller", "renderer", "end-to-end")
        for index, (operation, label, color) in enumerate(
            zip(operations, labels, colors)
        ):
            axes[1].bar(
                x + (index - 1) * width,
                [
                    float(
                        next(
                            row["median_ms"]
                            for row in benchmark
                            if row["topology"] == topology
                            and row["operation"] == operation
                        )
                    )
                    for topology in TOPOLOGIES
                ],
                width,
                label=label,
                color=color,
            )
        axes[1].set_xticks(x, TOPOLOGIES)
        axes[1].set_ylabel("CPU batch-1 median (ms)")
        axes[1].set_title("recorded runtime")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.25)

        accuracy = output_payload["accuracy_summary"]
        accuracy_labels: list[str] = []
        accuracy_values: list[float] = []
        for topology in TOPOLOGIES:
            for dtype in ("float64", "float32"):
                accuracy_labels.append(f"{topology}\nFP{dtype[-2:]}")
                accuracy_values.append(
                    max(
                        float(accuracy[topology][mode][dtype]["worst_max_abs_error"])
                        for mode in ("ideal", "position_amp_phase_coupling")
                    )
                )
        axes[2].bar(
            np.arange(len(accuracy_values)),
            accuracy_values,
            color=("#355C7D", "#C06C84", "#355C7D", "#C06C84"),
        )
        axes[2].set_yscale("log")
        axes[2].set_xticks(np.arange(len(accuracy_labels)), accuracy_labels)
        axes[2].set_ylabel("worst max absolute error")
        axes[2].set_title("independent-formula agreement")
        axes[2].grid(True, axis="y", alpha=0.25)
        figure.suptitle("V2.5 full-pattern performance and numerical accuracy")
    return _save_figure_bytes(figure)


def build_picture_bytes(
    input_payload: dict[str, Any], output_payload: dict[str, Any]
) -> dict[str, bytes]:
    del input_payload  # Inputs are embedded in the paired JSON; plots use recorded outputs.
    pictures = {
        f"golden_full_pattern_{topology}.png": _golden_picture_bytes(
            topology, output_payload["cases"][topology]
        )
        for topology in TOPOLOGIES
    }
    pictures["performance_comparison.png"] = _performance_picture_bytes(
        output_payload
    )
    return pictures


def write_bundle() -> dict[str, Any]:
    input_payload, output_payload = build_payloads()
    pictures = build_picture_bytes(input_payload, output_payload)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / INPUT_NAME).write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / OUTPUT_NAME).write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, payload in pictures.items():
        (PICTURE_DIR / name).write_bytes(payload)
    return {
        "version": "v2.5",
        "comparison_bundle": "written",
        "input_cases": len(input_payload["cases"]),
        "output_cases": len(output_payload["cases"]),
        "pictures": sorted(pictures),
    }


def check_bundle() -> dict[str, Any]:
    """Recompute expected JSON and PNG bytes in memory; never write files."""

    expected_input, expected_output = build_payloads()
    actual_input = _load_json(INPUT_DIR / INPUT_NAME)
    actual_output = _load_json(OUTPUT_DIR / OUTPUT_NAME)
    _assert_payload_close(actual_input, expected_input, "input")
    _assert_payload_close(actual_output, expected_output, "output")

    expected_pictures = build_picture_bytes(expected_input, expected_output)
    for name in PICTURE_NAMES:
        actual = (PICTURE_DIR / name).read_bytes()
        if not actual.startswith(_PNG_SIGNATURE):
            raise AssertionError(f"invalid V2.5 PNG signature: {name}")
        expected = expected_pictures[name]
        if not expected.startswith(_PNG_SIGNATURE):
            raise AssertionError(f"generated V2.5 PNG is invalid: {name}")
        actual_size = (int.from_bytes(actual[16:20], "big"), int.from_bytes(actual[20:24], "big"))
        expected_size = (int.from_bytes(expected[16:20], "big"), int.from_bytes(expected[20:24], "big"))
        if actual_size != expected_size:
            raise AssertionError(
                f"V2.5 comparison picture canvas is stale: {name}={actual_size}, expected={expected_size}"
            )
    return {
        "version": "v2.5",
        "comparison_bundle": "pass",
        "input_cases": len(expected_input["cases"]),
        "output_cases": len(expected_output["cases"]),
        "pictures": list(PICTURE_NAMES),
        "check_mode_writes": 0,
    }


def _assert_payload_close(actual: Any, expected: Any, path: str) -> None:
    """Compare frozen payloads exactly except for platform-level float drift."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"V2.5 comparison {path} keys are stale")
        for key in actual:
            _assert_payload_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"V2.5 comparison {path} length is stale")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_payload_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and (isinstance(actual, float) or isinstance(expected, float))
    ):
        if not math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise AssertionError(f"V2.5 comparison {path} value is stale")
        return
    if actual != expected:
        raise AssertionError(f"V2.5 comparison {path} value is stale")


def check_release_bundle() -> dict[str, Any]:
    """Stable validator-facing alias."""

    return check_bundle()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare all JSON/PNG bytes without writing",
    )
    args = parser.parse_args()
    result = check_bundle() if args.check else write_bundle()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
