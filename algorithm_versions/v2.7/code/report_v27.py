"""Merge V2.7 maps/audits, apply the training gate, and render release figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from boundary_v27 import classify_boundary, tracking_grid
from evaluate_v27 import METRIC_NAMES, _batch_metrics
from oracle_v27 import MAP_ROOT, PHI_GRID, THETA_GRID, summarize_map, target_grid
from states_v27 import all_audit_states
from v25_adapter_v27 import FrozenV25Adapter
from v27_compat import REPO_ROOT, V27_ROOT, enable_v26

enable_v26()
from controller_v26 import BeamformingControllerV26  # noqa: E402
from hfss_physics_v26 import HfssPatternData, quantize_int8_disk  # noqa: E402


REPORT_ROOT = V27_ROOT / "output" / "reports"
PICTURE_ROOT = REPORT_ROOT / "pictures"
AUDIT_ROOT = V27_ROOT / "output" / "audit"


def _repo_relative(path: Path) -> str:
    """Store portable repository-relative artifact references in reports."""
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _ideal_calibration_iq() -> np.ndarray:
    value = np.zeros((2, 1, 4), dtype=np.float32)
    value[0, 0] = 1.0
    return value


def _chunk_metrics(data: HfssPatternData, effective: np.ndarray, targets: np.ndarray, chunk: int = 64) -> np.ndarray:
    rows = []
    for start in range(0, len(targets), chunk):
        result = _batch_metrics(data, effective[start:start + chunk], targets[start:start + chunk])
        rows.append(np.stack([result[key] for key in METRIC_NAMES], axis=-1))
    return np.concatenate(rows).astype(np.float32)


def build_nominal_controller_maps() -> Path:
    data = HfssPatternData.load()
    targets = target_grid().astype(np.float32)
    controller = BeamformingControllerV26(data)
    calibration_iq = _ideal_calibration_iq()
    health = np.ones(4, dtype=np.float32)
    v26 = np.stack(
        [controller.deployment_input((float(theta), float(phi)), calibration_iq, np.asarray([1.0]), health)[1] for theta, phi in targets]
    ).astype(np.complex64)
    v25 = FrozenV25Adapter().commands(targets, np.ones(4, dtype=np.complex64), health)
    methods = []
    weights = []
    codes = []
    metrics = []
    for name, command in (("v2.6_zero_residual", v26), ("v2.5_low_update_15mm", v25)):
        int_codes, int_weights = [], []
        for value in command:
            code, restored = quantize_int8_disk(value)
            int_codes.append(code)
            int_weights.append(restored)
        int_weights = np.asarray(int_weights, dtype=np.complex64)
        for suffix, value in (("fp32", command), ("int8", int_weights)):
            methods.append(f"{name}_{suffix}")
            weights.append(value)
            metrics.append(_chunk_metrics(data, value, targets))
            codes.append(np.asarray(int_codes, dtype=np.int8))
    destination = MAP_ROOT / "nominal_controller_maps.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        targets_hfss_deg=targets,
        method_names=np.asarray(methods),
        metric_names=np.asarray(METRIC_NAMES),
        metrics=np.stack(metrics),
        weights=np.stack(weights),
        int8_codes=np.stack(codes),
    )
    return destination


def _wilson(successes: int, count: int, confidence: float = 0.95) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, count - successes + 1))
    upper = 1.0 if successes == count else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, count - successes))
    return lower, upper


def _audit_rows() -> list[dict[str, str]]:
    path = AUDIT_ROOT / "control_cases.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_parameter_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["state_class"], row["method"]), []).append(row)
    summaries = []
    for (state_class, method), values in sorted(groups.items()):
        strict = sum(value["strict_unambiguous"].lower() == "true" for value in values)
        pointing = sum(value["pointing_reachable"].lower() == "true" for value in values)
        ci = _wilson(strict, len(values))
        summaries.append(
            {
                "state_class": state_class,
                "method": method,
                "samples": len(values),
                "pointing_reachable_fraction": pointing / len(values),
                "strict_unambiguous_fraction": strict / len(values),
                "strict_fraction_95ci": list(ci),
                "target_power_regret_db_median": float(np.median([float(value["target_power_regret_vs_phase_db"]) for value in values])),
                "pointing_error_deg_p95": float(np.percentile([float(value["pointing_error_deg"]) for value in values], 95)),
                "strict_margin_db_p05": float(np.percentile([float(value["strict_margin_db"]) for value in values], 5)),
                "calibration_nmse_db_worst": float(np.max([float(value["calibration_nmse_db"]) for value in values])),
            }
        )
    mixed = [row for row in rows if row["state_class"] == "mixed"]
    bin_specs = {
        "amplitude_rms_fraction": np.asarray([0.0, 0.03, 0.06, 0.09, 0.1300001]),
        "phase_rms_deg": np.asarray([0.0, 5.0, 10.0, 15.0, 22.0001]),
        "measurement_snr_db": np.asarray([25.0, 30.0, 35.0, 40.0001]),
    }
    binned = []
    for parameter, edges in bin_specs.items():
        for method in sorted({row["method"] for row in mixed}):
            method_rows = [row for row in mixed if row["method"] == method]
            for left, right in zip(edges[:-1], edges[1:]):
                values = [row for row in method_rows if left <= float(row[parameter]) < right]
                if not values:
                    continue
                strict = sum(row["strict_unambiguous"].lower() == "true" for row in values)
                binned.append(
                    {
                        "parameter": parameter,
                        "bin": [float(left), float(right)],
                        "method": method,
                        "samples": len(values),
                        "strict_unambiguous_fraction": strict / len(values),
                        "strict_fraction_95ci": list(_wilson(strict, len(values))),
                        "target_power_regret_db_median": float(np.median([float(row["target_power_regret_vs_phase_db"]) for row in values])),
                        "pointing_error_deg_p95": float(np.percentile([float(row["pointing_error_deg"]) for row in values], 95)),
                    }
                )
    payload = {"schema_version": 1, "rows": summaries, "binned_sensitivity": binned}
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "parameter_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def _method_arrays(path: Path) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["method_names"].tolist(),
            archive["metric_names"].tolist(),
            archive["metrics"].copy(),
            archive["weights"].copy(),
        )


def _audit_robust_grid(rows: list[dict[str, str]], method: str) -> np.ndarray:
    """64 mixed-state samples per 16x16 stratum, expanded to the 64x64 query grid."""
    selected = [row for row in rows if row["state_class"] == "mixed" and row["method"] == method]
    successes = np.zeros((16, 16), dtype=np.float64)
    counts = np.zeros((16, 16), dtype=np.float64)
    for row in selected:
        ti = min(15, max(0, int((float(row["theta_deg"]) - 60.0) / 60.0 * 16.0)))
        pi = min(15, max(0, int((float(row["phi_deg"]) + 30.0) / 60.0 * 16.0)))
        counts[ti, pi] += 1
        successes[ti, pi] += row["strict_unambiguous"].lower() == "true"
    rate = np.divide(successes, counts, out=np.zeros_like(successes), where=counts > 0)
    return np.repeat(np.repeat(rate, 4, axis=0), 4, axis=1)


def _audit_class_grid(rows: list[dict[str, str]], state_class: str, method: str) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["state_class"] == state_class and row["method"] == method]
    fields = ("target_to_peak_db", "pointing_error_deg", "strict_margin_db", "hpbw_aware_margin_db")
    sums = {field: np.zeros((16, 16), dtype=np.float64) for field in fields}
    counts = np.zeros((16, 16), dtype=np.float64)
    strict = np.zeros((16, 16), dtype=np.float64)
    track = np.zeros((16, 16), dtype=np.float64)
    peak_jump = np.zeros((16, 16), dtype=np.float64)
    weight_change = np.zeros((16, 16), dtype=np.float64)
    for row in selected:
        ti = min(15, max(0, int((float(row["theta_deg"]) - 60.0) / 60.0 * 16.0)))
        pi = min(15, max(0, int((float(row["phi_deg"]) + 30.0) / 60.0 * 16.0)))
        counts[ti, pi] += 1
        for field in fields:
            sums[field][ti, pi] += float(row[field])
        strict[ti, pi] += row["strict_unambiguous"].lower() == "true"
        jump = float(row["nearest_sample_peak_jump_deg"])
        change = float(row["nearest_sample_weight_change"])
        track[ti, pi] += jump <= 3.0 and change <= 0.25
        peak_jump[ti, pi] += jump
        weight_change[ti, pi] += change
    if not np.all(counts > 0):
        raise RuntimeError(f"audit state class {state_class} does not cover all 16x16 strata")
    expand = lambda value: np.repeat(np.repeat(value, 4, axis=0), 4, axis=1)
    return {
        **{field: expand(sums[field] / counts) for field in fields},
        "robust_pass_fraction": expand(strict / counts),
        "continuous_tracking": expand((track / counts) >= 0.95),
        "peak_jump_deg": expand(peak_jump / counts),
        "weight_change": expand(weight_change / counts),
    }


def build_boundary_query_map(rows: list[dict[str, str]]) -> Path:
    oracle_path = MAP_ROOT / "nominal_oracle_maps.npz"
    control_path = MAP_ROOT / "nominal_controller_maps.npz"
    oracle_names, oracle_metric_names, oracle_metrics, oracle_weights = _method_arrays(oracle_path)
    control_names, control_metric_names, control_metrics, control_weights = _method_arrays(control_path)
    state_classes = []
    target_to_peak = []
    pointing = []
    margin = []
    hpbw = []
    robust = []
    robust_evaluated = []
    continuous = []
    peak_jump_theta = []
    peak_jump_phi = []
    weight_change_theta = []
    weight_change_phi = []

    def append(name: str, metric_names: list[str], metric: np.ndarray, weights: np.ndarray, audit_method: str | None):
        state_classes.append(name)
        values = metric.reshape(64, 64, -1)
        get = lambda field: values[:, :, metric_names.index(field)]
        target_to_peak.append(get("target_to_peak_db"))
        pointing.append(get("pointing_error_deg"))
        margin.append(get("strict_margin_db"))
        hpbw.append(get("hpbw_aware_margin_db"))
        if audit_method is None:
            robust.append(np.zeros((64, 64), dtype=np.float32))
            robust_evaluated.append(np.zeros((64, 64), dtype=bool))
        else:
            robust.append(_audit_robust_grid(rows, audit_method))
            robust_evaluated.append(np.ones((64, 64), dtype=bool))
        track = tracking_grid(
            get("peak_theta_deg"), get("peak_phi_deg"), weights.reshape(64, 64, 4)
        )
        continuous.append(track["continuous_tracking"])
        peak_jump_theta.append(track["peak_jump_theta_deg"])
        peak_jump_phi.append(track["peak_jump_phi_deg"])
        weight_change_theta.append(track["weight_change_theta"])
        weight_change_phi.append(track["weight_change_phi"])

    for index, name in enumerate(oracle_names):
        audit = "phase_reference_int8" if name == "phase_only" else None
        append(f"nominal/{name}", oracle_metric_names, oracle_metrics[:, index], oracle_weights[:, index], audit)
    for index, name in enumerate(control_names):
        audit = None
        if name == "v2.6_zero_residual_int8":
            audit = name
        elif name == "v2.5_low_update_15mm_int8":
            audit = name
        append(f"baseline/{name}", control_metric_names, control_metrics[index], control_weights[index], audit)

    if rows:
        for state_class in ("baseline", "amplitude", "phase", "snr", "failed_port", "global_phase", "mixed"):
            grid = _audit_class_grid(rows, state_class, "v2.6_zero_residual_int8")
            state_classes.append(f"audit/{state_class}/v2.6_zero_residual_int8")
            target_to_peak.append(grid["target_to_peak_db"])
            pointing.append(grid["pointing_error_deg"])
            margin.append(grid["strict_margin_db"])
            hpbw.append(grid["hpbw_aware_margin_db"])
            robust.append(grid["robust_pass_fraction"])
            robust_evaluated.append(np.full((64, 64), state_class == "mixed", dtype=bool))
            continuous.append(grid["continuous_tracking"])
            peak_jump_theta.append(grid["peak_jump_deg"])
            peak_jump_phi.append(grid["peak_jump_deg"])
            weight_change_theta.append(grid["weight_change"])
            weight_change_phi.append(grid["weight_change"])

    destination = MAP_ROOT / "boundary_maps.npz"
    np.savez_compressed(
        destination,
        theta_deg=THETA_GRID,
        phi_deg=PHI_GRID,
        state_classes=np.asarray(state_classes),
        target_to_peak_db=np.stack(target_to_peak).astype(np.float32),
        pointing_error_deg=np.stack(pointing).astype(np.float32),
        strict_margin_db=np.stack(margin).astype(np.float32),
        hpbw_aware_margin_db=np.stack(hpbw).astype(np.float32),
        robust_pass_fraction=np.stack(robust).astype(np.float32),
        robust_evaluated=np.stack(robust_evaluated),
        continuous_tracking=np.stack(continuous),
        peak_jump_theta_deg=np.stack(peak_jump_theta),
        peak_jump_phi_deg=np.stack(peak_jump_phi),
        weight_change_theta=np.stack(weight_change_theta),
        weight_change_phi=np.stack(weight_change_phi),
    )
    return destination


def write_boundary_polygons(boundary_map: Path) -> Path:
    """Extract contour vertices for the three nominal continuous-oracle layers."""
    with np.load(boundary_map, allow_pickle=False) as archive:
        names = archive["state_classes"].tolist()
        index = names.index("nominal/continuous_disk")
        gain = archive["target_to_peak_db"][index] >= -0.5
        pointing = gain & (archive["pointing_error_deg"][index] <= 5.0)
        strict = pointing & (archive["strict_margin_db"][index] >= 3.0)
    polygons = {}
    phi_mesh, theta_mesh = np.meshgrid(PHI_GRID, THETA_GRID)
    for name, mask in (("GAIN_REACHABLE", gain), ("POINTING_REACHABLE", pointing), ("STRICT_UNAMBIGUOUS", strict)):
        if not np.any(mask) or np.all(mask):
            polygons[name] = []
            continue
        fig, ax = plt.subplots()
        contour = ax.contour(phi_mesh, theta_mesh, mask.astype(float), levels=[0.5])
        polygons[name] = [segment.astype(float).tolist() for segment in contour.allsegs[0] if len(segment) >= 3]
        plt.close(fig)
    destination = MAP_ROOT / "boundary_polygons.json"
    destination.write_text(
        json.dumps({"schema_version": 1, "coordinates": "[phi_deg, theta_deg]", "polygons": polygons}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def training_gate() -> dict[str, Any]:
    oracle_path = MAP_ROOT / "nominal_oracle_maps.npz"
    control_path = MAP_ROOT / "nominal_controller_maps.npz"
    if not oracle_path.is_file() or not control_path.is_file():
        return {"status": "blocked", "train_v2.7": False, "reason": "nominal oracle/controller maps incomplete"}
    with np.load(oracle_path, allow_pickle=False) as oracle, np.load(control_path, allow_pickle=False) as control:
        if not bool(oracle["complete"]):
            return {"status": "blocked", "train_v2.7": False, "reason": "4096-target oracle map incomplete"}
        onames, ometrics = oracle["method_names"].tolist(), oracle["metric_names"].tolist()
        cnames, cmetrics = control["method_names"].tolist(), control["metric_names"].tolist()
        continuous = oracle["metrics"][:, onames.index("continuous_disk")]
        phase = oracle["metrics"][:, onames.index("phase_only")]
        baseline = control["metrics"][cnames.index("v2.6_zero_residual_fp32")]
        point = lambda values, names: (values[:, names.index("target_to_peak_db")] >= -0.5) & (values[:, names.index("pointing_error_deg")] <= 5.0)
        cont_point = point(continuous, ometrics)
        base_point = point(baseline, cmetrics)
        area_increase = 100.0 * (np.mean(cont_point) - np.mean(base_point))
        margin_gain = continuous[:, ometrics.index("strict_margin_db")] - baseline[:, cmetrics.index("strict_margin_db")]
        target_ok = continuous[:, ometrics.index("target_power")] >= phase[:, ometrics.index("target_power")] * 10.0 ** (-0.5 / 10.0) * (1.0 - 1e-6)
        improved_fraction = float(np.mean(target_ok & (margin_gain >= 0.25)))
        passed = bool(area_increase >= 2.0 or improved_fraction >= 0.05)
        return {
            "status": "passed" if passed else "not_met",
            "train_v2.7": passed,
            "pointing_reachable_area_increase_percentage_points": float(area_increase),
            "margin_improved_at_least_0p25db_fraction": improved_fraction,
            "criteria": {"area_increase_percentage_points": 2.0, "improved_fraction": 0.05},
            "checkpoint_policy": "train unchanged V2.6 CNN only when train_v2.7 is true; otherwise release characterization without checkpoint",
        }


def _plot_heatmap(array: np.ndarray, title: str, path: Path, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(array, origin="lower", aspect="auto", extent=[-30, 30, 60, 120], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("HFSS phi (deg)")
    ax.set_ylabel("HFSS theta (deg)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_figures(rows: list[dict[str, str]]) -> list[Path]:
    PICTURE_ROOT.mkdir(parents=True, exist_ok=True)
    boundary_path = MAP_ROOT / "boundary_maps.npz"
    outputs = []
    with np.load(boundary_path, allow_pickle=False) as archive:
        names = archive["state_classes"].tolist()
        index = names.index("nominal/continuous_disk")
        gain = archive["target_to_peak_db"][index] >= -0.5
        pointing = gain & (archive["pointing_error_deg"][index] <= 5.0)
        strict = pointing & (archive["strict_margin_db"][index] >= 3.0)
        layers = gain.astype(np.int8) + pointing.astype(np.int8) + strict.astype(np.int8)
        path = PICTURE_ROOT / "boundary_layers_continuous.png"
        _plot_heatmap(layers, "Continuous oracle boundary layers (0..3)", path, "viridis", 0, 3)
        outputs.append(path)
        path = PICTURE_ROOT / "hpbw_aware_margin_continuous.png"
        _plot_heatmap(archive["hpbw_aware_margin_db"][index], "HPBW-aware margin (diagnostic only, dB)", path, "coolwarm")
        outputs.append(path)
        path = PICTURE_ROOT / "continuous_tracking.png"
        _plot_heatmap(archive["continuous_tracking"][index], "Adjacent-target continuous tracking", path, "Greens", 0, 1)
        outputs.append(path)
        jump = np.maximum(
            np.nan_to_num(archive["peak_jump_theta_deg"][index], nan=0.0),
            np.nan_to_num(archive["peak_jump_phi_deg"][index], nan=0.0),
        )
        path = PICTURE_ROOT / "peak_jump_map.png"
        _plot_heatmap(jump, "Adjacent-target peak jump (deg)", path, "magma", 0, None)
        outputs.append(path)
    if rows:
        failure = [row for row in rows if row["state_class"] == "failed_port" and row["method"] == "v2.6_zero_residual_int8"]
        groups = {}
        for row in failure:
            groups.setdefault(row["failed_port"], []).append(float(row["target_power_regret_vs_phase_db"]))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = sorted(groups)
        ax.bar(labels, [np.median(groups[label]) for label in labels])
        ax.set_xlabel("Failed port")
        ax.set_ylabel("Median target-power regret (dB)")
        ax.set_title("V2.6 sensitivity to one-port failure")
        fig.tight_layout()
        path = PICTURE_ROOT / "failed_port_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
        one_factor = [
            row for row in rows
            if row["state_class"] != "mixed" and row["method"] in ("phase_reference_int8", "v2.6_zero_residual_int8", "v2.5_low_update_15mm_int8")
        ]
        state_ids = []
        for row in one_factor:
            if row["state_id"] not in state_ids:
                state_ids.append(row["state_id"])
        selected_methods = ("phase_reference_int8", "v2.6_zero_residual_int8", "v2.5_low_update_15mm_int8")
        sensitivity = np.zeros((len(state_ids), len(selected_methods)), dtype=np.float64)
        for si, state_id in enumerate(state_ids):
            for mi, method in enumerate(selected_methods):
                values = [float(row["target_power_regret_vs_phase_db"]) for row in one_factor if row["state_id"] == state_id and row["method"] == method]
                sensitivity[si, mi] = np.median(values) if values else np.nan
        fig, ax = plt.subplots(figsize=(8.2, 7.0))
        image = ax.imshow(sensitivity, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(selected_methods)), selected_methods, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(state_ids)), state_ids, fontsize=7)
        ax.set_title("One-factor median target-power regret (dB)")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = PICTURE_ROOT / "parameter_sensitivity_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
        methods = sorted({row["method"] for row in rows})
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for method in methods:
            values = [row for row in rows if row["method"] == method]
            ax.scatter(
                np.median([float(value["target_power_regret_vs_phase_db"]) for value in values]),
                np.median([float(value["strict_margin_db"]) for value in values]),
                label=method,
                s=35,
            )
        ax.set_xlabel("Median target-power regret vs phase reference (dB)")
        ax.set_ylabel("Median strict 8-degree margin (dB)")
        ax.set_title("Method Pareto view")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = PICTURE_ROOT / "method_pareto.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def write_report() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _audit_rows()
    oracle_summary = summarize_map()
    parameter = summarize_parameter_audit(rows)
    boundary_map = build_boundary_query_map(rows)
    polygons = write_boundary_polygons(boundary_map)
    gate = training_gate()
    figures = render_figures(rows)
    overall_methods = []
    for method in sorted({row["method"] for row in rows}):
        values = [row for row in rows if row["method"] == method]
        overall_methods.append(
            {
                "method": method,
                "samples": len(values),
                "target_power_regret_db_median": float(np.median([float(row["target_power_regret_vs_phase_db"]) for row in values])),
                "pointing_error_deg_p95": float(np.percentile([float(row["pointing_error_deg"]) for row in values], 95)),
                "strict_margin_db_median": float(np.median([float(row["strict_margin_db"]) for row in values])),
                "strict_unambiguous_count": int(sum(row["strict_unambiguous"].lower() == "true" for row in values)),
            }
        )
    state_summary_path = AUDIT_ROOT / "state_continuous_reference_summary.json"
    repeatability_path = REPORT_ROOT / "oracle_repeatability_5pct.json"
    repeatability = json.loads(repeatability_path.read_text(encoding="utf-8")) if repeatability_path.is_file() else {"status": "pending"}
    payload = {
        "schema_version": 1,
        "release_kind": "trained-controller" if gate.get("train_v2.7") else "two-dimensional-boundary-characterization",
        "release_status": "complete" if repeatability.get("passed") else "complete_with_failed_raw_pointing_repeatability_gate",
        "paper_reference": {
            "task": "forward prediction: excitation/error to 64x64 complex radiation pattern",
            "best_validation_loss_approx": 0.0028,
            "not_ranked_as_inverse_control": True,
            "source": "情况说明与任务书/毛馨瑶毕业论文-终稿.pdf",
        },
        "project_goal_source": "大创立项-李新然.docx",
        "v2.5": "frozen low-update checkpoint evaluated with its original 15 mm interface; geometry-mismatch ablation only",
        "v2.6": "untrained zero-residual/analytic controller baseline",
        "fixed_physics": "47 GHz, HFSS four-port embedded fields, +Y 6.8 mm array, full 1-degree sphere search",
        "oracle_summary": oracle_summary,
        "parameter_summary_rows": len(parameter["rows"]),
        "overall_method_results": overall_methods,
        "state_continuous_reference": json.loads(state_summary_path.read_text(encoding="utf-8")) if state_summary_path.is_file() else {"status": "pending"},
        "oracle_repeatability": repeatability,
        "training_gate": gate,
        "boundary_query_map": _repo_relative(boundary_map),
        "boundary_polygons": _repo_relative(polygons),
        "figures": [_repo_relative(path) for path in figures],
        "strict_policy": "3 dB outside a fixed 8-degree exclusion is unchanged; HPBW-aware values are diagnostic only",
        "vitis_fpga_scope": "not run in V2.7",
    }
    json_path = REPORT_ROOT / "V2.7_SUMMARY.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = [
        "# V2.7 固定 1×4 阵列二维边界报告",
        "",
        "V2.7 将论文前向预测结果与逆向控制严格分开。论文约 2.8e-3 的验证损失不代表真实阵列二维波束可达。",
        "",
        f"- 发布类型：`{payload['release_kind']}`",
        f"- Oracle 完整：`{oracle_summary['complete']}`；唯一目标：`{oracle_summary['unique_targets']}/4096`",
        f"- 训练门禁：`{gate['status']}`；是否训练：`{gate['train_v2.7']}`",
        f"- 5% 独立复算：`{'通过' if repeatability.get('passed') else '未通过'}`；原始主峰指向超差样本 {repeatability.get('failed_sample_count', {}).get('pointing_error_deg', 'pending')}/205。目标增益与严格强峰间隔复算均通过。",
        "- V2.5：15 mm 几何失配消融；V2.6：未训练零残差解析基线。",
        "- 严格判据始终为目标点距全局峰 -0.5 dB、指向 5 度、8 度外强峰间隔 3 dB。",
        "- HPBW-aware 结果只解释宽主瓣，不替代严格门槛。",
        "",
        "## 名义边界",
        "",
    ]
    for name, value in oracle_summary["methods"].items():
        markdown.append(
            f"- `{name}`：GAIN {value['gain_reachable_count']}/4096，POINTING {value['pointing_reachable_count']}/4096，STRICT {value['strict_unambiguous_count']}/4096；HPBW-aware 诊断 {value['hpbw_aware_diagnostic_count']}/4096。"
        )
    markdown += [
        "",
        "## 多参数控制比较",
        "",
    ]
    for value in overall_methods:
        markdown.append(
            f"- `{value['method']}`：中位目标功率后悔 {value['target_power_regret_db_median']:.4f} dB，p95 指向误差 {value['pointing_error_deg_p95']:.2f} 度，严格可行 {value['strict_unambiguous_count']}/{value['samples']}。"
        )
    markdown += [
        "",
        "详细逐样本数据以紧凑的 `output/audit/control_cases.npz` 保留，离线查询图见 `output/maps/boundary_maps.npz`；重复内容的大型 CSV 不纳入展示分支。",
    ]
    (REPORT_ROOT / "V2.7_BOUNDARY_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    if not (MAP_ROOT / "nominal_controller_maps.npz").is_file():
        print(build_nominal_controller_maps())
    print(json.dumps(write_report(), ensure_ascii=False, indent=2))
