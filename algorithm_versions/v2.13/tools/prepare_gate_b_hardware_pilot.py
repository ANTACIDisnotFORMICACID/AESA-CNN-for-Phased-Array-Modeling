"""Prepare a validation-only, statistically stratified Gate-B hardware pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import chi2, t


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.12" / "src"))
sys.path.insert(0, str(ROOT.parent / "v2.10" / "src"))

from v210.coordinates import direction_to_hfss_angles  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.inversion import _jacobian_diagnostics_batch  # noqa: E402


def repeat_precision(repeat_count: int, snr_values: tuple[float, ...]) -> dict[str, object]:
    """Normal-theory uncertainty factors for variance and repeated-mean power."""
    if repeat_count < 3:
        raise ValueError("repeat_count must be at least three")
    degrees = repeat_count - 1
    variance_factor = [
        float(degrees / chi2.ppf(0.975, degrees)),
        float(degrees / chi2.ppf(0.025, degrees)),
    ]
    standard_deviation_factor = [float(np.sqrt(value)) for value in variance_factor]
    t_critical = float(t.ppf(0.975, degrees))
    mean_half_width = {
        str(int(snr_db)): float(
            100.0 * t_critical * 10.0 ** (-snr_db / 20.0) / np.sqrt(repeat_count)
        )
        for snr_db in snr_values
    }
    return {
        "repeat_count": repeat_count,
        "degrees_of_freedom": degrees,
        "variance_95ci_factor_relative_to_sample_variance": variance_factor,
        "standard_deviation_95ci_factor_relative_to_sample_std": standard_deviation_factor,
        "mean_power_95ci_half_width_percent_by_repeat_snr_db": mean_half_width,
        "assumptions": "independent approximately Gaussian repeat errors with fixed calibration state",
    }


def _condition_numbers(g: torch.Tensor, commands: torch.Tensor, batch_size: int) -> np.ndarray:
    values: list[torch.Tensor] = []
    for start in range(0, len(g), batch_size):
        batch = g[start : start + batch_size]
        ones = torch.ones((len(batch), len(commands)), dtype=batch.real.dtype)
        _, condition, _ = _jacobian_diagnostics_batch(batch, commands, ones)
        values.append(condition.cpu())
    return torch.cat(values).numpy()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _choose_random(
    generator: np.random.Generator,
    candidates: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(candidates) < count:
        raise ValueError(f"stratum contains only {len(candidates)} candidates, fewer than {count}")
    return np.sort(generator.choice(candidates, size=count, replace=False))


def select_pilot(
    direction: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    signal_db_relative: np.ndarray,
    jacobian_condition: np.ndarray,
    *,
    seed: int,
    per_core_stratum: int = 40,
    coverage_supplement: int = 40,
) -> tuple[np.ndarray, list[str], list[str], dict[str, object]]:
    """Return 200 probability-sample rows plus 40 farthest-point diagnostics."""
    count = len(direction)
    if any(len(value) != count for value in (theta_deg, phi_deg, signal_db_relative, jacobian_condition)):
        raise ValueError("all pilot selection inputs must have the same length")
    generator = np.random.default_rng(seed)
    known_failure = (
        (theta_deg >= 50.0)
        & (theta_deg < 60.0)
        & (phi_deg >= 30.0)
        & (phi_deg < 40.0)
    )
    high_condition_threshold = float(np.quantile(jacobian_condition, 0.90))
    strong_threshold = float(np.quantile(signal_db_relative, 0.75))
    weak_threshold = float(np.quantile(signal_db_relative, 0.25))

    labels = np.full(count, "other_validation", dtype="U32")
    labels[signal_db_relative <= weak_threshold] = "weak_field"
    labels[signal_db_relative >= strong_threshold] = "strong_field"
    labels[jacobian_condition >= high_condition_threshold] = "high_condition"
    labels[known_failure] = "known_failure_angle"
    core_names = (
        "known_failure_angle",
        "high_condition",
        "strong_field",
        "weak_field",
        "other_validation",
    )

    selected_parts: list[np.ndarray] = []
    population_counts: dict[str, int] = {}
    for name in core_names:
        candidates = np.flatnonzero(labels == name)
        population_counts[name] = int(len(candidates))
        selected_parts.append(_choose_random(generator, candidates, per_core_stratum))
    core_selected = np.concatenate(selected_parts)

    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[core_selected] = True
    remaining = np.flatnonzero(~selected_mask)
    minimum_chord = np.min(
        np.linalg.norm(direction[remaining, None, :] - direction[core_selected][None, :, :], axis=2),
        axis=1,
    )
    diagnostic: list[int] = []
    for _ in range(coverage_supplement):
        position = int(np.argmax(minimum_chord))
        chosen = int(remaining[position])
        diagnostic.append(chosen)
        distance = np.linalg.norm(direction[remaining] - direction[chosen], axis=1)
        minimum_chord = np.minimum(minimum_chord, distance)
        minimum_chord[position] = -1.0
    diagnostic_selected = np.asarray(diagnostic, dtype=np.int64)

    selected = np.concatenate((core_selected, diagnostic_selected))
    primary_stratum = labels[selected].tolist()
    inference_role = ["core_stratified"] * len(core_selected) + ["coverage_diagnostic"] * len(diagnostic_selected)
    selected_counts = {name: int(np.sum(labels[core_selected] == name)) for name in core_names}
    weights = {
        name: float(population_counts[name] / selected_counts[name]) for name in core_names
    }
    report = {
        "seed": seed,
        "selection": "five mutually exclusive probability strata plus farthest-point coverage supplement",
        "core_strata": list(core_names),
        "per_core_stratum": per_core_stratum,
        "coverage_supplement": coverage_supplement,
        "population_counts": population_counts,
        "selected_counts": selected_counts,
        "poststratification_weight": weights,
        "thresholds": {
            "known_failure_theta_deg": [50.0, 60.0],
            "known_failure_phi_deg": [30.0, 40.0],
            "high_condition_p90": high_condition_threshold,
            "strong_field_p75_db_relative": strong_threshold,
            "weak_field_p25_db_relative": weak_threshold,
        },
        "inference_rule": (
            "Use only core_stratified rows with poststratification weights for whole-validation "
            "estimates; report coverage_diagnostic rows separately."
        ),
    }
    return selected, primary_stratum, inference_role, report


def _angular_coverage(direction: np.ndarray, pilot_direction: np.ndarray) -> dict[str, float]:
    maximum_dot = np.clip(direction @ pilot_direction.T, -1.0, 1.0).max(axis=1)
    nearest_deg = np.rad2deg(np.arccos(maximum_dot))
    return _quantiles(nearest_deg)


def _write_direction_table(
    path: Path,
    selected: np.ndarray,
    validation_global_index: np.ndarray,
    direction: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    signal_db: np.ndarray,
    condition: np.ndarray,
    primary_stratum: list[str],
    inference_role: list[str],
    selection_report: dict[str, object],
) -> None:
    weight_by_stratum = selection_report["poststratification_weight"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "pilot_id", "validation_local_index", "source_query_index", "theta_deg", "phi_deg",
                "qx", "qy", "qz", "primary_stratum", "inference_role",
                "poststratification_weight", "clean_signal_db_relative", "jacobian_condition",
            )
        )
        for pilot_id, local_index in enumerate(selected):
            stratum = primary_stratum[pilot_id]
            weight = weight_by_stratum[stratum] if inference_role[pilot_id] == "core_stratified" else 0.0
            writer.writerow(
                (
                    pilot_id,
                    int(local_index),
                    int(validation_global_index[local_index]),
                    float(theta[local_index]),
                    float(phi[local_index]),
                    *[float(value) for value in direction[local_index]],
                    stratum,
                    inference_role[pilot_id],
                    weight,
                    float(signal_db[local_index]),
                    float(condition[local_index]),
                )
            )


def _write_command_table(path: Path, codebook: MeasurementCodebook) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("command_id", "split", "kind", "edge_left", "edge_right", "element", "row", "col", "real", "imag", "amplitude", "phase_deg")
        )
        for command_id in range(80):
            split = "train" if command_id < 64 else "validation"
            edge = codebook.edges[command_id]
            for element, value in enumerate(codebook.weights[command_id].numpy()):
                writer.writerow(
                    (
                        command_id,
                        split,
                        codebook.kinds[command_id],
                        "" if edge is None else edge[0],
                        "" if edge is None else edge[1],
                        element,
                        element // 4,
                        element % 4,
                        float(value.real),
                        float(value.imag),
                        float(abs(value)),
                        float(np.rad2deg(np.angle(value))),
                    )
                )


def _write_schedule(path: Path, pilot_count: int, *, repeats: int, seed: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("repeat_id", "session_block", "sequence_in_repeat", "pilot_id", "command_id", "split"))
        for repeat in range(repeats):
            generator = np.random.default_rng(seed + repeat)
            direction_order = generator.permutation(pilot_count)
            sequence = 0
            for pilot_id in direction_order:
                for command_id in generator.permutation(80):
                    writer.writerow(
                        (
                            f"pilot_r{repeat + 1:02d}",
                            "session_A" if repeat < repeats // 2 else "session_B",
                            sequence,
                            int(pilot_id),
                            int(command_id),
                            "train" if command_id < 64 else "validation",
                        )
                    )
                    sequence += 1


def _plot(
    output: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    selected: np.ndarray,
    primary_stratum: list[str],
    inference_role: list[str],
    signal_db: np.ndarray,
    condition: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].scatter(phi, theta, s=3, alpha=0.15, color="gray", label="validation")
    names = sorted(set(primary_stratum))
    roles = np.asarray(inference_role)
    strata = np.asarray(primary_stratum)
    palette = plt.get_cmap("tab10")
    for index, name in enumerate(names):
        mask = (strata == name) & (roles == "core_stratified")
        axes[0].scatter(phi[selected][mask], theta[selected][mask], s=17, label=name, color=palette(index))
    diagnostic = roles == "coverage_diagnostic"
    axes[0].scatter(
        phi[selected][diagnostic],
        theta[selected][diagnostic],
        s=24,
        marker="x",
        linewidths=1.0,
        color="black",
        label="coverage_diagnostic",
    )
    axes[0].set_xlabel("HFSS phi (deg)")
    axes[0].set_ylabel("HFSS theta (deg)")
    axes[0].set_title("Pilot angular coverage")
    axes[0].legend(fontsize=6, loc="best")
    axes[1].hist(signal_db, bins=30, alpha=0.5, density=True, label="validation")
    axes[1].hist(signal_db[selected], bins=20, alpha=0.6, density=True, label="pilot")
    axes[1].set_xlabel("Clean power RMS relative level (dB)")
    axes[1].set_title("Signal-strength coverage")
    axes[1].legend()
    axes[2].hist(condition, bins=30, alpha=0.5, density=True, label="validation")
    axes[2].hist(condition[selected], bins=20, alpha=0.6, density=True, label="pilot")
    axes[2].set_xlabel("Unweighted Jacobian condition")
    axes[2].set_title("Identifiability coverage")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("V2.13 Gate-B hardware pilot design (not measurement evidence)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    precision = {row["repeat_count"]: row for row in report["repeat_precision"]}
    lines = [
        "# V2.13 Gate-B 硬件 pilot 与统计设计",
        "",
        "## 结论",
        "",
        "已冻结 240 个 validation-only 测量方向：200 个分层随机方向用于总体统计，",
        "另加 40 个最远点覆盖方向用于局部失效诊断。该文件是测量计划，不是实测结果；",
        "Gate B 仍为 `ideal-synthetic-only / hardware-blocked`。",
        "",
        "## Pilot 构成",
        "",
        "| 分层 | validation 总数 | 随机抽取 |",
        "|---|---:|---:|",
    ]
    selection = report["selection"]
    for name in selection["core_strata"]:
        lines.append(f"| `{name}` | {selection['population_counts'][name]} | {selection['selected_counts'][name]} |")
    lines.extend(
        [
            "| `coverage_diagnostic` | — | 40 |",
            "",
            f"全 validation 到最近 pilot 方向的角距离 P95 为 `{report['coverage']['nearest_pilot_angle_deg']['p95']:.2f}°`，最大为 `{report['coverage']['nearest_pilot_angle_deg']['maximum']:.2f}°`。",
            "整体统计只能使用 `core_stratified` 行并应用 CSV 中的 poststratification weight；",
            "40 个 coverage diagnostic 方向必须单独报告，不能混入无权总体平均。",
            "",
            "## 重复数统计",
            "",
            "下表给出正态、独立重复假设下，样本标准差的 95% CI 相对因子，以及 30 dB",
            "repeat SNR 时均值功率的 95% CI 半宽。",
            "",
            "| R | 标准差 95% CI 因子 | 30 dB 时均值半宽 | 观测数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for repeat_count in (3, 5, 10, 20):
        row = precision[repeat_count]
        factor = row["standard_deviation_95ci_factor_relative_to_sample_std"]
        observations = report["observation_budget"][str(repeat_count)]
        half_width = row["mean_power_95ci_half_width_percent_by_repeat_snr_db"]["30"]
        lines.append(f"| {repeat_count} | [{factor[0]:.2f}, {factor[1]:.2f}] | ±{half_width:.2f}% | {observations} |")
    lines.extend(
        [
            "",
            "`R=3` 只适合格式/链路冒烟，噪声标准差区间极宽；`R=10` 是本 pilot 的正式",
            "默认值，并拆成两个各 5 次的独立 session block。若 session 间漂移显著，必须",
            "使用分层/混合效应报告，而不能把 10 次当作同分布白噪声简单合并。",
            "",
            "## 执行边界",
            "",
            "- 每个 repeat 内随机化方向顺序和方向内命令顺序；",
            "- 64 条 train commands 恢复 Q，16 条 validation commands 只做 held-out 回放；",
            "- external-test observations 不在计划中；",
            "- 完成后先运行 Gate-B importer/validator，再计算协方差、power NMSE、Jacobian condition、角区和 bootstrap CI；",
            "- 若能在部分方向获得校准 complex reference，才额外报告 operator NMSE。",
            "",
            "当前 synthetic 30 dB 结果只作为接收链设计目标，不作为硬件通过阈值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-truth", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=21330)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    codebook = MeasurementCodebook.load(args.codebook)
    with np.load(args.complex_truth, allow_pickle=False) as archive:
        validation_global_index = np.asarray(archive["query_validation"], dtype=np.int64)
        direction = np.asarray(archive["direction"], dtype=np.float32)[validation_global_index]
        g = np.asarray(archive["g"], dtype=np.complex64)[validation_global_index]
    theta, phi = direction_to_hfss_angles(direction)
    g_torch = torch.from_numpy(g.copy())
    active_commands = codebook.weights[:80]
    clean_power = torch.einsum("qn,cn->qc", g_torch, active_commands).abs().square()
    signal_rms = clean_power.square().mean(dim=1).sqrt().numpy()
    signal_db = 10.0 * np.log10(np.maximum(signal_rms / signal_rms.max(), 1e-30))
    condition = _condition_numbers(g_torch, codebook.weights[:64], args.batch_size)

    selected, primary_stratum, inference_role, selection_report = select_pilot(
        direction,
        theta,
        phi,
        signal_db,
        condition,
        seed=args.seed,
    )
    repeat_report = [repeat_precision(count, (20.0, 30.0, 40.0)) for count in (3, 5, 10, 20)]
    report = {
        "schema": "v213-gate-b-hardware-pilot-design-1",
        "status": "measurement_plan_ready_gate_b_hardware_blocked",
        "external_test_used": False,
        "new_model_trained": False,
        "source_truth": str(args.complex_truth.resolve()),
        "source_split": "query_validation only",
        "validation_population": len(direction),
        "pilot_direction_count": len(selected),
        "command_count": 80,
        "train_command_count": 64,
        "heldout_validation_command_count": 16,
        "recommended_repeat_count": 10,
        "minimum_repeat_count": 3,
        "selection": selection_report,
        "coverage": {
            "nearest_pilot_angle_deg": _angular_coverage(direction, direction[selected]),
            "validation_signal_db_relative": _quantiles(signal_db),
            "pilot_signal_db_relative": _quantiles(signal_db[selected]),
            "validation_jacobian_condition": _quantiles(condition),
            "pilot_jacobian_condition": _quantiles(condition[selected]),
        },
        "repeat_precision": repeat_report,
        "observation_budget": {
            str(repeats): int(len(selected) * 80 * repeats) for repeats in (3, 5, 10, 20)
        },
        "analysis_contract": {
            "whole_population_estimates": "core_stratified_only_with_poststratification_weights",
            "coverage_diagnostics": "report_separately",
            "noise_covariance": "per_direction_diagonal_from_repeats_plus_pooled_relative_cross_command_diagnostic",
            "confidence_interval": "stratified_or_weighted_direction_bootstrap_95_percent",
            "session_effect": "report_session_A_vs_session_B_and_use_mixed_effect_if_drift_is_detected",
            "operator_nmse": "only_if_independent_calibrated_complex_reference_is_available",
        },
        "claim_boundary": "This is a privileged-HFSS-assisted measurement design, not hardware evidence.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_direction_table(
        args.output_dir / "pilot_directions.csv",
        selected,
        validation_global_index,
        direction,
        theta,
        phi,
        signal_db,
        condition,
        primary_stratum,
        inference_role,
        selection_report,
    )
    _write_command_table(args.output_dir / "pilot_commands_long.csv", codebook)
    _write_schedule(args.output_dir / "pilot_schedule_r10.csv", len(selected), repeats=10, seed=args.seed + 100)
    pilot_manifest = {
        "schema": "v213-gate-b-import-manifest-1",
        "dataset_id": "gate-b-hardware-pilot-fill-after-acquisition",
        "source_kind": "hardware_measurement",
        "independent_repeats": True,
        "frequency_hz": 47000000000.0,
        "receiver_channel": "calibrated_single_polarization_power",
        "power_unit": "W",
        "array_id": "fill-array-id",
        "instrument": "fill-receiver-and-detector",
        "calibration_reference_plane": "fill-reference-plane",
        "calibration_id": "fill-calibration-id",
        "source_files": ["fill-measurements-long-csv-path"],
        "generated_from_single_complex_g": False,
        "not_for_training": True,
        "external_test_used": False,
        "pilot_design": "PILOT_SUMMARY.json",
        "schedule": "pilot_schedule_r10.csv",
        "notes": "Ten independent repeats split across session_A and session_B; fill all placeholder fields.",
    }
    (args.output_dir / "PILOT_IMPORT_MANIFEST.template.json").write_text(
        json.dumps(pilot_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "PILOT_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(args.output_dir / "PILOT_STATISTICAL_DESIGN.md", report)
    _plot(
        args.output_dir / "PILOT_COVERAGE.png",
        theta,
        phi,
        selected,
        primary_stratum,
        inference_role,
        signal_db,
        condition,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
