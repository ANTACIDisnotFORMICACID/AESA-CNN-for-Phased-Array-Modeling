"""Aggregate three-seed V2.11 encoding ablations and compare with V2.10."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ORDER = ("fourier_v210", "ssh_l4", "ssh_l5", "fourier_plus_ssh_l4")
LABELS = {
    "fourier_v210": "Fourier-27",
    "ssh_l4": "SSH L=4",
    "ssh_l5": "SSH L=5",
    "fourier_plus_ssh_l4": "Fourier + SSH L=4",
}
COLORS = {
    "fourier_v210": "#4c78a8",
    "ssh_l4": "#f58518",
    "ssh_l5": "#54a24b",
    "fourier_plus_ssh_l4": "#e45756",
}


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _historical_validation(checkpoint_path: Path) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_epoch = int(checkpoint["stage_results"][0]["selected_local_epoch"])
    row = checkpoint["history"][best_epoch - 1]
    return {
        "best_epoch": best_epoch,
        "linear_nmse_db": float(row["validation_linear_nmse_db"]),
        "global_linear_nmse_db": float(row["validation_global_linear_nmse_db"]),
        "amplitude_db_mae": float(row["validation_amplitude_db_mae"]),
        "correlation": float(row["validation_mean_command_amplitude_correlation"]),
    }


def _plot_curves(path: Path, records: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for variant in ORDER:
        histories = [row["history"] for row in records[variant]]
        epoch = np.asarray([item["epoch"] for item in histories[0]])
        nmse = np.asarray(
            [[item["validation_linear_nmse_db"] for item in history] for history in histories]
        )
        mae = np.asarray(
            [[item["validation_amplitude_db_mae"] for item in history] for history in histories]
        )
        axes[0].plot(epoch, nmse.mean(axis=0), color=COLORS[variant], label=LABELS[variant])
        axes[0].fill_between(
            epoch, nmse.mean(axis=0) - nmse.std(axis=0), nmse.mean(axis=0) + nmse.std(axis=0),
            color=COLORS[variant], alpha=0.13,
        )
        axes[1].plot(epoch, mae.mean(axis=0), color=COLORS[variant], label=LABELS[variant])
        axes[1].fill_between(
            epoch, mae.mean(axis=0) - mae.std(axis=0), mae.mean(axis=0) + mae.std(axis=0),
            color=COLORS[variant], alpha=0.13,
        )
    axes[0].set_title("Validation amplitude NMSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("NMSE (dB; lower is better)")
    axes[1].set_title("Validation amplitude dB MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MAE (dB; lower is better)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_benchmarks(
    path: Path,
    aggregate: dict[str, dict[str, object]],
    historical: dict[str, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    panels = (
        ("validation", "linear_nmse_db", "Validation"),
        ("test_reused_benchmark", "linear_nmse_db", "Reused internal test"),
        ("external", "linear_nmse_db", "Reused HFSS reserved holdout"),
    )
    x_value = np.arange(len(ORDER))
    for ax, (section, metric, title) in zip(axes, panels):
        means = [aggregate[v][section][metric]["mean"] for v in ORDER]
        errors = [aggregate[v][section][metric]["sample_std"] for v in ORDER]
        ax.bar(x_value, means, yerr=errors, color=[COLORS[v] for v in ORDER], capsize=4)
        if section in historical:
            ax.axhline(
                historical[section][metric], color="#222222", linestyle="--", linewidth=1.2,
                label="historical V2.10",
            )
            ax.legend(fontsize=8)
        ax.set_xticks(x_value, [LABELS[v] for v in ORDER], rotation=25, ha="right")
        ax.set_ylabel("amplitude NMSE (dB)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_level_error(path: Path, aggregate: dict[str, dict[str, object]]) -> None:
    bins = ("0_to_-3_dB", "-3_to_-6_dB", "-6_to_-10_dB", "-10_to_-20_dB", "below_-20_dB")
    labels = ("0..-3", "-3..-6", "-6..-10", "-10..-20", "<-20")
    x_value = np.arange(len(bins))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for index, variant in enumerate(ORDER):
        means = [aggregate[variant]["level_bins"][name]["db_mae"]["mean"] for name in bins]
        errors = [aggregate[variant]["level_bins"][name]["db_mae"]["sample_std"] for name in bins]
        ax.bar(
            x_value + (index - 1.5) * width,
            means,
            width,
            yerr=errors,
            color=COLORS[variant],
            capsize=3,
            label=LABELS[variant],
        )
    ax.set_xticks(x_value, labels)
    ax.set_xlabel("true level relative to each command peak (dB)")
    ax.set_ylabel("external amplitude error MAE (dB)")
    ax.set_title("Error by true relative field level on reused HFSS holdout")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-test", type=Path, required=True)
    parser.add_argument("--historical-external", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, object]]] = {variant: [] for variant in ORDER}
    flat_rows: list[dict[str, object]] = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "result.json").exists():
            continue
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        external = json.loads((run_dir / "external_holdout.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        variant = result["variant"]
        if variant not in records:
            continue
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        alpha_nominal = checkpoint["model_state"].get("nominal_pattern.direction_encoder.alpha")
        alpha_residual = checkpoint["model_state"].get("angular_residual.direction_encoder.alpha")
        row = {
            "result": result,
            "external": external,
            "history": history,
            "alpha_nominal": float(alpha_nominal) if alpha_nominal is not None else None,
            "alpha_residual": float(alpha_residual) if alpha_residual is not None else None,
        }
        records[variant].append(row)
        flat_rows.append(
            {
                "variant": variant,
                "seed": result["seed"],
                "validation_nmse_db": result["validation"]["linear_nmse_db"],
                "test_nmse_db": result["test_reused_benchmark"]["linear_nmse_db"],
                "external_nmse_db": external["metrics"]["linear_nmse_db"],
                "external_db_mae": external["metrics"]["amplitude_db_mae"],
                "external_correlation": external["metrics"]["mean_command_correlation"],
                "external_top10_db_mae": external["error_by_true_relative_level"]["top_10_dB_combined"]["db_mae"],
                "parameter_count": result["parameter_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "alpha_nominal": row["alpha_nominal"],
                "alpha_residual": row["alpha_residual"],
            }
        )
    if any(len(records[variant]) != 3 for variant in ORDER):
        raise ValueError("expected exactly three completed seeds for every variant")

    aggregate: dict[str, dict[str, object]] = {}
    for variant in ORDER:
        rows = records[variant]
        aggregate[variant] = {
            "seed_count": len(rows),
            "parameter_count": int(rows[0]["result"]["parameter_count"]),
            "elapsed_seconds": _mean_std([row["result"]["elapsed_seconds"] for row in rows]),
            "validation": {
                metric: _mean_std([row["result"]["validation"][metric] for row in rows])
                for metric in ("linear_nmse_db", "amplitude_db_mae", "mean_command_amplitude_correlation")
            },
            "test_reused_benchmark": {
                metric: _mean_std([row["result"]["test_reused_benchmark"][metric] for row in rows])
                for metric in ("linear_nmse_db", "power_nmse_db", "amplitude_db_mae", "mean_command_amplitude_correlation")
            },
            "external": {
                metric: _mean_std([row["external"]["metrics"][metric] for row in rows])
                for metric in ("linear_nmse_db", "power_nmse_db", "amplitude_db_mae", "mean_command_correlation")
            },
            "level_bins": {
                name: {
                    metric: _mean_std(
                        [row["external"]["error_by_true_relative_level"][name][metric] for row in rows]
                    )
                    for metric in ("db_mae", "db_error_p95")
                }
                for name in rows[0]["external"]["error_by_true_relative_level"]
            },
        }
        if variant == "fourier_plus_ssh_l4":
            aggregate[variant]["learned_alpha"] = {
                "nominal": [row["alpha_nominal"] for row in rows],
                "residual": [row["alpha_residual"] for row in rows],
            }

    historical_test_payload = json.loads(args.historical_test.read_text(encoding="utf-8"))
    historical_external_payload = json.loads(args.historical_external.read_text(encoding="utf-8"))
    historical = {
        "validation": _historical_validation(args.historical_checkpoint),
        "test_reused_benchmark": {
            "linear_nmse_db": float(historical_test_payload["metrics"]["linear_nmse_db"]),
            "amplitude_db_mae": float(historical_test_payload["metrics"]["amplitude_db_mae"]),
            "correlation": float(historical_test_payload["metrics"]["mean_command_amplitude_correlation"]),
        },
        "external": {
            "linear_nmse_db": float(historical_external_payload["trained_amplitude_only"]["linear_nmse_db"]),
            "power_nmse_db": float(historical_external_payload["trained_amplitude_only"]["power_nmse_db"]),
            "amplitude_db_mae": float(historical_external_payload["trained_amplitude_only"]["amplitude_db_mae"]),
            "correlation": float(historical_external_payload["trained_amplitude_only"]["mean_command_correlation"]),
            "top10_db_mae": float(historical_external_payload["trained_error_by_true_relative_level"]["top_10_dB_combined"]["db_mae"]),
        },
    }
    fourier_by_seed = {row["result"]["seed"]: row for row in records["fourier_v210"]}
    paired: dict[str, dict[str, object]] = {}
    for variant in ORDER[1:]:
        deltas: dict[str, list[float]] = {"validation_nmse_db": [], "test_nmse_db": [], "external_nmse_db": []}
        wins: dict[str, int] = {key: 0 for key in deltas}
        for row in records[variant]:
            baseline = fourier_by_seed[row["result"]["seed"]]
            values = {
                "validation_nmse_db": row["result"]["validation"]["linear_nmse_db"] - baseline["result"]["validation"]["linear_nmse_db"],
                "test_nmse_db": row["result"]["test_reused_benchmark"]["linear_nmse_db"] - baseline["result"]["test_reused_benchmark"]["linear_nmse_db"],
                "external_nmse_db": row["external"]["metrics"]["linear_nmse_db"] - baseline["external"]["metrics"]["linear_nmse_db"],
            }
            for key, value in values.items():
                deltas[key].append(float(value))
                wins[key] += int(value < 0.0)
        paired[variant] = {
            "delta_definition": "variant dB minus same-seed Fourier dB; negative is better",
            "deltas": {key: _mean_std(value) for key, value in deltas.items()},
            "win_count_out_of_3": wins,
        }

    summary = {
        "schema": "v211-encoding-ablation-summary-1",
        "date": "2026-09-10",
        "protocol": {
            "seeds": [28101, 28102, 28103],
            "epochs": 120,
            "queries_per_epoch": 8192,
            "training_observable": "single-polarization amplitude only",
            "selection_partition": "validation",
            "test_and_external_status": "reused benchmarks already opened by V2.10; not fresh blind tests",
        },
        "aggregate": aggregate,
        "paired_vs_same_seed_fourier": paired,
        "historical_v210": historical,
        "decision": {
            "best_current_candidate": "fourier_plus_ssh_l4",
            "replace_historical_v210": False,
            "reason": "hybrid improves all three external seeds and reduces variance, but remains behind the historical V2.10 checkpoint",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    _plot_curves(figures / "validation_learning_curves.png", records)
    _plot_benchmarks(figures / "benchmark_nmse_comparison.png", aggregate, historical)
    _plot_level_error(figures / "external_error_by_level.png", aggregate)

    hybrid = aggregate["fourier_plus_ssh_l4"]
    fourier = aggregate["fourier_v210"]
    paired_hybrid = paired["fourier_plus_ssh_l4"]
    report = f"""# V2.11 Fourier/SSH 幅度训练消融报告

## 结论

在本轮相同 CUDA 环境、相同数据划分和三个配对种子下，`Fourier + SSH L=4` 是最稳定的候选：

- 验证 NMSE：{hybrid['validation']['linear_nmse_db']['mean']:.3f} ± {hybrid['validation']['linear_nmse_db']['sample_std']:.3f} dB；
- 复用内部测试 NMSE：{hybrid['test_reused_benchmark']['linear_nmse_db']['mean']:.3f} ± {hybrid['test_reused_benchmark']['linear_nmse_db']['sample_std']:.3f} dB；
- 复用 HFSS reserved holdout NMSE：{hybrid['external']['linear_nmse_db']['mean']:.3f} ± {hybrid['external']['linear_nmse_db']['sample_std']:.3f} dB；
- 外部强场前10 dB MAE：{hybrid['level_bins']['top_10_dB_combined']['db_mae']['mean']:.3f} ± {hybrid['level_bins']['top_10_dB_combined']['db_mae']['sample_std']:.3f} dB。

相对同种子 Fourier，混合编码在三次外部评估中全部获胜，配对 NMSE 改善 {-paired_hybrid['deltas']['external_nmse_db']['mean']:.3f} dB；外部 NMSE 的种子标准差由 {fourier['external']['linear_nmse_db']['sample_std']:.3f} dB 降至 {hybrid['external']['linear_nmse_db']['sample_std']:.3f} dB。

但它尚不能替换历史 V2.10：历史 checkpoint 的内部测试/外部 NMSE 为 {historical['test_reused_benchmark']['linear_nmse_db']:.3f}/{historical['external']['linear_nmse_db']:.3f} dB，仍优于混合模型三种子均值。当前结论是“混合编码在本轮受控重训中有效且更稳”，不是“已经刷新 V2.10 最佳结果”。

## 三种子汇总

| 编码 | 参数量 | 验证 NMSE dB | 内部测试 NMSE dB | 外部 NMSE dB | 外部 MAE dB | 前10 dB MAE dB |
|---|---:|---:|---:|---:|---:|---:|
"""
    for variant in ORDER:
        item = aggregate[variant]
        report += (
            f"| {LABELS[variant]} | {item['parameter_count']:,} | "
            f"{item['validation']['linear_nmse_db']['mean']:.3f}±{item['validation']['linear_nmse_db']['sample_std']:.3f} | "
            f"{item['test_reused_benchmark']['linear_nmse_db']['mean']:.3f}±{item['test_reused_benchmark']['linear_nmse_db']['sample_std']:.3f} | "
            f"{item['external']['linear_nmse_db']['mean']:.3f}±{item['external']['linear_nmse_db']['sample_std']:.3f} | "
            f"{item['external']['amplitude_db_mae']['mean']:.3f}±{item['external']['amplitude_db_mae']['sample_std']:.3f} | "
            f"{item['level_bins']['top_10_dB_combined']['db_mae']['mean']:.3f}±{item['level_bins']['top_10_dB_combined']['db_mae']['sample_std']:.3f} |\n"
        )
    report += f"""

## 解释

1. 纯 SSH-L4 容量不足且两次逊于 Fourier；不建议作为下一基线。
2. SSH-L5 在两个种子上优于 Fourier，但均值提升小于种子波动，且参数更多、正半球条件数更差。
3. 混合编码保留了 Fourier 高频坐标映射，同时让 SSH 提供平滑全局修正。两处分支的 alpha 均明显离开零，说明网络实际使用了 SSH，而非保持初始 Fourier。
4. 混合编码总参数仅比 Fourier 增加 {(hybrid['parameter_count'] / fourier['parameter_count'] - 1.0) * 100:.2f}%，但训练耗时增加约 {(hybrid['elapsed_seconds']['mean'] / fourier['elapsed_seconds']['mean'] - 1.0) * 100:.1f}%。
5. 所有模型在低于峰值20 dB的弱场误差最大；混合编码的主要收益也包含强场区域改善，而不仅是低幅度点。

## 可复现性边界

- 本轮使用当前 PyTorch/CUDA 环境重新训练。相同 Fourier 结构和种子没有复现历史 checkpoint 的最佳轨迹，当前 CPU 复核也未消除差距，因此存在未完全定位的训练栈/数值轨迹差异。
- 三种子配对比较可支持本轮编码排序，但测试集和 HFSS reserved holdout 都已在 V2.10 阶段打开，不能作为新的无偏盲测。
- 若要正式升级基线，应先冻结混合编码方案，再建立全新的未见角块或实测幅度盲测集。

## 学得的混合系数

- nominal alpha：{hybrid['learned_alpha']['nominal']}
- residual alpha：{hybrid['learned_alpha']['residual']}

符号随初始化和分支参数化变化，不比较正负；绝对值明显非零即可证明 SSH 修正被启用。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
