"""Fit training-only SSH statistics and audit scale/correlation on q_x > 0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_v211(v211_src: Path):
    sys.path.insert(0, str(v211_src))
    from v211.models import DirectionFourierEncoder, real_spherical_harmonics, ssh_mode_labels

    return DirectionFourierEncoder, real_spherical_harmonics, ssh_mode_labels


def _top_correlations(correlation: np.ndarray, labels: list[str], count: int = 20) -> list[dict[str, object]]:
    pairs: list[tuple[float, int, int]] = []
    for i_value in range(correlation.shape[0]):
        for j_value in range(i_value + 1, correlation.shape[1]):
            pairs.append((abs(float(correlation[i_value, j_value])), i_value, j_value))
    pairs.sort(reverse=True)
    return [
        {
            "abs_correlation": value,
            "mode_a": labels[i_value],
            "mode_b": labels[j_value],
            "signed_correlation": float(correlation[i_value, j_value]),
        }
        for value, i_value, j_value in pairs[:count]
    ]


def _plot_scale(path: Path, labels: list[str], mean: np.ndarray, std: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> None:
    x_value = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    axes[0].fill_between(x_value, minimum, maximum, color="#8bb8d8", alpha=0.45, label="train min..max")
    axes[0].plot(x_value, mean, color="#1f77b4", linewidth=1.2, label="train mean")
    axes[0].set_ylabel("raw real-SSH value")
    axes[0].set_title("Positive-hemisphere training statistics by SSH channel")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(x_value, std, color="#d62728", linewidth=1.4)
    axes[1].set_ylabel("population standard deviation")
    axes[1].set_xlabel("degree-major SSH channel")
    axes[1].grid(True, alpha=0.25)
    tick_step = max(1, len(labels) // 16)
    tick_index = x_value[::tick_step]
    axes[1].set_xticks(tick_index, [labels[index] for index in tick_index], rotation=55, ha="right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_correlation(path: Path, correlation: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = ax.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    tick_step = max(1, len(labels) // 12)
    tick_index = np.arange(0, len(labels), tick_step)
    ax.set_xticks(tick_index, [labels[index] for index in tick_index], rotation=55, ha="right")
    ax.set_yticks(tick_index, [labels[index] for index in tick_index])
    ax.set_title("Empirical SSH correlation on training directions (constant mode removed)")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--v211-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if args.degree < 1 or args.degree > 30:
        raise ValueError("degree must be in [1,30]")

    DirectionFourierEncoder, real_spherical_harmonics, ssh_mode_labels = _load_v211(args.v211_src.resolve())
    direction = np.load(args.dataset_dir / "direction.npy").astype(np.float64)
    train_index = np.load(args.split_dir / "query_train.npy").astype(np.int64)
    train_direction = direction[train_index]
    norms = np.linalg.norm(train_direction, axis=-1)
    if np.any(train_direction[:, 0] <= 0):
        raise ValueError("training split is not strictly q_x > 0")
    if not np.allclose(norms, 1.0, atol=2e-6):
        raise ValueError("training directions are not unit vectors")

    torch_direction = torch.from_numpy(train_direction).to(torch.float64)
    with torch.no_grad():
        ssh = real_spherical_harmonics(torch_direction, args.degree).cpu().numpy()
        fourier = DirectionFourierEncoder(4).to(torch.float64)(torch_direction[None])[0].cpu().numpy()
    labels = ssh_mode_labels(args.degree)
    mean = ssh.mean(axis=0)
    std = ssh.std(axis=0)
    minimum = ssh.min(axis=0)
    maximum = ssh.max(axis=0)
    active = std > args.epsilon
    standardized = (ssh[:, active] - mean[active]) / (std[active] + args.epsilon)
    correlation = np.corrcoef(standardized, rowvar=False)
    active_labels = [label for label, keep in zip(labels, active) if keep]
    eigenvalues = np.linalg.eigvalsh(correlation)
    top = _top_correlations(correlation, active_labels)
    degree_diagnostics: dict[str, dict[str, float | int]] = {}
    for candidate_degree in sorted({4, 5, args.degree}):
        if candidate_degree > args.degree:
            continue
        active_count = (candidate_degree + 1) ** 2 - 1
        candidate_correlation = correlation[:active_count, :active_count]
        candidate_eigenvalues = np.linalg.eigvalsh(candidate_correlation)
        candidate_off_diagonal = np.abs(
            candidate_correlation - np.eye(active_count, dtype=candidate_correlation.dtype)
        )
        degree_diagnostics[f"l{candidate_degree}"] = {
            "active_channel_count": active_count,
            "max_abs_off_diagonal": float(candidate_off_diagonal.max()),
            "correlation_eigenvalue_min": float(candidate_eigenvalues.min()),
            "correlation_eigenvalue_max": float(candidate_eigenvalues.max()),
            "condition_number": float(
                candidate_eigenvalues.max() / max(candidate_eigenvalues.min(), 1e-15)
            ),
        }

    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"ssh_train_stats_l{args.degree}.npz",
        degree=np.asarray(args.degree, dtype=np.int64),
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        train_query_indices=train_index,
    )
    np.save(output / f"ssh_train_correlation_l{args.degree}.npy", correlation)
    with (output / "ssh_channel_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "mean", "std", "min", "max", "constant_on_train_domain"])
        for row in zip(labels, mean, std, minimum, maximum, ~active):
            writer.writerow(row)

    fourier_std = fourier.std(axis=0)
    summary = {
        "schema": "v211-ssh-training-statistics-1",
        "date": "2026-09-10",
        "domain": "strict q_x > 0 (+X boresight)",
        "statistics_scope": "query_train.npy only; validation/test excluded",
        "train_query_count": int(train_index.size),
        "source_query_count": int(direction.shape[0]),
        "ssh_degree": args.degree,
        "ssh_channel_count": int(ssh.shape[1]),
        "constant_channel_count": int((~active).sum()),
        "normalization_epsilon": args.epsilon,
        "ssh_raw": {
            "global_min": float(ssh.min()),
            "global_max": float(ssh.max()),
            "active_channel_std_min": float(std[active].min()),
            "active_channel_std_max": float(std[active].max()),
            "max_abs_channel_mean": float(np.max(np.abs(mean))),
        },
        "ssh_standardized": {
            "active_channel_mean_max_abs": float(np.max(np.abs(standardized.mean(axis=0)))),
            "active_channel_std_min": float(standardized.std(axis=0).min()),
            "active_channel_std_max": float(standardized.std(axis=0).max()),
            "global_min": float(standardized.min()),
            "global_max": float(standardized.max()),
        },
        "positive_hemisphere_correlation": {
            "active_channel_count": int(active.sum()),
            "max_abs_off_diagonal": float(top[0]["abs_correlation"]),
            "correlation_eigenvalue_min": float(eigenvalues.min()),
            "correlation_eigenvalue_max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / max(eigenvalues.min(), 1e-15)),
            "top_pairs": top,
            "by_degree_prefix": degree_diagnostics,
        },
        "v210_fourier_control": {
            "channel_count": int(fourier.shape[1]),
            "global_min": float(fourier.min()),
            "global_max": float(fourier.max()),
            "channel_std_min": float(fourier_std.min()),
            "channel_std_max": float(fourier_std.max()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_scale(figures / "ssh_channel_scale.png", labels, mean, std, minimum, maximum)
    _plot_correlation(figures / "ssh_train_correlation.png", correlation, active_labels)

    strongest = top[0]
    report = f"""# V2.11 SSH 输入尺度与正半球相关性审计

## 数据边界

- 仅使用训练查询索引，共 {train_index.size} 个方向；验证集和测试集未参与统计。
- 所有训练方向均满足严格 `q_x > 0`。
- 实 SSH 采用 degree-major、`m=-l..l` 排列，本次拟合到 L={args.degree}，共 {(args.degree + 1) ** 2} 个通道。

## 输入尺度

- V2.10 Fourier：范围 [{fourier.min():.6f}, {fourier.max():.6f}]，逐通道标准差范围 [{fourier_std.min():.6f}, {fourier_std.max():.6f}]。
- SSH 原值：范围 [{ssh.min():.6f}, {ssh.max():.6f}]，非常量通道标准差范围 [{std[active].min():.6f}, {std[active].max():.6f}]。
- 使用训练集 `mu_lm/std_lm` 标准化后，非常量通道标准差范围 [{standardized.std(axis=0).min():.6f}, {standardized.std(axis=0).max():.6f}]。
- `Y_0_0` 在任意方向恒定，标准化后为零；Linear 层偏置承担等价常量自由度。

## 正半球相关性

- 标准化后最大非对角绝对相关系数为 {strongest['abs_correlation']:.6f}：`{strongest['mode_a']}` 与 `{strongest['mode_b']}`。
- 经验相关矩阵条件数为 {summary['positive_hemisphere_correlation']['condition_number']:.6e}。
- 首轮候选的条件数：L=4 为 {degree_diagnostics['l4']['condition_number']:.6e}，L=5 为 {degree_diagnostics['l5']['condition_number']:.6e}；高阶前缀的病态性明显增强。
- 因此保留 `Linear K->27`：它既统一 CNN 输入通道，又允许在 `q_x>0` 上重新组合相关模态。

## 实验设置

- `fourier`：原 V2.10 的 27 通道严格对照。
- `ssh_l4`：25 个标准化 SSH 通道，经 Linear 25->27。
- `ssh_l5`：36 个标准化 SSH 通道，经 Linear 36->27。
- `fourier_plus_ssh_l4`：`Fourier + alpha * Linear(SSH)`，`alpha=0` 初始化，初始输出与 V2.10 完全一致。

物理 VSH 的 L=30 是复矢量远场带宽判据，不直接作为 CNN 的 SSH 输入阶数；SSH 编码先比较 L=4/5，再根据验证集结果扩阶。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
