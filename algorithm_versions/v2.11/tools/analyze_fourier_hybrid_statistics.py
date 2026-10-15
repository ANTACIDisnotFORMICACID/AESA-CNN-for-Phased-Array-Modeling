from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


SEEDS = (28101, 28102, 28103)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Two-sided paired randomization test over all 2**n sign assignments."""
    observed = abs(mean(differences))
    randomized = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        randomized.append(abs(mean([s * d for s, d in zip(signs, differences)])))
    return sum(value >= observed - 1e-15 for value in randomized) / len(randomized)


def paired_summary(fourier: list[float], hybrid: list[float], lower_better: bool) -> dict:
    differences = [h - f for f, h in zip(fourier, hybrid)]
    n = len(differences)
    delta_mean = mean(differences)
    delta_sd = stdev(differences)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_half_width = t_critical * delta_sd / math.sqrt(n)
    t_result = stats.ttest_rel(hybrid, fourier)
    wilcoxon = stats.wilcoxon(hybrid, fourier, alternative="two-sided", method="exact")
    wins = sum((d < 0.0) if lower_better else (d > 0.0) for d in differences)
    return {
        "n_independent_training_seeds": n,
        "fourier_mean": mean(fourier),
        "hybrid_mean": mean(hybrid),
        "hybrid_minus_fourier_by_seed": differences,
        "mean_difference": delta_mean,
        "sample_sd_difference": delta_sd,
        "mean_difference_95pct_t_ci": [delta_mean - ci_half_width, delta_mean + ci_half_width],
        "paired_cohen_dz": delta_mean / delta_sd,
        "paired_t_test_two_sided_p": float(t_result.pvalue),
        "wilcoxon_exact_two_sided_p": float(wilcoxon.pvalue),
        "exact_sign_flip_two_sided_p": exact_sign_flip_pvalue(differences),
        "wins_out_of_3": wins,
        "lower_is_better": lower_better,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()

    fourier_docs = [
        load_json(root / "runs" / f"fourier_v210_seed{seed}" / "external_holdout.json")
        for seed in SEEDS
    ]
    hybrid_docs = [
        load_json(root / "runs" / f"fourier_plus_ssh_l4_seed{seed}" / "external_holdout.json")
        for seed in SEEDS
    ]

    metric_specs = {
        "linear_nmse_db": True,
        "power_nmse_db": True,
        "amplitude_db_mae": True,
        "amplitude_db_error_p95": True,
        "mean_command_correlation": False,
    }
    overall = {}
    for key, lower_better in metric_specs.items():
        overall[key] = paired_summary(
            [doc["metrics"][key] for doc in fourier_docs],
            [doc["metrics"][key] for doc in hybrid_docs],
            lower_better,
        )

    bin_names = ("0_to_-3_dB", "-3_to_-6_dB", "-6_to_-10_dB", "-10_to_-20_dB", "below_-20_dB")
    by_level = {}
    for bin_name in bin_names:
        by_level[bin_name] = paired_summary(
            [doc["error_by_true_relative_level"][bin_name]["db_mae"] for doc in fourier_docs],
            [doc["error_by_true_relative_level"][bin_name]["db_mae"] for doc in hybrid_docs],
            True,
        )
        by_level[bin_name]["observation_count_per_seed"] = fourier_docs[0]["error_by_true_relative_level"][bin_name]["count"]

    output = {
        "schema": "v211-fourier-hybrid-paired-statistics-1",
        "comparison": "Fourier + SSH L=4 versus Fourier-27",
        "primary_independent_unit": "training seed",
        "seeds": list(SEEDS),
        "external_observations_per_seed": fourier_docs[0]["metrics"]["observation_count"],
        "inference_warning": "direction-command observations are correlated and are not treated as independent replicates",
        "overall": overall,
        "error_by_true_relative_level": by_level,
    }
    (root / "fourier_hybrid_statistics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows = []
    for name, item in by_level.items():
        reduction = 100.0 * (item["fourier_mean"] - item["hybrid_mean"]) / item["fourier_mean"]
        ci = item["mean_difference_95pct_t_ci"]
        rows.append(
            f"| {name.replace('_to_', '…').replace('_dB', '').replace('below_', '<')} "
            f"| {item['observation_count_per_seed']:,} | {fmt(item['fourier_mean'])} | {fmt(item['hybrid_mean'])} "
            f"| {fmt(item['mean_difference'])} | {fmt(reduction, 1)}% | {item['wins_out_of_3']}/3 "
            f"| [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(item['exact_sign_flip_two_sided_p'])} |"
        )

    nmse = overall["linear_nmse_db"]
    mae = overall["amplitude_db_mae"]
    corr = overall["mean_command_correlation"]
    report = f"""# Fourier 与 Fourier+SSH L=4 配对统计分析

## 统计口径

- 对比对象：相同数据划分、训练轮数和种子的 Fourier-27 与 Fourier+SSH L=4。
- 独立重复单位：3 个训练种子（28101、28102、28103）。
- 每个种子在复用 HFSS reserved holdout 上有 {output['external_observations_per_seed']:,} 个方向-命令观测；这些观测相关，不能替代独立训练重复。
- 主非参数检验：精确配对符号翻转检验。n=3 时，即使三个种子全部同向改善，双侧 p 值下限仍为 0.25。

## 总体结果

| 指标 | Fourier 均值 | 混合均值 | 混合-Fourier | 95% t 区间 | Cohen dz | 获胜种子 | 精确双侧 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 线性幅度 NMSE (dB) | {fmt(nmse['fourier_mean'])} | {fmt(nmse['hybrid_mean'])} | {fmt(nmse['mean_difference'])} | [{fmt(nmse['mean_difference_95pct_t_ci'][0])}, {fmt(nmse['mean_difference_95pct_t_ci'][1])}] | {fmt(nmse['paired_cohen_dz'])} | {nmse['wins_out_of_3']}/3 | {fmt(nmse['exact_sign_flip_two_sided_p'])} |
| 幅度 dB MAE | {fmt(mae['fourier_mean'])} | {fmt(mae['hybrid_mean'])} | {fmt(mae['mean_difference'])} | [{fmt(mae['mean_difference_95pct_t_ci'][0])}, {fmt(mae['mean_difference_95pct_t_ci'][1])}] | {fmt(mae['paired_cohen_dz'])} | {mae['wins_out_of_3']}/3 | {fmt(mae['exact_sign_flip_two_sided_p'])} |
| 命令平均相关系数 | {fmt(corr['fourier_mean'], 5)} | {fmt(corr['hybrid_mean'], 5)} | {fmt(corr['mean_difference'], 5)} | [{fmt(corr['mean_difference_95pct_t_ci'][0], 5)}, {fmt(corr['mean_difference_95pct_t_ci'][1], 5)}] | {fmt(corr['paired_cohen_dz'])} | {corr['wins_out_of_3']}/3 | {fmt(corr['exact_sign_flip_two_sided_p'])} |

## 不同真实场强区间的幅度 dB MAE

| 相对峰值区间 (dB) | 每种子观测数 | Fourier | 混合 | 差值 | 相对下降 | 获胜种子 | 95% t 区间 | 精确双侧 p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 统计解释

混合编码在总体 NMSE、幅度 MAE、相关系数以及全部五个场强区间上均为 3/3 同向改善，方向一致性很强。总体外部 NMSE 改善 {fmt(-nmse['mean_difference'])} dB，Cohen dz 的绝对值为 {fmt(abs(nmse['paired_cohen_dz']))}，属于较大的配对效应。

但 n=3 无法提供传统 0.05 水平的非参数显著性：精确双侧 p=0.25，且 t 区间较宽。正确结论是“观察到稳定、幅度较大的工程改善”，而不是“已经完成统计显著性证明”。若要验证显著性，至少应增加独立训练种子，并优先建立未被模型选择使用过的新角块/实测幅度盲测集。
"""
    (root / "STATISTICAL_ANALYSIS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
