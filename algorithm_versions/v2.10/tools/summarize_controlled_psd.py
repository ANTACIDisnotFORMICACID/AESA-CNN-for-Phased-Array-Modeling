"""Create the human-readable V2.10 comparison report and compact figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v29-report", type=Path, required=True)
    args = parser.parse_args()
    report = read(args.root / "final_report.json")
    protocol = read(args.root / "protocol.json")
    development = read(args.root / "development_results.json")
    v29 = read(args.v29_report)
    aggregate = report["aggregate"]

    v29_forward = v29["forward"]
    v29_summary = {
        "amplitude_nmse_db_mean": float(np.mean([row["amplitude_nmse_db"] for row in v29_forward])),
        "amplitude_nmse_db_seed_std": float(v29["seed_nmse_std_db"]),
        "all_mae_db_mean": float(np.mean([row["bins"]["all"]["mae_db"] for row in v29_forward])),
        "strong_mae_db_mean": float(np.mean([row["bins"]["strong"]["mae_db"] for row in v29_forward])),
        "regret_median_mean": float(v29["targets"]["phase"]["median_regret"]),
        "regret_p95_mean": float(v29["targets"]["phase"]["p95_regret"]),
        "bottleneck_over_p95_db_mean": float(v29["targets"]["phase"]["bottleneck_over_p95_db"]),
    }
    delta_b = {key: aggregate["B10"][key] - aggregate["A10"][key] for key in aggregate["A10"]}
    delta_c = {key: aggregate["C10"][key] - aggregate["A10"][key] for key in aggregate["A10"]}
    summary = {
        "selected_recipe": report["selection"]["selected_recipe"],
        "representative_seed": report["selection"]["representative_seed"],
        "v29_frozen": v29_summary,
        "v210": aggregate,
        "B10_minus_A10": delta_b,
        "C10_minus_A10": delta_c,
        "acceptance": report["acceptance"],
        "training_seconds_total": float(sum(row["seconds"] for row in development)),
        "all_runs_equal_updates": report["selection"]["all_runs_equal_updates"],
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = ["V2.9 C", "A10", "B10", "C10"]
    values = [v29_summary, aggregate["A10"], aggregate["B10"], aggregate["C10"]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    plots = [
        ("strong_mae_db_mean", "Strong-field MAE", "dB", False),
        ("amplitude_nmse_db_seed_std", "Seed NMSE std", "dB", False),
        ("regret_p95_mean", "Phase-only regret P95", "%", True),
        ("bottleneck_over_p95_db_mean", "Bottleneck overestimate P95", "dB", False),
    ]
    colors = ["#6c757d", "#2c7fb8", "#41ab5d", "#d95f0e"]
    for axis, (key, title, unit, percent) in zip(axes, plots):
        y = [item[key] * (100 if percent else 1) for item in values]
        axis.bar(labels, y, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=25)
        for index, value in enumerate(y):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("AESA-CNN V2.10 controlled same-state comparison")
    fig.savefig(args.root / "comparison.png", dpi=170)
    plt.close(fig)

    rows = []
    for label, item in zip(labels, values):
        q_value = "—" if label == "V2.9 C" else f"{item['q_nmse_db_mean']:.3f}"
        success = "—" if label == "V2.9 C" else f"{100 * item['success_regret_le_10pct_mean']:.2f}%"
        rows.append(
            f"|{label}|{item['amplitude_nmse_db_mean']:.3f}|{item['amplitude_nmse_db_seed_std']:.3f}|"
            f"{q_value}|{item['all_mae_db_mean']:.3f}|{item['strong_mae_db_mean']:.3f}|"
            f"{100 * item['regret_median_mean']:.2f}%|{100 * item['regret_p95_mean']:.2f}%|"
            f"{item['bottleneck_over_p95_db_mean']:.3f}|{success}|"
        )
    run_rows = []
    for row in report["runs"]:
        run_rows.append(
            f"|{row['recipe']}|{row['seed']}|{row['forward']['amplitude_nmse_db']:.3f}|"
            f"{row['forward']['q_nmse_db']:.3f}|{row['forward']['bins']['strong']['mae_db']:.3f}|"
            f"{100 * row['trajectory']['p95_regret']:.2f}%|"
            f"{row['trajectory']['bottleneck_over_p95_db']:.3f}|"
            f"{100 * row['trajectory']['success_regret_le_10pct']:.2f}%|"
        )
    content = f"""# V2.10 受控 PSD 对照实验结果

## 结论

本轮冻结选择为 **A10 / seed {report['selection']['representative_seed']}**，即迁移后的 V2.9
Cartesian 横向等效场结构。B10 的低秩 PSD 表示没有在相同数据、seed、更新次数和优化预算下
稳定胜过 A10；C10 的乐观高估尾部 CVaR 代理导致明显的整体精度和四点尾部退化。因此本轮
**不替换 V2.9 的结构基线，也不把 C10 的损失写入后续默认配方**。

这是一项同一 47 GHz HFSS 状态上的受控回顾性实验，不是新物理状态盲测。V2.9 的 holdout
历史上已经打开；V2.10 代码仍做到训练阶段只读取 train/validation，冻结后才读取复制的
calibration/holdout，但不能把这种程序隔离重新命名为外部盲测。

## 实验合同

- 三个方法都使用 seed 21011/21012/21013，每组 3,600 epoch、每 epoch 4 次更新，共 14,400 次更新；
- A10 28,854 参数；B10/C10 28,420 参数；均低于 V2.9 的 34,973 参数预算；
- 训练和验证数据是 V2.9 `vector_beam_20260905` 四个分区的字节级副本，SHA-256 见 `protocol.json`；
- validation 按平均 phase-only regret P95 选型，并要求幅度 NMSE 不劣于 A10 超过 0.3 dB；
- holdout 每个模型使用 8 起点/160 步，HFSS 数值参考使用 32 起点/160 步；
- 总 GPU 训练时间约 {summary['training_seconds_total'] / 60:.1f} 分钟，不含测试和评估。

## 汇总对比

|模型|幅度 NMSE (dB)|seed 标准差 (dB)|Q-NMSE (dB)|全场 MAE (dB)|强场 MAE (dB)|regret 中位|regret P95|瓶颈高估 P95 (dB)|regret≤10%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

说明：V2.9 的前向指标为其三个冻结 C seed 的均值，而四点指标只来自冻结代表 seed 29513；
V2.10 行是三个 seed 的均值。两者训练方案和 seed 不完全相同，因此只能作工程基线对照，不能
把 A10 的 seed 标准差降低解释成结构性因果提升。

## B10 分析

相对 A10，B10 的 Q-NMSE 改善 {abs(delta_b['q_nmse_db_mean']):.3f} dB，强场 MAE 改善
{abs(delta_b['strong_mae_db_mean']):.3f} dB，瓶颈高估 P95 改善
{abs(delta_b['bottleneck_over_p95_db_mean']):.3f} dB；但总体幅度 NMSE 退化
{delta_b['amplitude_nmse_db_mean']:.3f} dB，全场 MAE 增加 {delta_b['all_mae_db_mean']:.3f} dB，
四点 regret P95 增加 {100 * delta_b['regret_p95_mean']:.2f} 个百分点，regret≤10% 成功率下降
{abs(100 * delta_b['success_regret_le_10pct_mean']):.2f} 个百分点，且 seed 标准差 0.273 dB 超过
0.25 dB 门槛。结论是：**PSD 物理输出可行，但当前 rank-2 因子和 0.25 权重的 Q 监督没有形成
净收益。**

## C10 失败分析

C10 相对 A10 的幅度 NMSE 退化 {delta_c['amplitude_nmse_db_mean']:.3f} dB，强场 MAE 增加
{delta_c['strong_mae_db_mean']:.3f} dB，四点 regret P95 增加
{100 * delta_c['regret_p95_mean']:.2f} 个百分点，瓶颈高估 P95 反而增加
{delta_c['bottleneck_over_p95_db_mean']:.3f} dB。当前 CVaR 项只处罚固定 teacher 权值上的正向误差，
模型可以通过改变整体标定和误差分布规避代理目标；它也没有对“模型自己优化出的权值”做可微
闭环训练。下一轮若继续，应改为对称的轨迹级误差 + 校准残差建模，或周期性冻结挖掘命令后
重新建立严格控制组，而不是继续增大单侧权重。

## 每个 seed 的保留结果

|方法|seed|幅度 NMSE (dB)|Q-NMSE (dB)|强场 MAE (dB)|regret P95|瓶颈高估 P95 (dB)|regret≤10%|
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(run_rows)}

## 门槛与决策

- 通过：强场 MAE ≤0.25 dB；冻结 A10 为 0.227 dB。
- 通过：三 seed NMSE 标准差 ≤0.25 dB；A10 为 0.177 dB。
- 未通过：瓶颈高估 P95 ≤0.5 dB；A10 为 0.937 dB。
- 未通过：四点最小增益 regret P95 ≤10%；A10 为 18.77%。

因此 V2.10 仍是研究候选而不是验收版本。下一项高价值工作不是增加网络宽度，而是取得新的
频率/公差/温度/安装或实测状态，并把轨道、站点、接收机、噪声和 SNR 门槛填入任务合同。
在此之前，继续在同一状态上调 loss 只能回答内部工程问题，不能回答卫星链路外推问题。

## 产物

- `protocol.json`：冻结配置、数据来源、数据/代码哈希和参数量；
- `selection_frozen.json`：仅由 validation 得出的选择；
- `final_report.json`：九个 checkpoint 的完整保留指标；
- `summary.json`：本报告使用的紧凑机器可读摘要；
- `comparison_arrays.npz`：逐帧 regret、误差和最小增益；
- `comparison.png`：关键指标图；
- `runs/*/best.pt`、`history.json`、`result.json`：九组训练产物。
"""
    write(args.root / "RESULTS.md", content)


if __name__ == "__main__":
    main()
