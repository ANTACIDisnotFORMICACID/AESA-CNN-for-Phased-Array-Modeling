"""Build human-readable tables, figures, and independent consistency checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
VERSION_ROOT = HERE.parents[3]
FRONT = VERSION_ROOT / "artifacts" / "front_hemisphere_20260906"
FREQUENCY_HZ = 47e9
C0 = 299_792_458.0
LAMBDA_M = C0 / FREQUENCY_HZ
DY_M = 6.8e-3
DZ_M = 15e-3


def positions() -> np.ndarray:
    y = (np.arange(4) - 1.5) * DY_M
    z = (np.arange(4) - 1.5) * DZ_M
    return np.asarray([(0.0, yy, zz) for zz in z for yy in y])


def factor(q: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    phase = np.exp(-1j * 2 * np.pi / LAMBDA_M * np.einsum("...d,nd->...n", q, positions()))
    value = np.zeros(q.shape[:-1] + (3, 16), dtype=np.complex64)
    value[..., 0, :] = amplitude[..., None] * phase
    return value


def main() -> None:
    data = pd.read_csv(HERE / "paired_cases.csv")
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    with np.load(HERE / "ideal_o3_raw_results.npz", allow_pickle=False) as stored:
        raw = {key: stored[key] for key in stored.files}

    verification = {
        "rows": int(len(data)),
        "expected_rows": 1680,
        "models": sorted(data.model.unique().tolist()),
        "splits": sorted(data.split.unique().tolist()),
        "checks": {},
    }
    verification["checks"]["paired_row_count"] = len(data) == 1680
    max_magnitude_error = 0.0
    max_gain_error = 0.0
    boresight_amplitude = float(summary["boresight_calibration"]["mean_embedded_element_field_norm"])
    for split in ("validation", "holdout"):
        suite_path = FRONT / f"{split}_front_suite.npz"
        with np.load(suite_path, allow_pickle=False) as suite:
            q = suite["q"].astype(np.float64)
            hfss_h = suite["h"].astype(np.complex64)
        envelope = np.sqrt(np.mean(np.sum(np.abs(hfss_h) ** 2, axis=-2), axis=-1))
        for model in ("common_element_envelope", "isotropic_boresight_calibrated"):
            weight = raw[f"{split}_{model}_weight"]
            stored_gain = raw[f"{split}_{model}_gain"]
            amplitude = envelope if model == "common_element_envelope" else np.full(q.shape[:-1], boresight_amplitude)
            ideal_h = factor(q, amplitude)
            recalculated = np.sum(np.abs(np.einsum("bkrn,bn->bkr", ideal_h, weight)) ** 2, axis=-1) / 60.0
            max_gain_error = max(max_gain_error, float(np.max(np.abs(recalculated - stored_gain))))
            max_magnitude_error = max(max_magnitude_error, float(np.max(np.abs(np.abs(weight) - 0.25))))
    verification["max_weight_magnitude_error"] = max_magnitude_error
    verification["max_recomputed_linear_gain_error"] = max_gain_error
    verification["checks"]["constant_modulus"] = max_magnitude_error < 1e-6
    verification["checks"]["gain_recomputation"] = max_gain_error < 2e-5
    verification["checks"]["same_target_rows"] = bool(
        data.groupby(["split", "case_index"])[[f"q{i}_{axis}" for i in range(1, 5) for axis in "xyz"]]
        .nunique().to_numpy().max() == 1
    )
    verification["all_checks_passed"] = all(verification["checks"].values())
    (HERE / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    records = []
    for (split, model), group in data.groupby(["split", "model"], sort=False):
        delta = group.ideal_gmin_db - group.hfss_gmin_db
        records.append({
            "split": split,
            "model": model,
            "cases": len(group),
            "hfss_gmin_p5_db": np.percentile(group.hfss_gmin_db, 5),
            "hfss_gmin_median_db": np.median(group.hfss_gmin_db),
            "ideal_gmin_p5_db": np.percentile(group.ideal_gmin_db, 5),
            "ideal_gmin_median_db": np.median(group.ideal_gmin_db),
            "paired_delta_median_db": np.median(delta),
            "ideal_higher_fraction": np.mean(delta > 0),
            "ideal_at_least_3db_higher_fraction": np.mean(delta >= 3),
            "hfss_ge5_fraction": np.mean(group.hfss_gmin_db >= 5),
            "ideal_ge5_fraction": np.mean(group.ideal_gmin_db >= 5),
            "hfss_ge8_fraction": np.mean(group.hfss_gmin_db >= 8),
            "ideal_ge8_fraction": np.mean(group.ideal_gmin_db >= 8),
            "hfss_ge10_fraction": np.mean(group.hfss_gmin_db >= 10),
            "ideal_ge10_fraction": np.mean(group.ideal_gmin_db >= 10),
            "hfss_solo_utilization_median_db": np.median(group.hfss_min_solo_utilization_db),
            "ideal_solo_utilization_median_db": np.median(group.ideal_min_solo_utilization_db),
        })
    pd.DataFrame(records).to_csv(HERE / "summary_table.csv", index=False, encoding="utf-8-sig")

    holdout = data[data.split == "holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    names = ["HFSS", "Common-element ideal", "Isotropic ideal"]
    common = holdout[holdout.model == "common_element_envelope"]
    isotropic = holdout[holdout.model == "isotropic_boresight_calibrated"]
    axes[0].boxplot([common.hfss_gmin_db, common.ideal_gmin_db, isotropic.ideal_gmin_db], tick_labels=names, showfliers=False)
    axes[0].set_ylabel("Optimized four-point minimum gain (dB)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].boxplot(
        [common.ideal_gmin_db - common.hfss_gmin_db, isotropic.ideal_gmin_db - isotropic.hfss_gmin_db],
        tick_labels=["Common - HFSS", "Isotropic - HFSS"], showfliers=False,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Paired gain change (dB)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.savefig(HERE / "holdout_paired_comparison.png", dpi=180)
    plt.close(fig)

    holdout_summary = summary["results"]["holdout"]
    common_summary = holdout_summary["common_element_envelope"]
    iso_summary = holdout_summary["isotropic_boresight_calibrated"]
    common_row = next(row for row in records if row["split"] == "holdout" and row["model"] == "common_element_envelope")
    iso_row = next(row for row in records if row["split"] == "holdout" and row["model"] == "isotropic_boresight_calibrated")
    examples = []
    for case in (81, 189):
        c = common[common.case_index == case].iloc[0]
        i = isotropic[isotropic.case_index == case].iloc[0]
        examples.append((case, c.stratum, c.hfss_gmin_db, c.ideal_gmin_db, i.ideal_gmin_db))

    report = f"""# 理想4x4阵列与HFSS四目标优化控制实验

## 结论

在完全相同的840组四点、恒模权值、总归一化输入功率、增益公式和O3预算下：

1. **仅去除端口间差异与非理想相位结构，提升很小。** 保留HFSS方向相关的共同单元包络时，holdout的四点最低增益中位数由 `{common_summary['hfss_gmin_db']['median']:.3f} dB` 提高到 `{common_summary['ideal_gmin_db']['median']:.3f} dB`，成对中位提升仅 `{common_summary['paired_delta_db']['median']:.3f} dB`，95% bootstrap区间为 `[{common_summary['paired_median_delta_95pct_bootstrap_ci_db'][0]:.3f}, {common_summary['paired_median_delta_95pct_bootstrap_ci_db'][1]:.3f}] dB`。没有案例提高3 dB以上。
2. **去除方向相关的单元扫描衰减后，提升非常显著。** 各向同性理想模型的holdout中位数为 `{iso_summary['ideal_gmin_db']['median']:.3f} dB`，相对同位置HFSS成对中位提升 `{iso_summary['paired_delta_db']['median']:.3f} dB`；`{100*iso_summary['fraction_ideal_higher_by_at_least_3db']:.2f}%` 的案例至少提高3 dB。
3. 因而，当前低值**不像主要由端口间互耦畸变或O3失效造成**；更主要的差异来自HFSS中共同存在的方向相关单元响应/大角度扫描衰减。阵列几何和四点共用一套相位仍造成约5 dB的多目标共享损失，但在本批四点上并未使理想各向同性模型出现HFSS那样的低绝对增益。
4. 这仍不能把第2项全部严格归因于“孤立单元方向图”：HFSS的平均嵌入单元包络本身可能包含平均互耦、失配和有源扫描效应。要单独测出互耦贡献，还需要同几何的无互耦全波模型或S参数/嵌入方向图去耦实验。

## 控制变量

- 阵列：4×4平面阵，法向为 `+x`，列沿 `+y`、间距6.80 mm，行沿 `+z`、间距15 mm；47 GHz下分别为 `{DY_M/LAMBDA_M:.3f}λ` 和 `{DZ_M/LAMBDA_M:.3f}λ`。
- 权值：16个阵元均为 `|w_n|=1/4`，因此 `||w||²=1`；仅调整相位。
- 目标：`max min_i G_i(w)`，四点共用同一组权值，`G=||Hw||²/60`。
- 优化器：与原HFSS结果一致的O3，即O1的8起点×160步Adam、40轮坐标候选，以及32起点×320步Adam soft-min；学习率0.065，温度1.2降至0.12，固定随机种子。
- 样本：validation 240组、holdout 600组，原有 `clustered/moderate/wide/grazing` 各占四分之一；CNN不参与。

## 两种理想模型

对方向单位向量 `q` 和阵元位置 `r_n`，理想阵列流形为

`a_n(q)=exp[-j(2π/λ) q·r_n]`。

### 共同单元包络模型

从同一HFSS目标方向计算

`e_rms(q)=sqrt((1/16) sum_n ||h_n^HFSS(q)||²)`，

再令全部阵元为 `h_n^ideal(q)=e_rms(q)a_n(q)`。它保留方向相关的平均单元强度和HFSS增益尺度，但删除端口间嵌入方向图差异、极化差异和非理想相位结构。

### 各向同性模型

把 `e_rms` 固定为最接近阵面法向的HFSS平均端口场强，再只保留几何相位。它同时删除方向扫描衰减，因此是“同孔径、同功率、理想无互耦各向同性阵元”的对照，不是对真实天线绝对增益的直接预测。

## 主要数值（holdout 600组）

|模型|Gmin P5|Gmin中位数|相对HFSS成对中位变化|达到5 dB|达到8 dB|达到10 dB|
|---|---:|---:|---:|---:|---:|---:|
|HFSS|{common_summary['hfss_gmin_db']['p5']:.3f}|{common_summary['hfss_gmin_db']['median']:.3f}|—|{100*common_row['hfss_ge5_fraction']:.2f}%|{100*common_row['hfss_ge8_fraction']:.2f}%|{100*common_row['hfss_ge10_fraction']:.2f}%|
|共同单元包络|{common_summary['ideal_gmin_db']['p5']:.3f}|{common_summary['ideal_gmin_db']['median']:.3f}|+{common_summary['paired_delta_db']['median']:.3f}|{100*common_row['ideal_ge5_fraction']:.2f}%|{100*common_row['ideal_ge8_fraction']:.2f}%|{100*common_row['ideal_ge10_fraction']:.2f}%|
|各向同性理想阵列|{iso_summary['ideal_gmin_db']['p5']:.3f}|{iso_summary['ideal_gmin_db']['median']:.3f}|+{iso_summary['paired_delta_db']['median']:.3f}|{100*iso_row['ideal_ge5_fraction']:.2f}%|{100*iso_row['ideal_ge8_fraction']:.2f}%|{100*iso_row['ideal_ge10_fraction']:.2f}%|

按各模型自身的单点最优能力归一化后，holdout中位四点共享损失为：HFSS `{common_row['hfss_solo_utilization_median_db']:.3f} dB`，共同单元包络 `{common_row['ideal_solo_utilization_median_db']:.3f} dB`，各向同性理想阵列 `{iso_row['ideal_solo_utilization_median_db']:.3f} dB`。这说明几何和单权值多目标约束本身仍有明显代价，但HFSS方向包络又进一步压低了大角度目标的绝对增益。

## 同位置案例

|原holdout案例|类型|HFSS Gmin|共同单元包络 Gmin|各向同性理想 Gmin|
|---:|---|---:|---:|---:|
"""
    for case, group, hfss, common_value, iso_value in examples:
        report += f"|{case}|{group}|{hfss:.4f} dB|{common_value:.4f} dB|{iso_value:.4f} dB|\n"
    report += """

## 物理解释与边界

- 各向同性理想阵列的高结果部分利用了大间距阵列的栅瓣/多峰自由度；它只说明四个指定点可以同时较高，不代表旁瓣、EIRP分布或覆盖质量更好。
- “共同单元包络≈HFSS”说明端口间非一致结构不是主要数dB差距来源，但共同包络仍含HFSS平均扫描效应，不能被称为严格无互耦单元方向图。
- 若要把互耦与单元方向图严格分开，下一项控制实验应在HFSS中保持阵元、坐标、边界和总功率不变，分别生成：完整阵列嵌入方向图、关闭互耦/孤立单元平移得到的方向图，以及对应S参数去嵌结果，再对同840组四点运行同一O3。

## 可复核文件

- `paired_cases.csv`：1680行逐案例成对结果及四点坐标。
- `ideal_o3_raw_results.npz`：理想模型权值、四点线性增益和最低值。
- `summary_table.csv`、`summary.json`：统计结果。
- `PROTOCOL.json`：模型定义和因果解释边界。
- `verification.json`：恒模约束、目标一致性和增益独立复算检查。
- `run_ideal_array_control.py`：完整实验脚本。
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")

if __name__ == "__main__":
    main()
