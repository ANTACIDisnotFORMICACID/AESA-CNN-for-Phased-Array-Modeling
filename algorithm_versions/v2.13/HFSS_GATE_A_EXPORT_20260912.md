# V2.13 Gate-A HFSS 自动导出结果（2026-09-12）

## 结果

已通过 PyAEDT 0.9.6 / AEDT 2025 R2 自动导出 8 组完整球面复数 `rEPhi`。数据包含
522728 个方向–命令组合，全部保存在 `v2.13/hfss_gate_a`，不作为 CNN 训练数据。

| 命令 | 类型 | 来源 | complex NMSE |
|---|---|---|---:|
| H00 | 等幅0°控制 | 复用 V2.9 A01 直接 HFSS 报表 | -293.54 dB |
| H90 | 等幅90°控制 | 本轮直接 HFSS API | -293.55 dB |
| P01 | 随机恒模相位 | 本轮直接 HFSS API | -226.75 dB |
| P02 | 随机恒模相位 | 本轮直接 HFSS API | -227.67 dB |
| P03 | 随机恒模相位 | 本轮直接 HFSS API | -229.71 dB |
| P04 | 随机恒模相位 | 本轮直接 HFSS API | -228.90 dB |
| A01 | 随机幅相 | 本轮直接 HFSS API | -229.36 dB |
| A02 | 随机幅相 | 本轮直接 HFSS API | -227.17 dB |

聚合 complex NMSE 为 `-229.34 dB`。工程文件前后摘要一致，说明自动化没有保存
`Edit Sources` 状态或改写原工程。

## 复用边界

V2.9 的 A01 原始 CSV 是 HFSS report engine 的直接输出，符合本轮要求，已复制为
`H00_hfss_direct.csv`。V2.9 其余24组 `complex_farfield` 的元数据明确标记为
`linear_superposition`，因此没有复制为独立观测。

## 单位事故与修正

HFSS 对本轮新报表输出 `mV`，历史 A01 输出 `V`。首次比较未处理这一差异，得到约
0 dB 的明显错误结果；导出器现按 CSV 表头逐表达式识别 V/mV/uV 并统一到 V，且加入
回归测试。原始 CSV 不做数值改写，保证问题可追溯。

## 数据位置

- 命令与冻结参数：`hfss_gate_a/command_manifest.json`、`hfss_gate_a/commands/`
- HFSS 原始直接报表：`hfss_gate_a/raw_direct_reports/`
- 紧凑审计数据：`hfss_gate_a/prepared/gate_a_hfss_direct_complex.npz`
- 机器可读导出报告：`hfss_gate_a/EXPORT_SUMMARY.json`
- 自动化脚本：`tools/prepare_hfss_gate_a_export.py`、`tools/export_hfss_gate_a_direct.py`

## 可成立的结论

当前结果关闭的是 HFSS 仿真层 Gate A：

\[
E_{\phi,\mathrm{HFSS-report}}(q,w)=\sum_nw_nE_\phi(q,e_n)
\]

在数值精度内成立。它验证 source/report API、端口顺序、相位、功率归一化和单位约定；
由于 HFSS 线性后处理复用同一已收敛多端口解，它不是功放非线性、开关状态依赖互耦或
真实硬件漂移的证据。
