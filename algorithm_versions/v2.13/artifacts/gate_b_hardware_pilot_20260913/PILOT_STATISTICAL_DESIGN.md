# V2.13 Gate-B 硬件 pilot 与统计设计

## 结论

已冻结 240 个 validation-only 测量方向：200 个分层随机方向用于总体统计，
另加 40 个最远点覆盖方向用于局部失效诊断。该文件是测量计划，不是实测结果；
Gate B 仍为 `ideal-synthetic-only / hardware-blocked`。

## Pilot 构成

| 分层 | validation 总数 | 随机抽取 |
|---|---:|---:|
| `known_failure_angle` | 100 | 40 |
| `high_condition` | 455 | 40 |
| `strong_field` | 1098 | 40 |
| `weak_field` | 1086 | 40 |
| `other_validation` | 2161 | 40 |
| `coverage_diagnostic` | — | 40 |

全 validation 到最近 pilot 方向的角距离 P95 为 `4.05°`，最大为 `5.37°`。
整体统计只能使用 `core_stratified` 行并应用 CSV 中的 poststratification weight；
40 个 coverage diagnostic 方向必须单独报告，不能混入无权总体平均。

## 重复数统计

下表给出正态、独立重复假设下，样本标准差的 95% CI 相对因子，以及 30 dB
repeat SNR 时均值功率的 95% CI 半宽。

| R | 标准差 95% CI 因子 | 30 dB 时均值半宽 | 观测数 |
|---:|---:|---:|---:|
| 3 | [0.52, 6.28] | ±7.86% | 57600 |
| 5 | [0.60, 2.87] | ±3.93% | 96000 |
| 10 | [0.69, 1.83] | ±2.26% | 192000 |
| 20 | [0.76, 1.46] | ±1.48% | 384000 |

`R=3` 只适合格式/链路冒烟，噪声标准差区间极宽；`R=10` 是本 pilot 的正式
默认值，并拆成两个各 5 次的独立 session block。若 session 间漂移显著，必须
使用分层/混合效应报告，而不能把 10 次当作同分布白噪声简单合并。

## 执行边界

- 每个 repeat 内随机化方向顺序和方向内命令顺序；
- 64 条 train commands 恢复 Q，16 条 validation commands 只做 held-out 回放；
- external-test observations 不在计划中；
- 完成后先运行 Gate-B importer/validator，再计算协方差、power NMSE、Jacobian condition、角区和 bootstrap CI；
- 若能在部分方向获得校准 complex reference，才额外报告 operator NMSE。

当前 synthetic 30 dB 结果只作为接收链设计目标，不作为硬件通过阈值。
