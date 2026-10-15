# AESA-CNN V2.14-Validation Complete

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 结构与采样覆盖策略验证版，不晋级新模型 |
| 继承 | V2.13 的 validation-only 结构压力与外部测试封存规则 |
| 主要变化 | 预注册 fusion/jitter 2×2 筛选、等预算角域覆盖交换和物理归因附录 |
| 冻结结论 | fusion/jitter 路线停止；覆盖因果假设通过；没有 V2.14 CNN 候选 |
| 后续关系 | 局部耦合迹象引出 V2.15 的方向条件化图消息实验 |

V2.14 是独立于 V2.13 validation-only 审计的结构与采样策略验证区。已完成
`cross-branch access on/off × dephased-envelope jitter regularization on/off` 的 2×2
可行性实验和 train-only 等预算角域覆盖因果实验；V2.10–V2.13 均保持不变，
external test 保持封存。

> 仓库状态：`BLOCKED`（完整复跑）。本仓库保留代码、协议和紧凑结果，但不包含本版
> 使用的大型训练数组；下文结论来自冻结记录，不能据此声称已在精简仓库内重新训练。

2026-09-13 又在同版本内追加并完成“物理残差阶梯 + 互补整数角标签”附录；没有创建
V2.15，也没有重新开启已关闭的模型筛选。

两个 fusion 条件都使用相同的 8→4→2、1×1、zero-start adapter 和完全相同的参数量。
对照条件屏蔽 SSH 输入，只允许 Fourier residual 与几何输入；实验条件额外允许 SSH 输入。
因此检验对象是跨分支信息，而不是额外参数量。

Stage-0 只允许三个配对种子和短程微调，用于检查实现、稳定性及效应方向。它不授予新模型
资格；是否进入十种子正式实验需依据预注册筛选规则决定。V2.14 已于 2026-09-13 完成
全部预注册验证并封版，不再继续结构或超参数搜索。

## Stage-0 结果

结果见 [V2.14_STAGE0_RESULTS_20260913.md](V2.14_STAGE0_RESULTS_20260913.md)。当前结论为
`screen_complete_stop`：最佳 C11 相对 C00 只改善 `0.026 dB`，且 0.5° envelope
稳健性反而变化 `-0.052 dB`，没有达到扩展阈值。下一实验转向角域覆盖的因果验证。

## 角域覆盖结果与最终结论

- 结果：[V2.14_COVERAGE_CAUSAL_RESULTS_20260913.md](V2.14_COVERAGE_CAUSAL_RESULTS_20260913.md)
- 最终验证报告：[V2.14_FINAL_VALIDATION_REPORT_20260913.md](V2.14_FINAL_VALIDATION_REPORT_20260913.md)
- 机器结果：[artifacts/coverage_causal_20260913/COVERAGE_CAUSAL_SUMMARY.json](artifacts/coverage_causal_20260913/COVERAGE_CAUSAL_SUMMARY.json)

等预算覆盖交换使共同 nested-gap 集平均改善 `4.463 dB`（3/3 为正），outer
validation 平均改善 `0.723 dB`（3/3 为正），角距离中位数由 `5.303°` 降至
`1.000°`。这支持“补齐有标签角域覆盖”是当前首要方向，但不授予新模型资格。

最终状态：fusion/jitter 结构路线停止，覆盖因果假设通过；没有 V2.14 CNN 候选进入
正式十种子或 external-test 资格。external test 始终封存，硬件 Gate B 仍 blocked。

## 物理归因与新标签附录

- 预注册：[V2.14_PHYSICS_ATTRIBUTION_PREREGISTRATION.md](V2.14_PHYSICS_ATTRIBUTION_PREREGISTRATION.md)
- 物理结果：[V2.14_PHYSICS_ATTRIBUTION_RESULTS_20260913.md](V2.14_PHYSICS_ATTRIBUTION_RESULTS_20260913.md)
- 新标签结果：[V2.14_NEW_ANGULAR_LABELS_RESULTS_20260913.md](V2.14_NEW_ANGULAR_LABELS_RESULTS_20260913.md)

物理阶梯 P1–P5 均未达到 `0.50 dB + 3/3` 门槛；残差却与阵元邻接和 `|Sij|`
分别相关 `0.638/0.631`，说明存在局部耦合迹象，但当前低阶归因模型预测力不足。
490 条新角标签的等预算 exchange 在独立 coverage_probe 上平均改善 `0.999 dB`
且 3/3 正向，全部 gate 通过。当前结论仍是优先优化采样覆盖，暂不改 CNN。

快速验收：

```powershell
python algorithm_versions/v2.14/tools/run_tests.py
```
