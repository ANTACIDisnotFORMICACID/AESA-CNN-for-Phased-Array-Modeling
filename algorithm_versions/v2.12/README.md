# AESA-CNN V2.12

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 后续受控实验的物理主干与正式基线 |
| 继承 | V2.10 amplitude-only 任务和 V2.11 方向编码研究 |
| 主要变化 | 解析传播相位、相干 rank-1 算子、filtration SSH、Fourier-27 与二维 residual CNN 联合建模 |
| 冻结结论 | SSH 预测价值与 latent specialization 通过；gauge-invariant operator 平滑性未通过 |
| 后续关系 | V2.13 审计物理闭环；V2.15/V2.16 只替换残差传播结构并保留本版骨架 |

V2.12 独立实现 filtration-preserving SSH 方向编码与单极化相干 rank-1
阵列算子。训练可观测量始终是幅度；潜在复场只定义到逐方向公共相位，主要物理
审计对象是 gauge-invariant 的去传播相位算子 `Q_tilde`。

解释分为两个层次：`a(q)` 保留为网络需要学习的去传播相位 latent
representation，用于说明 SSH、Fourier residual 与方向无关增益各自承担什么；
由数据支持的可辨识物理结论只建立在 `Q_tilde=a* a^T` 及幅度预测上。

- 冻结方案：[`V2.12_FROZEN_PLAN.md`](V2.12_FROZEN_PLAN.md)
- 实现、数据合同与运行入口：[`IMPLEMENTATION.md`](IMPLEMENTATION.md)
- 冻结机器可读配置：[`configs/v212_frozen_protocol.json`](configs/v212_frozen_protocol.json)
- SSH–Fourier 分离实现与验证：[`V2.12_S_IMPLEMENTATION.md`](V2.12_S_IMPLEMENTATION.md)
- SSH–Fourier 十种子结果：[`V2.12_S_RESULTS_20260912.md`](V2.12_S_RESULTS_20260912.md)
- 冻结表征审计：[`V2.12_S2_REPRESENTATION_AUDIT_20260912.md`](V2.12_S2_REPRESENTATION_AUDIT_20260912.md)
- 最终结果分析与下一步展望：[`V2.12_FINAL_ANALYSIS_AND_OUTLOOK_20260912.md`](V2.12_FINAL_ANALYSIS_AND_OUTLOOK_20260912.md)
- 后续物理闭环审计：[`../v2.13/README.md`](../v2.13/README.md)

快速验收：

```powershell
python algorithm_versions/v2.12/tools/run_tests.py
```

V2.10 和 V2.11 未被修改。O1 optimized codebook 与 G3 lifted-Q distillation
均是诊断实验，不替换 G2 主模型。

当前冻结结论：SSH 的 Level 1 预测价值与 Level 2 latent specialization 已通过，
Level 3 gauge-invariant operator 平滑性未通过。V2.12-S/S2 到此结束，不再追加
更强的 separation；后续模型将 SSH 作为结构化多尺度方向编码器使用。
