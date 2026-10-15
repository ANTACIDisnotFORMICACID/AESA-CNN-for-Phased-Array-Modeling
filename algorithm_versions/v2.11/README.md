# AESA-CNN V2.11

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 方向编码迁移与消融过渡版 |
| 继承 | V2.10 的 4×4 CNN 输入合同、训练器和物理传播层 |
| 主要变化 | 比较 Fourier-27、SSH 和 zero-start Fourier+SSH，不改变其他训练变量 |
| 冻结结论 | 编码迁移与无泄漏约束完成；本版不替代后续 V2.12 主干 |
| 后续关系 | V2.12 将 filtration SSH、Fourier residual 与 rank-1 物理算子统一冻结 |

V2.11 是从 V2.10 迁移到球谐方向编码的过渡实验版本。它保持原 4x4 CNN 前端的通道合同，
比较 Fourier-27、SSH 和 zero-start Fourier+SSH 三类编码，不迁移训练器、优化器、物理传播层
或既有权重。

## 约束

- SSH 归一化统计量只由训练方向计算；验证和测试只能读取冻结统计量。
- VSH 的物理带宽审计不直接决定 CNN 输入维数。
- 本版用于编码消融和迁移验证，不替代后续 V2.12 的冻结主干。

## 入口

- [编码迁移说明](SSH_ENCODING_MIGRATION.md)
- [VSH、SSH 与 HFSS 分析计划](VSH_SSH_HFSS_ANALYSIS_PLAN.md)
- `src/v211/`：方向编码和模型适配实现
- `tests/`：编码、形状和训练数据边界测试
- `artifacts/encoding_ablation_20260910/`：精简统计与报告
