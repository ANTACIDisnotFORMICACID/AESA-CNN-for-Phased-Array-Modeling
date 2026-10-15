# AESA-CNN V2.16：受控 DCAR 结构实验

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 最新完成控制变量验证的 CNN 结构线 |
| 继承 | 从 V2.12 物理主干重新分叉；不继承 V2.15 的 graph/S 分支 |
| 主要变化 | 用 Direction-Conditioned Axial Residual（DCAR）替换 residual block 内部 3×3 传播 |
| 冻结结论 | 近等参数 gated-DCAR 在原协议的 validation、coverage、operator 上均约 +1 dB、3/3 种子正向 |
| 证据边界 | 属于完成控制变量的探索性机制证据；旧 external test 未读取，新 confirmatory test 未定义 |
| 后续关系 | V2.17 进行跨阵列数据与可行性试验；V2.18 检查 4×4 canonical 迁移稳定性 |

## 研究问题

> 仓库状态：`BLOCKED`（完整复跑）。本仓库保留结构实现、协议和紧凑汇总，不包含本版
> 所需的大型训练数组与 checkpoint；冻结结论没有被改写为当前仓库内的复跑结果。

V2.15 表明局部消息传播有价值，但真实 S 图并未显著优于 shuffled-S，因此无法把收益归因于
真实互耦拓扑。V2.16 去除这一混杂，单独检验：在二维规则阵列上，按行/列轴分解并由方向
控制的残差传播，是否优于近等参数的普通 3×3 residual。

共同物理骨架保持不变：

\[
a_n(q)=s[b_n+a_{\mathrm{SSH},n}(q)+\delta_n(q)],
\qquad
g_n(q)=a_n(q)e^{jkq^Tr_n},
\qquad
\hat y(q,w)=\left|\sum_n g_n(q)w_n\right|.
\]

解析相位、逐阵元复基准、L=4 filtration SSH、Fourier-27、IQ 残差和相干求和均不改变；
实验变量仅位于 \(\delta_n(q)\) 的空间传播结构。

## DCAR 结构

普通 V2.12 block 为：

\[
B(H)=H+C_2^{3\times3}\sigma(C_1^{3\times3}H).
\]

受控 DCAR 将其替换为：

\[
B_D(H,q)=H+
\gamma_r(q)R_2^{3\times1}\sigma(R_1^{3\times1}H)+
\gamma_c(q)C_2^{1\times3}\sigma(C_1^{1\times3}H).
\]

`row` 和 `column` 分支分别建模两条阵列孔径轴，\(\gamma_r,\gamma_c\) 由方向编码生成。
两个 block 串联后可产生跨轴、对角和多跳路径，而不需要引入 V2.15 的 S 图。

最终 Stage-1 与 V2.12 共同保持：

- Fourier-27 + row/column/edge/corner，共 31 张输入图；
- width=24、3×3 stem、两个 residual blocks、3×3 IQ head；
- 相同激活位置、最长路径深度和 zero-start 初始化；
- 相同数据、训练预算、配对种子和审计协议。

近参数匹配版本只调整轴向分支内部宽度，使 gated-PM 共 28,755 参数，相对 V2.12
control 的 28,789 参数只差 `-0.118%`。

## 两阶段结果

### Stage-0：失败并停止

首版 DCAR 同时减少了空间深度、移除 edge/corner 图并降低约 23% 参数量。它相对 V2.12
control 在 validation 和 coverage 分别平均退化 `2.216 dB`、`2.103 dB`。由于一次改变了
多个因素，该结果不能单独证明轴向传播无效；本版保留失败证据，不覆盖原记录。

### Stage-1：恢复控制变量后通过探索门禁

| 模型 | 参数量 | validation NMSE / MAE | coverage NMSE / MAE | 复 operator NMSE |
|---|---:|---:|---:|---:|
| V2.12 control | 28,789 | −26.469 / 0.633 dB | −26.590 / 0.619 dB | −20.493 dB |
| DCAR static-PM | 28,643 | −26.859 / 0.620 dB | −26.949 / 0.609 dB | −20.788 dB |
| DCAR gated-PM | 28,755 | **−27.452 / 0.590 dB** | **−27.601 / 0.575 dB** | **−21.512 dB** |

gated-PM 相对 V2.12 control：

- validation amplitude NMSE 平均 `+0.983 dB`，3/3 种子正向；
- coverage probe 平均 `+1.011 dB`，3/3 种子正向；
- complex Ephi operator 平均 `+1.019 dB`，3/3 种子正向；
- 相对 static-PM 平均 `+0.592 dB`，说明 direction gate 提供独立增益。

## 结论边界

支持的表述是：

> 在 V2.16 原冻结协议内，深度匹配、近等参数的 gated DCAR 优于 V2.12 3×3 residual。

不支持以下表述：

- DCAR 已通过新的 confirmatory blind test；
- DCAR 在所有阵列规模或 canonical 数据协议中都优于各向同性模型；
- 原始 gate 数值可以直接解释为唯一的行/列物理贡献；
- 参数量近似相同意味着执行延迟相同。当前 CPU 实现约慢 21.2%。

## 文档与复现入口

- [路线预注册](V2.16_PREREGISTRATION.md)
- [Stage-0 失败结果](V2.16_STAGE0_RESULTS_20260914.md)
- [Stage-1 受控预注册](V2.16_STAGE1_CONTROLLED_DCAR_PREREGISTRATION.md)
- [Stage-1 结果](V2.16_STAGE1_CONTROLLED_DCAR_RESULTS_20260914.md)
- [V2.12–V2.16 完整技术汇报](V2.12_V2.16_HFSS完整汇报_20261015.md)
- `src/v216/axial_operator.py`：DCAR 实现

```powershell
python algorithm_versions/v2.16/tools/run_tests.py
```

当前精简分支缺少部分训练 NPZ/checkpoint，完整 artifact 回归当前为 `BLOCKED`；详见仓库根目录
`docs/TEST_MATRIX.md`。这不改变已有实验回执，也不能计作当前环境中的完整 PASS。
