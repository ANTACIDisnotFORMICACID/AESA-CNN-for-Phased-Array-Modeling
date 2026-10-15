# V2.12 Stage 2 结果（2026-09-11）

## 结论

V2.12 的主要增益来自相干 rank-1 测量算子与正确的物理幅度尺度，而不是单纯增加 SSH 通道。当前最稳健的主模型是 G1；G2 的平均精度更高，但没有达到预先冻结的 8/10 配对胜率，因此保留为 SSH 物理候选；G3 lifted-operator distillation 仅保留为诊断实验。

物理解释分为两层：

\[
a(q):\ \text{去传播相位后的 latent representation，用于解释网络学习内容；}
\]

\[
\widetilde Q(q)=a^*(q)a^T(q):\ \text{amplitude-only 数据可辨识的 physical operator，用于数据支持的物理结论。}
\]

因此不废弃对 \(a(q)\) 的结构分析，但不把其 gauge-dependent 复相位或 SSH 系数解释为唯一真实物理量。

## Rank-1 数据合同审计

- 训练方向数：22,600；拟合仅使用 64 条 train commands。
- rank-1 接受率：99.9956%。
- rank-2 held-out power NMSE 改善的中位数及 P95 均为 0 dB。
- Jacobian 最小非零奇异值：P05 29,779.49，中位数 79,129.15。
- condition number：P05 9.83，中位数 14.23，P95 19.82。
- 结论：当前 HFSS 单频、相干、单极化数据满足 rank-1 主线合同。

## 幅度尺度修正

HFSS 去相位包络幅度的训练集统计量明显不在单位尺度附近。固定 \(a\approx1\) 且把残差 I/Q 限制在较小范围会形成表达与优化瓶颈。现使用仅由 q_train 的 16 条 singles 估计的正尺度 \(s_a\)：

\[
a(q)=s_a\left[b+a_{\rm SSH}(q)+\Delta a_F(q)\right].
\]

该尺度不使用 validation/test 数据，也不引入方向相关自由度。修正后，单种子 120 epoch 的 G1/G2 已达到约 -22.54/-23.29 dB，显著优于未修正尺度的 -8.25/-14.34 dB。

## 10 种子 validation-D 结果

| 模型 | 平均 NMSE (dB) | 相对 G0/E0 改善 |
|---|---:|---:|
| G0 / V2.10 amplitude baseline | -17.1431 | — |
| Stage 1 E2-F5 | -17.8514 | +0.7084 |
| G1：Fourier residual + rank-1 physics | -23.7855 | +6.6424 |
| G2：SSH + Fourier residual + rank-1 physics | -24.1728 | +7.0297 |

G2 相对 G1：

- 平均改善 0.3873 dB；paired bootstrap 95% CI 为 [0.0752, 0.7102] dB。
- 7/10 种子获胜；双侧配对置换检验 p=0.0605。
- 主瓣 MAE 平均改善 0.0188 dB，没有发生主瓣退化。
- 由于未达到预冻结的至少 8/10 胜率，G2 不判定为正式胜出。

G2 相对 G0 的分场强线性 NMSE 改善为：主瓣 +5.49 dB、[-10,-3) dB 区间 +6.31 dB、[-20,-10) dB 区间 +7.27 dB、低于 -20 dB 区间 +9.45 dB。物理重构对弱场和零陷区域的收益尤其明显。

## Gauge-invariant operator 频谱

使用 validation directions、实体角权重和 raw SSH L=12 审计 \(\widetilde Q\)：

| 对象 | L≤4 累计能量 | L≤5 累计能量 | L≤10 累计能量 | 归一化梯度能量 |
|---|---:|---:|---:|---:|
| HFSS truth | 0.93144 | 0.98023 | 1.00000 | 9.24266 |
| G1 best validation run | 0.92346 | 0.97693 | 1.00000 | 9.64440 |
| G2 best validation run | 0.92552 | 0.97762 | 1.00000 | 9.52627 |
| G3 best validation run | 0.92251 | 0.97662 | 1.00000 | 9.69920 |

G2 比 G1 略接近 HFSS 的低阶能量分布，说明 SSH 分支具有正确方向的归纳偏置；但 G2 的 Fourier residual 低 filtration 泄漏仍为 0.8938，表明两个分支存在较强功能重叠。这可以解释 SSH 的边际平均收益只有约 0.39 dB。

## G3 operator distillation

单种子筛选在 \(\lambda_Q\in\{0.01,0.1,1\}\) 中选择 validation-D 最好的 0.01。随后用与 G2 相同的 10 个种子正式比较：

- G2 平均 -24.1728 dB；G3 平均 -24.3272 dB。
- G3 平均改善 0.1544 dB，中位改善 0.0349 dB。
- 95% CI 为 [-0.2158, 0.5498] dB；5/10 获胜；p=0.4805。
- 不满足正置信区间与 8/10 胜率要求；G3 不替换 G2，也不进入主线模型。

这说明 train-command inversion 得到的 \(\widehat Q\) 可作为合理的 gauge-invariant 辅助目标，但目前的置信加权与固定 \(\lambda_Q\) 没有带来跨种子的稳定泛化收益。

## 冻结状态与下一步

- Stage 2 external test 尚未打开。
- 当前稳健主模型：G1；物理候选/消融模型：G2；诊断模型：G3。
- 下一步应先减少 G2 中 SSH 与 Fourier residual 的低阶重叠，例如软泄漏惩罚或分阶段训练，再做新的 validation-only 消融；不能用当前 test 调参。
- \(a(q)\) 继续用于 latent branch、尺度与去相位包络解释；所有“数据证明的物理频谱”结论继续以 \(\widetilde Q\) 为准。
