# V2.8 HFSS 单极化幅度 CNN 物理修正版

日期：2026-09-03

## 结论

在不增加任何不可实测信息的前提下，修正版把测试集全局幅度 NMSE 从 -2.92 dB 提升到
-16.02 dB，幅度 dB MAE 从 6.14 dB 降至 2.28 dB，功率 NMSE 从 +1.16 dB 提升到
-15.32 dB。提升覆盖 DFT、随机相位和单阵元三类未见命令，属于明确且大幅的预测性能改善。

本实验仍只使用一个固定接收极化的标量幅度。HFSS 当前通道为 `|Ephi|`；接收相位、
`Etheta`、双极化总场和 S16P 均未进入训练目录或损失。模型内部的复数变量只负责描述已知
阵元命令的相干叠加，不是测量标签。

## 固定数据与防泄漏条件

- 数据：`data/processed/hfss_47ghz_ephi_amplitude`；
- 固定划分：`data/splits/hfss_47ghz_ephi_seed28001`；
- 训练/验证/测试方向：45,110 / 9,660 / 9,670；
- 训练/验证/测试命令：92 / 18 / 18；
- 方向按 10°×10° 角块隔离，命令在每个命令族内部隔离；
- 训练过程只依据 validation 选择最佳权重；模型结构和超参数未依据 test 修改。

测试集在 120 epoch 阶段评估过一次，随后因为第 120 轮仍是 validation 最佳边界，按预先
约定把同一配置延长到 200 epoch。因此 200 epoch 的测试结果不是绝对意义上的首次开封盲测，
但延长期间没有改变结构、损失或超参数。下一轮正式模型比较应另留新的命令和角块作为一次性
最终盲测集。

## 网络修正

1. 在理想 steering、逐阵元复增益和局部互耦之间，加入单极化嵌入式阵元方向图分支。
   分支根据方向和 4×4 行列/边缘/角点地图输出逐阵元复乘性修正，直接对应
   `单极化方向图 × 理想传播相位`。
2. 所有阵元域卷积仍为二维 3×3；没有把 4×4 阵元展平成长度 16，也没有 GAP。
3. 同一方向只计算一次完整 `[2,4,4]` 潜在复算子，并同时与全部 92 个训练命令显式复数
   收缩。这样同一潜在信道必须同时解释单阵元、DFT 和随机相位的幅度。
4. 每个命令族在损失中具有相同总权重，避免 68 个随机相位训练命令淹没各 12 个 DFT 和
   单阵元命令。
5. 损失由逐命令幅度 NMSE、功率 NMSE 和噪声底以上的稳健 dB 误差组成。功率项约束相干
   峰值，dB 项兼顾较弱信号区域。
6. 潜在复信道的全局相位用整个孔径与理想算子的复内积固定，不再依赖单个参考阵元。
7. 每轮计算完整 validation；保存 validation 最佳权重，并配置学习率平台衰减和 20 轮
   early stopping。

修正版共有 62,098 个可训练参数；禁用的 multihop 分支仍不参与前向。相比首版，增加参数
主要来自方向图分支，不改变输入和输出合同。

## 训练配置

- checkpoint：`artifacts/checkpoints/hfss_47ghz_ephi_physics_v2_200ep.pt`；
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch 2.7.1+cu128；
- 最大 200 epoch，最佳 epoch 为 200，未触发早停；
- 每轮随机抽取 8,192 个训练方向，每个方向使用全部 92 个训练命令；
- 每轮有效监督组合数 753,664，200 轮累计 150,732,800；
- direction batch size 2,048，validation batch size 4,096；
- AdamW，初始学习率 1e-3；最佳轮前验证目标持续改善，学习率未触发衰减；
- 幅度 dB / 功率损失权重分别为 0.03 / 0.10。

第 200 轮抽样训练幅度 NMSE 为 0.01059，训练 dB MAE 为 1.47 dB。按命令族平衡的
validation 幅度 NMSE 为 -15.64 dB、dB MAE 为 2.37 dB，最佳权重位于预算边界，说明
200 epoch 尚未表现出过拟合；但 120 到 200 epoch 的边际收益已明显变小。

## 统一口径结果

下表的“全局 NMSE”先汇总该集合中的全部命令-方向观测再计算，与首版报告口径一致。

| 模型 | Validation 幅度 NMSE | Test 幅度 NMSE | Test 功率 NMSE | Test dB MAE | Test 相关系数 | Test 绝对误差 P95 |
|---|---:|---:|---:|---:|---:|---:|
| 首版 CNN，20 epoch | -4.16 dB | -2.92 dB | +1.16 dB | 6.14 dB | 0.553 | 9.076 |
| 修正版，120 epoch | -14.42 dB | -14.83 dB | -14.12 dB | 2.56 dB | 0.957 | 2.236 |
| 修正版，200 epoch | -15.63 dB | -16.02 dB | -15.32 dB | 2.28 dB | 0.967 | 1.955 |

从 20 epoch 首版到 200 epoch 修正版：

- 测试幅度 NMSE 改善 13.11 dB；
- 测试功率 NMSE 改善 16.48 dB；
- 测试 dB MAE 降低约 62.9%；
- 测试绝对误差 P95 降低约 78.5%；
- 平均逐命令幅度相关系数从 0.553 提升到 0.967。

## 未见命令族拆解

以下 NMSE 为逐命令 NMSE 的算术平均，每个命令等权。

| Test 命令族 | 首版 NMSE | 修正版 NMSE | 首版 dB MAE | 修正版 dB MAE |
|---|---:|---:|---:|---:|
| DFT phase | +1.30 dB | -15.99 dB | 8.85 dB | 2.90 dB |
| Random phase | -4.08 dB | -15.92 dB | 5.64 dB | 2.26 dB |
| Single element | -4.74 dB | -16.61 dB | 6.87 dB | 1.74 dB |

首版最明显的问题是对未见 DFT 命令的相干关系失效。修正版三类命令的 NMSE 已集中在
-15.9 至 -16.6 dB，说明改善不是依靠随机相位样本数量获得的平均优势。

## 结构消融

采用同一分组训练、同一损失和最大 200 epoch，但关闭新增单极化方向图分支：

- 第 25 epoch 达到最佳 validation 目标；
- 第 45 epoch 触发 early stopping；
- validation 全局幅度 NMSE 仅 -4.22 dB，dB MAE 6.34 dB；
- 启用方向图分支后的对应结果为 -15.63 dB 和 2.37 dB。

这说明仅仅增加 epoch 和修改采样方式不足以解释主要提升。显式建模逐阵元、随方向变化的
单极化响应，是本轮性能改善的关键结构因素。

## 当前判断与下一步

当前模型已经达到先前建议的 NMSE 小于 -10 dB、功率 NMSE 为明显负值、不同命令族表现
接近的阶段性标准，可以开始小规模四目标优化的数值闭环验证。但 2.28 dB 的全空间平均幅度
误差仍高于精确代理模型所需水平，尤其 DFT 命令的 dB MAE 仍接近 2.9 dB。

下一轮不应继续单纯堆叠 epoch。建议按顺序执行：

1. 冻结当前 checkpoint，预留新的命令和角块做真正一次性盲测；
2. 按主工作扇区、强信号区和零陷区分别统计误差，确认 2.28 dB 是否主要来自低幅度区域；
3. 在 validation 上比较 200 到 300 epoch 的边际收益，或采用更小学习率精调；
4. 增加符合未来硬件量化规则的幅相命令，而不是扩大网络宽度；
5. 暗室阶段把 `receiver_channel` 改为实际固定单极化通道定义，并从实测幅度重新训练，
   不迁移 HFSS 权重。

## 结果文件

- `artifacts/reports/hfss_47ghz_ephi_physics_v2_200ep_validation.json`
- `artifacts/reports/hfss_47ghz_ephi_physics_v2_200ep_test.json`
- `artifacts/reports/hfss_47ghz_ephi_ablation_no_pattern_validation.json`
- `artifacts/reports/hfss_47ghz_ephi_baseline_validation_v2metrics.json`
- `artifacts/reports/hfss_47ghz_ephi_baseline_test_v2metrics.json`
