# AESA-CNN V2.17：跨阵列数据与可行性试验

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 数据获取、跨阵列准备和 validation-only feasibility pilot |
| 继承 | V2.16 的“解析物理 + 方向门控几何残差”思想，不修改 V2.16 冻结结果 |
| 主要变化 | 为 1×4、1×8、2×4、4×4 建立 canonical HFSS 数据，并分别训练 geometry-matched 模型 |
| 数据结论 | 四阵列数据门禁全部完成：`1×4/1×8/2×4=PASS`，`4×4=PASS_REUSED_FROZEN_DATA` |
| 模型结论 | pilot 为 `PARTIAL_2_OF_4`：1×8、2×4 PASS；1×4、4×4 FAIL |
| 证据边界 | 没有正式模型晋级，不是共享参数或零样本跨尺寸模型，blind test 未读取 |
| 后续关系 | V2.18 回到 4×4 canonical 协议，执行迁移归因与单变量辅助监督筛选 |

## 本版到底完成了什么

V2.17 的首要成果是**可读、可审计、可复用的跨阵列 HFSS 数据链**。在四个数据门禁完成后，
本版又按预注册协议运行了分阵列可行性试验。这里的“训练”只用于判断同一抽象思想是否值得
进入后续正式版本；它不替代 V2.16，也不授权打开 blind test。

四种阵列只共享以下建模原则，不共享参数或 checkpoint：

1. 保留解析阵列相位和复数相干求和；
2. 从本阵列 train 数据学习低秩 EEP 骨架；
3. CNN 只学习复包络残差；
4. 空间传播按阵列几何定制，并由方向编码产生 gate；
5. candidate 与本阵列 control 使用相同输入、预算和近等参数量；
6. 同时审计 amplitude、complex operator、Q 和强弱场分桶。

因此，V2.17 不能描述为“一个网络跨四种尺寸泛化”。它验证的是：四个独立模型能否在各自
阵列上复用同一物理建模思想。

## HFSS 数据门禁

所有新阵列都遵循：

```text
build → solve → export_s → export_eep → direct audit → canonicalize
```

保存工程、API 返回 `True` 或文件存在均不等于 PASS；必须实际读取完整 S 矩阵和两套端口
EEP，并由独立激励命令验证直接组合场。

| 阵列 | 数据状态 | 直接组合审计 | canonical 状态 |
|---|---|---:|---|
| 1×4 | PASS | aggregate complex NMSE `−286.324 dB` | 64,441 个有限、唯一前半球方向 |
| 1×8 | PASS | aggregate complex NMSE `−286.165 dB` | 64,441 个有限、唯一前半球方向 |
| 2×4 | PASS | aggregate complex NMSE `−285.409 dB` | 64,441 个有限、唯一前半球方向 |
| 4×4 | PASS_REUSED_FROZEN_DATA | 旧 Gate-A 约 `−229.46 dB` | 64,441 个冻结方向，未修改母版 |

数据只生成 train/validation 派生视图；没有创建 blind-test 低精度缓存。自动化异常、GUI
进程内恢复方案和禁用路线见
[`hfss/cross_array_47ghz_v1/AUTOMATION_RECOVERY_20261013.md`](hfss/cross_array_47ghz_v1/AUTOMATION_RECOVERY_20261013.md)。

## 分阵列模型设计

train-only 预分析表明，每种阵列的最优复 rank-1 骨架可解释超过 95% 的加权能量，但不支持
跨阵列复用同一个 rank-1 因子。完整模型使用：

```text
Fourier-27 → MLP-64 → direction-dependent complex scalar
                       + static per-element complex gain
                       + analytic propagation phase
                       + geometry-matched CNN residual
```

| 阵列 | control / candidate 的几何设计 |
|---|---|
| 1×4 | 两层 column-only 1×3 residual；candidate 增加逐 block 方向 gate |
| 1×8 | 三层 column-only 1×3 residual，dilation 1/2/1；candidate 增加方向 gate |
| 2×4 | row-pair 全高混合 + column 1×3；candidate 增加方向 gate |
| 4×4 | V2.16 深度保持的 static DCAR / gated-PM 思路 |

每组 candidate/control 参数差绝对值不超过 `0.5%`，并通过初始输出一致、有限梯度和参数独立
的训练前静态审计。

## Validation-only feasibility pilot

四个阵列分别使用 paired seeds `21721/21722/21723`，每个 arm 训练 40 epochs，共 24 个
独立 run；checkpoint 只按本阵列 validation amplitude linear NMSE 选择。

| 阵列 | 判定 | amplitude 改善 | complex 改善 | Q 改善 | 主要原因 |
|---|---|---:|---:|---:|---|
| 1×4 | FAIL | +1.645 dB | +1.718 dB | −0.419 dB | Q 与最强场稳定性未过门禁 |
| 1×8 | PASS | +1.350 dB | +0.886 dB | +0.993 dB | 3/3 主指标正向，全部门禁通过 |
| 2×4 | PASS | +1.312 dB | +1.206 dB | +1.131 dB | 3/3 主指标正向，全部门禁通过 |
| 4×4 | FAIL | −0.401 dB | −0.572 dB | −0.510 dB | 当前轴向 candidate 不如各向同性 control |

最终结论为 `PARTIAL_2_OF_4`。结果支持针对阵列几何设计残差传播，但不支持“同一方向门控
结构对所有阵列普适”。失败几何不得通过打开 blind test 或事后修改门槛来修复。

## 入口与复现

- [数据获取与 pilot 最终状态](V2.17_DATA_ACQUISITION_STATUS_20261013.md)
- [数据处理和分阵列模型协议](V2.17_DATA_PROCESSING_AND_MODEL_PLAN.md)
- `src/v217/cross_array.py`：FFD、Touchstone、canonical、split 和命令实现
- `src/v217/per_array_models.py`：分阵列 geometry-matched residual 模型
- `tools/prepare_cross_array_dataset.py`：数据冻结
- `tools/analyze_cross_array_dataset.py`：质量、耦合和极化分析
- `tools/analyze_train_only_feasibility.py`：只读 train cache 的低秩预分析
- `tools/audit_per_array_model_shapes.py`：训练前结构与参数匹配审计

```powershell
python algorithm_versions/v2.17/tools/run_tests.py
```

当前精简分支的 V2.17 代码级回归为 `8/8 PASS`。这证明合同与隔离逻辑可运行，不代表
重新执行了 HFSS 求解或 24 个训练 run。

## 冻结边界

- V2.16 仍是最新完成控制变量验证的 CNN 线。
- V2.17 在 `PARTIAL_2_OF_4` 处收口，不发布正式新 CNN。
- 不读取 blind test 做归一化、阈值、结构、模型或 checkpoint 选择。
- 若继续研究 1×4 稳定性或 4×4 control/candidate，应建立新版本并先预注册新假设。
