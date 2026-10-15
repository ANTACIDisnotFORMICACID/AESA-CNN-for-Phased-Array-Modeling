# V2.9 代码结构与实现规划

## 1. 文档目的与设计结论

本文定义 V2.9 的数学链路、代码边界、文件结构、迁移顺序和验收标准。规划已于
2026-09-04 完成首轮实现和 200 epoch HFSS 训练；实际文件职责见 `IMPLEMENTATION.md`，
运行结果见 `artifacts/reports/HFSS_V29_200EP_20260904.md`。

V2.9 的首要目标是提高未见方向、未见命令和优化器生成权值上的前向远场幅度预测精度。
“互耦层”“器件层”和“方向残差层”用于提供归纳偏置与容量分工，不把参数唯一可辨识
作为前置条件。只有独立的 S 参数、相干 OTA 标定或器件状态实验支持时，才进一步检验其
物理真实性。

当前证据给出的实现优先级为：

1. 保留 V2.8 已验证的幅度-only 输入输出主干与精确复数收缩；
2. 将方向相关名义阵元响应作为主要模型能力；
3. 用低容量结构化矩阵描述稳定的方向无关端口混合；
4. 只保留一个受限方向残差分支，消除 V2.8 多分支表达重叠；
5. 器件层在 HFSS 阶段保持恒等，等到真实状态数据存在后再启用。

## 2. 范围

### 2.1 首轮实现范围

- 固定 V2.9 的数学前向链路和复数约定；
- 识别 V2.8 可复用的输入、输出、数据、评估和优化模块；
- 实现重写 CNN 的模块边界和文件树；
- 定义分阶段训练、消融、测试和退出条件；
- 防止复相位/S 参数泄漏、互耦重复计算和分支不可辨识。

### 2.2 当前仍不包含

- V2.8 数据、checkpoint 或运行目录的复制；
- 对现有 V2.8 代码的修改；
- 把 HFSS 等效参数解释成实际器件参数；
- OTA 器件非线性、温漂和时间记忆模型的实现。

## 3. 冻结的输入输出合同

V2.9 继续兼容 V2.8 的幅度数据目录：

```text
direction.npy        [Q,3]
theta_deg.npy        [Q]
phi_deg.npy          [Q]
command_iq.npy       [N,2,4,4]
command_family.npy   [N]
amplitude.npy        [N,Q]
manifest.json
```

### 3.1 模型输入

| 名称 | 形状 | 含义 |
|---|---:|---|
| `unit_direction` | `[B,Q,3]` | 算法坐标系中的单位观察方向 |
| `command_iq` | `[B,2,4,4]` 或共享命令组 `[N,2,4,4]` | 已知发射控制命令 |
| `state` | 可选，默认不存在 | 未来 OTA 的温度、频率、功率档或会话状态 |

`command_iq` 中的相位是已知控制量，不是接收端测得相位。HFSS 和暗室训练目录仍不得包含
`operator_iq.npy` 或 `s_parameters_iq.npy`。

### 3.2 模型输出

```text
field_iq              [B,Q,2]
amplitude             [B,Q]
operator_iq           [B,Q,2,4,4]
complex_terms_iq      dict[str, [B,Q,2,4,4]]
```

`complex_terms_iq` 中各项必须在取绝对值之前严格相加得到 `operator_iq`。全局相位对齐仅供
诊断使用，不放进正式前向主链；否则分项归因和跨方向相位关系会被隐藏的逐方向 gauge 改写。
这里的 `operator_iq` 明确定义为相对于器件输出 `u` 的天线侧端口到场算子；未来即使
`F_dev` 为非线性，该算子的物理接口也不随命令改变。

### 3.3 版本兼容策略

- 幅度数据 schema 保持兼容 `v2.8-amplitude-1`，避免重新生成同一数据；
- V2.9 checkpoint 使用新 schema，例如 `v2.9-amplitude-operator-1`；
- checkpoint 必须记录模型分层、基底类型、启用模块、数据指纹、命令/角域划分和训练阶段；
- V2.8 checkpoint 不直接加载进 V2.9，仅允许在显式迁移工具中按名称转换兼容参数。

## 4. 数学前向链路

### 4.1 复数约定

V2.9 固定发射源约定：

\[
E(q;w)=h(q)^T w=\sum_{n=1}^{16}h_n(q)w_n.
\]

代码继续以 I/Q 双实数表示复数。不得把该式无说明地改写成接收波束形成约定
\(w^Hh\)。所有矩阵转置、端口编号和单元测试均以此约定为准。

### 4.2 物理分层

定义：

\[
\begin{aligned}
a_0(q) &= \exp\!\left(jk\,\hat q\cdot r_n\right),\\
b(q) &= g_0(q)\odot a_0(q),\\
u &= F_{\mathrm{dev}}(w;s),\\
h_{\mathrm{ant}}(q) &= C_{\mathrm{ant}}^T b(q)+r_\psi(q),\\
\widehat E(q;w,s) &= h_{\mathrm{ant}}(q)^T u,\\
\widehat A(q;w,s) &= |\widehat E(q;w,s)|.
\end{aligned}
\]

各层职责如下：

| 层 | 责任 | HFSS 阶段 | OTA 扩展 |
|---|---|---|---|
| `a0` | 几何传播相位 | 固定解析层 | 固定解析层 |
| `g0` | 单极化名义阵元方向响应 | 主要学习路径 | 可由 HFSS 预训练后实测微调 |
| `C_ant` | 方向无关的有效端口混合 | 可选、低容量 | 可按安装状态校准 |
| `r_psi` | 方向相关结构散射和未建模误差 | 单一受限 CNN | 可微调 |
| `F_dev` | 控制命令到端口激励 | 恒等映射 | 线性校准或器件非线性 |
| 收缩与绝对值 | 相干叠加和测量量 | 固定解析层 | 固定解析层 |

### 4.3 禁止重复计算互耦

基底类型必须是显式枚举并写入 checkpoint：

- `ideal_phase`：只有 `a0`，允许启用 `g0`、`C_ant` 和 `r_psi`；
- `isolated_element_pattern`：孤立阵元方向图乘 `a0`，允许启用 `C_ant`；
- `common_pattern`：公共方向图乘 `a0`，允许受限 `C_ant`；
- `embedded_element_basis`：完整嵌入式阵元场已经包含指定终端下的互耦和结构影响，
  必须关闭额外 `C_ant`，只作为 HFSS 精确重组 oracle 或诊断基底。

由复数 FFD 构造的 `common_pattern` 也只能用于 HFSS 研究对照；可部署 OTA 模型不得借此
绕过 amplitude-only 数据合同。正式 checkpoint 必须记录名义基底的来源和可部署资格。

模型构造器遇到 `embedded_element_basis + enable_coupling=true` 必须直接报错，而不是警告。

### 4.4 当前 HFSS 中的器件层

当前 HFSS 数据描述固定、线性、无源天线模型，没有 PA、移相器、馈线温漂或控制码字误差。
因此：

```text
source_type == hfss_amplitude  ->  F_dev = IdentityDeviceModel
```

训练入口必须拒绝在 HFSS 数据上启用可训练器件模型。以后只有 manifest 含有明确的状态字段、
器件参考面和跨状态数据时，才允许启用：

```text
LinearDeviceCalibration:   u = D_dev(s) w
NonlinearDeviceModel:      u_n = f_n(w_n, power, temperature, code, history)
```

## 5. 模型内部设计（原规划，现已实现）

### 5.1 名义阵元响应 `NominalElementPattern`

输入为方向 Fourier 特征、4×4 行列坐标以及边缘/角点确定性标记。输出分成：

```text
log_amplitude_delta [B,Q,4,4]
relative_phase      [B,Q,4,4]
```

随后解析构造：

\[
g_0(q)=\exp(\Delta\log A(q))\exp(j\Delta\phi(q)).
\]

约束：

- 初始化为 `log_amplitude_delta=0`、`relative_phase=0`；
- 单阵元命令主要约束幅度头，DFT/随机相位命令主要约束相对相位；
- 相位不直接以角度回归，始终通过 I/Q 或单位复数构造；
- 边缘和角点作为输入特征，不再各建一个能够独立修改完整算子的自由分支。

### 5.2 结构化端口混合 `StructuredPortCoupling`

建议参数化：

\[
C_{\mathrm{ant}}=I+P_{\mathrm{off}}\!\left(M\odot K+UV^T\right),
\]

其中 \(P_{\mathrm{off}}\) 将对角元素置零。

- `M`：固定稀疏连接掩码，由 4×4 几何和独立 S16P 诊断生成；
- `K`：掩码内的小复数修正，对角线固定为零；
- `UV^T`：可选低秩非局部项，默认关闭，从秩 1 开始消融；其对角分量同样被投影掉；
- 初始化严格满足 `C_ant=I`；
- 若宣称互易有效耦合，加入 `||C-C^T||²` 软约束，不误用 Hermitian 约束；
- S16P 只用于选择掩码、初始化或独立诊断，不进入正式 amplitude-only 训练批次。

不采用两个同时自由的 `D_out` 和 `D_in`。逐端口器件校准统一由未来的 `DeviceModel`
承担；`C_ant` 对角线始终固定为 1，以减少尺度和相位重参数化。

### 5.3 单一残差 `AngularOperatorResidual`

残差输出为逐端口复算子修正 `[B,Q,2,4,4]`，但容量必须显著小于 V2.8 多分支总容量。
首选顺序：

1. 低秩角域基底；
2. 窄通道 4×4 二维残差 CNN；
3. 只有消融证明存在非局部误差时，才启用一个全孔径上下文支路。

禁止同时恢复“方向残差”“边缘残差”“multihop 残差”三个均可修改完整算子的自由分支。
残差以零初始化，并受能量、角域平滑和容量约束。

### 5.4 器件模型 `DeviceModel`

统一接口：

```text
forward(command_iq, optional_state) -> effective_command_iq
```

第一版只实现和启用 `IdentityDeviceModel`。线性对角校准和非线性器件模型仅预留接口，
不进入 HFSS 基线实验，避免用天线误差替代不存在的器件误差标签。

### 5.5 顶层 `V29AmplitudeForwardModel`

顶层只负责固定顺序：

```text
direction
  -> ideal propagation
  -> nominal element pattern
  -> structured port coupling
  -> add one angular residual
command
  -> device model
operator + effective command
  -> exact complex contraction
  -> amplitude
```

顶层不得包含隐藏的卷积层或方向特征网络。所有可训练参数必须属于上述具名模块，便于冻结、
消融和 checkpoint 审计。

## 6. 计划文件结构

以下目标结构已完成首轮实现；训练产物集中在 `artifacts/`，没有复制 V2.8 大体积数据。

```text
v2.9/
├── README.md
├── ARCHITECTURE_PLAN.md
├── pyproject.toml                         # 从 V2.8 迁移，包名/入口改为 v29
├── configs/
│   ├── geometry_4x4_47ghz.toml           # 复用几何合同
│   ├── model_v29.toml                    # 新：基底、g0、C、残差、器件开关
│   ├── train_staged.toml                 # 新：分阶段冻结与损失权重
│   └── optimize_four_targets.toml        # 复用优化约束
├── schemas/
│   └── amplitude_manifest.schema.json    # 保持 V2.8 数据兼容
├── src/v29/
│   ├── __init__.py
│   ├── contracts.py                      # 迁移并扩展预测分项/模型状态合同
│   ├── coordinates.py                    # 原样复用
│   ├── complex_ops.py                    # 复用，增加发射约定单测
│   ├── io/
│   │   ├── amplitude_dataset.py          # 原样复用幅度-only隔离
│   │   ├── amplitude_split.py            # 原样复用数据指纹和固定划分
│   │   ├── amplitude_synthesis.py        # 复用，后续增加幅相/量化码本
│   │   ├── hfss_ffd.py                   # 只用于离线 HFSS 转换/审计
│   │   ├── touchstone.py                 # 只用于掩码诊断，不进训练批次
│   │   └── dataset_store.py              # 保留 HFSS 复基底审计
│   ├── physics/
│   │   ├── ideal_far_field.py            # 迁移 IdealScalarFarFieldOperator
│   │   ├── basis_policy.py               # 新：基底类型与防重复互耦规则
│   │   └── phase_alignment.py            # 新：仅诊断使用的全局相位对齐
│   ├── models/
│   │   ├── prediction.py                 # 新：AmplitudePrediction 与分项合同
│   │   ├── nominal_pattern.py            # 新：log幅度/相对相位头
│   │   ├── port_coupling.py              # 新：I + sparse + low-rank
│   │   ├── angular_residual.py            # 新：唯一受限方向残差
│   │   ├── device.py                      # 新：Identity；OTA接口预留
│   │   ├── contraction.py                 # 从 V2.8 迁移精确复收缩
│   │   └── amplitude_forward.py           # 新：只编排物理层级
│   ├── training/
│   │   ├── datasets.py                    # 迁移方向分组、多命令监督
│   │   ├── losses.py                      # 新：输出损失和结构正则
│   │   ├── stages.py                      # 新：M1/M2/M3冻结与解冻策略
│   │   ├── engine.py                      # 复用训练循环，改为配置驱动阶段
│   │   └── metrics.py                     # 扩展分区、命令族和稳定性指标
│   ├── optimization/
│   │   ├── problem.py                     # 原样复用四目标合同
│   │   ├── projected_gradient.py          # 复用，适配 V29 顶层接口
│   │   └── basis_verification.py          # 复用 HFSS 精确重组验证
│   └── cli/
│       ├── prepare_hfss_amplitude.py       # 薄迁移
│       ├── prepare_measured_amplitude.py   # 薄迁移
│       ├── prepare_amplitude_split.py      # 薄迁移
│       ├── train_forward.py                # 重写：读取阶段配置
│       ├── evaluate_forward.py             # 扩展新盲测/分区指标
│       ├── optimize_targets.py             # 薄迁移
│       └── diagnose_structure.py           # 新：C、残差、gauge和S16P诊断
└── tests/
    ├── test_contract_compatibility.py
    ├── test_no_phase_or_sparameter_leakage.py
    ├── test_transmit_complex_convention.py
    ├── test_identity_initialization.py
    ├── test_basis_policy.py
    ├── test_nominal_pattern.py
    ├── test_structured_port_coupling.py
    ├── test_angular_residual.py
    ├── test_stage_freezing.py
    ├── test_term_additivity.py
    ├── test_unseen_command_angle_evaluation.py
    └── test_four_target_optimization.py
```

## 7. V2.8 复用与重写矩阵

### 7.1 原样复用或仅修改包名

| V2.8 模块 | V2.9 处理 | 理由 |
|---|---|---|
| `coordinates.py` | 原样迁移 | 坐标定义已验证，与 CNN 无关 |
| `io/amplitude_dataset.py` | 原样迁移 | 幅度-only 数据隔离是部署边界 |
| `io/amplitude_split.py` | 原样迁移 | 数据指纹与固定划分必须保留 |
| `io/hfss_ffd.py` | 原样迁移 | 仅离线生成/审计，不进入训练输入 |
| `io/touchstone.py` | 原样迁移 | 仅独立物理诊断 |
| `optimization/problem.py` | 原样迁移 | 四目标约束未因 CNN 改写而变化 |

### 7.2 薄封装迁移

| V2.8 模块 | V2.9 调整 |
|---|---|
| `contracts.py` | 保留几何/查询，重定义标量预测与可加分项 |
| `complex_ops.py` | 保留计算，补充普通转置发射约定说明和测试 |
| `amplitude_synthesis.py` | 保留现有码本，增加随机幅相、稀疏及量化码字策略 |
| `training/amplitude.py` | 拆成 dataset、loss、stage 和 engine |
| `evaluate_forward.py` | 保留基本指标，增加角域/命令族/零陷分区 |
| `projected_gradient.py` | 只适配新模型调用，不改变约束投影数学 |

### 7.3 必须重写

| V2.8 模块 | 原因 |
|---|---|
| `models/amplitude_forward.py` | 当前 gain、pattern、direction、edge 分支可重叠修改同一潜在信道 |
| `models/blocks_2d.py` | 新结构需要 log幅度/相对相位头和受限残差块 |
| `cli/train_forward.py` | 需要显式训练阶段、基底策略和模块冻结审计 |
| 模型配置 | 需要区分基底、耦合、低秩项、残差和器件状态 |

不建议让 `v29` 在运行时直接 import `v28`。实施时应复制经过测试的稳定模块、保留来源提交
和文件哈希，再在 V2.9 内独立演进，避免两个版本的安装路径和 checkpoint schema 互相耦合。

## 8. 分阶段实现与训练

### 阶段 P0：迁移主干，不改变数值

- 迁移输入输出、数据加载、坐标、复数收缩、评估和优化；
- 建立 `v29` 包与 CLI；
- 使用恒等初始化复现理想传播幅度；
- 对相同输入验证 V2.8 固定解析函数与 V2.9 数值一致。

退出条件：所有合同测试通过，正式训练加载器仍拒绝复相位和 S 参数。

### 阶段 M1：名义方向响应

- 只启用 `NominalElementPattern`；
- `C_ant=I`、`r_psi=0`、`F_dev=Identity`；
- 单阵元、DFT、随机相位命令共同训练，但分别报告命令族指标。

退出条件：新 validation 上明显优于纯理想算子，并验证增益来自方向响应而非数据泄漏。

### 阶段 M2：结构化端口混合

- 加入稀疏 `C_ant`，保持残差关闭；
- 先冻结 M1，再拟合 C；随后只进行小学习率联合微调；
- 比较几何近邻掩码、S16P 诊断掩码以及可选秩 1 非局部项。

退出条件：未见命令和未见角块均改善；若只改善训练集或 C 在不同种子间剧烈漂移，则关闭
该项或仅把它作为正则化路径。

### 阶段 M3：单一方向残差

- 固定或低学习率更新 M1/M2；
- 从零训练 `AngularOperatorResidual`；
- 从低秩版本开始，只有证据支持时才扩大为窄 CNN。

退出条件：残差在新盲测集和四目标 regret 上继续改善，而不只是降低训练误差。

### 阶段 M4：受控联合微调

- 全部分层以较小学习率联合训练；
- 保留残差能量、角域平滑、C 稀疏/低秩和可选互易正则；
- 不启用 OTA 器件模型。

退出条件：多个随机种子均优于 M1/M2，且输出对分层开关、端口排列和全局相位变换符合预期。

### 阶段 O1：OTA 器件扩展

仅在具有跨温度、功率、频率、码字或会话状态的数据后进行。冻结或限制天线层，新增
`DeviceModel`，通过跨状态泛化而不是当前 HFSS 拟合误差检验器件层。

## 9. 损失与正则化规划

正式监督仍只作用于可测幅度：

\[
\mathcal L =
\mathcal L_{\mathrm{amplitude\ NMSE}}
+\lambda_{\mathrm{power}}\mathcal L_{\mathrm{power\ NMSE}}
+\lambda_{\mathrm{dB}}\mathcal L_{\mathrm{robust\ dB}}
+\lambda_r\lVert r_\psi\rVert_2^2
+\lambda_s\mathcal L_{\mathrm{angular\ smooth}}
+\lambda_C\mathcal R(C_{\mathrm{ant}}).
\]

约束原则：

- 单阵元命令负责稳定幅度头，但不强迫其学习不可观测绝对相位；
- DFT 和随机相位命令提供阵元间相干约束；
- 命令族继续做总权重平衡；
- 低幅度点的 dB 损失使用噪声门限和稳健截断；
- 不以朴素复向量正交作为硬约束；幅度-only 的逐方向全局相位自由度会使其解释不稳定；
- 如需减少结构层与残差竞争，优先采用分阶段冻结、残差能量惩罚和容量限制。

## 10. 数据与划分规划

### 10.1 保留现有数据原则

- HFSS 与 OTA 使用相同幅度合同，但 checkpoint 和训练结果严格隔离；
- 方向块和命令集合同时留出；
- 标准化、降维和任何统计量只能在训练集合拟合；
- 已被观察过的 V2.8 test 不再作为 V2.9 最终盲测集。

### 10.2 新命令覆盖

除单阵元、二维 DFT 和随机相位外，增加：

- 随机幅度＋随机相位；
- 稀疏多端口组合；
- 实际硬件量化码字；
- 由四目标优化器常见解分布生成的困难命令。

训练、validation 和 test 必须按命令 ID 显式隔离；不得先生成相近命令再随机逐样本拆分。

### 10.3 球面采样

全球规则 theta/phi 网格评估应使用固体角权重，避免高纬区域被过度计数。最终应用若只关心
特定扇区，应同时报告全局指标和目标扇区指标，不能用单一平均值掩盖目标区域退化。

## 11. 评估与消融矩阵

保持训练预算、参数量级、数据划分和随机种子可比：

| 实验 | `g0` | `C_ant` | `r_psi` | `F_dev` | 目的 |
|---|---|---|---|---|---|
| M0 | V2.8 当前结构 | 当前局部层 | 当前多分支 | 无 | 固定基线 |
| M1 | 新 | 关闭 | 关闭 | Identity | 名义方向响应贡献 |
| M2 | 新 | 稀疏 | 关闭 | Identity | 稳定端口混合贡献 |
| M2-LR | 新 | 稀疏+低秩 | 关闭 | Identity | 非局部项是否必要 |
| M3 | 新 | 最佳 M2 | 单一受限残差 | Identity | 最终 V2.9 候选 |
| M3-no-C | 新 | 关闭 | 单一受限残差 | Identity | 判断 C 是否真实提供泛化 |

至少报告：

- 全局、目标扇区和固体角加权幅度 NMSE；
- 功率 NMSE、稳健 dB MAE、绝对误差 P50/P95；
- 单阵元、DFT、随机相位、随机幅相、稀疏和量化命令族指标；
- 主瓣、旁瓣和深零陷分区指标；
- 三个以上随机种子的均值、标准差和最差值；
- 四目标预测最优值与 HFSS 嵌入式阵元基底精确重组之间的 regret；
- 结构化参数跨种子稳定性，但不以参数一致作为预测模型的唯一成功条件。

## 12. 测试设计

### 12.1 必须先于训练通过的单元测试

1. `IdentityDeviceModel`、`C=I`、`g0=1`、`r=0` 时严格退化为理想远场算子；
2. 手算两个复端口，验证使用 `h^T w` 而非 `w^Hh`；
3. I/Q 收缩与原生复数计算在容差内一致；
4. 结构分项在复数域严格相加得到最终算子；
5. `embedded_element_basis` 与可训练 C 同时启用时构造失败；
6. HFSS 数据启用可训练 `DeviceModel` 时构造或训练失败；
7. 正式加载器继续拒绝 `operator_iq.npy` 和 `s_parameters_iq.npy`；
8. 4×4 行列维度在全部卷积和矩阵映射中保持，不被无说明展平；
9. 全局相位变换不改变预测幅度；
10. 不同命令 batch 组织方式得到相同预测。

### 12.2 集成测试

- 同一 V2.8 幅度目录可被 V2.9 加载并复现数据指纹；
- 新固定 split 可重复生成，并与 checkpoint 严格绑定；
- 一轮 CPU smoke training 可保存、加载、评估 checkpoint；
- 各阶段的冻结参数名和实际 `requires_grad` 完全一致；
- 四目标优化输出满足总功率和单阵元幅度限制；
- 关闭任一可选层后 checkpoint 配置和状态字典无隐式兼容错误。

## 13. 实施顺序与提交边界

为便于回退和审阅，建议按以下独立提交实施：

1. `v2.9 scaffold and frozen contracts`：包骨架、数据兼容、解析收缩和合同测试；
2. `v2.9 nominal element response`：M1 模型及消融；
3. `v2.9 structured port coupling`：M2、掩码和正则；
4. `v2.9 constrained angular residual`：M3；
5. `v2.9 staged training and evaluation`：阶段调度、新盲测和分区指标；
6. `v2.9 four-target closed-loop validation`：HFSS regret 和最终候选冻结。

每个提交只增加一个可解释能力，并附对应测试或消融结果；不得在同一提交中同时扩大 CNN、
更换数据划分并调整损失，否则无法确定性能来源。

## 14. 结构逻辑自检

### 14.1 从需求到实现的闭环

| 需求 | 数学位置 | 代码位置 | 验证方式 | 状态 |
|---|---|---|---|---|
| 保留幅度-only部署合同 | `|E|` 标签 | `io/amplitude_dataset.py` | 泄漏拒绝测试 | 已实现并测试 |
| 保留相干叠加 | `h^T w` | `models/contraction.py` | 手算复数测试 | 已实现并测试 |
| 提升方向误差表达 | `g0(q)` | `models/nominal_pattern.py` | staged nominal/后验移除 | 已实现并训练 |
| 约束稳定端口混合 | `C_ant` | `models/port_coupling.py` | staged coupling/后验移除 | 已实现并训练 |
| 表达剩余结构散射 | `r_psi(q)` | `models/angular_residual.py` | staged residual/后验移除 | 已实现并训练 |
| 不伪造 HFSS 器件误差 | `F_dev=I` | `models/device.py` | 构造拒绝测试 | 已实现并测试 |
| 防止重复互耦 | 基底策略 | `physics/basis_policy.py` | 非法组合测试 | 已实现并测试 |
| 服务四目标优化 | `A(q,w)` | `optimization/` | 直接 FFD regret | 已实现并验证 |

### 14.2 依赖方向检查

依赖保持单向：

```text
contracts/coordinates
        -> io + physics
        -> models
        -> training
        -> evaluation/optimization CLI
```

- `io` 不依赖模型；
- `physics` 不依赖训练；
- 模型不读取文件、不决定数据划分；
- 训练器通过公开模型接口冻结层，不访问私有张量；
- 优化器只依赖模型的幅度输出和问题约束；
- S16P/FFD 审计路径不反向进入正式训练数据目录。

### 14.3 已发现并在规划中消除的逻辑冲突

1. **物理解释与幅度-only不可辨识冲突**：改为“有效分层”，以输出泛化为主验收；
2. **HFSS EEP 已含互耦与额外 C 冲突**：加入基底策略硬错误；
3. **发射 `h^T w` 与接收 `w^Hh` 冲突**：冻结单一复数约定并加手算测试；
4. **V2.8 多方向分支表达重叠**：合并为名义响应头加单一受限残差；
5. **当前 HFSS 无器件误差与可训练器件层冲突**：HFSS 强制 Identity；
6. **逐方向 gauge 与分项可加性冲突**：gauge 移到诊断模块，前向输出保留原始复数分项；
7. **复用 V2.8 与版本独立性冲突**：实施时复制稳定模块并记录来源，不运行时跨版本 import；
8. **调结构与复用旧 test 冲突**：建立新的盲测命令和角块。

### 14.4 自检结论

当前实现已在数据合同、数学顺序、模块职责、训练、评估和四目标应用之间形成闭环，
没有要求幅度-only 数据提供不可观测的绝对相位或真实器件标签。文件依赖为单向，V2.8
可运行基线不会被修改。

首轮实现已经冻结：`ideal_phase` 名义基底、几何近邻 C 掩码和旧 seed 28001 工程对照。
下一轮论文级实验仍需冻结：

1. 新盲测命令数、角块和固体角/目标扇区权重；
2. 是否生成只用于 HFSS 对照的 `common_pattern`；
3. 是否将当前几何近邻掩码替换为由 S16P 离线生成并固化的诊断掩码。

这三个决定影响实验配置，但不改变本文定义的代码边界和依赖关系。

## 15. 首轮实现后的证据修订

原计划的 80/40/60/20 staged 训练已完整执行，但测试全局幅度 NMSE 只有 -14.14 dB。
日志显示独立 residual 阶段不能改善 validation，而最后 20 个联合 epoch 尚在持续改善。

随后使用相同模型、数据、损失和 200 epoch 总预算执行全联合对照，得到：

- validation 全局幅度 NMSE：-15.83 dB；
- test 全局幅度 NMSE：-16.49 dB；
- test 全局功率 NMSE：-16.05 dB；
- test 幅度 dB MAE：2.16 dB；
- 参数量：34,973。

因此结构规划保持不变，但训练结论修订为：**分层用于约束表示和消融，不意味着各层必须
长时间独立优化**。当前 CLI 默认 `joint=200`，staged 模式保留为诊断。下一轮若重新研究
分阶段策略，应增加联合阶段预算，并只使用新的 validation/盲测划分作决策。
