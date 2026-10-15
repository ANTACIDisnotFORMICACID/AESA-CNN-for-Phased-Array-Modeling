# V2.9 实现说明

## 1. 项目边界

V2.9 是独立的 `v29` Python 包。稳定的数据、坐标、优化和 HFSS 审计模块由 V2.8 迁移，
运行时不 import `v28`；训练数据暂时直接引用 V2.8 的兼容数据目录，避免复制大文件。

正式训练可观测量仍只有：

```text
known direction + known command I/Q -> measured scalar amplitude
```

接收复相位和 S 参数不会进入模型输入、损失或 checkpoint 输入合同。

## 2. 前向实现

顶层位于 `src/v29/models/amplitude_forward.py`：

\[
\begin{aligned}
b(q)&=g_0(q)\odot a_0(q),\\
h(q)&=C_{\mathrm{ant}}^T b(q)+r_\psi(q),\\
E(q;w)&=h(q)^T w,\\
A(q;w)&=|E(q;w)|.
\end{aligned}
\]

### 2.1 理想传播 `physics/ideal_far_field.py`

根据冻结的 4×4 坐标和 47 GHz 波数生成 `exp(+j k q·r)`。该层没有可训练参数。

### 2.2 名义方向响应 `models/nominal_pattern.py`

方向 Fourier 特征与行、列、边缘、角点确定性地图进入二维 3×3 CNN。网络输出：

- 有界 `log_amplitude`；
- 以 `(1+delta_real, delta_imag)` 表示并归一化的单位复相位因子。

因此相位没有 `-180/180°` 跳变，零初始化严格给出 `g0=1`。

### 2.3 端口混合 `models/port_coupling.py`

实现：

\[
C=I+P_{\mathrm{off}}(M\odot K+UV^T).
\]

`M` 是 4×4 几何近邻掩码，`K` 是复数稀疏修正，`UV^T` 是秩 1 非局部项。对角线始终
固定为 `1+0j`；互易性目前为软正则，而非硬矩阵对称。

### 2.4 单一残差 `models/angular_residual.py`

只保留一个方向相关 4×4 复残差 CNN，宽度 12、深度 1、零初始化。边缘和角点只是输入
特征，不再各自拥有能修改完整算子的自由分支。

### 2.5 器件层 `models/device.py`

当前只实现 `IdentityDeviceModel`。模型构造器在 `source_type=hfss_amplitude` 时拒绝任何
非恒等器件模型，防止把天线误差伪装成不存在的器件误差。

### 2.6 固定收缩 `models/contraction.py`

以双实数 I/Q 精确执行普通复数乘积 `h^T w`，不对命令取共轭。模型分项为：

```text
ideal
+ nominal_pattern_delta
+ port_coupling_delta
+ angular_residual_delta
= operator_iq
```

全局相位对齐移至 `physics/phase_alignment.py`，只供诊断，不改变训练前向链。

## 3. 文件职责

```text
src/v29/
├── contracts.py, coordinates.py, complex_ops.py
├── io/                         幅度合同、固定划分、FFD/S16P审计
├── physics/                    理想算子、基底策略、诊断相位对齐
├── models/                     V2.9 名义响应、C、残差、器件和收缩
├── training/
│   ├── amplitude.py            幅度损失、训练和评估
│   └── stages.py               联合/阶段训练冻结规则
├── optimization/               四目标约束与投影梯度
└── cli/                        数据、训练、评估、诊断和优化入口
```

`forward_model.py` 与 `operator_residual.py` 保留为迁移自 V2.8 的三分量 HFSS 审计基线，
不属于正式 amplitude-only V2.9 模型。

## 4. 训练实现

`train_forward.py` 保存新 schema：

```text
v2.9-amplitude-operator-1
```

checkpoint 记录：

- 数据和 split 指纹；
- 输入合同与 `transmit_h_transpose_w` 复数约定；
- 模型各层配置；
- requested/executed epoch；
- 每个训练阶段的可训练参数名、初始 validation 和选中轮次；
- 完整 history 与固定 command/query indices；
- `measured_phase_used=false`、`s_parameters_used=false`。

实现了两种 200 epoch 调度：

- `joint`：全部三层共同训练 200 epoch；当前推荐并已形成正式候选；
- `default`：80 nominal + 40 coupling + 60 residual + 20 joint；作为阶段归因对照。

首轮证据显示阶段方案把过多预算分给单层冻结训练，尤其独立 residual 阶段不能改善
validation；全联合训练在 200 epoch 仍持续改善，因此 CLI 默认改为 `joint`。

## 5. 正则和评估

训练目标包含幅度 NMSE、功率 NMSE、稳健 dB 损失，以及：

- 残差输出能量；
- 耦合矩阵偏离单位阵的 L2；
- `C-C^T` 互易软约束。

validation 选模仍以可观测输出损失为准，不以正则项或隐藏参数“更像物理”替代预测性能。

`diagnose_structure.py` 支持在固定 split 上进行 full、去 C、去 residual 和 nominal-only
后验移除诊断，并输出各层参数量、复算子 RMS 和耦合正则统计。

## 6. 安全约束

- `embedded_element_basis + enable_coupling` 直接报错；
- 当前可训练模型只接受 `ideal_phase`，其它基底明确保留为审计策略，避免配置名与实现不符；
- HFSS 训练强制恒等器件层；
- 正式数据加载器拒绝复相位和 S 参数文件；
- 4×4 行列轴在全部 CNN 中保留；
- checkpoint 与数据指纹、固定 split 严格绑定。

## 7. 当前未完成的研究项

- 新的真正未读取盲测 split；
- 随机幅相、稀疏和硬件量化命令的新数据采集；
- 三个以上训练种子的统计显著性；
- 由 S16P 自动生成固定稀疏掩码；
- OTA 器件状态与非线性模型；
- FPGA/Vitis AI 量化和板端验证。

这些项目不影响当前 HFSS 前向模型、评估和四目标优化链路的完整运行，但属于论文级结论和
实物部署前必须完成的验证。
