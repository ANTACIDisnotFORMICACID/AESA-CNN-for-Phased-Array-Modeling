# V2.8 幅度信息正向模型实现

## 1. 两条严格隔离的数据路径

### HFSS 可行性路径

```text
raw FFD H_hfss[Q,XYZ,IQ,4,4]
        ↓ 指定单一接收极化投影
h_hfss[Q,IQ,4,4]
        ↓ 与已知命令码本 w[N,IQ,4,4] 复数收缩
z_hfss[N,Q]
        ↓ 绝对值；在此处永久丢弃测量相位
amplitude_hfss[N,Q]
```

复数 FFD 只由 `prepare_hfss_amplitude.py` 在内存中使用。输出训练目录不含复场和 S16P。

### 实物部署路径

```text
同一已知命令码本 w[N,IQ,4,4]
        ↓ 实物阵列 + 单通道接收天线 + 球面扫描
amplitude_measured[N,Q]
        ↓ prepare_measured_amplitude.py
同一 AmplitudeDataset 合同
```

二者使用同一个 `train_forward.py`。实际部署从头训练实物模型，不混入 HFSS 样本和
HFSS checkpoint。

## 2. 学习接口

| 对象 | 形状 | 是否为测量信息 | 用途 |
|---|---:|---|---|
| `direction` | `[Q,3]` | 云台坐标 | 查询输入 |
| `command_iq` | `[N,2,4,4]` | 否，是已知控制设置 | 完整模型输入 |
| `amplitude` | `[N,Q]` | 是，唯一标签 | 幅度损失 |
| `H_ideal(q)` | `[B,Q,2,4,4]` | 否，几何计算 | 物理主值 |
| `H_latent(q)` | `[B,Q,2,4,4]` | 不可直接观测 | CNN 输出的潜在复信道 |
| `predicted_amplitude` | `[B,Q]` | 预测量 | 与标签比较 |

训练器永远不会接收到 HFSS/实测相位标签。命令 I/Q 中的相位是已知控制量，实物部署时
同样可获得，不构成信息泄漏。

## 3. 为什么仍保留潜在复数信道

测量幅度满足：

```text
A(q,w) = |sum_(r,c) h(q,r,c) * w(r,c)|
```

如果让普通 CNN 直接从命令预测幅度，它可能记忆相位码且破坏阵元相干关系。当前结构让
CNN 只预测 `h(q)`，再用固定复数收缩计算场，最后取绝对值。这样可表达相长和相消，且
阵元身份始终保持为 `4×4`。

幅度观测无法确定 `h(q)` 的共同全局相位，但共同相位不会改变任何 `|h·w|`。实现通过
参考阵元固定这个无物理影响的自由度。

## 4. 必需的相位码本

逐端口幅度只能得到 16 个阵元各自的模，无法确定阵元之间的相对传播相位。因此代码生成：

- 16 个单阵元命令：检查端口映射和死端口；
- 16 个二维 DFT 相位命令：提供结构化相干组合；
- 默认 96 个随机相位命令：补充相位恢复约束和优化分布覆盖。

暗室必须执行相同的 `command_iq.npy`。若硬件不支持关闭单阵元或连续相位，应在采集前
按真实码字重新生成码本，HFSS 可行性数据也必须使用同一个硬件可实现码本。

## 5. 物理层级

| 层级 | 幅度模型实现 | 意义 |
|---|---|---|
| L0 | `IdealScalarFarFieldOperator` | 指定接收通道下的理想阵列相位主值 |
| L1 | `ScalarComplexElementGain` | 阵元个体幅相偏差的潜变量 |
| L2 | `ScalarComplexLocalCoupling3x3` | 同排、跨排和对角近邻耦合 |
| L3a | direction 3×3 branch | 方向相关嵌入式响应修正 |
| L3b | edge/corner 3×3 branch | 有限阵面边缘和角点修正 |
| L4 | optional multihop | 有证据时才启用的远程上下文 |
| 固定层 | complex contraction + absolute value | 全部 16 阵元相干汇聚及可测幅度 |

所有空间卷积保持 `3×3` 和 `[4,4]`，命令不会被展平成长度 16。

## 6. 文件职责

- `io/amplitude_dataset.py`：严格幅度数据合同，并拒绝复场/S参数文件。
- `io/amplitude_synthesis.py`：HFSS 接收极化投影、码本和幅度观测生成。
- `models/amplitude_forward.py`：潜在复数二维算子及幅度输出。
- `training/amplitude.py`：幅度损失、相位码/角域双重留出、训练和评估。
- `cli/prepare_hfss_amplitude.py`：只为可行性测试生成幅度标签。
- `cli/prepare_measured_amplitude.py`：将暗室数组转换为同一合同。
- `cli/train_forward.py`：正式幅度训练入口。
- `cli/evaluate_forward.py`：未见相位码和未见角域联合评估。
- `cli/optimize_targets.py`：四目标最弱功率最大化。

原 `BasisDataset`、复数算子 CNN 和 S16P 适配器保留为 HFSS 审计及方法消融，不再是
正式可部署训练入口。

## 7. 可移植成立的边界

软件接口相同不等于物理数据自动等价。HFSS 与实物仍必须一致地定义：

- 接收极化通道；
- 阵元命令参考面和功率归一化；
- 未激励端口终端；
- 球面半径与坐标原点；
- 幅度单位和接收链校准；
- 实际可实现的相位/幅度码字。

这些字段在 amplitude manifest 中是必填项。真实值未知时不得用占位值产生正式训练数据。

