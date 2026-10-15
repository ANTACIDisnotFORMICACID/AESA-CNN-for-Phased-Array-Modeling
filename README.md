# AESA-CNN：面向有源相控阵的物理约束神经网络

本项目研究 47 GHz 有源相控阵的远场幅度预测与波束控制基础问题，打通
**HFSS 全波仿真 → 可审计数据集 → 物理约束复算子 → CNN 残差建模 → 受控验证**
的完整工程链路。当前工作围绕三个可复现的问题展开：

1. 只有幅度观测时，如何保留阵列的相干叠加物理结构？
2. 哪些误差应由解析模型解释，哪些残差适合交给神经网络学习？
3. 一个结构改进是否在数据隔离、等预算对照和预注册门禁下仍然成立？

> 当前边界：**V2.16** 是最新完成控制变量验证的 CNN 设计；**V2.17** 完成跨阵列
> 数据门禁并进行 validation-only 可行性试验；**V2.18** 完成规范化数据迁移筛选，
> 结果为 `ENGINEERING_CANDIDATE_NONE`。canonical blind test 始终未用于模型选择。

## 项目概览

| 项目维度 | 内容 |
|---|---|
| 研究对象 | 47 GHz、1×4 / 1×8 / 2×4 / 4×4 有源相控阵 |
| 输入 | 观察方向、阵元几何、已知复 IQ 激励命令 |
| 训练监督 | 单极化远场幅度；复场只用于独立物理审计或明确授权的辅助实验 |
| 核心模型 | 解析传播相位 + 相干 rank-1 算子 + SSH/Fourier 方向表示 + CNN 复残差 |
| 代表性结构 | V2.12 3×3 residual、V2.15 局部图消息、V2.16 方向门控轴向残差 DCAR |
| 验证方式 | 训练/验证/盲测隔离、配对种子、近等参数对照、强弱场分桶、停止门禁 |
| 当前结论 | DCAR 在原 V2.16 协议中稳定改善约 1 dB，但未在 V2.18 canonical 迁移中形成新工程候选 |
| 工程状态 | 仿真和离线审计完成到文档所列边界；尚无板级、暗室 OTA 或 FPGA 部署验收 |

## 系统流程

```text
HFSS 阵列工程
   │
   ├─ build → solve → export S 参数 → export embedded element patterns
   │                                           │
   └──────────── direct field audit ───────────┘
                         │
                  canonical complex operator
                         │
          train / validation / blind-test 隔离
                         │
   解析相位 + rank-1 骨架 + SSH/Fourier + CNN residual
                         │
             amplitude / operator / Q 多层审计
                         │
             预注册 gate 决定继续、停止或晋级
```

HFSS 阶段采用严格的
`build -> solve -> export_s -> export_eep -> direct audit` 顺序。工程文件存在、脚本返回成功
或配置写入均不算数据 PASS；只有 S 参数和端口 EEP 可读，并且直接组合场交叉验证通过后，
数据才允许进入后续处理。

## 算法设计

### 1. 任务定义：从方向和阵列命令预测远场幅度

令观察方向为 \(q\in\mathbb R^3\)，第 \(n\) 个阵元位置为 \(r_n\)，已知复激励命令为
\(w\in\mathbb C^N\)。模型不直接把一条命令映射为一个幅度，而是先预测该方向上的
逐阵元复响应 \(g(q)\in\mathbb C^N\)：

\[
g_n(q)=a_n(q)\exp\!\left(jkq^Tr_n\right),
\]

其中 \(\exp(jkq^Tr_n)\) 是由阵元位置和频率确定的解析传播相位，\(a_n(q)\) 是需要学习的
去传播相位复包络。随后严格使用发射约定完成相干叠加：

\[
E(q,w)=g(q)^T w,
\qquad
\hat y(q,w)=|E(q,w)|.
\]

这样，同一个方向算子 \(g(q)\) 可以解释多组 DFT、随机相位和优化命令，网络不能针对某条
命令单独记忆答案。

### 2. 幅度监督下的可辨识对象

幅度观测无法确定每个方向的公共复相位，因此项目不把潜在复场的绝对相位当作唯一真值。
平方幅度可以写为：

\[
\hat y^2(q,w)=w^H Q(q)w,
\qquad
Q(q)=g(q)^*g(q)^T.
\]

\(Q(q)\) 对逐方向公共相位不变，是 amplitude-only 条件下可审计的 Hermitian rank-1
二次算子。项目因此同时报告：

- **amplitude NMSE / MAE**：有限命令下的最终预测质量；
- **phase-aligned operator NMSE**：去除公共相位后的复算子误差；
- **Q-NMSE**：幅度数据真正可辨识的完整二次算子误差；
- **强场、弱场和深零陷分桶**：避免总体均值掩盖相干抵消区的失效。

### 3. 物理骨架：已知规律解析计算，网络只学习剩余项

去相位复包络使用以下结构：

\[
a_n(q)=s\left[
b_n+
\sum_{l=1}^{4}\alpha_l\sum_{\mu}\phi_{l\mu}(q)\beta_{l\mu n}
+\delta_n(q)
\right].
\]

| 模块 | 作用 | 设计原因 |
|---|---|---|
| 解析传播相位 | 根据频率、方向与阵元坐标计算确定性相位 | 避免 CNN 重学已知阵列几何 |
| 逐阵元复基准 \(b_n\) | 表示方向无关的增益和相位偏置 | 提供稳定、低复杂度的起点 |
| filtration SSH | 表示结构化、多尺度、慢变的角向成分 | 统计量只由训练方向拟合，防止验证泄漏 |
| Fourier-27 编码 | 提供方向坐标的多频正弦/余弦特征 | 缓解普通坐标网络的高频谱偏置 |
| CNN 残差 \(\delta_n(q)\) | 学习低阶物理项尚未解释的局部复包络 | zero-start 初始化保证模型起点等于物理基线 |

SSH 在 V2.12 的等参数十种子实验中表现出稳定预测价值，但其更强的
gauge-invariant operator 平滑性解释没有通过，因此本项目只把 SSH 作为结构化方向编码器，
不把它宣称为唯一真实的低频物理场。

### 4. CNN 结构如何演进

#### V2.12：二维残差基线

V2.12 将 Fourier-27 特征复制到 4×4 阵元平面，并拼接 row、column、edge、corner
四张几何图，形成 31 通道输入。网络使用 3×3 stem、两个 3×3 residual blocks 和
zero-start IQ head，共 28,789 个可训练参数。它建立了后续结构实验共同使用的物理主干和
数据边界。

#### V2.15：方向条件化局部图消息

V2.15 将一个普通残差块替换为方向门控的行邻接、列邻接和稀疏 S 图消息块。候选相对
control 在 validation、coverage probe 和一次性 test 上分别平均改善
`1.433 / 1.388 / 1.334 dB`；但真实 S 图相对 shuffled-S 只改善 `0.043 dB`，没有通过
机理门禁。因此支持的结论是“局部消息传播偏置有效”，而不是“网络学到了真实互耦拓扑”。

#### V2.16：Direction-Conditioned Axial Residual

V2.16 不继承 V2.15 的 S 图，而是把每个 3×3 residual block 分解为并联的
3×1 行分支和 1×3 列分支：

\[
B_D(H,q)=H+
\gamma_r(q)R_2\,\sigma(R_1H)+
\gamma_c(q)C_2\,\sigma(C_1H).
\]

方向编码生成 \(\gamma_r(q)\) 与 \(\gamma_c(q)\)，控制不同观察方向下两条孔径轴的贡献。
两个 DCAR block 串联后能够形成跨轴、对角和多跳交互。最终受控版本恢复了 V2.12 的
输入、深度、stem/head、激活位置和零初始化，并将参数差控制为 `-0.118%`，从而把比较变量
收敛到残差传播方式本身。

在原 V2.16 协议内，近等参数 gated-DCAR 相对 V2.12 control 的平均改善为：

| 指标 | 改善 | 配对种子 |
|---|---:|---:|
| validation amplitude NMSE | `+0.983 dB` | 3/3 正向 |
| coverage probe | `+1.011 dB` | 3/3 正向 |
| complex Ephi operator | `+1.019 dB` | 3/3 正向 |

该结论属于完成控制变量的探索性证据，不等于已经通过新的 confirmatory blind test。

### 5. 数据隔离与模型选择

```text
train
  ├─ 拟合归一化、SSH basis、模型参数
  └─ 生成训练统计

validation / coverage probe
  ├─ checkpoint 与预注册结构门禁
  └─ 不反向传播，不拟合训练统计

blind test
  └─ 仅在结构、阈值和 checkpoint 全部冻结后才允许一次性读取
```

V2.17 和 V2.18 的 canonical blind test 仍保持封存。`PENDING`、`BLOCKED`、配置完成、模型已保存
或测试程序能启动都不计为 `PASS`。

## 版本演进

项目演进不是简单堆叠网络，而是逐步关闭数据、物理和实验设计中的不确定性：

```text
V2.5–V2.7   轻量控制器、固定 1×4 阵列与物理可达边界
      ↓
V2.8–V2.10  4×4 amplitude-only 数据合同、前向模型与严格基线
      ↓
V2.11–V2.12 SSH/Fourier 方向表示与相干 rank-1 主干
      ↓
V2.13–V2.14 物理闭环、噪声、覆盖与结构筛选
      ↓
V2.15–V2.16 图消息与受控 DCAR 结构实验
      ↓
V2.17        四阵列 canonical 数据与分阵列 feasibility pilot
      ↓
V2.18        4×4 canonical 迁移归因和辅助监督筛选
```

完整的“继承关系—主要变化—冻结结论—下一版本”见
[版本索引](algorithm_versions/README.md)。

## 关键结果与诚实边界

| 版本 | 已验证结果 | 不支持的结论 |
|---|---|---|
| V2.12 | 建立可运行的解析相位 + rank-1 + SSH/Fourier + residual CNN 主干 | SSH 是唯一真实低频场 |
| V2.15 | 局部图消息改善前向预测 | 真实 S 拓扑是收益来源 |
| V2.16 | gated-DCAR 在原协议中约 1 dB、3/3 种子改善 | 已通过新 blind test 或已替代全部基线 |
| V2.17 | 四种阵列 HFSS 数据门禁全部完成；pilot 为 `PARTIAL_2_OF_4` | 单一模型已实现跨尺寸零样本泛化 |
| V2.18 | canonical 迁移归因完成；最终 `ENGINEERING_CANDIDATE_NONE` | canonical 数据上存在已晋级的新结构 |

当前证据来自 HFSS 仿真、离线数值审计和受控验证，不等同于真实馈电链、板级、暗室 OTA、
Vitis AI、XModel、bitstream 或实时系统验收。

## 仓库结构

```text
README.md                    算法设计、版本主线、证据边界和检查入口
requirements.txt            轻量回归所需的 Python 依赖
algorithm_versions/
  README.md                 V2.5–V2.18 版本索引
  run_all_checks.py         不依赖大型外部数据的统一检查入口
  v2.5 ... v2.18/           每版独立保存说明、代码、协议、测试和精简证据
    v2.6/data/               1×4 HFSS 参考数据
    v2.17/hfss/              跨阵列工程、S 参数、EEP 与审计回执
```

大型 HFSS solver cache、逐方向原始导出、canonical NPZ 和训练 checkpoint 不进入普通 Git
历史。仓库保留源代码、数据合同、配置、测试、精简结果和关键审计回执；需要完整训练数组
或 checkpoint 的 V2.14–V2.16、V2.18 在本仓库中明确记为 `BLOCKED`。

本仓库不继承接手项目时的旧代码和提交历史，也不收录批量临时 JSON、逐 epoch 图片、训练
history 和重复格式逐样本 CSV。保留内容以最终报告、机器汇总、必要回执和能解释物理现象的
代表图为限。

## 快速检查

当前分支已在 Python 3.13、NumPy 和 PyTorch 环境验证。克隆后可从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -X utf8 -B algorithm_versions/run_all_checks.py
```

该入口运行 V2.8–V2.13 与 V2.17 中不依赖外部大型数据包的 177 项测试，并另外执行
V2.5 的 35 项单元测试和可移植产物校验、V2.6 的 16 项测试、V2.7 的 16 项测试。
代码测试通过不代表重新完成 HFSS 求解、模型训练或硬件验证。

## 推荐阅读顺序

1. [版本索引](algorithm_versions/README.md)：先建立版本继承和结论边界。
2. [V2.12](algorithm_versions/v2.12/README.md)：理解当前物理骨架。
3. [V2.15](algorithm_versions/v2.15/README.md) 与
   [V2.16](algorithm_versions/v2.16/README.md)：查看两种结构实验及控制变量设计。
4. [V2.17](algorithm_versions/v2.17/README.md)：查看跨阵列数据门禁和可行性试验。
5. [V2.18](algorithm_versions/v2.18/README.md)：查看迁移失败归因、停止规则和最终筛选结论。

## 版本与发布规则

- 已冻结版本不回写模型选择；勘误使用独立提交并说明原因。
- 新假设进入新版本，并在训练前冻结数据边界、对照、种子、预算和停止门禁。
- 大型可再生数据、checkpoint 与 solver cache 独立发布，不与代码提交混合。
- 结果统一使用 `PASS`、`FAIL`、`BLOCKED`、`PENDING`，不把缺失证据包装为成功。
- 每个版本对应一个同名 Git 标签；发布前核对提交、标签、本地链接和轻量回归结果。
