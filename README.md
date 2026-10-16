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

下面的说明以 4×4 阵列为主，因为 V2.12–V2.18 的受控结构实验都在这一协议下完成。
1×4、1×8 和 2×4 数据用于跨阵列数据门禁与分阵列可行性检查，不应被理解为同一个网络
已经完成跨尺寸泛化。

| 符号 | 含义 | 4×4 情况下的形状 |
|---|---|---|
| \(q=(q_x,q_y,q_z)\) | 归一化观察方向 | \(3\) |
| \(r_n\) | 第 \(n\) 个阵元的三维坐标 | \(3\) |
| \(w_n\) | 第 \(n\) 个阵元的已知复激励 | \(1\) 个复数 |
| \(a_n(q)\) | 去除解析传播相位后的逐阵元复包络 | \(16\) 个复数 |
| \(g_n(q)\) | 包含传播相位的逐阵元复响应 | \(16\) 个复数 |
| \(E(q,w)\) | 所有阵元相干叠加后的总复场 | \(1\) 个复数 |
| \(Q(q)\) | 幅度观测可辨识的二次算子 | \(16\times16\) 复矩阵 |

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

### 4. 方向编码与阵列编码

#### 4.1 Fourier-27：给小网络提供多尺度角度坐标

直接输入 \(q_x,q_y,q_z\) 时，小型卷积网络要自行合成高频角向变化。项目采用四个固定
二倍频带：

\[
\Gamma(q)=\left[
q_d,\,
\sin(2^b\pi q_d),\,
\cos(2^b\pi q_d)
\right]_{d\in\{x,y,z\},\,b=0,1,2,3}.
\]

每个坐标由原值、4 个正弦项和 4 个余弦项组成，三个坐标一共
\(3(1+2\times4)=27\) 个通道。频率是固定的，Fourier 编码本身没有可训练参数。它的作用
不是代替物理模型，而是把不同角度尺度展开到网络容易使用的坐标系中。

#### 4.2 filtration SSH：适配正半球采样的结构化基

标准球谐函数在完整球面上正交，但本项目只使用严格正半球方向，采样密度也不一定均匀。
如果直接使用原始球谐基，同一阶内部以及不同阶之间可能出现数值相关。V2.12 因此在训练
方向和训练损失对应的经验测度上构造 filtration basis。

令 \(Y_l(q)\) 为第 \(l\) 阶实球谐块，\(\Phi_{<l}\) 为已经保留的低阶基。首先去除当前阶在
低阶空间中的投影：

\[
\widetilde Y_l
=Y_l-\Phi_{<l}
\left(\Phi_{<l}^{T}W\Phi_{<l}\right)^{-1}
\Phi_{<l}^{T}WY_l ,
\]

再对 \(W^{1/2}\widetilde Y_l\) 做 SVD，并按平方奇异值比例 \(10^{-4}\) 截断不可稳定识别
的方向。由此得到固定变换 \(T_l\) 和：

\[
\Phi_l(q)=Y_{\le l}(q)T_l.
\]

本项目使用 \(L=4\)，各阶保留秩为 \((1,3,5,7,9)\)，共 25 个模式。常数模式由
\(b_n\) 承担，所以包络中实际使用 24 个非恒定模式。每阶通过可学习门
\(\alpha_l\) 接入，每个模式再用复系数 \(\beta_{l\mu n}\) 映射到 16 个阵元。
basis、截断阈值和尺度只由训练方向拟合；validation、coverage 和 blind test 不参与。

filtration 的意义是保持“从低阶到高阶逐步扩展”的嵌套关系，同时处理受限方向域上的
数值病态。它提供的是适合当前采样域的结构化坐标，而不是对真实电磁场频谱的唯一分解。

#### 4.3 四张孔径几何图

27 个方向特征会复制到每个阵元位置，再拼接四张固定的 4×4 几何图：

- row：行坐标，从 \(-1\) 到 \(1\)；
- column：列坐标，从 \(-1\) 到 \(1\)；
- edge：边缘阵元指示；
- corner：角点阵元指示。

因此 CNN 输入为：

\[
X(q)\in\mathbb R^{31\times4\times4},
\qquad 31=27+4.
\]

4×4 两个空间轴始终对应真实阵列的行和列，不是任意重排后的图像像素。卷积共享权重因此
表达的是孔径上的局部规律；edge 和 corner 图则显式告诉网络有限阵列边界在哪里，避免它
把内部阵元与边缘阵元视为完全等价。

#### 4.4 SSH 与 Fourier 为什么同时保留

两条表示分工不同：

- SSH 分支直接生成逐阵元复包络中的低复杂度结构项；
- Fourier-27 进入 CNN，生成剩余的逐阵元 I/Q 残差；
- 可选 separation loss 只约束两条分支少做重复工作，不把这种分工包装成真实场的唯一
  物理解耦。

V2.12 的等参数实验说明 SSH 具有独立预测价值；后续 gauge-invariant 审计又否定了更强的
“SSH 必然对应真实低频算子”说法。这两个结果在 README 中同时保留。

### 5. CNN 结构如何演进

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

### 6. 从输入到输出：完整张量与计算链

以批量大小 \(B\)、方向数 \(Q\) 为例，V2.12 的前向过程如下：

| 步骤 | 计算 | 张量形状 |
|---:|---|---|
| 1 | 方向归一化并计算 Fourier-27 | \([B,Q,3]\rightarrow[B,Q,27]\) |
| 2 | 复制方向特征并拼接四张几何图 | \([B,Q,27]\rightarrow[BQ,31,4,4]\) |
| 3 | 3×3 stem 和两个 residual block | \([BQ,31,4,4]\rightarrow[BQ,24,4,4]\) |
| 4 | 3×3 I/Q head 与有界映射 | \([BQ,24,4,4]\rightarrow[B,Q,2,4,4]\) |
| 5 | I/Q 合成为 CNN 复残差 \(\delta(q)\) | \([B,Q,16]\) complex |
| 6 | 基准、SSH 与 CNN 残差相加 | \(a(q)\in\mathbb C^{B\times Q\times16}\) |
| 7 | 乘解析传播相位 | \(g(q)\in\mathbb C^{B\times Q\times16}\) |
| 8 | 与复激励逐阵元相乘并求和 | \(E(q,w)\in\mathbb C^{B\times Q}\) |
| 9 | 取复模得到监督量 | \(\hat y(q,w)\in\mathbb R_{\ge0}^{B\times Q}\) |

把这条链写成一组连续公式，就是：

\[
\begin{aligned}
z(q) &= \Gamma(q),\\
\delta(q) &= 2\tanh\!\left(\frac{
  \operatorname{CNN}([\operatorname{tile}(z(q)),M_{\mathrm{geo}}])
}{2}\right),\\
a(q) &= s\,[b+a_{\mathrm{SSH}}(q)+\delta(q)],\\
g_n(q) &= a_n(q)e^{jkq^Tr_n},\\
E(q,w) &= \sum_{n=1}^{16}g_n(q)w_n,\\
\hat y(q,w) &= |E(q,w)|.
\end{aligned}
\]

其中 \(M_{\mathrm{geo}}\) 表示四张几何图。限制到 \([-2,2]\) 的是每个 I/Q 残差分量，
不是最终场幅度；整体物理尺度由只使用训练数据估计的正数 \(s\) 恢复。

#### 6.1 V2.12 逐层结构

| 层 | 配置 | 输出 | 可训练参数 |
|---|---|---|---:|
| Fourier 编码 | 4 个固定频带 | 27 个方向特征 | 0 |
| 几何拼接 | row / column / edge / corner | 31×4×4 | 0 |
| stem | Conv2d 31→24，3×3，SiLU | 24×4×4 | 6,720 |
| residual block 1 | 两个 24→24 的 3×3 卷积 | 24×4×4 | 10,416 |
| residual block 2 | 两个 24→24 的 3×3 卷积 | 24×4×4 | 10,416 |
| I/Q head | Conv2d 24→2，3×3 | 2×4×4 | 434 |
| CNN 小计 | 6 次最长路径空间卷积 | 16 个复残差 | 27,986 |
| 基准与 SSH | 复基准、24×16 个复系数、4 个阶门 | 16 个复包络 | 803 |
| 总计 | V2.12 control | — | **28,789** |

配置中的 depth=2 是两个 residual block，不是整个模型只有两层。沿任意一条最长路径实际会
经过 stem、每个 block 的两次卷积和 head，共 6 次空间卷积。

#### 6.2 规范自由度与参数锚定

16 个复响应共有 32 个实自由度，但公共相位无法由幅度数据确定，所以可辨识维数是
\(32-1=31\)。实现中把第一个阵元的基准虚部固定为 0，并用 softplus 保证其实部为正；
其余 15 个阵元保留实部和虚部。这个锚点只是选择一个稳定的相位规范，不会增加幅度数据
本身不具备的信息。

#### 6.3 zero-start 为什么重要

两个普通 residual block 的第二个卷积和最终 I/Q head 都从零开始。DCAR 中每条轴向分支
的第二个卷积同样从零开始。因此初始化时：

\[
\delta(q)=0,
\]

模型严格从“复基准 + SSH + 解析传播相位”出发。CNN 只有在训练证据要求时才逐步增加修正，
这既减轻了小数据训练的不稳定，也使消融实验可以回答“学习残差带来了什么”，而不是比较
两个完全不同的随机函数。

### 7. 训练目标与评价指标

对一个方向—命令网格，令真值幅度为 \(y_i\)，预测为 \(\hat y_i\)。主训练目标由三个互补
部分组成：

\[
\mathcal L_{\mathrm{lin}}
=\frac{\sum_i(\hat y_i-y_i)^2}{\sum_i y_i^2+\varepsilon},
\]

\[
\mathcal L_{\mathrm{pow}}
=\frac{\sum_i(\hat y_i^2-y_i^2)^2}{\sum_i y_i^4+\varepsilon},
\]

\[
\mathcal L_{\mathrm{dB}}
=\operatorname{Huber}
\left(20\log_{10}\max(\hat y_i,\varepsilon),
      20\log_{10}\max(y_i,\varepsilon)\right).
\]

冻结实现的基本组合为：

\[
\mathcal L
=\mathcal L_{\mathrm{lin}}
+0.1\mathcal L_{\mathrm{pow}}
+0.03\mathcal L_{\mathrm{dB}}
+\lambda_r\|\delta\|_2^2
+\lambda_{\mathrm{sep}}\mathcal L_{\mathrm{sep}}.
\]

- linear NMSE 是 checkpoint 选择的主指标；
- power NMSE 更关注高功率区域和主瓣形状；
- dB Huber 让弱场误差进入训练，同时避免极深零陷产生无界梯度；
- 残差能量项抑制 CNN 无必要地覆盖物理基线；
- separation 项只在相应的 SSH–Fourier 实验中启用，用训练方向上的固定 basis 限制两条
  分支重复表示。

主模型仍是 amplitude-only 监督。V2.12 的 Q distillation 是诊断分支，V2.18 的
phase-aligned complex auxiliary 也是单独预注册的筛选变量；二者都不能反向改写
amplitude-only 基线。

单个总体 NMSE 不足以描述相控阵误差。本项目同时检查：

| 指标 | 回答的问题 |
|---|---|
| amplitude NMSE | 给定命令时，方向图整体幅度是否准确 |
| dB MAE / P95 | 误差在工程常用对数尺度上有多大 |
| 强场、弱场、深零陷分桶 | 总体均值是否掩盖相干抵消区失效 |
| phase-aligned operator NMSE | 去除公共相位后，逐阵元复响应是否接近参考 |
| Q-NMSE | 幅度真正可辨识的二次算子是否准确 |
| gradient cosine / relative error | 模型能否为波束权重优化提供正确局部方向 |
| decision gap / regret | 用模型选择的权重在独立参考上损失多少 |

### 8. 模型与数理建模的主要改进

这里的“改进”指本项目版本之间经过对照验证的设计变化，不主张这些模块在所有相关研究中
首次出现。

#### 8.1 从命令回归改为算子建模

普通黑盒可以直接学习 \((q,w)\mapsto y\)，但容易把训练命令的分布写进网络。AESA-CNN
学习的是 \(q\mapsto g(q)\)，命令 \(w\) 只在最后的解析相干求和中出现。这样做同时保留：

- 对新命令的复用能力；
- 场对激励的线性关系；
- 对波束权重的可微梯度；
- 对单个阵元响应和组合场的分层审计入口。

#### 8.2 用结构保证 rank-1，而不是训练后修补

模型先产生复向量 \(g\)，再构造 \(Q=g^*g^T\)。因此对任意向量 \(v\)：

\[
v^HQv=|g^Tv|^2\ge0.
\]

Hermitian、半正定和 rank-1 都由结构自动满足，不需要给一个自由 \(16\times16\) 矩阵增加
软惩罚。参数自由度也从一般复矩阵显著压缩为一个去除公共相位后的 31 维实对象。

#### 8.3 解析相位与学习包络解耦

\(e^{jkq^Tr_n}\) 对 47 GHz 阵列随角度变化很快，却完全由几何决定。把它解析计算后，
网络只拟合较平滑的 \(a(q)\)，减少了学习难度，也让跨频率或几何变化时需要替换的部分更加
明确。去相位算子：

\[
\widetilde Q(q)=a(q)^*a(q)^T
\]

还可以单独审计网络学到的包络结构，避免已知传播相位掩盖误差。

#### 8.4 方向域与孔径域分开编码

Fourier 和 filtration SSH 描述“往哪里看”，4×4 几何图及卷积描述“阵元位于孔径哪里”。
两类坐标在进入局部残差网络前保持明确分工。相比将方向、阵元编号和命令展平成一个长向量，
这种表示保留了角域多尺度结构与阵列二维邻域。

#### 8.5 从图消息实验收敛到 DCAR

V2.15 证明局部消息传播有用，但真实 S 图相对 shuffled-S 的增益只有 0.043 dB，无法支持
“收益来自真实互耦拓扑”。V2.16 因此去掉 S 图，把假设缩小为规则阵列的两条孔径轴：

\[
B_D(H,q)=H+
\gamma_r(q)R_2\sigma(R_1H)+
\gamma_c(q)C_2\sigma(C_1H).
\]

每个 block 的 \(\gamma_r,\gamma_c\) 由 Fourier-27 经线性层和 tanh 生成，允许不同观察方向
选择、抑制或反向补偿行列传播。两个 block 串联后会出现先行后列和先列后行的组合项，所以
并联轴向卷积仍能形成对角与二维多跳交互。

最终参数匹配版本交替使用 \((36,35)\) 和 \((35,36)\) 的行列分支宽度，使两条轴总容量对称：

| 对照 | 参数量 | 与 V2.12 的差异 |
|---|---:|---:|
| V2.12 3×3 control | 28,789 | 基准 |
| DCAR static-PM | 28,643 | −0.507% |
| DCAR gated-PM | 28,755 | −0.118% |

因此 V2.16 Stage-1 的比较主要回答传播结构是否有效，而不是“更多参数是否有效”。它也保留
了失败的 Stage-0：当深度、几何输入和参数量同时变化时，validation 退化 2.216 dB；恢复
控制变量后，gated-PM 才在原协议取得约 1 dB 改善。

#### 8.6 把停止规则纳入算法研究

数据迁移、命令覆盖、参数预算、配对种子和 blind-test 开启条件都在训练前冻结。V2.18
没有因为 canonical 迁移结果不理想而继续追加 arm 或延长训练，而是按规则停在
`ENGINEERING_CANDIDATE_NONE`。这部分不增加网络表达能力，但使“结构改进”与
“反复试到偶然成功”可以区分。

### 9. 数据隔离与模型选择

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
4. [V2.12–V2.16 完整技术汇报](algorithm_versions/v2.16/V2.12_V2.16_HFSS完整汇报_20261015.md)：
   查看逐层参数、HFSS 验证流程和完整数值结果。
5. [V2.17](algorithm_versions/v2.17/README.md)：查看跨阵列数据门禁和可行性试验。
6. [V2.18](algorithm_versions/v2.18/README.md)：查看迁移失败归因、停止规则和最终筛选结论。

## 版本与发布规则

- 已冻结版本不回写模型选择；勘误使用独立提交并说明原因。
- 新假设进入新版本，并在训练前冻结数据边界、对照、种子、预算和停止门禁。
- 大型可再生数据、checkpoint 与 solver cache 独立发布，不与代码提交混合。
- 结果统一使用 `PASS`、`FAIL`、`BLOCKED`、`PENDING`，不把缺失证据包装为成功。
- 每个版本对应一个同名 Git 标签；发布前核对提交、标签、本地链接和轻量回归结果。
