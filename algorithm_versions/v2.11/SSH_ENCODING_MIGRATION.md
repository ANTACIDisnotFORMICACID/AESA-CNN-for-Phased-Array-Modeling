# V2.11 CNN 方向编码迁移说明

## 范围

本轮只从 V2.10 迁移方向编码和 4×4 孔径 CNN 前端，不迁移训练器、优化器、物理传播层或已有权重。V2.10 文件保持不变。

保持的结构合同为：

`27 个方向通道 + 4 个几何通道 -> 31 通道 -> 原 3×3 CNN`。

## 三种编码

1. `fourier`：逐式保留 V2.10 的 4-band Fourier 编码，作为严格控制组。
2. `ssh`：实球谐基 `K=(L+1)^2`，先用训练查询的均值和标准差归一化，再用 `Linear(K,27)` 投影。
3. `fourier_plus_ssh`：`Fourier + alpha * Linear(normalized SSH)`，其中 `alpha=0` 初始化，初始输出严格等于 V2.10。

SSH 使用 degree-major、`m=-l..l` 排列。`m>0` 为 `sqrt(2) Re(Y_lm)`，`m<0` 为 `sqrt(2) Im(Y_l|m|)`，包含 Condon-Shortley 相位。

## 数据泄漏约束

`mu_lm` 和 `sigma_lm` 只能由 `query_train.npy` 指向的训练方向计算。统计量作为模型 buffer 保存，验证、测试和部署只读取，不能重新拟合。

`Y_0_0` 是常量，标准化后为零；投影层偏置保留等价的常量自由度。

## 首轮消融

- Fourier-27：V2.10 基线；
- SSH L=4：25->27；
- SSH L=5：36->27；
- Fourier+SSH L=4：零起点残差。

VSH 得到的 L=30 是完整复矢量远场的物理带宽，不应直接令 CNN 使用 961 个 SSH 输入通道。编码阶数由上述受控消融的验证集表现选择。
