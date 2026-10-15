# AESA-CNN V2.15

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 方向条件化局部图 CNN 结构实验 |
| 继承 | V2.12 的解析相位、rank-1、SSH/Fourier 骨架和冻结数据协议 |
| 主要变化 | 将一个普通 3×3 residual block 替换为行、列和稀疏 S 图消息块 |
| 冻结结论 | 预测门禁通过；真实 S 图相对 shuffled-S 只改善 0.043 dB，机理门禁未通过 |
| 后续关系 | V2.16 去除 S 图混杂，从 V2.12 重新分叉并严格控制轴向结构变量 |

V2.15 验证一个完整的新 CNN：在原解析相位与 SSH/Fourier 骨架内，将一个普通 3×3 residual block 替换为方向条件化的局部阵元图块，并用真实/打乱 S 参数作物理反事实。

> 仓库状态：`BLOCKED`（完整复跑）。本仓库没有纳入本版依赖的大型训练数组和 checkpoint；
> 冻结报告可审阅，但精简仓库不能据此宣称重新完成训练。

协议见 [V2.15_PREREGISTRATION.md](V2.15_PREREGISTRATION.md)。训练只读取独立的 train/validation 包；coverage_probe 在训练后审计；test 由独立的一次性评估程序在结构和 checkpoint 全部冻结后读取。

当前 CNN 的逐层张量、精确参数量、图传播公式及与 V2.12 的结构差异见 [V2.15_CNN_ARCHITECTURE_20260914.md](V2.15_CNN_ARCHITECTURE_20260914.md)。

## 结果

完整结果见 [V2.15_RESULTS_20260913.md](V2.15_RESULTS_20260913.md)。主候选相对 control 在 validation、coverage_probe、一次性 test 上分别平均改善 `1.433/1.388/1.334 dB`，均为 3/3 种子正向，达到预测资格门槛。

后续图机理拆解、complex-S/EEP 物理链、强场约束和波束形成下游指标的推导见 [V2.15_FUTURE_OUTLOOK_PHYSICS_20260913.md](V2.15_FUTURE_OUTLOOK_PHYSICS_20260913.md)。该文档属于 test 开启后的研究展望，不改变 V2.15 结果，也不授权继续用旧 test 调参。

真实 S 图相对 shuffled-S 只改善 `0.043 dB`，未达到物理机理 gate。因此有效的是方向条件化局部图结构，尚不能把收益归因于真实 S 拓扑。test 已冻结，V2.15 不再允许根据 test 调参。

机器结果：

- `artifacts/forward_validation_20260913/FORWARD_VALIDATION_SUMMARY.json`
- `artifacts/final_test_20260913/FINAL_TEST_SUMMARY.json`

回归测试：

```powershell
python algorithm_versions/v2.15/tools/run_tests.py
```

原实验工作区记录的回归结果为：V2.12 `31/31`、V2.13 `19/19`、V2.14 `16/16`、V2.15 `7/7`。
在本精简仓库中，依赖已排除大型数组的用例按 `BLOCKED` 处理。
