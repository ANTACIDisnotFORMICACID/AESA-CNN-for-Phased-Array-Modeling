# V2.10 后续 CNN 训练政策

## 唯一活动训练路线

从本文件生效起，后续 CNN 训练只允许使用严格正半球的单极化标量幅度监督：

- 已知输入：发射指令 `command_iq` 与方向 `direction/theta/phi`；
- 唯一测量标签：非负实数 `amplitude = |E_rx(q,w)|` 或经校准的 `|S21|`；
- 物理域：所有训练、验证和测试方向必须满足 `q_x > 0`；
- 禁止标签或训练输入：HFSS 复场、测得场相位、逐端口复算子、Q矩阵、S参数；
- 允许模型内部使用复数潜变量和已知发射指令的 IQ 表示，但它们只能由幅度损失反演，不能作为测量标签输入。

幅度监督下的复数潜变量存在方向相关公共相位不确定性：对任意方向函数 `alpha(q)`，
`h(q)` 与 `exp(j alpha(q)) h(q)` 会给出完全相同的幅度。因此，足够多的已知相位编码指令可以
恢复预测新指令幅度所需的端口相对相位，但不能唯一恢复物理绝对远场相位。当前正式输出与验收
对象只能是 `amplitude`/`power`；内部 `field_iq`/`operator_iq` 不得作为已辨识复场发布。

活动入口为：

- `v210-prepare-hfss-amplitude`：把HFSS数据一次性转换成不含复场的幅度表；
- `v210-prepare-measured-amplitude`：准备真实测量幅度表；
- `v210-prepare-amplitude-split`：按方向角块和指令划分；
- `v210-train-forward`：唯一正式训练入口；
- `v210-evaluate-forward`：幅度正向预测评估。

训练加载器只接受 `direction.npy`、`command_iq.npy`、`amplitude.npy`、`command_family.npy`、
`theta_deg.npy`、`phi_deg.npy` 和 `manifest.json`。训练目录中出现其他 NPY/NPZ 文件将直接报错。

## 当前活动基线

当前活动基线是 `artifacts/strict_single_pol_amplitude_20260908/` 中的
`v210_nominal_coupling_residual_1`，共34,973个参数。后续结构、容量、损失和数据改进均必须与
该幅度基线在独立角块、独立指令和独立物理状态上比较。

主要持续优化指标为：幅度NMSE、功率NMSE、全场/强场/弱场dB误差、方向图相关系数，以及
幅度代理进入优化器后产生的目标增益误差和瓶颈高估尾部。不得再以复场MSE或Q矩阵损失作为
正式选型指标。

## A10历史封存边界

`artifacts/controlled_psd_20260906/`、`artifacts/front_hemisphere_20260906/` 及
`src/v210/models/quadratic_psd.py` 中的A10/B10/C10属于历史复数据研究参考：

- 保留源码、checkpoint、哈希、报告和评估能力；
- 不删除、不覆盖、不在原artifact目录重训；
- 不作为默认模型、默认训练入口或后续正式选型候选；
- 仅可用于论文追溯、数值对照和解释旧结论。

若需要复核历史结果，应明确标注 `legacy complex-data reference`，不得把复核运行并入当前
幅度路线的排行榜或基线。
