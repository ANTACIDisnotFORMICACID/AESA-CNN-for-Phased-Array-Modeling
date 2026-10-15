# AESA-CNN V2.10

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 严格幅度监督的活动基线版 |
| 继承 | 整理 V2.9 的多条复数据和幅度数据实验路线 |
| 主要变化 | 冻结 `q_x>0` 正半球、单极化、纯幅度监督和 34,973 参数前向基线 |
| 冻结结论 | 同一 HFSS 状态内的前向预测与四点优化对照完成；无 FPGA、硬件或 OTA 验收 |
| 后续关系 | V2.11 保持物理和训练链不变，仅比较方向编码；V2.12 冻结新主干 |

## 当前状态

`amplitude_only_training_default / strict_front_hemisphere / active_baseline_34973 / legacy_A10_archived / no_fpga_or_ota_validation`

后续唯一正式训练路线已经切换为**严格正半球、单极化、纯幅度监督**。当前需要持续优化的
正向预测基线是 [CURRENT_FORWARD_BASELINE.md](CURRENT_FORWARD_BASELINE.md)；训练数据和
历史封存边界见 [TRAINING_POLICY.md](TRAINING_POLICY.md)。

这里的“纯幅度”是指测量监督只有非负标量幅度。已知发射指令仍以两通道IQ表示，否则无法
表达相位编码；模型内部也可使用复数潜变量，但训练入口不得读取HFSS复场、测得相位、逐端口
复算子、Q矩阵或S参数。

V2.10 已完成 V2.9 复数据路线的历史对照，并完成严格幅度训练。A10/B10/C10及正半球
F0/FS/FM/FSM现均作为封存参考，不再是默认训练或模型选型路线。

详细结论见：

- [正半球 CNN 与四点优化受控实验](artifacts/front_hemisphere_20260906/RESULTS.md)
- [受控实验结果](artifacts/controlled_psd_20260906/RESULTS.md)
- [严格单极化幅度训练报告](artifacts/strict_single_pol_amplitude_20260908/REPORT.md)
- [四目标恒模相位优化完整记录](artifacts/four_point_optimization_complete_record_20260908/README.md)
- [A/B/C 整理与去重清单](CONSOLIDATION_MANIFEST.md)
- [迁移清单](MIGRATION_MANIFEST.md)
- [任务合同与缺失输入](MISSION_CONTRACT.md)
- [原始优化计划](NEXT_OPTIMIZATION_PLAN.md)

## 2026-09-08 活动幅度基线

- 模型：`v210_nominal_coupling_residual_1`，34,973参数；
- 数据：只含已知指令、`q_x>0`方向和单极化标量幅度；
- 未见角块×未见指令测试：幅度NMSE −17.903 dB、MAE 1.579 dB、相关系数0.974；
- 独立HFSS reserved holdout：幅度NMSE −19.055 dB、MAE 1.687 dB、相关系数0.984；
- 半球峰值前10 dB强场MAE 0.536 dB。

## 2026-09-06 正半球复数据专项实验（历史封存）

按物理应用边界，新活动把训练、验证、四点和局部邻域严格限制为 `q_x>0`，反半球参与数为
0；阵列结构和 HFSS 状态保持不变。12 个配对训练运行与 600 组前半球四点留出评估已完成。

- 固定状态前半球正向 NMSE 从旧 A10/21011 的 −26.52 dB 提高到约 −27.1 dB；
- 开发集曾冻结选择 FM（最坏 `ΔQ` 相位难例），但留出集配方排序不稳定；该路线现仅作参考；
- 真实 HFSS 低预算四点求解仍以 O0 最好，regret P95 为 2.36%；
- 谱上界证明 95% 的测试四点即使放宽恒模限制，也不可能同时达到各自单点数值上限的 70%；
- 局部鲁棒 O4 显著减少约 2° 邻域下降，但牺牲名义点性能，应在真实误差分布确定后调权。

该活动仍是同一物理状态内的数值研究，不构成轨道、跨状态、硬件或 OTA 验收。

## 历史复数据冻结结果摘要

|方法|参数|幅度 NMSE|强场 MAE|seed 标准差|四点 regret P95|瓶颈高估 P95|
|---|---:|---:|---:|---:|---:|---:|
|V2.9 C|28,854|−24.822 dB|0.214 dB|0.520 dB|18.75%|1.020 dB|
|A10|28,854|−24.635 dB|0.227 dB|0.177 dB|18.77%|0.937 dB|
|B10|28,420|−24.338 dB|0.222 dB|0.273 dB|19.84%|0.929 dB|
|C10|28,420|−22.127 dB|0.298 dB|0.126 dB|28.19%|1.819 dB|

V2.9 的前向指标是三个 C seed 的均值，四点指标来自代表 seed 29513；V2.10 行是三个 seed
均值。数据都来自同一个已打开的 47 GHz HFSS 状态，所以这里只能比较工程行为，不能声称
外部泛化。

## 结构

```text
v2.10/
├─ README.md
├─ NEXT_OPTIMIZATION_PLAN.md
├─ MISSION_CONTRACT.md
├─ MIGRATION_MANIFEST.md
├─ CONSOLIDATION_MANIFEST.md
├─ pyproject.toml
├─ configs/
├─ schemas/
├─ data/
├─ src/v210/
│  ├─ models/quadratic_psd.py
│  ├─ research/v210_experiment.py
│  └─ cli/v210_campaign.py
├─ tests/
├─ tools/
└─ artifacts/
   ├─ front_hemisphere_20260906/          冻结HFSS数据的唯一现行副本
   ├─ controlled_psd_20260906/             历史复数据对照
   ├─ strict_single_pol_amplitude_20260908/ 当前幅度基线
   └─ four_point_optimization_complete_record_20260908/
```

## 关键设计

- A10 精确保留 V2.9 的 Cartesian 横向等效场结构；
- B10/C10 输出 rank-2 等效因子 (L(q))，只把
  (Q(q)=L(q)^H L(q)/60) 解释为物理可识别对象；
- 所有命令继续只通过复线性收缩进入，不进入 CNN；
- `Q` 由构造保证 Hermitian PSD，预测增益非负；
- A10/B10/C10 使用相同 seed、数据、batch、14,400 次更新和四点优化预算；
- 训练代码不读 calibration/holdout，checkpoint 哈希冻结后才进行保留评估。

## 验证

- 45/45 单元测试通过；
- CUDA selftest 通过 A10/B10/C10 的有限值、公共相位不变、二次型等价、非零梯度、
  Hermitian PSD 和 phase-only 恒模约束；
- 9/9 训练均完整执行 3,600 epoch；
- 数据、源码和 checkpoint SHA-256 已记录；
- 通过强场 MAE 和 seed 稳定性门槛，但未通过瓶颈高估和四点 regret P95 门槛。

## 复现

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = "D:\Miniconda3\envs\rl_env\python.exe"

& $python -B -m unittest discover -s tests -v
& $python -B -m v210.cli.v210_campaign selftest `
  --root artifacts\controlled_psd_20260906 --device cuda
```

训练活动已经冻结。完整重跑请按 `MIGRATION_MANIFEST.md` 创建新的 artifact 目录；不要删除
freeze marker 后原地重训，否则会破坏实验可追溯性。

## 下一决策

后续只优化幅度正向预测路线。优先获取新的频率、公差、温度、安装或实测幅度状态，并补齐
接收极化、噪声底和校准合同。结构或容量变化必须在独立角块、独立指令及独立物理状态上与
当前34,973参数幅度基线进行配对比较；历史A10/B10/C10不再参与正式选型。
