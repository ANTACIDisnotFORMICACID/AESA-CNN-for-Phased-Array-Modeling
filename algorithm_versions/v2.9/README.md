# V2.9：物理分层的幅度信息前向远场模型

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 物理分层前向模型与四目标优化实验版 |
| 继承 | V2.8 的 amplitude-only 数据合同、固定划分和命令码本 |
| 主要变化 | 将解析传播、名义阵元响应、端口耦合和角向残差分层建模 |
| 冻结结论 | 训练活动完成但未全部验收；旧 blind 已开启并冻结，禁止继续据其调参 |
| 后续关系 | V2.10 收紧数据域和监督形式，建立严格正半球、单极化活动基线 |

## 最新：总增益向量实验已完成

新活动 `vector_beam_20260905` 已完成 5 种训练配方 × 3 seeds，全部达到预设收敛规则。
验证集冻结选择 C / seed 29513，28,854 参数；尚未全部验收，不具备 FPGA 或实测部署资格。

- [完整结果与局限](artifacts/vector_beam_20260905/RESULTS.md)
- [物理分析、模型与训练协议](VECTOR_BEAM_CAMPAIGN.md)
- [冻结CNN的随机四点优化实验](artifacts/four_point_optimizer_20260905/RESULTS.md)
- 研究入口：`tools/research/beam_campaign.py`、`beam_diagnostics.py`。
- 本轮预测 RealizedGainTotal；选中模型的复向量仅为等效因子，不能作为真实 Etheta/Ephi 极化场，也不能直接替换旧 Ephi 接口。
- 角域、命令和轨迹保留集未参与训练/选择，但共享同一 HFSS 状态，不是独立实物盲测。
- 保留集已打开，禁止据其结果继续调参。下列数学主链、数据合同和测试描述属于保留的旧标量实现。

## 历史：标量 Ephi 活动状态

`convergence_campaign_complete / dual_blind_complete / acceptance_failed / ota_not_validated`

V2.9 已形成独立 `v29` Python 包，复用 V2.8 的幅度-only数据合同、坐标、HFSS/暗室入口、
固定划分、评估和四目标优化，重写了前向 CNN 与训练调度。
2026-09-05 已完成固定结构的 300 / 800 / 最多 2,000 轮策略实验和三 seed 收敛训练，
实际正式训练为 764 / 923 / 847 轮。仅按验证集选出的研究候选是 R1 / seed 29003 / epoch 725。

三 seed 整数盲测幅度 NMSE 均值 -20.1397 dB，错位保留角块均值 -19.7673 dB；
相对本轮各自 epoch200 平均改善 3.1509 dB。尚未全部验收通过：seed 标准差 0.6444 dB，
四目标相对数值参考损失中位数 7.2098%、P95 22.1033%。本轮盲测已打开，禁止继续调参。

- 本轮完整结论与隔离审计：[RESULTS_20260905.md](artifacts/research_20260905/RESULTS_20260905.md)
- 冻结训练协议：[TRAINING_CAMPAIGN_20260905.md](TRAINING_CAMPAIGN_20260905.md)
- 下一轮前向模型、连续四目标优化、手动数据准备与逐阶段验收计划：[NEXT_OPTIMIZATION_PLAN.md](NEXT_OPTIMIZATION_PLAN.md)
- 下一轮 HFSS 选定相位、源表和导出目录：[data/external_next/README.md](data/external_next/README.md)
- 研究候选：`artifacts/checkpoints/hfss_47ghz_ephi_v29_research_20260905_selected.pt`
  （大型 checkpoint 未纳入当前精简分支，需随对应版本数据包挂载）
- 本轮状态入口：`python -B -m v29.cli.research_campaign status --root artifacts/research_20260905`

旧 200 epoch 联合训练和分阶段消融结果均保留为历史基线，不覆盖、不与本轮新划分直接比较。

- 设计依据：[ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md)
- 实现说明：[IMPLEMENTATION.md](IMPLEMENTATION.md)
- 历史训练报告：[artifacts/reports/HFSS_V29_200EP_20260904.md](artifacts/reports/HFSS_V29_200EP_20260904.md)

## 数学主链

```text
direction q
  -> ideal propagation a0(q)
  -> nominal element response g0(q)
  -> structured port coupling C_ant
  -> one angular residual r(q)

known command w
  -> IdentityDeviceModel（HFSS）

E(q;w) = [C_ant^T(g0(q)⊙a0(q)) + r(q)]^T w
amplitude(q;w) = |E(q;w)|
```

所有可训练层只读取方向和固定阵列几何，不读取命令；同一潜在复算子必须同时解释全部已知
命令。复数收缩永久使用发射约定 `h^T w`。

## 冻结的数据合同

V2.9 直接读取 `v2.8-amplitude-1`：

```text
direction.npy        [Q,3]
theta_deg.npy        [Q]
phi_deg.npy          [Q]
command_iq.npy       [N,2,4,4]
command_family.npy   [N]
amplitude.npy        [N,Q]
manifest.json
```

正式训练仍拒绝 `operator_iq.npy` 和 `s_parameters_iq.npy`。HFSS 复数 FFD 与 S16P 仅用于
离线生成幅度标签、物理诊断和最终 HFSS 重组验证。

## 历史 200 轮结果（旧划分）

同一 HFSS 幅度数据、同一 seed 28001 划分、每轮 8192 个训练方向、共 200 epoch：

| 模型 | 参数量 | Test 全局幅度 NMSE | Test 全局功率 NMSE | Test dB MAE |
|---|---:|---:|---:|---:|
| V2.8 physics-v2 200ep | 62,098 | -16.02 dB | -15.32 dB | 2.28 dB |
| V2.9 staged 200ep | 34,973 | -14.14 dB | -13.85 dB | 2.80 dB |
| **V2.9 joint 200ep** | **34,973** | **-16.49 dB** | **-16.05 dB** | **2.16 dB** |

V2.9 joint 相对 V2.8 减少约 43.7% 参数，同时测试幅度 NMSE 改善约 0.47 dB。由于当前
test 已用于工程比较，该结果不能再作为后续调参的全新盲测结论；下一轮结构调整必须生成
新的命令/角域盲测集合。

## 历史 200 轮流程（不是本轮冻结候选的评估入口）

以下命令在 `algorithm_versions/v2.9` 中执行。当前数据继续引用 V2.8，避免复制约定相同的
大体积数据：

```powershell
$env:PYTHONPATH = "$PWD\src"

python -B -m v29.cli.train_forward `
  --dataset ..\v2.8\data\processed\hfss_47ghz_ephi_amplitude `
  --split-dir ..\v2.8\data\splits\hfss_47ghz_ephi_seed28001 `
  --checkpoint artifacts\checkpoints\hfss_47ghz_ephi_v29_joint_200ep.pt `
  --epochs 200 --stage-plan joint --queries-per-epoch 8192

python -B -m v29.cli.evaluate_forward `
  --dataset ..\v2.8\data\processed\hfss_47ghz_ephi_amplitude `
  --checkpoint artifacts\checkpoints\hfss_47ghz_ephi_v29_joint_200ep.pt `
  --split test `
  --output artifacts\reports\hfss_47ghz_ephi_v29_joint_200ep_test.json

python -B -m v29.cli.diagnose_structure `
  --dataset ..\v2.8\data\processed\hfss_47ghz_ephi_amplitude `
  --checkpoint artifacts\checkpoints\hfss_47ghz_ephi_v29_joint_200ep.pt `
  --split validation `
  --output artifacts\reports\hfss_47ghz_ephi_v29_joint_200ep_structure_validation.json
```

`train_forward` 默认使用已验证较好的 `joint` 策略。原计划的 80/40/60/20 阶段训练仍可用
`--stage-plan default` 运行，作为结构诊断而不是默认训练方式。

## 验证

```powershell
$env:PYTHONPATH = "$PWD\src"
python -B -m unittest discover -s tests -v
```

当前 42 项测试通过，覆盖数据隔离、坐标、HFSS I/O、4×4 拓扑、恒等初始化、端口耦合、复数约定、
分项可加性、训练阶段冻结、checkpoint 评估、四目标优化，以及新协议预算、盲测门禁和中断恢复一致性。

## 使用边界

- 当前 checkpoint 来自 HFSS，不具备实物部署资格；
- 当前 HFSS 数据没有真实器件误差，因此器件层强制为恒等；
- `C_ant` 和残差是预测用有效参数，没有独立标定时不能等同于真实 S 参数或器件误差；
- 训练模型目前只实现 `ideal_phase` 基底；完整嵌入式阵元基底只用于精确重组审计，不能再
  叠加一个可训练互耦矩阵。
