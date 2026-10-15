# V2.8：幅度测量可移植的 4×4 正向预测与四目标优化

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 4×4 amplitude-only 数据合同与前向基线 |
| 继承 | 从控制问题转向可由 HFSS 或暗室幅度数据驱动的正向预测 |
| 主要变化 | 训练入口只接受方向、已知 IQ 命令和标量幅度，禁止复场与 S 参数泄漏 |
| 冻结结论 | HFSS amplitude-only 基线已训练；暗室采集合同尚未冻结，不能声称实测可迁移 |
| 后续关系 | V2.9 在同一幅度合同上增加物理分层与更完整的训练/优化活动 |

## 当前状态

`hfss_amplitude_baseline_trained / chamber_contract_not_yet_frozen`

V2.8 已冻结为**幅度信息训练合同**。HFSS 只用于验证网络能否弥补理想阵列误差；
实际部署重新使用目标实物阵列的幅度数据训练，不使用 HFSS 数据、HFSS checkpoint
或由 HFSS 拟合的参数。

详细数据流、输入输出和限制见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，原始计划及
幅度合同修订见 [PLAN.md](PLAN.md)。

## 正式训练合同

训练数据目录只允许包含：

```text
direction.npy        [Q,3]
theta_deg.npy        [Q]
phi_deg.npy          [Q]
command_iq.npy       [N,2,4,4]   已知控制命令，不是测量相位
command_family.npy   [N]
amplitude.npy        [N,Q]       唯一测量标签
manifest.json
```

正式加载器若发现 `operator_iq.npy` 或 `s_parameters_iq.npy` 会拒绝训练。S16P 和 FFD
复相位只能保留在原始 HFSS/审计目录，不能进入幅度训练目录。

这里“只能测幅度”仍要求阵列命令的相位是已知的，因为它是控制器设置值。训练标签
不需要测量相位。仅有 16 次单阵元幅度扫描不足以确定阵元间相干关系，因此测量必须
包含 DFT 和随机多阵元相位码；代码会生成并保存可供暗室重复使用的同一码本。

## HFSS 可行性数据

HFSS 的 16 个逐端口 FFD 仅在离线转换器内部使用：

```text
16 FFD复场
→ 投影到指定接收极化（etheta/ephi/x/y/z）
→ 施加已知4×4相位码本
→ 复数叠加
→ 只保留绝对值
→ amplitude-only训练目录
```

转换时必须明确接收通道、参考距离、激励归一化、未激励端口终端和参考面。例如：

```powershell
python -m v28.cli.prepare_hfss_amplitude `
  --mapping data/manifests/ffd_ports.json `
  --output data/processed/hfss_amplitude `
  --receiver-channel ephi `
  --reference-distance-m 1.0 `
  --amplitude-unit V/m `
  --excitation-reference "按HFSS导出设置填写" `
  --inactive-port-termination "按HFSS工程填写" `
  --calibration-reference-plane "按HFSS端口面填写" `
  --array-id hfss_4x4_47ghz `
  --session-id hfss_export_001
```

`receiver-channel` 必须与未来暗室接收天线的极化通道一致。不能用 HFSS 总场幅度训练，
再用单极化接收机幅度替换。

## 实物数据入口

暗室控制程序应复用 HFSS 阶段生成的 `command_iq.npy`，对每个命令扫描同一球面坐标，
最终提供 `[N,Q]` 标量幅度。仪器输出转换为 NPY 后运行：

```powershell
python -m v28.cli.prepare_measured_amplitude `
  --direction raw/direction.npy `
  --theta-deg raw/theta_deg.npy `
  --phi-deg raw/phi_deg.npy `
  --command-iq raw/command_iq.npy `
  --command-family raw/command_family.npy `
  --amplitude raw/amplitude.npy `
  --output data/processed/chamber_amplitude `
  --frequency-hz 47e9 `
  --receiver-channel ephi `
  --reference-distance-m 1.0 `
  --amplitude-unit "按仪器标定填写" `
  --noise-floor-amplitude 0.0 `
  --excitation-reference "按实物链路填写" `
  --inactive-port-termination "按实物终端填写" `
  --calibration-reference-plane "按校准面填写" `
  --array-id physical_array_001 `
  --session-id chamber_session_001
```

HFSS 与实物目录随后使用完全相同的划分、训练、评估和优化入口。划分必须先落盘并
绑定数据文件指纹，训练命令不再自行临时划分：

```powershell
python -m v28.cli.prepare_amplitude_split --dataset data/processed/chamber_amplitude --output data/splits/chamber_seed28001
python -m v28.cli.train_forward --dataset data/processed/chamber_amplitude --split-dir data/splits/chamber_seed28001 --checkpoint runs/physical/model.pt --epochs 200
python -m v28.cli.evaluate_forward --dataset data/processed/chamber_amplitude --checkpoint runs/physical/model.pt --output runs/physical/test_metrics.json
python -m v28.cli.optimize_targets --checkpoint runs/physical/model.pt --targets data/fixtures/four_targets.json --output runs/physical/weights.json
```

优化结果包含 `deployment_eligible`。只有由 `anechoic_chamber_amplitude` 数据训练的
checkpoint 才标记为可部署；HFSS checkpoint 只用于算法测试。

## 2026-09-03 HFSS 幅度基线

- 输入：16 个 47 GHz embedded-element FFD；S16P 未进入训练目录；
- 通道：暂定 `Ephi`，数值按 HFSS `rE` 换算为 1 m 处 `V/m`；
- 网格：删除极点和重复的 `phi=+180°` 接缝后为 64,440 个唯一方向；
- 命令：16 个单阵元、16 个二维 DFT、96 个随机相位码，共 128 个；
- 固定划分：方向 45,110/9,660/9,670，命令 92/18/18；
- 基线测试：理想阵列 NMSE -1.06 dB、幅度 MAE 14.16 dB；20 轮 CNN 为
  NMSE -2.92 dB、幅度 MAE 6.14 dB。

该结果证明幅度合同中存在可学习的 HFSS 修正信号，但精度尚不足以进入四目标参数求解。
完整审计见 `artifacts/reports/HFSS_AMPLITUDE_BASELINE_20260903.md`。后续结构或超参数选择
只使用 validation；test 已用于本次冻结基线，不作为逐轮调参反馈。

## 2026-09-03 HFSS 单极化物理修正版

- 保持单个固定接收极化的标量幅度标签；不使用接收相位、双极化总场或 S16P；
- 增加逐阵元单极化方向图乘性分支，同一方向同时用全部训练命令约束同一潜在复信道；
- 损失按命令族平衡，并联合幅度 NMSE、功率 NMSE 和稳健 dB 误差；
- 最多训练 200 epoch，由 validation 保存最佳轮并支持降学习率和早停；
- 200 epoch 测试全局幅度 NMSE -16.02 dB、功率 NMSE -15.32 dB、幅度 MAE 2.28 dB。

完整结果和消融分析见 `artifacts/reports/HFSS_AMPLITUDE_PHYSICS_V2_20260903.md`。

冻结微调入口支持 `--initial-checkpoint`、`--freeze-element-gain` 和
`--freeze-local-coupling`。从 200-epoch 模型冻结增益/耦合后低学习率微调 50 轮，只使
validation 全局幅度 NMSE 从 -15.630 dB 改善到 -15.715 dB，说明当前优化方式已接近平台。
本次未读取 test；详见 `artifacts/reports/HFSS_AMPLITUDE_FREEZE_FINETUNE_20260903.md`。

## 模型原则

- 输入阵元命令保持 `[IQ,4,4]`，阵元不可展平。
- CNN 阵元域卷积全部为二维 `3×3`，无 GAP。
- CNN 由坐标预测潜在复数逐阵元信道，再与已知命令显式复数收缩；训练损失只看幅度。
- 潜在复信道的全局相位不可由幅度观测确定，代码用整个 4×4 孔径与理想算子的内积固定
  gauge，避免依赖可能位于零陷的单个参考阵元；这不改变任意命令的预测幅度或四目标功率。
- 训练按方向分组，一次 CNN 前向得到完整 `[2,4,4]` 潜在算子，再与同一方向的多组已知
  命令逐阵元显式收缩；没有展平阵元，也没有用命令驱动方向图分支。
- 训练和测试同时隔离角域块与相位码，避免只记忆已测命令。
- S16P 不参与正式训练；实测 S16P 以后只作为独立诊断或明确验证后的物理先验。

## 验证

```powershell
python -B -m unittest discover -s tests -v
```

当前回归测试覆盖幅度目录隔离、禁用相位文件、HFSS 幅度转换、码本功率、单方向多命令
联合训练、单极化幅度/功率损失、二维 CNN、4×4 拓扑、四目标优化，以及原有复数 HFSS
诊断链。
