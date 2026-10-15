# V2.12 实现与运行合同

## 代码映射

| 合同 | 实现 |
|---|---|
| `FiltrationSSHTransform` | `src/v212/basis.py` |
| Stage-1 八种编码 | `src/v212/encoding.py`, `src/v212/stage1.py` |
| `MeasurementCodebook` | `src/v212/codebook.py` |
| `ReceiverObservation` | `src/v212/observation.py` |
| `CoherentRank1Operator` | `src/v212/operator.py` |
| rank-1/rank-2 审计 | `src/v212/inversion.py` |
| `OperatorTarget` 与 G3 | `src/v212/distillation.py`, `src/v212/training.py` |
| `Q_tilde` 频谱 | `src/v212/audits.py` |
| O1 D-opt codebook | `src/v212/codebook_optimization.py` |
| A/B/C/D 与统计 | `src/v212/evaluation.py`, `src/v212/statistics.py` |

V2.12 复用 V2.10 的 CNN blocks、严格 amplitude-only 数据读取和训练器，但不
修改 V2.10/V2.11 文件。Stage 1 只替换 V2.10 两个方向编码器；G1/G2/G3 是
独立 rank-1 operator。

输出解释遵循双层合同：`envelope_iq`/`a(q)` 是用于网络结构解释的 latent
representation；`Q_tilde=a* a^T` 是用于数据支持结论、跨模型物理比较和球谐
频谱审计的 identifiable physical operator。不得把 latent `a(q)` 的绝对复相位
或其按阶复系数解释成 amplitude-only 数据唯一确定的物理量。

Stage 2 在模型外观公式中显式保留固定正实尺度 `s_a`：它只由训练方向上的
16 条 single-element command 幅度中位数估计。网络学习的是无量纲
`b + a_SSH + Delta a_F`，物理单位下的 latent envelope 为二者乘积；该尺度不会
访问 validation/test，也不会改变 gauge 结论。

## 数据生成

HFSS 复场只用于离线合成相干命令幅度和物理审计。写入训练目录的标签只有
`abs(E_phi^T w)`；manifest 强制 `phase_available=false`。三次完全相同的
仿真 repeat 只表示零注入噪声基线，不能冒充真实 repeatability 测量。

```powershell
python algorithm_versions/v2.12/tools/prepare_hfss_operator_dataset.py `
  --source algorithm_versions/v2.10/artifacts/front_hemisphere_20260906/datasets/dev/train.npz `
  --source algorithm_versions/v2.10/artifacts/front_hemisphere_20260906/datasets/dev/validation.npz `
  --source algorithm_versions/v2.10/artifacts/front_hemisphere_20260906/datasets/reserved/calibration.npz `
  --source algorithm_versions/v2.10/artifacts/front_hemisphere_20260906/datasets/reserved/holdout.npz `
  --output-root algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910 `
  --write-complex-audit
```

训练的经验等权测度由 split 中的样本人口隐式定义；物理审计用的实体角权重单独
保存为 `audit/solid_angle_weights.npy`，不会再次乘进训练 loss。

## Basis 与 Stage 1

```powershell
python algorithm_versions/v2.12/tools/build_filtration_basis.py `
  --dataset algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/amplitude_dataset `
  --split-dir algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/split `
  --output-dir algorithm_versions/v2.12/artifacts/filtration_basis_20260910

python algorithm_versions/v2.12/tools/train_stage1_variant.py `
  --dataset <dataset> --split-dir <split> --output-dir <runs> `
  --recipe E2-F4 --seed 21201
```

对每个 recipe 使用配置中的全部 10 个配对种子。`analyze_paired_runs.py` 在全部
种子完成后生成 bootstrap、sign-flip、Holm、胜率和 SSH 验收门槛结果。只有
最终胜出 checkpoint 才运行 `evaluate_stage1_variant.py --open-test`；该入口拒绝
覆盖已有 test report。

## Rank 审计、G1/G2/G3

快速 rank 审计可用 `--query-limit`，但受限运行不会生成可训练的 G3 target：

```powershell
python algorithm_versions/v2.12/tools/run_rank_audit.py `
  --dataset <dataset> --split-dir <split> --codebook <codebook.json> `
  --power-repeats <power_repeats.npy> --output-dir <audit> --query-limit 32
```

完整运行省略 `--query-limit`，输出
`operator_target_train_commands_only.npz`。目标工件保存且校验 query/command
索引；validation/test commands 不能参与 inversion target。

```powershell
python algorithm_versions/v2.12/tools/train_rank1_operator.py `
  --dataset <dataset> --split-dir <split> --codebook <codebook.json> `
  --output-dir <runs> --recipe G2 --seed 21201
```

G3 还需传入 `--operator-target` 和 `--lambda-q`。最终只对胜出模型运行一次
`evaluate_rank1_operator.py --open-test`，它报告 A/B/C/D 与四个场强区间。

## Gauge-invariant 与 O1

```powershell
python algorithm_versions/v2.12/tools/audit_gauge_spectrum.py `
  --complex-audit <hfss_complex_operator_audit_only.npz> --output-dir <spectrum>

python algorithm_versions/v2.12/tools/audit_model_operator.py `
  --checkpoint <checkpoint.pt> --dataset <dataset> --split-dir <split> `
  --solid-angle-weights <solid_angle_weights.npy> --output-dir <model-audit>

python algorithm_versions/v2.12/tools/optimize_codebook.py `
  --complex-audit <hfss_complex_operator_audit_only.npz> `
  --baseline-codebook <codebook.json> --power-repeats <power_repeats.npy> `
  --output-dir <o1>
```

O1 默认不访问 test directions。只有冻结一个候选后才使用 `--open-test`。

## 验收

无需安装 pytest：

```powershell
python algorithm_versions/v2.12/tools/run_tests.py
```

## V2.12-S2 frozen representation audit

The S2 audit never retrains and never opens the external test. It compares the
ten frozen G1-M5, G2-J and soft-specialization checkpoints, producing per-degree
post-hoc reliance, continuous distance-benefit, tangent-Jacobian specialization
and absolute low-order energy-reallocation statistics:

```powershell
python algorithm_versions/v2.12/tools/audit_v212s_s2_representation.py `
  --dataset algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/amplitude_dataset `
  --split-dir algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/split `
  --codebook algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/codebook.json `
  --screen-root algorithm_versions/v2.12/artifacts/v212s_screen_20260911 `
  --soft-root algorithm_versions/v2.12/artifacts/v212s_softsep_20260911 `
  --formal-root algorithm_versions/v2.12/artifacts/v212s_formal_20260911 `
  --candidate-fix-root algorithm_versions/v2.12/artifacts/v212s_formal_l5fix_20260911 `
  --output algorithm_versions/v2.12/artifacts/v212s_s2_representation_audit_20260912.json `
  --plot algorithm_versions/v2.12/artifacts/v212s_s2_representation_audit_20260912.png
```

All branch and Jacobian quantities are latent architectural diagnostics. Physical
interpretation remains restricted to the gauge-invariant `Q_tilde` audit.

测试覆盖 filtration span/projector/SVD、经验测度、参数量匹配、E0/G1 零门控
等价、pair 相位符号、Q 的 Hermitian/PSD/rank-1/gauge invariance、解析去相位、
G3 索引防泄漏、A/B/C/D 防交叉和 observation 标签保护。
