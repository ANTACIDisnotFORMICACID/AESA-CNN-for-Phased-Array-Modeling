# V2.13 Gate-B 幅度/功率重复测量合同

## 1. 适用边界

本合同只服务于 V2.13 的 amplitude-only `Q` 可辨识性与噪声审计。它复用冻结的
V2.12 `MeasurementCodebook`、rank-one inversion、Jacobian 和 `OperatorTarget`
语义，但不训练 CNN，也不读取 16 条 external-test commands。

可接收的来源分三类，验证结果必须保持分级：

1. `hardware_measurement`：暗室、OTA 或校准探针功率测量；只有这一类在重复独立、
   非相同且 provenance 完整时，才可能支持硬件 Gate B。
2. `direct_hfss_repeated_power`：由彼此独立的 HFSS 重算/重导出得到的重复功率；只
   支持仿真数值重复性，不支持硬件可辨识性。
3. `synthetic`：从同一个复场 `g` 或同一个 `Q` 加噪生成；只能用于算法自检、测量
   设计和失效边界。

把同一个 `g` 线性合成多条命令、把一份数组复制三次、重新加载同一个 HFSS 解或
把 H00/H90 全局相位控制当作两次功率测量，都不属于独立重复。

## 2. 原始 CSV

UTF-8 `measurements_long.csv` 每行是一条方向—命令—重复功率记录，必须形成完整的
`Q × 80 × R` 长表。80 条命令严格对应冻结 V2.12 的 64 条 train commands 与 16 条
validation commands；命令编号只能为 0–79，external test 的 80–95 不得出现。

| 列 | 类型 | 含义 |
|---|---|---|
| `direction_id` | int | 从 0 连续编号；每个方向唯一 |
| `theta_deg`, `phi_deg` | float | HFSS 球坐标角度，单位度 |
| `qx`, `qy`, `qz` | float | HFSS Cartesian 单位方向；必须满足 `qx>0` |
| `command_id` | int | 冻结 V2.12 命令编号 0–79 |
| `repeat_id` | string | 独立扫频/扫角采集编号 |
| `power` | float | 非负线性功率；单位由 manifest 指定 |
| `acquired_at_utc` | ISO-8601 | 带时区且各 repeat 唯一的采集时刻 |
| `session_id` | string | 原始仪器/仿真会话编号 |

同一 `repeat_id` 应是一整套方向—命令扫描。每次 repeat 必须重新触发真实接收积分或
独立 HFSS run/export，并保留原始文件；不能由上一 repeat 数值复制得到。

## 3. 导入 manifest

以 [manifest.template.json](manifest.template.json) 为起点。以下字段不可缺失：

- `schema = v213-gate-b-import-manifest-1`；
- `dataset_id`、`array_id`、`frequency_hz`、`receiver_channel`、`power_unit`；
- `instrument`、`calibration_reference_plane`、`calibration_id`；
- `source_kind`、`independent_repeats`、`source_files`；
- `generated_from_single_complex_g`；
- `not_for_training = true`、`external_test_used = false`。

`independent_repeats=true` 是需要原始记录支撑的 provenance 声明，不会覆盖数值检查。
验证器同时要求所有整次 repeat 不得 bit-identical，并报告发生变化的方向—命令单元
比例。任何 synthetic 来源即使数值不同，也不会被升级为硬件重复。

## 4. 规范化输出

导入器生成一个目录：

- `manifest.json`：规范化 schema、轴定义和完整 provenance；
- `measurements.npz`：`allow_pickle=False` 可读的数组；
- `VALIDATION.json`：结构、命令、重复差异和 eligibility 结果。

NPZ 数组合同：

| 数组 | shape | dtype/约束 |
|---|---|---|
| `direction` | `[Q,3]` | float，单位向量，`qx>0` |
| `theta_deg`, `phi_deg` | `[Q]` | float，与 `direction` 一致 |
| `command_indices` | `[80]` | int64，严格为 0–79 |
| `command_iq` | `[80,2,4,4]` | float，与冻结 codebook 数值一致 |
| `power_repeats` | `[Q,80,R]` | float，有限且非负 |
| `repeat_id` | `[R]` | 固定宽度 Unicode，唯一 |
| `acquired_at_utc` | `[R]` | 固定宽度 Unicode，唯一且带时区 |
| `session_id` | `[R]` | 固定宽度 Unicode，非空 |

验证命令示例：

```powershell
python algorithm_versions/v2.13/tools/import_gate_b_measurements.py `
  --csv measurements_long.csv `
  --manifest algorithm_versions/v2.13/measurement_contract/manifest.template.json `
  --codebook algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/codebook.json `
  --output-dir algorithm_versions/v2.13/measurement_contract/imported/session_001

python algorithm_versions/v2.13/tools/validate_gate_b_measurements.py `
  --package algorithm_versions/v2.13/measurement_contract/imported/session_001 `
  --codebook algorithm_versions/v2.12/artifacts/hfss_operator_96_20260910/codebook.json
```

## 5. 重复数与协方差最低要求

- **硬下限：`R=3`。** 这只足以运行逐方向、逐命令的对角方差估计和基础异常检查，
  置信区间仍很宽。
- **首轮正式硬件审计建议：`R>=10`。** 应覆盖至少两个独立会话或重新上电/校准块，
  并保留会话标签，以区分接收噪声与慢漂移。
- 冻结 V2.12 的 Jacobian 加权使用 `per_direction_diagonal_from_repeats`，因此不要求
  未正则化的 64×64 全协方差。若要在单一方向直接估计满秩 64×64 sample covariance，
  理论最低为 `R>=65`，工程上仍需更多样本或明确的 shrinkage 模型。

验证器额外给出跨方向汇聚、按方向 RMS 归一化的 64×64 相对噪声协方差，用于发现
公共增益漂移等跨命令相关性；它不能替代每个方向的独立满秩协方差。

## 6. Gate-B 执行规则

只用 64 条 train commands 恢复 rank-one `Q_hat`，只用 16 条 validation commands
回放 power NMSE。若有 privileged HFSS complex truth，可额外报告 operator NMSE；
硬件数据没有复场 truth 时不得制造该指标。报告至少包含：

- operator NMSE（仅在合法 privileged truth 存在时）；
- held-out validation power NMSE；
- 未加权与噪声加权 Jacobian condition；
- 10°角区和强/弱场失败分区；
- 方向 bootstrap 95% 置信区间；
- 数据来源分级与 `hardware_identifiability_claim`。

在真实非相同重复到位前，Gate B 必须保持 `ideal-synthetic-only / hardware-blocked`。
