# 复现说明与数据字典

## 1. 依赖

- Python 3.10或更高；
- NumPy、SciPy、PyTorch；
- v2.10源码可由当前目录的 `src` 导入；
- 不需要CNN checkpoint，也不需要重新运行HFSS求解器。

当前归档结果在CPU PyTorch环境完成。留出集H1/H2/H3分别约用157/128/129秒；不同硬件会变化。

## 2. 冻结命令

在 `algorithm_versions/v2.10` 下运行：

```powershell
python tools/research/hfss_four_strong_baseline.py `
  --suite validation `
  --methods H1_ratio_only,H2_preserve_0p5db,H3_preserve_1db `
  --starts 12 `
  --outer-steps 6 `
  --inner-steps 300 `
  --active-topk 48 `
  --case-batch 6 `
  --output artifacts/hfss_four_strong_highsearch_20260906
```

验证方法行为后，以完全相同预算运行留出集：

```powershell
python tools/research/hfss_four_strong_baseline.py `
  --suite holdout `
  --methods H1_ratio_only,H2_preserve_0p5db,H3_preserve_1db `
  --starts 12 `
  --outer-steps 6 `
  --inner-steps 300 `
  --active-topk 48 `
  --case-batch 6 `
  --output artifacts/hfss_four_strong_highsearch_20260906
```

## 3. 一键完整性检查

```powershell
python tools/research/verify_hfss_four_strong.py
```

通过时输出 `HFSS_FOUR_STRONG_VERIFICATION_OK`。检查内容包括：

- 协议与结果中的协议副本完全一致；
- 未加载CNN文件；
- 四点及约束网格均为正半球；
- 输入NPZ与算法脚本哈希匹配冻结协议；
- 所有权值满足 `|w_n|=1/4`；
- H2/H3逐样本目标损失没有越过0.5/1.0 dB限制；
- 实际 `u`没有超过其数学谱上界；
- JSON汇总成功率与逐样本NPZ重新计算一致。

## 4. 输入文件

位于 `artifacts/front_hemisphere_20260906`：

|文件|内容|
|---|---|
|`validation_front_suite.npz`|240组四点、HFSS因子、分层标签和邻域信息|
|`holdout_front_suite.npz`|600组留出四点及HFSS因子|
|`validation_optimizer_arrays.npz`|验证集H0/O0–O4权值、增益和单点结果|
|`holdout_optimizer_arrays.npz`|留出集H0/O0–O4权值、增益和单点结果|
|`datasets/dev/*.npz`与`datasets/reserved/*.npz`|合并得到32,400方向正半球约束网格|

具体输入SHA-256保存在 `validation_protocol.json` 和 `holdout_protocol.json`。脚本每次运行都会重新写入
当前输入和自身哈希，便于识别结果是否由不同版本生成。

## 5. 输出JSON

`validation_results.json`和`holdout_results.json`包含：

- `protocol`：控制变量、算法预算、输入哈希和“无CNN”声明；
- `methods`：H0–H3整体、分层、峰值距离、恒模误差和运行时间；
- `comparisons`：相对H0的成对成功率差异及bootstrap 95%区间；
- `spectral_solo_upper_bound`：谱上界分位数和被证明低于0.5的比例；
- `cause_breakdown`：已证明物理限制、旧目标漏解、牺牲型均衡和未决类别；
- `position_relationship`：坐标特征与 `rho` 的Spearman相关性。

## 6. 输出NPZ字段

公共字段：

|字段|形状/含义|
|---|---|
|`source_index`|原四点套件中的样本序号|
|`q`|`[case,4,3]`，四个单位方向向量|
|`stratum`|每组的clustered/moderate/wide/grazing标签|
|`spectral_solo_upper_bound`|每组 `u*` 的有效必要上界|

对每个方法前缀 `H0_target_maxmin`、`H1_ratio_only`、`H2_preserve_0p5db`、
`H3_preserve_1db`，包含：

|后缀|含义|
|---|---|
|`_weight`|`[case,16]`最终恒模复权值|
|`_target_gain`|`[case,4]`四个目标HFSS增益，功率域|
|`_minimum_gain_db`|四点最小增益，dB|
|`_hemisphere_peak_gain`|完整正半球最高HFSS增益，功率域|
|`_peak_index`|峰值在32,400方向合并网格中的序号|
|`_minimum_rho`|四点最小增益/正半球峰值|
|`_minimum_solo_utilization`|四点中最差的单点上限利用率|
|`_target_loss_db`|相对H0四点最小增益的下降|
|`_spatial_strong`|是否满足 `rho>=0.5`|
|`_strict_four_strong`|是否同时满足 `rho>=0.5`和`u>=0.5`|
|`_nearest_target_to_peak_deg`|半球峰值到最近目标的球面角距|
|`_farthest_target_to_peak_deg`|半球峰值到最远目标的球面角距|

## 7. 结果文件哈希

冻结时主要结果哈希如下：

|文件|SHA-256|
|---|---|
|`holdout_arrays.npz`（当前仓库未保留）|`8DF2F1FF7A7DB6B4B844FA9B6ACD253BE0C1C1C90D42E3FAF249FF9AD526D077`|
|`holdout_protocol.json`|`10840A0E5F26CB7EDA927177C5FE97F7FFA886EC296572E870CFF871D5FCD232`|
|`holdout_results.json`|`C9E459EC99ECB27ED72C6175F5EBD78378E590E12D79636703202DBCC95B03FD`|
|`validation_arrays.npz`|`D061F2C7ECAF48580D62FEC8E1F1289EA830932B690637F3C3F496DB1C587C68`|
|`validation_protocol.json`|`43294D269940E2AA45B6F559537F31A9A23C9A7415662903BB97108D03E7E7DE`|
|`validation_results.json`|`DDBB7430509A0C0757FFBE199BC133FDBB996C9DD6C39E7757F8F7F00165A73A`|
|基线脚本|`78E25D79198973196B8261A89F3213B35F1CB81324DDCFC40AAC2733E1EC58CA`|

新增说明文档不会改变上述数值结果文件；若重新运行基线，协议中的输入和脚本哈希是判断是否仍为同一
实验的首要依据。

## 8. 复现边界

- O3、H1–H3都是非凸数值可行解，不保证全局最优；
- 谱上界低于门槛可证明不可能，上界高于门槛不能证明可行；
- 当前峰值来自1°网格，连续角域可能在采样点之间出现更高峰；
- 所有HFSS因子来自单一电磁状态，复现同一结果不等于外部物理泛化；
- 若输入数据、端口顺序、增益归一化、方向坐标或相位符号改变，结果不可直接比较。
