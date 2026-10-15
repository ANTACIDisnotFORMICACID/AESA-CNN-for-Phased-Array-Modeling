# 数据字典

## 完整HFSS方向数据

唯一现行位置：`../front_hemisphere_20260906/datasets/`

- `q [N,3] float32`：单位方向向量；本研究前半球条件为 `q_x>0`。
- `h [N,3,16] complex64`：每个方向的三分量、16端口嵌入远场矩阵。
- `theta/phi [N]`：HFSS角度，单位degree。
- `ids [N]`：源方向编号。
- `commands [256,16] complex64`：数据集中保存的历史激励命令；O3四点实验不依赖CNN。
- `trajectory_*`：其他轨迹数据，不参与当前四点主要统计。

各文件总方向数/前半球数：train 36000/18000，validation 10800/5300，calibration 7200/3600，holdout 10800/5500；前半球合计32400且无重复。

## 四点HFSS输入

唯一现行位置：`../front_hemisphere_20260906/*_front_suite.npz`

- `q [cases,4,3]`：四个目标方向；
- `h [cases,4,3,16]`：四点真实HFSS复矩阵；
- `source_indices [cases,4]`：完整方向数据中的来源索引；
- `stratum [cases]`：四点类型；
- `neighbour_q/neighbour_h/neighbour_angle_deg`：每点邻域方向及矩阵。

## O3结果

位置：`evidence/hfss_o3_sdp_cross_validation/raw/o3_results/`

- `weight [cases,16] complex64`：最终恒模权值；
- `target_gain [cases,4]`：四点线性增益；
- `minimum_gain_db [cases]`：四点最低增益dB；
- `solo_gain [cases,4]`：每个目标单独服务时的相位only能力；
- `solo_utilization`：最弱目标相对单点能力的利用率；
- `rho`、`target_loss_db`、`hemisphere_peak_gain`：归一化空间性能指标；
- `method`：统一方法名称；
- `cnn_files_loaded`：应为空；
- `source_sha256`：来源哈希。

逐案例可读CSV位于 `tables/o3_validation_cases.csv` 和 `tables/o3_holdout_cases.csv`，包括四点坐标、四个增益、单点能力以及16个权值的幅度和相位。

## SDP结果

位置：`evidence/hfss_o3_sdp_cross_validation/raw/sdp_upper_bound/`

- `o3_minimum`：O3最低线性增益；
- `constant_modulus_sdp_upper`：保留每阵元固定幅度的SDP上界；
- `arbitrary_amplitude_trace_upper`：只固定总功率的更宽松上界；
- `constant_modulus_gap_db/trace_gap_db`：上界与O3差；
- `rank_99/effective_rank/leading_share`：松弛矩阵秩特征；
- `dual_repair/diagonal_error/minimum_lifted_eigenvalue/status`：数值可行性记录。

## HFSS直接复核

位置：`evidence/hfss_o3_sdp_cross_validation/hfss_cross_validation/`

包含案例189输入、16端口功率与相位、HFSS四点复场、RealizedGainTotal、24个邻域候选历史、自动化脚本和失败的原生Optimetrics尝试记录。

## 理想阵列控制

位置：`evidence/ideal_array_control/`

- `paired_cases.csv`：840组×2个理想模型，共1680行成对结果；
- `ideal_o3_raw_results.npz`：理想模型最终权值、四点线性增益和最低值；
- `summary.json/summary_table.csv`：总体、分层和成对统计；
- `PROTOCOL.json`：模型定义与因果解释边界；
- `verification.json`：恒模、目标一致性和增益复算检查。

## 单点谱与随机四点证明数据

位置：`analysis/`

- `spectral_single_direction_bounds.csv`：32400行；每个方向的来源、坐标、角度、任意幅度单位功率谱上界；
- `random_four_probability_bounds.csv`：不同阈值下的弱方向数、精确组合数和概率下界；
- `spectral_probability_summary.json`：分位数和最弱方向摘要。
