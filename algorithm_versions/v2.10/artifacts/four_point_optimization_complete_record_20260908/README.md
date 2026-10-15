# V2.10 四目标恒模相位优化完整记录

本目录汇总截至2026-09-08与四目标优化有关的模型、原始数据、逐案例结果、数学上界、HFSS直接复核、理想阵列控制实验和HFSS几何证据。归档目标是让不了解项目历史的读者能够独立回答以下问题：

1. 当前求解的优化问题是什么；
2. O3结果是否可信、距离全局上界多远；
3. 四点增益较低主要来自优化器、阵列几何还是真实单元方向响应；
4. 当前HFSS模型有哪些已经验证的结构条件，哪些仍需针对ground连通性继续验证；
5. 所有结论如何从原始矩阵和逐案例数据复算。

## 当前主要结论

- 统一问题为 `max_{|w_n|=1/4} min_i G_i(w)`；16个阵元幅度均固定为1/4，仅调整相位，CNN不参与。
- 当前正式广泛结果使用统一Adam版O3：O1的8起点×160步、40轮坐标候选、O3的32起点×320步soft-min。
- 600组holdout的HFSS四点最低增益中位数为4.497 dB，P5为1.146 dB；达到5/8/10 dB的比例分别为39.67%/13.83%/9.00%。
- 恒模SDP上界与O3的中位差为0.261 dB，99.17%的holdout案例差距不超过1 dB，说明普通优化器通常只剩零点几dB空间。
- HFSS案例189直接复核得到4.3965 dB，相对离线O3的4.4969 dB相差0.1004 dB；24个邻域候选仅改善0.0300 dB。
- 同840组四点的理想阵列控制显示：保留HFSS共同方向包络时中位数仅由4.497升至4.676 dB；各向同性理想阵列达到10.998 dB。主要数dB差距来自方向相关单元响应/大角度扫描效应，而不是端口间非一致性本身。
- 当前HFSS工程具有16端口、6.8 mm×15 mm的4×4空间网格和一个统一空气区域；但它由完整1×4排沿Z复制而成。

## 推荐阅读顺序

1. `COMPLETE_TECHNICAL_REPORT.md`：完整问题、方法、数据和结论；
2. `METHODS_AND_MATHEMATICS.md`：优化器、SDP和有限方向集合证明；
3. `IDEAL_ARRAY_NULL_THEORY.md`：理想阵因子零陷及其适用边界；
4. `HFSS_MODEL_AND_GROUND.md`：AEDT结构、空气盒及ground建模边界；
5. `DECISION_LOG.md`：重要口径修正和结论演变；
6. `DATA_DICTIONARY.md`：CSV/NPZ字段和单位；
7. `REPRODUCIBILITY.md`：复算与校验方法。

## 目录结构

```text
analysis/
  spectral_single_direction_bounds.csv   32400个前半球方向的单点谱上界
  random_four_probability_bounds.csv     随机四点低增益概率下界
  spectral_probability_summary.json
evidence/
  hfss_o3_sdp_cross_validation/           原840组O3、SDP和HFSS交叉验证证据包
  ideal_array_control/                    同位置理想阵列成对控制实验
  hfss_geometry_baseline/                 AEDT工程快照、构建脚本和对象清单
tools/                                    复算和整包校验脚本
../front_hemisphere_20260906/datasets/    四个完整HFSS方向矩阵数据集的唯一现行副本
../front_hemisphere_20260906/*_front_suite.npz  四点HFSS输入的唯一现行副本
```

## 结论边界

- O3是高预算可行解，不是严格全局最优证明；SDP是上界，不一定可实现。
- 理想各向同性阵列的高四点增益可能利用大间距产生的栅瓣；不表示旁瓣或整体辐射质量更好。
- “共同单元包络接近HFSS”只能说明端口间差异不是主要因素；共同包络仍可能包含平均互耦、失配和有源扫描效应。
- 未完成“当前独立ground”和“符合实物的统一ground”两套HFSS工程的重新求解，因此不能把差异单独归因于ground。
- `evidence/hfss_geometry_baseline/project_snapshot/`只含AEDT工程文件，不含约752 MB的求解结果目录；四点研究所用完备远场矩阵保存在同一V2.10目录的上述唯一数据位置。
- 为避免Git仓库保存字节完全相同的大文件，本记录包通过相对路径读取共享的冻结数据；原始独立归档形态可从整理前提交恢复。
