# AESA-CNN V2.18：4×4 canonical 迁移归因与单变量筛选

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 独立的 4×4 迁移复现、失败归因与 validation-only 工程筛选 |
| 继承 | 复用 V2.16 builders 和 V2.17 canonical 4×4 数据；不修改任一冻结版本 |
| Stage-1 | V2.16 baseline canonical reproduction 未通过预注册门禁，raw-complex opaux 未训练 |
| Stage-2 | 数据等价归因通过，只增加 phase-aligned complex auxiliary 这一变量进行 B/D 对照 |
| 冻结结论 | `PHASE_AUX_EXPERIMENT_FAIL`，最终 `ENGINEERING_CANDIDATE_NONE` |
| 当前基线 | canonical 4×4 保留 V2.12 control；V2.16 gated-PM 只保留其原协议结论 |
| 证据边界 | canonical blind test 未开启；禁止继续调参、延长预算或追加第五个 arm |

## 研究目的

> 仓库状态：`BLOCKED`（完整复跑）。本仓库保留实现、预注册协议和紧凑结果，但不包含
> canonical 训练数组与 checkpoint；冻结结论来自原实验记录，不能视为在精简仓库内重跑通过。

V2.16 gated-DCAR 在原数据协议内取得约 1 dB 的受控改善。V2.18 不直接假设该优势会迁移，
而是先在 V2.17 canonical 4×4 数据上复现相同 builders，再判断变化来自：

1. canonical 数据适配错误；
2. 命令分布变化；
3. 优化不稳定；
4. amplitude-only 监督不足。

整个版本只在 train/validation 和冻结的诊断交集上工作，不提供 blind-test 路径。

## Stage-1：canonical baseline reproduction

执行顺序被冻结为：

```text
prepare_v218_4x4.py
  → run_tests.py
  → train_v218.py --stage baseline
  → audit_v218.py --stage baseline
  → 只有 BASELINE_REPRODUCTION_PASS 才允许 raw-complex opaux
```

数据适配、几何、合同测试均通过；9 个 240-epoch baseline run 完成。但 gated-PM 相对
control 的三种子 amplitude 改善为 `[+0.457, -0.284, -1.140] dB`，均值 `−0.322 dB`；
complex operator 和 Q 也分别平均退化 `0.026 dB`、`0.205 dB`。因此状态为
`BASELINE_REPRODUCTION_FAIL`，原计划的 raw-complex opaux 按停止规则不训练。

## Stage-2：迁移归因与 phase-aligned auxiliary

Stage-2 是单独预注册的追加实验，不回写 Stage-1。数据归因首先验证：

- 3,200 个 canonical-train 和 800 个 canonical-validation 重合方向全部匹配；
- 最大方向误差 `6.00685e-8`；
- validation 重建 amplitude NMSE `−142.015 dB`；
- 最差命令 NMSE `−140.413 dB`；
- 所有数据等价门禁通过。

因此 canonical 适配错误被排除。命令迁移信号未通过，而 gated-PM 的 cross-seed 波动明显更大，
优化不稳定信号通过。Stage-2 只获准比较下面四个固定 arm：

| Arm | 结构 | 监督 |
|---|---|---|
| A | V2.12 control | amplitude-only |
| B | V2.12 control | amplitude + phase-aligned complex auxiliary |
| C | V2.16 gated-PM | amplitude-only |
| D | V2.16 gated-PM | amplitude + phase-aligned complex auxiliary |

这里的辅助监督先消除每个方向的公共相位，不把 amplitude-only 无法辨识的绝对相位作为目标。

## 最终结果

- B 相对 A 的 solid-angle amplitude 只改善 `0.103 dB`，弱场平均退化 `0.179 dB`；
- D 相对 C 的 phase-aligned operator、Q、amplitude 分别改善 `0.301/0.561/0.355 dB`，
  但最差种子弱场退化 `0.308 dB`，超过 `0.30 dB` 门槛；
- D 相对 A 的 amplitude 只改善 `0.025 dB`，command-family macro 改善 `0.014 dB`，
  不足以抵偿 DCAR 的执行代价；
- 所有 arm 均为 `BUDGET_SENSITIVE`，但预注册规则禁止延长训练预算。

最终状态：

```text
MIGRATION_ATTRIBUTION_PASS
PHASE_AUX_EXPERIMENT_FAIL
ENGINEERING_CANDIDATE_NONE
CONFIRMATION_NOT_OPENED
```

这不推翻 V2.16 在原协议内的结果，但说明其优势没有稳定迁移到 canonical 4×4 协议。当前
canonical 工程基线为 Arm A，即 V2.12 control。

## 文档与执行入口

- [完整状态与版本归因](V2.18_STATUS_20261014.md)
- [Stage-1 预注册](V2.18_PREREGISTRATION.md)
- [Stage-2 预注册](V2.18_STAGE2_PREREGISTRATION.md)
- [Stage-2 最终结果](V2.18_STAGE2_RESULTS_20261015.md)

Stage-2 的冻结执行顺序：

```powershell
python algorithm_versions/v2.18/tools/run_stage2_attribution.py
python algorithm_versions/v2.18/tools/run_tests.py
python algorithm_versions/v2.18/tools/train_v218_stage2.py
python algorithm_versions/v2.18/tools/audit_v218_stage2.py
```

当前精简分支未包含 Stage-2 checkpoint，完整回执测试当前为 `BLOCKED`；不能把未执行项计为
PASS。已有权威回执保留在 `artifacts/stage2_audit_4x4_20261014/STAGE2_FINAL_AUDIT.json`。

## 关闭状态

V2.18 已在 validation-only 工程筛选处完成。后续不得在本版本中继续调辅助权重、扩展 epoch、
新增 arm 或打开 blind test。任何新模型假设都必须进入独立版本并重新预注册。
