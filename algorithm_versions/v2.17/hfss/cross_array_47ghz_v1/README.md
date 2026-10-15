# cross_array_47ghz_v1

这是 V2.17 前置的长期 HFSS 数据集。原始复场不进入算法版本目录；`dataset_manifest.json` 是唯一冻结 manifest。

## 当前状态（2026-10-14）

| 阵列 | build | solve | S/EEP | canonical |
|---|---|---|---|---|
| 1×4 | PASS | PASS（复用 GUI 解并实读 S） | S4P、integer/offset EEP、16-command audit PASS | PASS |
| 2×4 | PASS | PASS（复用 GUI 解并实读 S） | S8P、integer/offset EEP、16-command audit PASS | PASS |
| 4×4 | 复用已冻结工程 | 历史 PASS | 历史 PASS | PASS |
| 1×8 | PASS | PENDING（等待 GUI 求解） | 未执行 | 未生成 |

1×4 与 2×4 均复用用户 GUI 解完成全数据门禁；直接组合聚合 complex NMSE 分别为 `-286.324 dB` 与 `-285.409 dB`。1×8 首次构建暴露了 validation 元组误判和 unite/purge 改写端口积分线的问题；修复后已取得真实 Design Validation PASS，并准备好独立 GUI 求解副本。

## 数据边界

- `canonical/<array_id>/operator_47ghz.npz`：绝对复场 `complex128`，不平滑、不插值、不归一化。
- `raw_exports/<array_id>/`：分阶段 HFSS 收据与未来的 S/EEP 原始导出。
- `direct_audit/<array_id>/`：16 条冻结组合命令；只用于直接 HFSS 与 EEP 线性合成审计。
- V2.17 的低精度缓存只包含 train/validation；不会生成 blind-test cache。

## 恢复入口

当前从普通桌面打开独立副本 `projects/manual_gui_work/a1x8_47ghz_manual_gui.aedt`，在普通交互式 GUI 中手动求解并保存；build-PASS 快照位于 `projects/build_snapshots/a1x8_build_pass_20261014/`。求解后复用 1×4/2×4 已验证的解回读和 S/EEP/直接组合审计路线。

外部 gRPC/COM/PID 求解附着分支均未形成可靠闭环，当前主线不得重复尝试。已证明路线、失败分类和恢复门禁见 `AUTOMATION_RECOVERY_20261013.md`。

只有 1×8 的全部 64 条 S 参数可读、S8P 与两个 EEP 网格完整且 16 条审计通过，才能把 1×8 改为 PASS。
