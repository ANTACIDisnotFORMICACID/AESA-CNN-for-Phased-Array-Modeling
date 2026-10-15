# HFSS 求解、恢复与自动化故障经验（更新于 2026-10-14）

## 默认复用路线

当前完整证明可行的流程分为标准路线和 1×8 的进程内回退路线：

1. 自动化只负责从冻结母版构建独立阵列工程，并以真实 Design Validation 结果作为 build 门禁；
2. 在 `FORMIC-ACID\ASUS` 普通交互式 AEDT GUI 中打开 `projects/manual_gui_work/` 下的独立副本，手动求解 `Setup2` 并保存；
3. 优先由 `run_verified_recovery.ps1` 建立时间戳恢复副本，再用 `recover_existing_solution.py` 只读实取全部 S 参数、integer/offset EEP 和 16 条直接场；该路线已在 1×4/2×4 闭环；
4. 若外部 gRPC 在工程解析前崩溃，停止重试，在用户正常打开的 GUI 恢复副本内运行 `export_existing_solution_in_gui.py` 导出 S/EEP，再运行 `export_direct_fields_in_gui.py` 通过 `GetSolutionDataPerVariation` 导出 16 条直接场；
5. `finalize_inprocess_export.py --direct-api-run` 或标准恢复脚本完成内容与数值验证，最后由 `promote_recovery_run.py` 晋级官方 raw exports、stage receipts 和 canonical。

保存工程、生成文件、API 无异常或 API 返回成功都不能单独构成 PASS。最终门禁始终是内容可读、矩阵/网格完整、数值有限和审计阈值全部通过。

## 已证明成功

- 1×4：全部 16 条 S 参数可读，S4P 与两套 4 端口 EEP 完整；16 条审计聚合 `-286.324 dB`、最差 `-282.111 dB`、幅度最大相对误差 `8.58e-15`。
- 2×4：全部 64 条 S 参数可读，S8P 与两套 8 端口 EEP 完整；16 条审计聚合 `-285.409 dB`、最差 `-282.229 dB`、幅度最大相对误差 `8.76e-15`。
- 1×8：全部 64 条 S 参数可读，S8P 与两套 8 端口 EEP 完整；GUI 进程内底层 API 约 38 秒导出 16 条直接场。审计聚合 `-286.165 dB`、最差 D14 `-278.611 dB`、幅度最大相对误差 `1.19e-14`。S8P SHA256 为 `5B98E95D5F74A58CA35B3AF48018B1B61282B3ECF3D491755643548D54D621BA`，canonical SHA256 为 `9563E0CEB6EA6599EBA4158739B9987DE432E96692383B974D2932A4CCB55181`。
- 两者的 HFSS 原生 Touchstone 导出都可能因 saved-version 状态拒绝。可接受的回退是先实际读取全部 N² 条 S 参数，再重建并重新解析完整 `.sNp`；不能由少量对角项推断完整矩阵。
- EEP 非零端口 phase reference 必须按 `exp(+j k q·r_port)` 恢复到公共全局参考，然后才能进行线性组合审计。

## 自动化为什么出问题

### 已证实的脚本缺陷

1. PyAEDT 0.9.6 的 `validate_full_design` 返回 `(messages, passed)`。旧代码把整个非空元组当成真，导致包含端口错误的 1×8 被误报 build PASS。现已按第二项判定，失败时抛出错误并保留消息。
2. 1×8 的 `_unite_touching_same_material`/purge 会在合并复制模块后改写边界引用，使 Port 1、Port 5 的积分线端点落到相邻列位置。1×8 现保留各复制模块，不再做该合并；重新构建已通过真实 Design Validation。
3. 求解路径调用 `oDesign.Analyze("Setup2", True)` 后曾无条件写 `solved=True`，没有保存实际返回码。现改为保留 `analyze_return`，非零整数立即 FAIL；即使返回 0 或 `None`，仍须成功读取 `S(1,1)` 才能 PASS。

以上缺陷会造成错误报告或错误几何，但不能解释 HFSSCOMENGINE 自身为何崩溃；两类问题必须分开记录。

### 已证实的引擎故障边界

- 2026-10-13 的六次外部自动化求解均为 `HFSSCOMENGINE.exe` 在 `ucrtbase.dll` 中异常终止：exception `0xc0000409`、fast-fail subcode `5`、BEX64，且 WER bucket 均为 `292ee832c503f92aa370cdee4d74cdfd`。最近一次 report ID：`1a6f38fd-1580-4344-af49-c0b06cffebdb`。
- PyAEDT 日志显示失败运行由外部 Python 启动全新的 AEDT gRPC 端口和进程；崩溃后没有可读 `Setup2 : LastAdaptive`，所以脚本最终以读回失败收口。
- 外部失败的 2×4 结果目录没有生成任何 `.profile`，而 GUI 成功副本生成了完整 profile；故崩溃发生在能够形成有效自适应求解记录之前。
- 同一个 2×4 工程由普通 GUI 启动同一个 `HFSSCOMENGINE.exe` 可成功完成。成功 profile 为本机 shared-memory、16 tasks/16 cores、MPI tasks=1，最大内存约 `3.47 GB`。因此不是“模型本身不可解”，也没有内存耗尽证据。
- 官方 `Analyze` 接口允许第二个 `isBlocking` 布尔参数，且约定返回 0 为成功；所以 `Analyze("Setup2", True)` 的参数个数不是已证实根因。
- 本机为 Windows 11 Home，而 Ansys 2025 R2 官方支持表只列 Professional/Enterprise/Education 并明确不支持 Home。它是显著兼容性风险，但由于 GUI 路线成功，不能据此宣称它是唯一根因。

最窄且证据充分的结论是：**外部 PyAEDT 新建 gRPC 会话与 HFSS 求解引擎之间存在稳定复现的启动/运行上下文故障；具体由哪一个传入参数、环境继承或 AEDT 缺陷触发，目前未被单独隔离。** 数据获取主线不再以重复崩溃试验换取更细定位。

### 1×8 保存后的恢复启动故障（2026-10-14）

- 普通 GUI 求解保存已产生 177 个结果文件、`179,790,072` bytes 和 1 个 `.profile`；工程 SHA256 为 `DBA4788D596EAE8AF7C7B7B5ECDF7559D142CCB4794D037DDD8408F9CFDFAC3B`。
- 对两个独立时间戳恢复副本运行已在 2×4 成功的 `run_verified_recovery.ps1`，均只到达 PyAEDT 日志 `New AEDT session is starting on gRPC port ...`，随后出现 `ansysedt.exe` 读取空地址 `0x0` 的应用程序错误。日志从未写出 AEDT PID、版本或 `Parsing project`。
- 因而这两次失败发生在工程打开前；它既不证明 1×8 结果损坏，也不是 S/EEP 内容错误。更换 gRPC 端口仍同型复现，已停止重试。
- Codex 工具启动的普通 AEDT 进程虽然建立了恢复副本 `.lock`，但没有任何顶层窗口，属于不可交互桌面；不能代替用户从 Windows 桌面正常启动 GUI。

### 已验证的 1×8 回退路线：普通 GUI 进程内导出

1. 从 Windows 桌面正常打开独立的时间戳恢复副本；最终成功副本为 `projects/recovery_work/a1x8_directapi_20261014T024216Z/a1x8_47ghz_directapi_20261014T024216Z.aedt`。
2. 在 AEDT 中选择 `Tools -> Run Script`，运行 `automation/export_existing_solution_in_gui.py`。
3. 该脚本严格核对活动工程名，不调用 `Analyze` 或 `Save`；它读取全部 64 条 S 参数并导出 integer/offset EEP。不要再让它创建可视 Data Table 直接场报告。
4. 在同一 GUI 恢复副本内运行 `automation/export_direct_fields_in_gui.py`，直接调用 `GetSolutionDataPerVariation` 导出 16 条复场；关闭工程时不要保存。
5. 外部运行 `automation/finalize_inprocess_export.py <S/EEP-run> --direct-api-run <direct-run>`，重建和解析 S8P、验证两套 EEP，并计算直接组合审计。
6. 只有离线 `FULL_GATE_REPORT.json` 的内容门禁全部 PASS，才允许晋级官方 raw exports/canonical。

这条路线与失败过的 `RunScriptAndExit`、外部 COM/PID 附着不同：AEDT 由用户以普通受支持 GUI 上下文启动，脚本由 GUI 菜单在现有进程内执行，不创建新的 gRPC 控制会话。

第一次进程内运行已经证明 64 条 S 参数以及 integer/offset 各 8 个 EEP 可在数秒内导出。随后用 `CreateReport(..., "Data Table", ...)` 生成第一条 65,341 点直接报告时，AEDT UI 被同步调用占用约 12 分钟；进程持续消耗 CPU、未崩溃，但没有产生首个 CSV。为避免按该速度重复 16 次，已终止的仅是恢复副本会话，已落盘 S/EEP 保留。该可视报告路线不再用于 1×8 主线。

直接组合审计改用 `automation/export_direct_fields_in_gui.py`：在新的时间戳恢复副本中通过 `ReportSetup.GetSolutionDataPerVariation` 直接读取 `re(rEPhi)`/`im(rEPhi)`，写紧凑 float64 数据，不创建或渲染 Data Table。离线 `finalize_inprocess_export.py --direct-api-run ...` 复用前序 S/EEP；离散核对唯一选择 `theta_phi_transpose` 网格顺序和 `0.001` raw-to-V 工程量纲比例，D00 探针线性 NMSE 约 `1.82e-30`，随后 16 条命令全部通过。该选择不是连续拟合。

## 不再重复的路线

除非 AEDT/PyAEDT/Windows edition 或许可环境发生明确变化并另立诊断任务，下列方法不得在数据主线上再次尝试：

- PyAEDT 新建 gRPC 图形或非图形会话直接求解；
- 改为单引擎、减少核心数、固定 Python 3.10、绕过 `analyze_setup` 直接调用 `oDesign.Analyze`；
- AEDT `RunScriptAndExit`（现有许可上下文以只读方式打开 HFSS Design）；
- `Ansoft.ElectronicsDesktop` 活动对象附着；
- PID + PyAEDT 0.9.6 附着（历史环境缺少 `pythoncom`，且从未形成已验证闭环）。

这些失败不否定模型。需要继续取数时，直接使用普通 GUI 求解和已验证的只读恢复流程。

## 1×8 已完成入口与回执

- build-PASS 快照：`projects/build_snapshots/a1x8_build_pass_20261014/`
- 已求解手动 GUI 副本：`projects/manual_gui_work/a1x8_47ghz_manual_gui.aedt`
- GUI 保存后工程 SHA256：`DBA4788D596EAE8AF7C7B7B5ECDF7559D142CCB4794D037DDD8408F9CFDFAC3B`
- GUI 保存后结果目录：177 文件、`179,790,072` bytes、1 个 `.profile`；内容实读门禁已 PASS。
- 成功恢复副本：`projects/recovery_work/a1x8_directapi_20261014T024216Z/a1x8_47ghz_directapi_20261014T024216Z.aedt`
- S/EEP 与最终门禁：`raw_exports/a1x8/inprocess_gate_20261014T032500Z/FULL_GATE_REPORT.json`
- 直接 API 原始场：`raw_exports/a1x8/direct_api_gate_20261014T024216Z/`
- 晋级回执：`raw_exports/a1x8/recovery_promotion_20261014.json`

外部 gRPC 恢复已同型崩溃两次，以下命令仅保留为 1×4/2×4 的历史成功入口，不得在本机 1×8 主线上继续重试：

```powershell
& .\automation\run_verified_recovery.ps1 -ArrayId a1x8 -Rows 1 -Cols 8
```

本轮已用 `promote_recovery_run.py` 晋级并按相同阈值完成 canonical。以后复用时，任何一步失败仍保留恢复副本和回执，不覆盖 build 快照或已成功的前序数据。

## 直接审计阈值

1. `Setup2 : LastAdaptive` 的全部 N² 条 S 参数可读且有限；
2. `.sNp` 可重新解析为完整 N×N 复矩阵；
3. integer/offset 各有 N 个非空、有限复数 FFD；
4. 16 条命令全部完成；
5. 聚合 complex NMSE `< -100 dB`；
6. 每条命令 complex NMSE `< -80 dB`；
7. 幅度最大相对误差 `< 1e-6`。
