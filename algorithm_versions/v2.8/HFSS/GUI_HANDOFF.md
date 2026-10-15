# V2.8 HFSS GUI 接力说明

当前 Codex 进程运行于 `FORMIC-ACID\CodexSandboxOffline`，该身份不能安全激活本机
HFSS Design。以下操作必须在你的交互式 `FORMIC-ACID\ASUS` 会话中完成。

## 首选：在普通 PowerShell 中运行现成脚本

注意：不要直接双击或从包含中文目录的完整 `D:\Desktop\...` 路径打开 AEDT 工程。
AEDT 2025 R2 的 gRPC 保存可能将中文路径显示为 `?????` 并报“无法写入文件”。
下面两个脚本会临时把当前 HFSS 目录映射为 ASCII 的 `V:` 盘符后再启动 AEDT。

进入本目录后依次运行：

```powershell
.\run_build_4x4.ps1 -Graphical
```

构建成功后先确认：

- `projects/v2_8_array_4x4.aedt` 存在；
- `artifacts/build_4x4_report.json` 中 `passed=true`；
- 端口共 16 个，逻辑坐标正好覆盖 `[row_z,col_y]=[4,4]`；
- `Y` 间距为 6.8 mm，`Z` 间距为 15 mm；
- 只有一个 Radiation 边界，且只绑定一个六面长方体空气盒；
- `artifacts/build_4x4_report.json` 的 `radiation_box` 记录显示旧 `Box9` 已删除，
  新盒完整包围四排，六个方向的最小余量均不少于 47 GHz 的 `lambda/4`（约 1.595 mm）；
- `Infinite Sphere1` 仍存在。

然后运行：

```powershell
.\run_solve_export.ps1 -Graphical -Cores 4
```

该步骤只对 4×4 工程执行一次 `Analyze Setup2`，随后导出：

- `exports/automatic/v2_8_array_4x4.s16p`；
- `exports/automatic/element_patterns/` 中的映射 TXT 与 16 个 FFD；
- `artifacts/solve_export_report.json`。

## 若 4×4 自动构建失败

若需要在 GUI 中手工操作，先在同一个普通 PowerShell 窗口建立临时映射：

```powershell
subst V: (Resolve-Path .)
```

然后从 `V:\projects\` 打开工程。完成后可用 `subst V: /D` 解除映射。当前已经从真实中文路径打开的工程窗口不要继续保存；若没有需要保留的未保存修改，应先关闭并选择“不保存”，再从 `V:\projects\v2_8_array_4x4.aedt` 重新打开。

1. 在 AEDT 2025 R2 打开 `projects/source_1x4_20250326.aedt`；
2. 立即另存为 `projects/v2_8_array_4x4.aedt`；
3. 选择 `ANT1` 组内全部模型对象，并额外选择 `Box12`；不要选择 `Box9` 或
   `NF_Plane_Z65_75`；
4. 执行 `Duplicate Along Line`：向量 `(0,0,15mm)`，`NumClones=4`，勾选
   `Duplicate Assignments`；
5. 确认得到原排加三排，并且波端口由 4 个增加为 16 个；
6. 删除旧 `Rad1` 和旧空气盒 `Box9`，在 47 GHz 下重新建立一个且仅一个覆盖整个
   阵列的长方体 `Radiation/Open Region`；检查它有 6 个面，并在 `±X/±Y/±Z` 六个
   方向相对阵列包围盒均留出至少 `lambda/4≈1.595 mm`；
7. 保留 `Infinite Sphere1`：Theta `0°..180°/1°`，Phi `-180°..180°/1°`；
8. 运行 `Validate`，保存工程。不要把旧 1×4 结果目录复制为 4×4 求解结果。

如果手工构建完成，请告诉 Codex；Codex 会重新生成端口清单和工程哈希后再安排求解。

## 若求解成功但自动球面导出失败

优先在 `Radiation > Infinite Sphere1` 中执行 `Compute Antenna Parameters`，选择：

- Solution：`Setup2 : LastAdaptive`；
- Frequency：`47GHz`；
- Export：`Element Pattern / Far Field Data (*.ffd)`。

一次导出应生成一个源映射 TXT 和 16 个 FFD。把整个导出结果原样放入
`exports/gui_drop/`，不要根据文件序号猜端口映射，也不要重排 FFD 正文。

若只能导出报表，则对每个端口依次设置 `1 W / 0°`，其余 15 个端口为 `0 W / 0°`，
导出完整 Theta/Phi 网格的：

```text
re(rETheta), im(rETheta), re(rEPhi), im(rEPhi)
```

同时导出 16×16 复 S 参数。不要只导出 `GainTotal`、dB 幅值或归一化方向图。
