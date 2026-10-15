# V2.8 HFSS 工作区

本目录隔离保存 V2.8 的 AEDT 母版副本、4×4 工作工程、自动化脚本、审计记录和球面
复场导出。外层 `HFSS/20250326.aedt` 只作为来源，不在原位修改。

## 目录

- `projects/`：母版副本和生成的 4×4 AEDT 工程；
- `automation/`：只用于 V2.8 HFSS 构建、审计和导出的脚本；
- `artifacts/`：JSON 清单、端口/几何/求解审计和规范化数据；
- `exports/gui_drop/`：自动导出失败时，HFSS GUI 导出文件的唯一投放位置；
- `logs/`：AEDT/PyAEDT 运行日志。

## 当前冻结条件

- AEDT/HFSS：2025 R2；
- 频率：47 GHz；
- 原 1×4 沿 HFSS `+Y`，列间距 6.8 mm；
- 四排沿 HFSS `+Z`，排间距 15 mm；
- 阵面为 `Y-Z`，名义端射方向为 `+X`；
- 端口逻辑索引为 `[row_z,col_y]`，数据进入算法后保持 `[4,4]`，不得展平。

## 安全规则

1. `projects/source_1x4_20250326.aedt` 是带哈希的母版副本，只读审计；
2. 所有几何修改保存为新的 `projects/v2_8_array_4x4.aedt`；
3. 旧 1×4 求解结果不复制为 4×4 结果，4×4 必须重新验证并求解；
4. 自动导出失败时不补零、不伪造复相位，改用 `exports/gui_drop/` 的 GUI 流程；
5. 球面数据必须包含每个端口的复 `rETheta/rEPhi`、统一 Theta/Phi 网格和 16×16
   复 S 参数。
6. 4×4 工程只能有一个 Radiation 边界和一个六面长方体 `rad_box`；旧 `Box9` 必须
   删除，新盒须完整包含阵列并在六个方向各保留至少 47 GHz 的 `lambda/4` 余量。
