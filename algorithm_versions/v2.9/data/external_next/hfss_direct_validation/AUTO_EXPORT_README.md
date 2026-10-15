# 自动导出实施记录（v2.9）

## 1. 已验证的自动化边界

`tools/hfss/batch_linear_superposition.py` 不打开、不修改、不保存 AEDT 工程，仅读取 v2.8 已导出的 16 个嵌入式方向图和 `commands/sources/cmd_*_sources.csv`。每个端口的复权重严格为：

`amplitude_coefficient * exp(j * phase_deg)`；`power_w` 仅用于总功率检查，不能直接作为复幅度。

FFD 四列已核对为 `[re(rETheta), im(rETheta), re(rEPhi), im(rEPhi)]`。A01 的 integer-grid 线性叠加与直接 HFSS 报表 65341 点逐点一致；`RealizedGainTotal_linear = (|rETheta|² + |rEPhi|²) / 60` 在 A01 全网格恒定校准，适用前提是同一 47 GHz、1 W 源归一化和同一 FFD 导出约定。

这些结果是 HFSS 嵌入式模式的线性合成，不是 24 次独立 HFSS 直接仿真；因此不能用于证明非线性、功放压缩或状态相关互耦已经被覆盖。

## 2. 运行与输出

使用已有 HFSS Python 环境运行：

```powershell
D:\Desktop\wireless\BigChuang\floquet\.venv_hfss\Scripts\python.exe `
  v2.9\tools\hfss\batch_linear_superposition.py `
  --v29-root v2.9 --priority all --force
```

输出位置：

- `data/external_next/hfss_direct_validation/exports/complex_farfield/<command_id>_complex_farfield.csv`
- `data/external_next/hfss_direct_validation/exports/realized_gain/<command_id>_realized_gain.csv`
- `data/external_next/hfss_direct_validation/exports/metadata/<command_id>_meta.json`
- `data/external_next/hfss_direct_validation/exports/batch_export_report.json`
- `data/external_next/hfss_direct_validation/exports/batch_manifest.json`

## 3. 本轮实际结果

- core 12 组先完成，全部通过 A01 直接 HFSS 校准和 A01/A02 公共相位关系检查。
- 随后完成 extended 12 组；共 24 组复场文件和 24 组增益文件。
- 每个输出均为 64800 行，网格为 theta 0.5°…179.5°、phi -179.5°…179.5°，无 NaN/Inf，坐标对唯一且完整，总端口功率通过约 1 W 检查，文件非空。
- A01/A02 关系为 `E_A02 = j E_A01`，最大复场误差约 `1.0e-11 V`。

## 4. 后续证据边界

若要确认 RealizedGainTotal 在不同幅相状态下仍与直接 HFSS 一致，至少应另行对 A02 或一组幅相命令做直接 HFSS 报表并与对应合成结果对齐；当前 `1/60` 仅由 A01 直接报表校准，不应扩大解释为所有设置下的实测保证。
