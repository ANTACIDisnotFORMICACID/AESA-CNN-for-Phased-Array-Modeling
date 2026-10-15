# HFSS 直接整阵列导出清单

## 1. 命令集合

`commands/selected_commands_summary.csv` 列出 24 组命令。`priority=core` 的 12 组先导出；闭环
通过后再完成 `extended`。每组逐端口功率和相位位于 `commands/sources/`。

命令分为：

- A：等幅结构化相位，用于检查公共相位、端口顺序、行列方向和渐进相位；
- B：固定随机种子的 4-bit 等幅相位，用于离散相位压力测试；
- C：V2.9 现有四目标优化的连续幅相命令，覆盖不同目标间距和 regret；
- D：与 C 相同相位但改为每端口 1/16 W，用于 phase-only 对照。

当前硬件位数未知，所以 B 组的 4-bit 不是硬件假设，只是可复现的离散测试。

## 2. 每组操作

1. 使用与现有嵌入式 FFD 相同的 `Setup2 / LastAdaptive / 47 GHz`。
2. 确认源名严格为 `1:1` 到 `16:1`，按 4×4 行优先映射。
3. 打开 `commands/sources/cmd_*_sources.csv`。
4. 在 HFSS `Edit Sources` 中逐端口填写 `power_w` 和 `phase_deg`。
5. 再次核对 `total_power_w=1`；不要把 `amplitude_coefficient` 当作瓦特。
6. 保持公共相位中心、端口终端、远场 setup 和坐标系不变。
7. 导出错位网格：theta=0.5°…179.5°，phi=-179.5°…179.5°，步长1°。
8. 导出复数 `rETheta`、`rEPhi` 的实部和虚部到
   `exports/complex_farfield/<command_id>_complex_farfield.csv`。
9. 导出 `RealizedGainTotal` 到
   `exports/realized_gain/<command_id>_realized_gain.csv`。
10. 若方便，同时导出 `GainTotal`、incident power、accepted power；不要覆盖前两项。
11. 复制 `exports/metadata/command_metadata_template.json`，填写命令ID和HFSS设置，保存为
    `<command_id>_meta.json`。
12. 不修改HFSS原始列名、顺序、单位或数值；处理和归一化由后续程序完成。

## 3. 推荐执行顺序

1. A01 与 A02：验证公共相位只改变复场相位、不改变 RealizedGainTotal；
2. A03、A04、A06、A08：检查端口、行列和相位正负号；
3. B01、B02：检查一般离散相位叠加；
4. C03：历史最差 case 26 连续幅相命令；
5. D03：case 26 的等幅 phase-only 对照；
6. C04、D04：低 regret case 22 对照；
7. 核心12组闭环通过后，完成其余 extended 命令。

## 4. 核心闭环门槛

- 相对各命令峰值高于 -20 dB 区域，FFD合成与HFSS直接结果幅度MAE ≤0.10 dB；
- 幅度误差P95 ≤0.20 dB；
- 公共相位对齐后复场相位误差P95 ≤2°；
- RealizedGainTotal MAE ≤0.10 dB，P95 ≤0.20 dB；
- 四目标命令在对应四点上的最小功率/增益差 ≤1%。

任一项失败时停止新增训练数据，先检查源单位、端口顺序、相位符号、Etheta/Ephi组合、总入射
功率、相位中心和HFSS归一化。
