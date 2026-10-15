# 已选 HFSS 直接验证命令

本目录包含 24 组总入射功率约为 1 W 的源设置：

- `selected_commands_summary.csv`：命令用途、优先级和逐命令源文件；
- `selected_commands_matrix.csv`：每组16端口的功率/相位横向总表；
- `sources/cmd_*_sources.csv`：适合逐组录入 HFSS Edit Sources 的纵向表；
- `commands_manifest.json`：端口顺序、随机种子、历史案例和来源哈希；
- `generate_selected_commands.py`：可复现这些表的生成程序。

先完成 `priority=core` 的12组：A01、A02、A03、A04、A06、A08、B01、B02、C03、C04、D03、
D04。核心组覆盖公共相位不变性、端口行列、结构化/随机相位、历史最差 case 26、低 regret
case 22，以及连续幅相/phase-only 对照。

C/D组来自已经打开的历史诊断，只用于验证HFSS直接整阵列结果与现有复数基底的闭环，不能
作为下一轮新盲测，也不能在下一轮训练中冒充未见数据。

每个源文件的 `power_w` 才是 HFSS 使用 W 作为源单位时需要录入的数值；
`amplitude_coefficient` 是算法复权值模长，仅用于审计。端口按 `1:1` 到 `16:1`、4×4行优先排列。
