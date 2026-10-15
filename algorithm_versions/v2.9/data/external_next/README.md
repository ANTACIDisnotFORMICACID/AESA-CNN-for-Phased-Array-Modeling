# 下一轮外部输入准备区

本目录只用于保存下一轮优化所需的人工输入和 HFSS 直接验证导出，不改变 V2.9 已冻结的
`research_20260905` 数据和结果。

当前已确认：阵列部署在卫星，四个目标是地面接收站；真实轨道、接收站坐标、FPGA 资源、
射频控制限制和测量误差暂时未知；天线层目标采用 RealizedGainTotal，最终控制目标优先采用
链路余量，其次为 SNR 或接收功率。

目录：

```text
external_next/
  trajectories/               合成/真实四目标轨迹及坐标定义
  hardware/                   FPGA、相移器和误差条件
  link_budget/                接收站和链路预算
  hfss_direct_validation/
    HFSS_EXPORT_CHECKLIST.md   人工导出步骤
    commands/                  已选 24 组命令和逐端口源表
    exports/complex_farfield/  HFSS 复 Etheta/Ephi 原始导出
    exports/realized_gain/     HFSS RealizedGainTotal 原始导出
    exports/metadata/          每组设置记录
    s_parameters/              已有或新导出的 Touchstone
    checksums/                 人工导出完成后的哈希清单
```

原则：原始 HFSS 文件不手工归一化、不改列顺序；处理后的文件另存。所有新训练/验证/校准/
盲测划分按完整轨迹或状态分组，禁止把相邻帧随机拆分。
