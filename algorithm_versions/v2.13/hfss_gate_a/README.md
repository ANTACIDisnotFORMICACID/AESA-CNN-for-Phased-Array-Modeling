# V2.13 Gate-A HFSS 直接复场导出

本目录用于验证：

\[
E_{\phi,\mathrm{HFSS}}(q,w)\overset{?}{=}
\sum_{n=1}^{16}w_nE_\phi(q,e_n).
\]

命令集固定为 8 组：零相位和 90° 公共相位控制各一组、4 组连续随机恒模相位、
2 组随机幅相权重；全部满足 `sum(|w|^2)=1 W`。8 组乘完整 181×361 网格，约
52 万个方向–命令复场样本，足够用于 Gate-A 数值合同审计，不作为 CNN 训练数据。

导出器复用 V2.9 已验证的 PyAEDT 0.9.6 路径：

1. `hfss.edit_sources(..., include_port_post_processing=True)`；
2. `Far Fields / Infinite Sphere1 / Setup2 : LastAdaptive / 47 GHz`；
3. 导出 `re(rEPhi)` 与 `im(rEPhi)` 的 HFSS Data Table；
4. 关闭工程时 `save=False`，并核对工程文件前后摘要一致。

`H00` 允许复用 V2.9 已存在的 A01 直接 HFSS 报表。V2.9 的其余 24 组
`complex_farfield` 明确由 Python 线性合成，因此不复用为独立证据。
其原始 PyAEDT 冒烟报告复制在 `reused_evidence/H00_prior_direct_hfss_smoke_report.json`。

实际导出时 HFSS 对新报表选择了 `mV`，历史 H00 报表为 `V`。解析器根据每个字段
表头显式换算到 `V`；禁止依赖固定的隐含缩放。该差异由专门单元测试覆盖。

组合复场保存在 `raw_direct_reports/`，其 provenance 是 HFSS report engine，而不是
本地 `g @ w`。`prepared/gate_a_hfss_direct_complex.npz` 同时保存已有单端口 FFD
基，仅用于两个独立代码路径的事后比较。这里的“直接”不表示每个 `w` 重新网格求解；
HFSS 的线性多端口后处理本身会复用同一已收敛场解。它能验证 HFSS 内部源组合、端口
顺序、幅相和单位约定，不能验证功放非线性或真实硬件状态漂移。
