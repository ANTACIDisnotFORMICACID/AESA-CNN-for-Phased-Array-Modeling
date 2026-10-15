# V2.9 固定结构、GPU 收敛训练与双盲测

本轮严格保留 34,973 参数结构和原损失组成。预算固定为：
R1/R2/R3 各 300 轮筛选；最优两种各两个 seed、800 轮确认；
胜出方案三个 seed 各最多 2,000 轮。不得将预算静默缩减。
400 轮前不早停；之后连续 150 轮最佳验证 NMSE 改善不足 0.05 dB 才能早停。
到上限仍改善时必须报告未收敛。

## 隔离链路

1. 准备程序核验 16 个端口映射、47 GHz、公共相位参考点和两套角网格。
2. 错位 FFD 此阶段只读取四行头部与计算文件哈希，不分析其场值。
3. 416 条命令分为 292/62/62；整数角网格按绝对原点的 10 度角块分割。
4. 仅写入 dev/train.npz 和 dev/validation.npz 的幅度标签。
   blind/spec.npz 仅有命令、分区、角块，无场标签。训练程序不读取原始 FFD 或 blind。
5. 训练策略和代表 seed 都只依验证集选择。三个最终 checkpoint 与 epoch200 checkpoint
   全部写入 frozen_selection.json 的哈希后，才能打开两级盲测。
6. blind_opened.json 存在后，训练入口拒绝启动。双盲测报告完成后拒绝重复评估。
   失败评估仅允许在同一冻结哈希下恢复，禁止因此更换模型或调整阈值。

## 不能夸大的结论

- 旧整数网格和有限单阵元/DFT命令在历史开发中已出现；本轮不能抹去这种历史暴露。
  报告另列 58 条新型测试命令的指标。
- 错位网格全部方向是未见采样点，但其中训练/验证角块内的点属于插值。
  主要二级盲测只统计 test 角块，其余全网格结果明确作为附加诊断。
- 两套网格来自同一个 HFSS 电磁解，不是独立器件状态或 OTA 实测。
- Ephi 幅度与经校准的单极化接收幅度属于同类观测，但当前模拟没有器件链和测量噪声。
- 四目标参考为 32 起点数值优化，不保证全局最优；后悔值只是相对此数值参考。
  30 组方向仅用于算法诊断，不能替代尚未确定的实物目标坐标。

## 固定验收条件

三 seed 均收敛；相对各自 epoch200 盲测 NMSE 平均改善至少 0.3 dB，
样本标准差不超过 0.25 dB，单 seed 不倒退超过 0.1 dB；
功率 NMSE 和 dB MAE 不整体恶化；
错位 test 角块相对整数 test 角块平均退化不超过 0.5 dB；
四目标相对数值参考的后悔值中位数不超过 5%、P95 不超过 10%。
未达标必须如实报告，不能在开启盲测后继续调参。

## 运行环境和产物

使用既有 D:/Miniconda3/envs/rl_env/python.exe（PyTorch CUDA），不修改 CPU 基线环境。
入口为 python -B -m v29.cli.research_campaign，动作依次为 prepare、train、evaluate。
本轮产物写入 artifacts/research_20260905；原 v2.8 数据与旧 v2.9 checkpoint 不覆盖。
协议保存源代码、原始 FFD、开发数据和盲测规范的 SHA-256，运行时拒绝协议或代码漂移。

FFD 行序和相干重组对照了 Ansys 官方实现：
[PyAEDT farfield_visualization](https://github.com/ansys/pyaedt/blob/main/src/ansys/aedt/core/visualization/advanced/farfield_visualization.py)。
全方向指标按现有等角采样点统计，不冒充等立体角积分。
