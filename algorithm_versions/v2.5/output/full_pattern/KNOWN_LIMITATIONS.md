# V2.5 发布结果边界

1. V2.5 的 CNN 控制能力继承自 V2.4；本版新增完整方向图渲染与兼容包装，不代表控制器经过新训练或波束指标再次提升。
2. V2.5 只在 `output/controller_training/` 保存一套 V2.4 checkpoint；内部
   `algorithm_version` 保持 `2.4.0`，不得用 V2.5 标签覆盖来源语义，也不得重新
   `torch.save()` 冻结产物。完整图侧直接引用该唯一副本。
3. 当前完整包可直接运行 V2.4 控制链 22 项和 V2.5 完整场 13 项，共 35 项；
   8 个 checkpoint 门槛由同一分支总 validator 检查，不依赖切换到 V2.4。
4. 历史 JSON 已以原始 Git blob 放入 `input/full_pattern/historical_replay_input.json`，
   运行时验证 blob SHA-1 后读取；不再依赖祖先提交、完整克隆或联网取数。
5. FP64/FP32 的极小误差证明物理渲染器与独立阵列公式一致，不是实物天线误差。真实阵元方向性、射频非线性、温漂和未建模互耦仍须在 1×4 实物上校准验证。
6. FPGA 在线闭环通常只需输出 4 或 16 个复权值；64×64 完整图可留在主机侧作为观察器，除非系统明确要求硬件实时渲染。
7. `comparison_inputs.json`、`comparison_results.json` 和三张图是由冻结 NPZ、CSV、
   JSON 与 checkpoint 元数据确定性生成的仿真证据，不是暗室、VNA 或 OTA 实测。
