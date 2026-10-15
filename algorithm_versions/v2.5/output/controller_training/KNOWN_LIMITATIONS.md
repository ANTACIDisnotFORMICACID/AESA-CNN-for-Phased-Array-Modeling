# V2.4 发布结果边界

1. 当前 8 个 checkpoint、golden vectors、INT8 输出、测点设计和单板曲线已通过结构与数值校验；不要重新 `torch.save()` 这些冻结 checkpoint。
2. 4×4 与 1×4 的 low-update 发布控制器由 validation 选择 `zero_shot`（分别对应已记录 seed），表示在给定预算下继续更新没有改善验证指标，不表示微调流程缺失。
3. 所有板级数据仍是受控非理想仿真，不是真实 OTA 测量；V2.4 证明的是算法闭环与少测点可行性，不等于完成 FPGA 部署或实物泛化验证。
4. 本目录是 V2.5 自包含项目中的 V2.4 数学阶段；它完整保存控制器训练、8 个
   checkpoint 和发布证据。远端 `v2.4` 分支仍保留该阶段的独立历史版本，但执行
   V2.5 不需要切换分支或复制 checkpoint。
5. `input/controller_training/comparison_inputs.json`、本目录的
   `comparison_results.json` 和 `pictures/` 是从冻结 golden vectors/metrics
   确定性导出的查看层，不代替完整训练数据，也不声称是实测 OTA 数据。
