# V2.10 A/B/C 整理与去重清单

## 整理规则

- 版本冲突以B（`_staging/v2.10`）和C（中文路径Git工作副本中的V2.10归档）为准；
- A中独有内容必须保留；若C的完整记录中已有字节相同副本，则保留C中的规范归档位置；
- Python缓存不属于项目资料，不进入Git；
- 删除现行重复项前，先用Git提交保存A原貌和B/C原始导入形态。

## Git恢复点

- `181f798`：整理前A目录快照；
- `b9b0a1b`：B与C按原始内容导入后的快照；
- 当前分支：`v2.10`。

这两个提交保证旧版本、旧路径和被去重的副本均可恢复；最终工作树只保留规范现行位置。

## 唯一现行位置

|内容|唯一现行位置|被合并的旧位置|
|---|---|---|
|train/validation/calibration/holdout四个冻结HFSS分区|`artifacts/front_hemisphere_20260906/datasets/`|`artifacts/controlled_psd_20260906/datasets/`；完整记录包内的`full_hfss_direction_datasets/`|
|validation/holdout四点HFSS套件|`artifacts/front_hemisphere_20260906/*_front_suite.npz`|完整记录包内的`raw/hfss_input/`|
|历史A10 seed 21011 checkpoint|`artifacts/controlled_psd_20260906/runs/A10_seed21011/best.pt`|`artifacts/front_hemisphere_20260906/reference/legacy_A10_21011.pt`|
|活动研究脚本|`tools/research/`|完整记录包内两个字节相同的`source_snapshot/`脚本|
|A的四点优化证据|`artifacts/four_point_optimization_complete_record_20260908/evidence/hfss_o3_sdp_cross_validation/`|A根目录下的`four_point_optimization_limit_evidence_20260906/`|
|理想阵列控制|`artifacts/four_point_optimization_complete_record_20260908/evidence/ideal_array_control/`|C外层的`ideal_array_control_20260907/`|

完整记录包的复核工具已改为通过相对路径读取上述共享数据。历史协议中的绝对来源字符串保留为实验溯源信息，不代表现行文件仍需重复存放。
