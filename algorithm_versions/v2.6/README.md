# V2.6：固定 1×4 HFSS 阵列控制器

## 版本摘要

| 项目 | 内容 |
|---|---|
| 类型 | 固定阵列控制器实现版 |
| 继承 | V2.5 的轻量控制器接口和完整复场评估思路 |
| 主要变化 | 面向已求解的 47 GHz、1×4 HFSS 阵列，以解析权值为主干，CNN 只预测复残差 |
| 冻结结论 | 主机实现与物理审计完成；正式发布训练、Vitis 编译和硬件验证未完成 |
| 后续关系 | V2.7 不立即训练新模型，而是先审计固定阵列在二维角域中的可达边界 |

## 定位与边界

V2.6 is an independent experimental version for the solved 47 GHz HFSS array.
It does not modify V2.5, its checkpoints, golden vectors, or the original AEDT
project.  The canonical electromagnetic data remains read-only at
`data/hfss_reference_1x4/hfss_active_element_patterns_47ghz.npz`。

The model is a residual weight controller, not a replacement electromagnetic
solver.  Host code builds a 15-channel physical input, the CNN produces only a
complex residual, and host/PS code performs health masking, complex projection,
INT8 disk quantization, full-sphere evaluation, and feasibility reporting.

Quick verification from the repository root:

```powershell
.\AESA-CNN\.venv\Scripts\python.exe algorithm_versions\v2.6\tests\run_tests.py
```

Create the deterministic calibration design and a small oracle smoke report:

```powershell
.\AESA-CNN\.venv\Scripts\python.exe algorithm_versions\v2.6\code\run_v26_pipeline.py --stage smoke
.\AESA-CNN\.venv\Scripts\python.exe algorithm_versions\v2.6\code\run_v26_pipeline.py --stage profiles
.\AESA-CNN\.venv\Scripts\python.exe algorithm_versions\v2.6\code\run_v26_pipeline.py --stage audit
```

The production oracle deliberately defaults to the specified 32 starts and 300
iterations.  It supports resumable shards because the complete 64x64 map is a
large CPU job:

```powershell
.\AESA-CNN\.venv\Scripts\python.exe algorithm_versions\v2.6\code\run_v26_pipeline.py --stage oracle --shard-index 0 --shard-count 16
```

No result is labelled accurate two-dimensional scanning unless the physical
feasibility tests pass.  When they do not, the API returns finite best-effort
weights plus a reason bit mask.

The current evidence and unfinished release gates are summarized in
`output/reports/V2.6_IMPLEMENTATION_REPORT.md` and `KNOWN_LIMITATIONS.md`.
