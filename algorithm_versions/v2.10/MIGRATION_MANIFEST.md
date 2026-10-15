# V2.9 → V2.10 迁移清单

## 已迁移

- `src/v29` 已复制为独立 `src/v210` 包，包名、CLI 入口和测试引用已改为 `v210`；
- V2.9 的坐标、数据合同、HFSS I/O、物理层、幅度训练、优化器和研究工具均保留；
- 配置、schema、测试、fixture 和端口 manifest 已迁移；
- `pyproject.toml` 已改为 `aesa-cnn-v210`，并增加 `v210-campaign` 入口；
- V2.9 原目录未写入或覆盖。

## V2.10 新增

- `models/quadratic_psd.py`：A10 的迁移场基线与 B10/C10 的 rank-2 PSD 因子；
- `research/v210_experiment.py`：固定三 seed、等更新预算、冻结后评估的完整活动；
- `cli/v210_campaign.py`：prepare/selftest/train/evaluate/all 入口；
- `tests/test_v210_quadratic_psd.py`：PSD、Hermitian、二次型、横向性和相位不变性测试；
- `MISSION_CONTRACT.md` 与机器可读合同；
- `artifacts/controlled_psd_20260906`：数据哈希、九组训练、冻结选择和完整报告。

## 未直接复制

- V2.9 的旧 checkpoint 和历史报告没有伪装成 V2.10 产物；V2.10 新 checkpoint 全部重新训练；
- 完整原始 HFSS FFD 没有重复复制，受控活动只复制 V2.9 已冻结的四个 NPZ 分区；
- 新物理状态、轨道/站点和硬件数据因不存在而未生成占位标签。

## 复现入口

在 `algorithm_versions/v2.10` 下执行：

```powershell
$env:PYTHONPATH = "$PWD\src"
$python = "D:\Miniconda3\envs\rl_env\python.exe"
$source = "..\v2.9\artifacts\vector_beam_20260905"
$root = "artifacts\controlled_psd_20260906"

& $python -B -m unittest discover -s tests -v
& $python -B -m v210.cli.v210_campaign prepare --root $root --source-root $source
& $python -B -m v210.cli.v210_campaign selftest --root $root --device cuda
& $python -B -m v210.cli.v210_campaign train --root $root --device cuda
& $python -B -m v210.cli.v210_campaign evaluate --root $root --device cuda
```

活动已经冻结；同一 `root` 在 `selection_frozen.json`/`holdout_opened.json` 存在时会拒绝再次训练。
如要做下一轮实验，应创建新的 artifact 目录和新协议，不得覆盖本轮结果。

## 2026-09-09 A/B/C 整理

整理后，四个冻结 HFSS NPZ 分区仅保留在
`artifacts/front_hemisphere_20260906/datasets/`。历史协议中的旧相对路径继续作为来源记录，
读取和校验代码会在旧副本不存在时解析到上述唯一现行位置。完整映射、恢复提交和删除边界见
`CONSOLIDATION_MANIFEST.md`。
