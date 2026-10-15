"""按唯一正确顺序重建 V2.4 发布结果；默认只展示计划。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PIPELINE_STEPS = (
    ("train_sparse_controller", "run_v24_sparse_end_to_end_study.py", ()),
    ("refine_low_update", "run_v24_finetune_refinement.py", ()),
    ("finalize_integer_release", "finalize_v24_results.py", ()),
    ("export_input_output_comparison", "export_v24_controller_bundle.py", ()),
    (
        "validate_release",
        "validate_v24_controller_artifacts.py",
        ("--version", "v2.4"),
    ),
)


def command_plan() -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "command": [sys.executable, "-X", "utf8", "-B", str(CODE_DIR / script), *args],
        }
        for step, script, args in PIPELINE_STEPS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-overwrite",
        action="store_true",
        help="确认重写 output/controller_training 中的冻结研究结果；完整训练耗时数分钟。",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印执行顺序。")
    args = parser.parse_args()
    plan = command_plan()
    if args.dry_run or not args.confirm_overwrite:
        print(json.dumps({"dry_run": True, "steps": plan}, ensure_ascii=False, indent=2))
        if not args.dry_run and not args.confirm_overwrite:
            raise SystemExit("拒绝覆盖：实际重跑必须显式传入 --confirm-overwrite")
        return
    for item in plan:
        subprocess.run(item["command"], cwd=CODE_DIR, check=True)


if __name__ == "__main__":
    main()
