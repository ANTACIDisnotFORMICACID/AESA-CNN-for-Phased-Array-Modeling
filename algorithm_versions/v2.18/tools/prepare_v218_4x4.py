"""Prepare the sealed V2.18 4x4 canonical adapter package."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
VERSION = Path(__file__).resolve().parents[1]
for relative in ("v2.10/src", "v2.12/src", "v2.16/src", "v2.18/src"):
    sys.path.insert(0, str(ROOT / "algorithm_versions" / relative))

from v218.canonical import prepare


if __name__ == "__main__":
    result = prepare(
        train_cache_path=ROOT / "algorithm_versions/v2.17/artifacts/training_views/a4x4/train_operator_cache.npz",
        validation_cache_path=ROOT / "algorithm_versions/v2.17/artifacts/training_views/a4x4/validation_operator_cache.npz",
        v216_exchange_path=ROOT / "algorithm_versions/v2.16/artifacts/prepared_20260914/train_exchange.npz",
        validation_commands_path=ROOT / "algorithm_versions/v2.17/splits/a4x4/validation_commands.npz",
        output_dir=VERSION / "artifacts/prepared_4x4_20261014",
    )
    print(json.dumps(result, indent=2))
