from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.cross_array import (  # noqa: E402
    ArraySpec,
    canonicalize_array,
    file_sha256,
    parse_ffd_mapping,
    write_split_and_commands,
    write_training_views,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze one HFSS array into the V2.17 data contract")
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--integer-mapping", type=Path, required=True)
    parser.add_argument("--offset-mapping", type=Path, required=True)
    parser.add_argument("--touchstone", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--row-pitch-mm", type=float, default=15.0)
    parser.add_argument("--col-pitch-mm", type=float, default=6.8)
    parser.add_argument("--frequency-hz", type=float, default=47e9)
    parser.add_argument("--split-seed", type=int, default=21701)
    parser.add_argument("--validation-command-seed", type=int, default=21702)
    parser.add_argument("--test-command-seed", type=int, default=21703)
    return parser.parse_args()


def fingerprint_sources(mapping: Path) -> list[dict[str, object]]:
    files = [mapping, *(Path(value) for value in parse_ffd_mapping(mapping).values())]
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in files
    ]


def update_manifest(root: Path, entry: dict[str, object]) -> None:
    path = root / "dataset_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema": "cross_array_47ghz_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arrays": {},
        }
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    previous = manifest["arrays"].get(entry["array_id"], {})
    manifest["arrays"][entry["array_id"]] = {**previous, **entry}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    root = args.dataset_root.resolve()
    canonical_path = root / "canonical" / args.array_id / "operator_47ghz.npz"
    spec = ArraySpec(
        array_id=args.array_id,
        rows=args.rows,
        cols=args.cols,
        row_pitch_m=args.row_pitch_mm * 1e-3,
        col_pitch_m=args.col_pitch_mm * 1e-3,
        frequency_hz=args.frequency_hz,
    )
    summary = canonicalize_array(
        spec,
        integer_mapping=args.integer_mapping,
        offset_mapping=args.offset_mapping,
        touchstone=args.touchstone,
        output=canonical_path,
    )
    split = write_split_and_commands(
        canonical_path,
        ROOT / "splits" / args.array_id,
        split_seed=args.split_seed,
        validation_command_seed=args.validation_command_seed,
        test_command_seed=args.test_command_seed,
    )
    training_views = write_training_views(
        canonical_path,
        ROOT / "splits" / args.array_id,
        ROOT / "artifacts" / "training_views" / args.array_id,
    )
    sources = (
        fingerprint_sources(args.integer_mapping)
        + fingerprint_sources(args.offset_mapping)
        + [{
            "path": str(args.touchstone.resolve()),
            "bytes": args.touchstone.stat().st_size,
            "sha256": file_sha256(args.touchstone),
        }]
    )
    entry = {
        **summary,
        "spec": spec.to_json(),
        "split": split,
        "training_views": training_views,
        "sources_fingerprint": sources,
        "status": "canonical_complete",
    }
    update_manifest(root, entry)
    report_path = root / "canonical" / args.array_id / "canonicalization_report.json"
    report_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
