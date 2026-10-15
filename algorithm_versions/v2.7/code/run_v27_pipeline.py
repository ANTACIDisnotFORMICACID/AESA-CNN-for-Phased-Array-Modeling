"""Safe stage runner for the characterization-first V2.7 workflow."""

from __future__ import annotations

import argparse
import json

from evaluate_v27 import merge_audit, run_audit
from environment_v27 import run as run_environment
from oracle_v27 import merge_nominal_map, run_shards, summarize_map
from phase_map_v27 import run as run_phase_map
from report_v27 import build_nominal_controller_maps, write_report
from state_oracle_v27 import merge as merge_state_oracle, run as run_state_oracle
from states_v27 import write_state_manifest
from verify_oracle_v27 import run as run_repeatability, summarize as summarize_repeatability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("states", "phase-map", "oracle", "audit", "state-oracle", "repeatability", "report", "all"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="diagnostic limit for new oracle targets or audit states")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = ("states", "phase-map", "oracle", "audit", "state-oracle", "repeatability", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "states":
            run_environment()
            print(json.dumps(write_state_manifest()["counts"], ensure_ascii=False, indent=2))
        elif stage == "phase-map":
            print(json.dumps(run_phase_map(), ensure_ascii=False, indent=2))
        elif stage == "oracle":
            progress = run_shards(args.workers, args.limit)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            if progress["remaining"] == 0:
                print(json.dumps(summarize_map(merge_nominal_map()), ensure_ascii=False, indent=2))
        elif stage == "audit":
            progress = run_audit(args.limit, args.workers)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            if progress["completed_states"] == 83:
                print([str(path) for path in merge_audit()])
        elif stage == "state-oracle":
            progress = run_state_oracle(args.workers, args.limit)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            if progress["remaining"] == 0:
                print(merge_state_oracle())
        elif stage == "repeatability":
            progress = run_repeatability(args.workers, args.limit)
            print(json.dumps(progress, ensure_ascii=False, indent=2))
            if progress["completed"] == progress["total"]:
                print(json.dumps(summarize_repeatability(), ensure_ascii=False, indent=2))
        elif stage == "report":
            build_nominal_controller_maps()
            print(json.dumps(write_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
