from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def nearest(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, float]:
    distance, index = cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64), k=1
    )
    maximum = float(np.max(distance))
    if maximum >= 1e-6:
        raise ValueError(f"Maximum direction alignment error {maximum} is not <1e-6")
    return np.asarray(index, dtype=np.int64), maximum


def nmse_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ratio = float(np.vdot(reference - candidate, reference - candidate).real / np.vdot(reference, reference).real)
    return 10 * math.log10(max(ratio, 1e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.canonical, allow_pickle=False) as archive:
        canonical = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.physics, allow_pickle=False) as archive:
        physics = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.gate_a, allow_pickle=False) as archive:
        gate = {name: np.asarray(archive[name]) for name in archive.files}
    old_direction = np.concatenate((physics["direction_train"], physics["direction_validation"]))
    old_g = np.concatenate((physics["g_ephi_train"], physics["g_ephi_validation"]))
    selected, offset_alignment = nearest(canonical["direction"], old_direction)
    offset_nmse = nmse_db(old_g, canonical["g_ephi"][selected])
    integer = np.flatnonzero(canonical["source_grid"] == "integer")
    gate_selected, integer_alignment = nearest(gate["direction_hfss"], canonical["direction"][integer])
    integer_nmse = nmse_db(gate["g_single_port_ephi"][gate_selected], canonical["g_ephi"][integer])
    synthesis = canonical["g_ephi"][integer] @ gate["commands"].T
    direct_nmse = nmse_db(gate["ephi_complex"][gate_selected], synthesis)
    report = {
        "passed": offset_nmse < -100 and integer_nmse < -100 and direct_nmse < -100,
        "offset_train_validation_queries": len(old_direction),
        "offset_ephi_complex_nmse_db": offset_nmse,
        "offset_direction_alignment_max": offset_alignment,
        "integer_front_queries": len(integer),
        "integer_ephi_complex_nmse_db": integer_nmse,
        "integer_direction_alignment_max": integer_alignment,
        "gate_a_front_complex_nmse_db": direct_nmse,
        "threshold_db": -100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
