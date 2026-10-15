"""Convert 16 explicitly mapped HFSS FFD files into the canonical dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import ArrayGeometry
from ..io import (
    BasisDataset,
    DatasetManifest,
    load_hfss_ffd_basis,
    load_touchstone,
    save_basis_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path, help="JSON object mapping HFSS port name to FFD path")
    parser.add_argument("--output", required=True, type=Path, help="canonical dataset directory")
    parser.add_argument("--touchstone", type=Path, help="optional solved 16-port S16P file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
    ):
        raise ValueError("mapping JSON must be an object from port names to file paths")
    mapping_base = args.mapping.resolve().parent
    resolved_mapping = {}
    for port, raw_path in mapping.items():
        path = Path(raw_path)
        resolved_mapping[port] = str(path if path.is_absolute() else (mapping_base / path).resolve())
    mapping = resolved_mapping
    geometry = ArrayGeometry()
    export = load_hfss_ffd_basis(mapping, geometry)
    network = load_touchstone(args.touchstone) if args.touchstone is not None else None
    provenance = {"mapping_file": str(args.mapping.resolve()), "source_files": list(export.source_files)}
    if args.touchstone is not None:
        provenance["touchstone_file"] = str(args.touchstone.resolve())
    manifest = DatasetManifest.for_geometry(
        geometry,
        source_type="hfss_ffd",
        provenance=provenance,
    )
    dataset = BasisDataset(
        direction=export.direction_algorithm,
        operator_iq=export.operator_iq,
        theta_deg=export.theta_deg,
        phi_deg=export.phi_deg,
        manifest=manifest,
        s_frequency_hz=network.frequency_hz if network is not None else None,
        s_parameters_iq=network.s_parameters_iq if network is not None else None,
    )
    save_basis_dataset(dataset, args.output)
    print(f"saved {dataset.direction.shape[0]} directions and 16 source responses to {args.output.resolve()}")


if __name__ == "__main__":
    main()
