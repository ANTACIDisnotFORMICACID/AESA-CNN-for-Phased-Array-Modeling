"""Convert raw HFSS FFD into deployment-equivalent amplitude-only observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import ArrayGeometry
from ..io import (
    AmplitudeManifest,
    BasisDataset,
    DatasetManifest,
    RECEIVER_CHANNELS,
    canonical_spherical_query_mask,
    generate_command_codebook,
    load_hfss_ffd_basis,
    save_amplitude_dataset,
    synthesize_amplitude_observations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receiver-channel", required=True, choices=RECEIVER_CHANNELS)
    parser.add_argument("--reference-distance-m", required=True, type=float)
    parser.add_argument("--amplitude-unit", required=True)
    parser.add_argument("--excitation-reference", required=True)
    parser.add_argument("--inactive-port-termination", required=True)
    parser.add_argument("--calibration-reference-plane", required=True)
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--random-commands", type=int, default=96)
    parser.add_argument("--seed", type=int, default=28001)
    return parser


def _load_mapping(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not all(
        isinstance(port, str) and isinstance(value, str) for port, value in raw.items()
    ):
        raise ValueError("mapping must be a JSON object from port names to FFD paths")
    base = path.resolve().parent
    resolved = {}
    for port, value in raw.items():
        source = Path(value)
        resolved[port] = str(source if source.is_absolute() else (base / source).resolve())
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    geometry = ArrayGeometry()
    export = load_hfss_ffd_basis(_load_mapping(args.mapping), geometry)
    keep = canonical_spherical_query_mask(export.theta_deg, export.phi_deg)
    raw_basis = BasisDataset(
        direction=export.direction_algorithm[keep],
        operator_iq=export.operator_iq[keep],
        theta_deg=export.theta_deg[keep],
        phi_deg=export.phi_deg[keep],
        manifest=DatasetManifest.for_geometry(
            geometry,
            "hfss_ffd",
            provenance={"source_files": list(export.source_files)},
        ),
    )
    commands, families = generate_command_codebook(
        random_commands=args.random_commands,
        seed=args.seed,
    )
    manifest = AmplitudeManifest(
        schema_version="v2.8-amplitude-1",
        source_type="hfss_amplitude",
        frequency_hz=geometry.frequency_hz,
        receiver_channel=args.receiver_channel,
        amplitude_unit=args.amplitude_unit,
        reference_distance_m=args.reference_distance_m,
        excitation_reference=args.excitation_reference,
        inactive_port_termination=args.inactive_port_termination,
        calibration_reference_plane=args.calibration_reference_plane,
        array_id=args.array_id,
        session_id=args.session_id,
        provenance={
            "generator": "HFSS FFD projected to scalar receiver amplitude",
            "mapping_file": str(args.mapping.resolve()),
            "command_seed": args.seed,
            "raw_query_count": int(keep.size),
            "canonical_query_count": int(keep.sum()),
            "query_canonicalization": "drop theta=0/180 and duplicate phi=+180 seam",
            "raw_ffd_quantity": "r_times_E in volt",
            "amplitude_conversion": "E at reference_distance_m = abs(rE) / reference_distance_m",
            "measured_phase_exported_to_training_dataset": False,
            "s_parameters_exported_to_training_dataset": False,
        },
    )
    dataset = synthesize_amplitude_observations(raw_basis, commands, families, manifest)
    save_amplitude_dataset(dataset, args.output)
    print(
        f"saved amplitude-only data: {commands.shape[0]} commands x "
        f"{int(keep.sum())} canonical directions to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
