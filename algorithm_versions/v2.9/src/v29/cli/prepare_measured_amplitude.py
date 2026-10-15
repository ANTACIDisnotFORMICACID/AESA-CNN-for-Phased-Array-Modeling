"""Validate instrument-converted arrays and create the same amplitude-only dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import AmplitudeDataset, AmplitudeManifest, RECEIVER_CHANNELS, save_amplitude_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, type=Path)
    parser.add_argument("--theta-deg", required=True, type=Path)
    parser.add_argument("--phi-deg", required=True, type=Path)
    parser.add_argument("--command-iq", required=True, type=Path)
    parser.add_argument("--command-family", required=True, type=Path)
    parser.add_argument("--amplitude", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency-hz", required=True, type=float)
    parser.add_argument("--receiver-channel", required=True, choices=RECEIVER_CHANNELS)
    parser.add_argument("--reference-distance-m", required=True, type=float)
    parser.add_argument("--amplitude-unit", required=True)
    parser.add_argument("--noise-floor-amplitude", required=True, type=float)
    parser.add_argument("--excitation-reference", required=True)
    parser.add_argument("--inactive-port-termination", required=True)
    parser.add_argument("--calibration-reference-plane", required=True)
    parser.add_argument("--array-id", required=True)
    parser.add_argument("--session-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = AmplitudeManifest(
        schema_version="v2.8-amplitude-1",
        source_type="anechoic_chamber_amplitude",
        frequency_hz=args.frequency_hz,
        receiver_channel=args.receiver_channel,
        amplitude_unit=args.amplitude_unit,
        reference_distance_m=args.reference_distance_m,
        excitation_reference=args.excitation_reference,
        inactive_port_termination=args.inactive_port_termination,
        calibration_reference_plane=args.calibration_reference_plane,
        array_id=args.array_id,
        session_id=args.session_id,
        noise_floor_amplitude=args.noise_floor_amplitude,
        provenance={"measured_phase_available": False},
    )
    dataset = AmplitudeDataset(
        direction=np.load(args.direction, allow_pickle=False),
        command_iq=np.load(args.command_iq, allow_pickle=False),
        amplitude=np.load(args.amplitude, allow_pickle=False),
        command_family=np.load(args.command_family, allow_pickle=False),
        theta_deg=np.load(args.theta_deg, allow_pickle=False),
        phi_deg=np.load(args.phi_deg, allow_pickle=False),
        manifest=manifest,
    )
    save_amplitude_dataset(dataset, args.output)
    print(f"saved validated chamber amplitude data to {args.output.resolve()}")


if __name__ == "__main__":
    main()

