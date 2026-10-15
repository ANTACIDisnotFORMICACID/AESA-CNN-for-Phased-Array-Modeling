"""Build a positive-hemisphere, single-polarization amplitude-only data table.

The source campaign NPZ files may contain coherent HFSS fields, but those files
are used only by this one-way conversion step.  The destination follows the
same contract as chamber magnitude measurements and deliberately contains no
complex field operator, measured phase, or S-parameter arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from v210.coordinates import spherical_basis  # noqa: E402
from v210.io import (  # noqa: E402
    AmplitudeDataset,
    AmplitudeManifest,
    generate_command_codebook,
    save_amplitude_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receiver-channel", choices=("etheta", "ephi"), default="ephi")
    parser.add_argument("--random-commands", type=int, default=192)
    parser.add_argument("--seed", type=int, default=28101)
    parser.add_argument("--reference-distance-m", type=float, default=1.0)
    parser.add_argument("--array-id", default="hfss-47ghz-4x4")
    parser.add_argument("--session-id", default="strict-single-pol-amplitude-20260908")
    return parser


def _load_front_sources(paths: list[Path]) -> tuple[np.ndarray, ...]:
    directions: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    phis: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            required = {"q", "h", "theta", "phi"}
            missing = required.difference(source.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            q = np.asarray(source["q"], dtype=np.float64)
            h = np.asarray(source["h"])
            theta = np.asarray(source["theta"], dtype=np.float64)
            phi = np.asarray(source["phi"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or h.shape != (q.shape[0], 3, 16):
            raise ValueError(f"unexpected q/h shape in {path}: {q.shape}, {h.shape}")
        if theta.shape != (q.shape[0],) or phi.shape != (q.shape[0],):
            raise ValueError(f"unexpected angle shape in {path}")
        front = q[:, 0] > 0.0
        directions.append(q[front])
        fields.append(h[front])
        thetas.append(theta[front])
        phis.append(phi[front])

    q_all = np.concatenate(directions, axis=0)
    h_all = np.concatenate(fields, axis=0)
    theta_all = np.concatenate(thetas)
    phi_all = np.concatenate(phis)
    rounded = np.round(q_all, decimals=10)
    if np.unique(rounded, axis=0).shape[0] != q_all.shape[0]:
        raise ValueError("source files contain duplicate positive-hemisphere directions")
    if not np.all(q_all[:, 0] > 0.0):
        raise AssertionError("non-front direction escaped q_x > 0 filter")
    return q_all, h_all, theta_all, phi_all


def _scalar_amplitude(
    h: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    command_iq: np.ndarray,
    receiver_channel: str,
    reference_distance_m: float,
) -> np.ndarray:
    e_theta, e_phi = spherical_basis(theta_deg, phi_deg)
    receiver = e_theta if receiver_channel == "etheta" else e_phi
    scalar_operator = np.einsum("qx,qxn->qn", receiver, h, optimize=True)
    commands = command_iq[:, 0] + 1j * command_iq[:, 1]
    scalar_field = np.einsum(
        "qn,cn->cq", scalar_operator, commands.reshape(commands.shape[0], 16), optimize=True
    )
    return (np.abs(scalar_field) / reference_distance_m).astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    if args.reference_distance_m <= 0:
        raise ValueError("reference distance must be positive")
    q, h, theta, phi = _load_front_sources(args.source)
    commands, families = generate_command_codebook(
        random_commands=args.random_commands,
        seed=args.seed,
    )
    amplitude = _scalar_amplitude(
        h,
        theta,
        phi,
        commands,
        args.receiver_channel,
        args.reference_distance_m,
    )
    manifest = AmplitudeManifest(
        schema_version="v2.8-amplitude-1",
        source_type="hfss_amplitude",
        frequency_hz=47.0e9,
        receiver_channel=args.receiver_channel,
        amplitude_unit="V/m",
        reference_distance_m=args.reference_distance_m,
        excitation_reference="known unit-power 4x4 complex command at HFSS port planes",
        inactive_port_termination="50 ohm matched",
        calibration_reference_plane="HFSS lumped-port reference planes",
        array_id=args.array_id,
        session_id=args.session_id,
        phase_available=False,
        provenance={
            "conversion": "coherent HFSS field projected once, then abs(.) retained",
            "source_files": [str(path.resolve()) for path in args.source],
            "hemisphere": "strict q_x > 0 (+X boresight)",
            "receiver_channel": args.receiver_channel,
            "raw_complex_data_available_to_training": False,
            "measured_phase_exported_to_training_dataset": False,
            "s_parameters_exported_to_training_dataset": False,
            "command_seed": args.seed,
            "random_command_count": args.random_commands,
        },
    )
    dataset = AmplitudeDataset(
        direction=q,
        command_iq=commands,
        amplitude=amplitude,
        command_family=families,
        theta_deg=theta,
        phi_deg=phi,
        manifest=manifest,
    )
    save_amplitude_dataset(dataset, args.output)
    written = sorted(path.name for path in args.output.iterdir() if path.is_file())
    forbidden = {"operator_iq.npy", "s_parameters_iq.npy"}.intersection(written)
    if forbidden:
        raise AssertionError(f"phase-bearing files were exported: {sorted(forbidden)}")
    print(
        f"saved {commands.shape[0]} commands x {q.shape[0]} q_x>0 directions; "
        f"receiver={args.receiver_channel}; files={written}"
    )


if __name__ == "__main__":
    main()
