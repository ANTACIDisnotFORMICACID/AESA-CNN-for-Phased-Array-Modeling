"""Package train/validation-only HFSS complex fields for V2.14 attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V210 = ROOT.parent / "v2.10"
V212 = ROOT.parent / "v2.12"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V212 / "src"))
sys.path.insert(0, str(V210 / "src"))

from v210.contracts import ArrayGeometry  # noqa: E402
from v210.coordinates import spherical_basis  # noqa: E402
from v210.io.hfss_ffd import load_hfss_ffd_basis  # noqa: E402
from v214.physics_metrics import transverse_diagnostic, polarization_diagnostic, weighted_nmse_db  # noqa: E402


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry(path: Path) -> ArrayGeometry:
    with path.open("rb") as stream:
        item = tomllib.load(stream)
    return ArrayGeometry(
        rows=int(item["array"]["rows"]), cols=int(item["array"]["cols"]),
        frequency_hz=float(item["array"]["frequency_hz"]),
        row_pitch_m=float(item["array"]["row_pitch_m"]),
        col_pitch_m=float(item["array"]["col_pitch_m"]),
        rotation_algorithm_to_hfss=np.asarray(item["array"]["algorithm_to_hfss_rotation"]),
        port_names=tuple(item["ports"]["names"]),
    )


def _mapping(directory: Path, prefix: str) -> dict[str, Path]:
    return {f"{port}:1": directory / f"{prefix}_{port}.ffd" for port in range(1, 17)}


def _parse_s16p(path: Path) -> tuple[float, np.ndarray]:
    option = next(line for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.lstrip().startswith("#"))
    tokens = option.lower().split()
    if "s" not in tokens or "ma" not in tokens:
        raise ValueError("V2.14 requires Touchstone S-parameter magnitude-angle format")
    unit = tokens[1]
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}[unit]
    numeric: list[float] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.split("!", 1)[0].strip()
        if line and not line.startswith("#"):
            numeric.extend(float(value) for value in line.split())
    if len(numeric) != 1 + 2 * 16 * 16:
        raise ValueError(f"expected one 16-port frequency row, found {len(numeric)} numeric values")
    frequency = numeric[0] * scale
    pair = np.asarray(numeric[1:], dtype=np.float64).reshape(-1, 2)
    value = pair[:, 0] * np.exp(1j * np.deg2rad(pair[:, 1]))
    return frequency, value.reshape((16, 16), order="F")


def _align_angles(source_theta: np.ndarray, source_phi: np.ndarray, target_theta: np.ndarray, target_phi: np.ndarray) -> np.ndarray:
    lookup = {
        (round(float(theta), 8), round(float(phi), 8)): index
        for index, (theta, phi) in enumerate(zip(source_theta, source_phi))
    }
    try:
        index = np.asarray([
            lookup[(round(float(theta), 8), round(float(phi), 8))]
            for theta, phi in zip(target_theta, target_phi)
        ], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"target angle absent from FFD grid: {exc}") from exc
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffd-dir", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--touchstone", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gate-a-npz", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometry = _geometry(args.geometry)
    half = load_hfss_ffd_basis(_mapping(args.ffd_dir, "exportelement_new"), geometry)
    integer = load_hfss_ffd_basis(_mapping(args.ffd_dir, "exportelement"), geometry)
    half_h = half.operator_iq[:, :, 0].reshape(len(half.theta_deg), 3, 16) + 1j * half.operator_iq[:, :, 1].reshape(len(half.theta_deg), 3, 16)
    integer_h = integer.operator_iq[:, :, 0].reshape(len(integer.theta_deg), 3, 16) + 1j * integer.operator_iq[:, :, 1].reshape(len(integer.theta_deg), 3, 16)
    half_et, half_ep = spherical_basis(half.theta_deg, half.phi_deg)
    int_et, int_ep = spherical_basis(integer.theta_deg, integer.phi_deg)
    half_g_phi = np.einsum("qx,qxn->qn", half_ep, half_h)
    half_g_theta = np.einsum("qx,qxn->qn", half_et, half_h)
    integer_g_phi = np.einsum("qx,qxn->qn", int_ep, integer_h)

    direction = np.load(args.dataset_root / "amplitude_dataset" / "direction.npy")
    theta = np.load(args.dataset_root / "amplitude_dataset" / "theta_deg.npy")
    phi = np.load(args.dataset_root / "amplitude_dataset" / "phi_deg.npy")
    train_index = np.load(args.dataset_root / "split" / "query_train.npy")
    validation_index = np.load(args.dataset_root / "split" / "query_validation.npy")
    audit = np.load(args.dataset_root / "audit" / "hfss_complex_operator_audit_only.npz", allow_pickle=False)
    solid = np.asarray(audit["solid_angle_weights"], dtype=np.float64)
    aligned = _align_angles(half.theta_deg, half.phi_deg, theta, phi)
    maximum_direction_error = float(np.max(np.abs(half.direction_algorithm[aligned] - direction)))
    if maximum_direction_error >= 1e-7:
        raise ValueError(f"direction alignment error {maximum_direction_error} exceeds 1e-7")
    h = half_h[aligned]
    g_phi = half_g_phi[aligned]
    g_theta = half_g_theta[aligned]
    audit_nmse_db = weighted_nmse_db(g_phi, np.asarray(audit["g"], dtype=np.complex128))
    if audit_nmse_db > -100.0:
        raise ValueError(f"Ephi audit alignment failed: {audit_nmse_db:.3f} dB")

    with np.load(args.gate_a_npz, allow_pickle=False) as gate:
        gate_direction = np.asarray(gate["direction_hfss"])
        gate_index = _align_angles(integer.theta_deg, integer.phi_deg, gate["theta_deg"], gate["phi_deg"])
        if float(np.max(np.abs(integer.direction_algorithm[gate_index] - gate_direction))) >= 1e-7:
            raise ValueError("Gate-A direction alignment exceeds 1e-7")
        gate_g = integer_g_phi[gate_index]
        commands = np.asarray(gate["commands"])
        direct = np.asarray(gate["ephi_complex"])
        synthetic = gate_g @ commands.T
        gate_nmse_db = weighted_nmse_db(synthetic, direct)
    if gate_nmse_db > -120.0:
        raise ValueError(f"Gate-A direct-report cross-check failed: {gate_nmse_db:.3f} dB")

    s_frequency, s_matrix = _parse_s16p(args.touchstone)
    if abs(s_frequency - geometry.frequency_hz) > 1.0:
        raise ValueError("Touchstone does not contain the 47 GHz solved point")
    if not np.isfinite(s_matrix).all():
        raise ValueError("S parameters are not finite")

    def selected(value: np.ndarray, index: np.ndarray) -> np.ndarray:
        return np.asarray(value[index])

    output = args.output_dir / "physics_train_validation.npz"
    np.savez_compressed(
        output,
        direction_train=selected(direction, train_index), direction_validation=selected(direction, validation_index),
        theta_deg_train=selected(theta, train_index), theta_deg_validation=selected(theta, validation_index),
        phi_deg_train=selected(phi, train_index), phi_deg_validation=selected(phi, validation_index),
        g_ephi_train=selected(g_phi, train_index), g_ephi_validation=selected(g_phi, validation_index),
        g_etheta_train=selected(g_theta, train_index), g_etheta_validation=selected(g_theta, validation_index),
        h_cartesian_train=selected(h, train_index), h_cartesian_validation=selected(h, validation_index),
        solid_angle_weights_train=selected(solid, train_index), solid_angle_weights_validation=selected(solid, validation_index),
        positions_m=geometry.positions_m.reshape(16, 3), s_parameters_47ghz=s_matrix,
        frequency_hz=np.asarray(geometry.frequency_hz),
    )
    project_hash = _hash(args.project)
    diagnostics = {
        "ephi_audit_complex_nmse_db": audit_nmse_db,
        "gate_a_direct_complex_nmse_db": gate_nmse_db,
        "transverse_train": transverse_diagnostic(direction[train_index], h[train_index]),
        "transverse_validation": transverse_diagnostic(direction[validation_index], h[validation_index]),
        "polarization_train": polarization_diagnostic(g_phi[train_index], g_theta[train_index]),
        "polarization_validation": polarization_diagnostic(g_phi[validation_index], g_theta[validation_index]),
    }
    manifest = {
        "schema": "v214-physics-source-manifest-1", "frequency_hz": geometry.frequency_hz,
        "train_count": int(len(train_index)), "validation_count": int(len(validation_index)),
        "test_values_exported": False, "external_test_used": False,
        "project": str(args.project.resolve()), "project_sha256_before": project_hash,
        "project_sha256_after": _hash(args.project), "project_unchanged_on_disk": project_hash == _hash(args.project),
        "geometry": str(args.geometry.resolve()), "touchstone": str(args.touchstone.resolve()),
        "half_grid_ffd_prefix": str((args.ffd_dir / "exportelement_new").resolve()),
        "integer_grid_ffd_prefix": str((args.ffd_dir / "exportelement").resolve()),
        "prepared_dataset": str(output.resolve()), "diagnostics": diagnostics,
    }
    (args.output_dir / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
