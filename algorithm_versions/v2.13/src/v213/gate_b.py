"""Gate-B measurement contract, repeat audit, and frozen V2.12 inversion wrapper.

The module deliberately keeps amplitude-only evidence separate from privileged
HFSS complex truth.  It imports the frozen V2.12 sensing and inversion code but
does not modify, train, or evaluate any CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v212.audits import rank_one_q_matrix
from v212.codebook import MeasurementCodebook
from v212.inversion import (
    _fit_factor_batch,
    _graph_synchronization_initialization_batch,
    _jacobian_diagnostics_batch,
    _powers_from_factor_batch,
)


CANONICAL_SCHEMA = "v213-gate-b-power-repeats-1"
IMPORT_SCHEMA = "v213-gate-b-import-manifest-1"
SUPPORTED_SOURCE_KINDS = {
    "hardware_measurement",
    "direct_hfss_repeated_power",
    "synthetic",
}
REQUIRED_IMPORT_COLUMNS = (
    "direction_id",
    "theta_deg",
    "phi_deg",
    "qx",
    "qy",
    "qz",
    "command_id",
    "repeat_id",
    "power",
    "acquired_at_utc",
    "session_id",
)


@dataclass(frozen=True)
class GateBMeasurementPackage:
    manifest: dict[str, Any]
    direction: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    command_indices: np.ndarray
    command_iq: np.ndarray
    power_repeats: np.ndarray
    repeat_id: np.ndarray
    acquired_at_utc: np.ndarray
    session_id: np.ndarray


@dataclass(frozen=True)
class GateBRecovery:
    q_hat: torch.Tensor
    predicted_validation_power: torch.Tensor
    observed_validation_power: torch.Tensor
    operator_error_numerator: torch.Tensor | None
    operator_error_denominator: torch.Tensor | None
    clean_power_error_numerator: torch.Tensor | None
    clean_power_error_denominator: torch.Tensor | None
    observed_power_error_numerator: torch.Tensor
    observed_power_error_denominator: torch.Tensor
    jacobian_condition_unweighted: torch.Tensor
    jacobian_condition_noise_weighted: torch.Tensor
    minimum_jacobian_singular_value_noise_weighted: torch.Tensor


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parse_utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _complex_commands(command_iq: np.ndarray) -> np.ndarray:
    command_iq = np.asarray(command_iq)
    if command_iq.ndim != 4 or command_iq.shape[1:] != (2, 4, 4):
        raise ValueError("command_iq must have shape [C,2,4,4]")
    count = len(command_iq)
    return command_iq[:, 0].reshape(count, 16) + 1j * command_iq[:, 1].reshape(count, 16)


def load_gate_b_package(directory: str | Path) -> GateBMeasurementPackage:
    root = Path(directory)
    manifest = _read_json(root / "manifest.json")
    with np.load(root / "measurements.npz", allow_pickle=False) as archive:
        required = {
            "direction",
            "theta_deg",
            "phi_deg",
            "command_indices",
            "command_iq",
            "power_repeats",
            "repeat_id",
            "acquired_at_utc",
            "session_id",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"measurements.npz is missing arrays: {missing}")
        values = {name: np.asarray(archive[name]).copy() for name in required}
    return GateBMeasurementPackage(manifest=manifest, **values)


def validate_gate_b_package(
    package: GateBMeasurementPackage,
    codebook: MeasurementCodebook,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = package.manifest
    source_kind = str(manifest.get("source_kind", ""))

    if manifest.get("schema") != CANONICAL_SCHEMA:
        errors.append(f"manifest.schema must be {CANONICAL_SCHEMA}")
    if source_kind not in SUPPORTED_SOURCE_KINDS:
        errors.append(f"source_kind must be one of {sorted(SUPPORTED_SOURCE_KINDS)}")
    for key in (
        "dataset_id",
        "frequency_hz",
        "receiver_channel",
        "power_unit",
        "array_id",
        "instrument",
        "calibration_reference_plane",
        "calibration_id",
        "source_files",
        "independent_repeats",
        "generated_from_single_complex_g",
    ):
        if key not in manifest or manifest[key] in (None, "", []):
            errors.append(f"manifest.{key} is required")
    if manifest.get("external_test_used") is not False:
        errors.append("external_test_used must be false")
    if manifest.get("not_for_training") is not True:
        errors.append("not_for_training must be true during Gate-B validation")
    frequency_hz = manifest.get("frequency_hz")
    if isinstance(frequency_hz, (int, float)) and not np.isclose(float(frequency_hz), 47e9):
        errors.append("frequency_hz must match the frozen 47 GHz V2.12 contract")
    if manifest.get("power_unit") not in {"W", "mW", "uW", "normalized_linear_power"}:
        errors.append("power_unit must name a supported linear power unit, never dB/dBm")

    direction = np.asarray(package.direction, dtype=np.float64)
    theta = np.asarray(package.theta_deg, dtype=np.float64)
    phi = np.asarray(package.phi_deg, dtype=np.float64)
    repeats = np.asarray(package.power_repeats, dtype=np.float64)
    command_indices = np.asarray(package.command_indices, dtype=np.int64)
    repeat_id = np.asarray(package.repeat_id).astype(str)
    acquired = np.asarray(package.acquired_at_utc).astype(str)
    sessions = np.asarray(package.session_id).astype(str)

    query_count = direction.shape[0] if direction.ndim == 2 else 0
    repeat_count = repeats.shape[2] if repeats.ndim == 3 else 0
    if direction.ndim != 2 or direction.shape[1] != 3 or query_count == 0:
        errors.append("direction must have non-empty shape [Q,3]")
    if theta.shape != (query_count,) or phi.shape != (query_count,):
        errors.append("theta_deg and phi_deg must have shape [Q]")
    if repeats.shape[:2] != (query_count, 80):
        errors.append("power_repeats must have shape [Q,80,R] for train+validation commands")
    if repeat_count < 3:
        errors.append("at least three independently acquired repeats are required")
    if repeat_id.shape != (repeat_count,):
        errors.append("repeat_id must have shape [R]")
    if acquired.shape != (repeat_count,):
        errors.append("acquired_at_utc must have shape [R]")
    if sessions.shape != (repeat_count,):
        errors.append("session_id must have shape [R]")

    if direction.size:
        if not np.isfinite(direction).all() or not np.isfinite(theta).all() or not np.isfinite(phi).all():
            errors.append("direction and angle arrays must be finite")
        norms = np.linalg.norm(direction, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            errors.append("direction vectors must be unit norm")
        if np.any(direction[:, 0] <= 0.0):
            errors.append("all directions must satisfy the frozen strict q_x > 0 hemisphere")
        expected_theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1.0, 1.0)))
        expected_phi = np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0]))
        phi_error = np.abs((phi - expected_phi + 180.0) % 360.0 - 180.0)
        if np.max(np.abs(theta - expected_theta), initial=0.0) > 1e-4 or np.max(phi_error, initial=0.0) > 1e-4:
            errors.append("theta_deg/phi_deg are inconsistent with direction in the HFSS Cartesian convention")
        rounded = np.round(direction, decimals=10)
        if len(np.unique(rounded, axis=0)) != query_count:
            errors.append("direction rows must be unique")

    if repeats.size:
        if not np.isfinite(repeats).all():
            errors.append("power_repeats must be finite")
        if np.any(repeats < 0.0):
            errors.append("power_repeats must be non-negative")

    try:
        commands = _complex_commands(package.command_iq)
        expected_indices = torch.cat((codebook.indices("train"), codebook.indices("validation"))).numpy()
        if command_indices.shape != (80,) or not np.array_equal(command_indices, expected_indices):
            errors.append("command_indices must be exactly the frozen train+validation indices 0..79")
        frozen = codebook.weights.detach().cpu().numpy()[expected_indices]
        if not np.allclose(commands, frozen, atol=1e-6, rtol=1e-6):
            errors.append("command_iq does not exactly implement the frozen V2.12 train+validation order")
    except ValueError as exc:
        errors.append(str(exc))

    if repeat_id.shape == (repeat_count,):
        if np.any(repeat_id == "") or len(set(repeat_id.tolist())) != repeat_count:
            errors.append("repeat_id values must be non-empty and unique")
    if acquired.shape == (repeat_count,):
        if len(set(acquired.tolist())) != repeat_count:
            errors.append("acquired_at_utc values must be unique across repeat sweeps")
        if not all(_parse_utc(value) for value in acquired):
            errors.append("acquired_at_utc values must be timezone-aware ISO-8601 timestamps")
    if sessions.shape == (repeat_count,) and np.any(sessions == ""):
        errors.append("session_id values must be non-empty")

    exact_equal_pairs = 0
    repeat_pairs = 0
    if repeats.ndim == 3 and repeat_count >= 2:
        for left in range(repeat_count):
            for right in range(left + 1, repeat_count):
                repeat_pairs += 1
                exact_equal_pairs += int(np.array_equal(repeats[..., left], repeats[..., right]))
        variable_cells = np.ptp(repeats, axis=-1) > 0.0
        variable_cell_fraction = float(np.mean(variable_cells))
        mean_power = repeats.mean(axis=-1)
        repeat_standard_deviation = repeats.std(axis=-1, ddof=1)
        rms_signal = float(np.sqrt(np.mean(mean_power**2)))
        rms_repeat_noise = float(np.sqrt(np.mean(repeat_standard_deviation**2)))
        repeat_snr_db: float | None = (
            float(20.0 * np.log10(rms_signal / rms_repeat_noise))
            if rms_repeat_noise > 0.0
            else None
        )
    else:
        variable_cell_fraction = 0.0
        rms_signal = 0.0
        rms_repeat_noise = 0.0
        repeat_snr_db = None

    no_identical_sweeps = repeat_pairs > 0 and exact_equal_pairs == 0
    if repeat_pairs and not no_identical_sweeps:
        warnings.append("one or more complete repeat sweeps are bit-identical")
    if repeat_pairs and variable_cell_fraction == 0.0:
        warnings.append("all power repeats are identical; noise covariance is exactly zero")
    if 0.0 < variable_cell_fraction < 0.01:
        warnings.append("fewer than one percent of direction-command cells vary across repeats")

    independent_claim = manifest.get("independent_repeats") is True
    not_single_g = manifest.get("generated_from_single_complex_g") is False
    numeric_repeat_evidence = no_identical_sweeps and variable_cell_fraction > 0.0
    provenance_eligible = independent_claim and not_single_g
    structurally_valid = not errors
    hardware_eligible = (
        structurally_valid
        and source_kind == "hardware_measurement"
        and provenance_eligible
        and numeric_repeat_evidence
    )
    hfss_repeat_eligible = (
        structurally_valid
        and source_kind == "direct_hfss_repeated_power"
        and provenance_eligible
        and numeric_repeat_evidence
    )
    if not structurally_valid:
        status = "invalid"
    elif hardware_eligible:
        status = "eligible_hardware_repeats"
    elif hfss_repeat_eligible:
        status = "eligible_hfss_repeats_simulation_only"
    else:
        status = "blocked_not_independent_nonidentical_repeats"

    return {
        "schema": "v213-gate-b-package-validation-1",
        "status": status,
        "valid": structurally_valid,
        "hardware_gate_b_eligible": hardware_eligible,
        "hfss_repeat_audit_eligible": hfss_repeat_eligible,
        "external_test_used": False,
        "query_count": query_count,
        "command_count": 80,
        "repeat_count": repeat_count,
        "source_kind": source_kind,
        "errors": errors,
        "warnings": warnings,
        "repeat_evidence": {
            "repeat_pair_count": repeat_pairs,
            "bit_identical_repeat_pairs": exact_equal_pairs,
            "all_repeat_sweeps_nonidentical": no_identical_sweeps,
            "variable_direction_command_fraction": variable_cell_fraction,
            "rms_signal_power": rms_signal,
            "rms_repeat_standard_deviation": rms_repeat_noise,
            "empirical_repeat_snr_db": repeat_snr_db,
        },
        "claim_boundary": (
            "HFSS repeats remain simulation-only; only eligible hardware repeats can support "
            "hardware Gate-B noise claims."
        ),
    }


def pooled_relative_noise_covariance(
    power_repeats: np.ndarray,
    command_indices: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray | float | int]:
    repeats = np.asarray(power_repeats, dtype=np.float64)
    command_indices = np.asarray(command_indices, dtype=np.int64)
    if repeats.ndim != 3 or repeats.shape[2] < 3:
        raise ValueError("power_repeats must have shape [Q,C,R>=3]")
    if command_indices.ndim != 1 or len(command_indices) == 0:
        raise ValueError("command_indices must be a non-empty vector")
    if np.any(command_indices < 0) or np.any(command_indices >= repeats.shape[1]):
        raise ValueError("command_indices are outside the package command axis")
    selected = repeats[:, command_indices]
    centered = selected - selected.mean(axis=-1, keepdims=True)
    scale = np.sqrt(np.mean(selected.mean(axis=-1) ** 2, axis=1)).clip(min=epsilon)
    samples = np.transpose(centered / scale[:, None, None], (0, 2, 1)).reshape(-1, len(command_indices))
    covariance = samples.T @ samples / max(len(samples) - 1, 1)
    variance = np.diag(covariance).copy()
    root = np.sqrt(np.maximum(variance, epsilon))
    correlation = covariance / np.outer(root, root)
    off_diagonal = correlation[~np.eye(len(command_indices), dtype=bool)]
    return {
        "pooled_relative_covariance": covariance,
        "per_direction_diagonal_variance": centered.var(axis=-1, ddof=1),
        "pooled_sample_count": int(len(samples)),
        "median_relative_variance": float(np.median(variance)),
        "p95_absolute_offdiagonal_correlation": float(np.quantile(np.abs(off_diagonal), 0.95)),
    }


def import_long_csv(
    csv_path: str | Path,
    import_manifest_path: str | Path,
    output_directory: str | Path,
    codebook: MeasurementCodebook,
) -> GateBMeasurementPackage:
    """Import the documented rectangular long CSV into the no-pickle NPZ contract."""
    csv_path = Path(csv_path)
    input_manifest = _read_json(Path(import_manifest_path))
    if input_manifest.get("schema") != IMPORT_SCHEMA:
        raise ValueError(f"import manifest schema must be {IMPORT_SCHEMA}")

    directions: dict[int, tuple[float, float, float, float, float]] = {}
    repeats: list[str] = []
    repeat_metadata: dict[str, tuple[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("measurement CSV is missing a header")
        missing = sorted(set(REQUIRED_IMPORT_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"measurement CSV is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            try:
                direction_id = int(row["direction_id"])
                command_id = int(row["command_id"])
                direction_value = tuple(float(row[name]) for name in ("theta_deg", "phi_deg", "qx", "qy", "qz"))
                power = float(row["power"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric field at CSV line {line_number}") from exc
            if direction_id < 0 or not 0 <= command_id < 80 or not np.isfinite(power) or power < 0:
                raise ValueError(f"invalid direction, command, or power at CSV line {line_number}")
            previous = directions.setdefault(direction_id, direction_value)
            if not np.allclose(previous, direction_value, atol=1e-10, rtol=0.0):
                raise ValueError(f"direction metadata changes at CSV line {line_number}")
            repeat_id = row["repeat_id"].strip()
            metadata = (row["acquired_at_utc"].strip(), row["session_id"].strip())
            if repeat_id not in repeat_metadata:
                repeats.append(repeat_id)
                repeat_metadata[repeat_id] = metadata
            elif repeat_metadata[repeat_id] != metadata:
                raise ValueError(f"repeat metadata changes at CSV line {line_number}")

    if sorted(directions) != list(range(len(directions))):
        raise ValueError("direction_id values must be contiguous from zero")
    if len(repeats) < 3:
        raise ValueError("the import requires at least three repeat sweeps")
    repeat_lookup = {value: index for index, value in enumerate(repeats)}
    power_repeats = np.empty((len(directions), 80, len(repeats)), dtype=np.float64)
    occupied = np.zeros(power_repeats.shape, dtype=bool)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            index = (int(row["direction_id"]), int(row["command_id"]), repeat_lookup[row["repeat_id"].strip()])
            if occupied[index]:
                raise ValueError(f"duplicate direction-command-repeat row at CSV line {line_number}")
            power_repeats[index] = float(row["power"])
            occupied[index] = True
    if not occupied.all():
        missing_count = int(np.size(occupied) - np.count_nonzero(occupied))
        raise ValueError(f"measurement CSV is not rectangular; {missing_count} cells are missing")

    metadata = np.asarray([directions[index] for index in range(len(directions))], dtype=np.float64)
    command_indices = torch.cat((codebook.indices("train"), codebook.indices("validation"))).numpy()
    commands = codebook.weights.detach().cpu().numpy()[command_indices]
    command_iq = np.stack((commands.real, commands.imag), axis=1).reshape(80, 2, 4, 4)
    manifest = dict(input_manifest)
    manifest.update(
        {
            "schema": CANONICAL_SCHEMA,
            "import_schema": IMPORT_SCHEMA,
            "axes": {
                "power_repeats": ["direction", "command", "repeat"],
                "command_iq": ["command", "iq", "row", "col"],
                "direction": ["direction", "xyz"],
            },
            "query_count": len(directions),
            "command_count": 80,
            "repeat_count": len(repeats),
            "command_protocol": "v212-measurement-codebook-1",
            "source_files": [str(csv_path.resolve()), *input_manifest.get("source_files", [])],
        }
    )
    package = GateBMeasurementPackage(
        manifest=manifest,
        direction=metadata[:, 2:5],
        theta_deg=metadata[:, 0],
        phi_deg=metadata[:, 1],
        command_indices=command_indices,
        command_iq=command_iq.astype(np.float32),
        power_repeats=power_repeats,
        repeat_id=np.asarray(repeats, dtype="U128"),
        acquired_at_utc=np.asarray([repeat_metadata[value][0] for value in repeats], dtype="U64"),
        session_id=np.asarray([repeat_metadata[value][1] for value in repeats], dtype="U128"),
    )
    report = validate_gate_b_package(package, codebook)
    if not report["valid"]:
        raise ValueError("imported package failed validation: " + "; ".join(report["errors"]))

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "measurements.npz",
        direction=package.direction,
        theta_deg=package.theta_deg,
        phi_deg=package.phi_deg,
        command_indices=package.command_indices,
        command_iq=package.command_iq,
        power_repeats=package.power_repeats,
        repeat_id=package.repeat_id,
        acquired_at_utc=package.acquired_at_utc,
        session_id=package.session_id,
    )
    (output / "VALIDATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return package


def synthetic_power_repeats(
    clean_power: torch.Tensor,
    *,
    noise_model: str,
    snr_db: float,
    repeats: int,
    seed: int,
) -> torch.Tensor:
    """Apply an explicit power-domain noise model without touching complex g."""
    if clean_power.ndim != 2 or torch.any(clean_power < 0):
        raise ValueError("clean_power must be non-negative with shape [Q,C]")
    if repeats < 3:
        raise ValueError("at least three repeats are required")
    if noise_model not in {"additive_gaussian_power", "multiplicative_lognormal_power", "mixed_equal_variance_power"}:
        raise ValueError("unsupported power noise model")
    generator = torch.Generator(device=clean_power.device).manual_seed(seed)
    ratio = float(10.0 ** (-snr_db / 20.0))
    clean = clean_power[:, :, None]
    shape = (*clean_power.shape, repeats)
    if noise_model == "additive_gaussian_power":
        sigma = clean_power.square().mean(dim=1, keepdim=True).sqrt()[:, :, None] * ratio
        noisy = clean + sigma * torch.randn(shape, generator=generator, device=clean.device, dtype=clean.real.dtype)
    elif noise_model == "multiplicative_lognormal_power":
        sigma_log = float(np.sqrt(np.log1p(ratio**2)))
        normal = torch.randn(shape, generator=generator, device=clean.device, dtype=clean.real.dtype)
        factor = torch.exp(sigma_log * normal - 0.5 * sigma_log**2)
        noisy = clean * factor
    else:
        component_ratio = ratio / np.sqrt(2.0)
        sigma = clean_power.square().mean(dim=1, keepdim=True).sqrt()[:, :, None] * component_ratio
        additive = sigma * torch.randn(shape, generator=generator, device=clean.device, dtype=clean.real.dtype)
        sigma_log = float(np.sqrt(np.log1p(component_ratio**2)))
        normal = torch.randn(shape, generator=generator, device=clean.device, dtype=clean.real.dtype)
        factor = torch.exp(sigma_log * normal - 0.5 * sigma_log**2)
        noisy = clean * factor + additive
    return noisy.clamp_min(0.0)


def recover_gate_b_rank1(
    power_repeats: torch.Tensor,
    codebook: MeasurementCodebook,
    *,
    truth_g: torch.Tensor | None = None,
    clean_validation_power: torch.Tensor | None = None,
    steps: int = 120,
    batch_size: int = 256,
    seed: int = 21321,
) -> GateBRecovery:
    """Recover Q from train commands and evaluate only held-out validation commands."""
    if power_repeats.ndim != 3 or power_repeats.shape[1] != 80 or power_repeats.shape[2] < 3:
        raise ValueError("power_repeats must have shape [Q,80,R>=3] for train+validation only")
    if torch.any(power_repeats < 0) or batch_size <= 0 or steps <= 0:
        raise ValueError("power repeats, batch size, and steps are invalid")
    if truth_g is not None and (not torch.is_complex(truth_g) or truth_g.shape != (len(power_repeats), 16)):
        raise ValueError("truth_g must have complex shape [Q,16]")

    device = power_repeats.device
    weights = codebook.weights.to(device)
    train_index = codebook.indices("train").to(device)
    validation_index = codebook.indices("validation").to(device)
    active_index = torch.cat((train_index, validation_index))
    if not torch.equal(active_index, torch.arange(80, device=device)):
        raise ValueError("frozen V2.12 train+validation commands must occupy indices 0..79")
    train_commands = weights[train_index]
    validation_commands = weights[validation_index]
    mean_power = power_repeats.mean(dim=-1)
    observed_validation = mean_power[:, 64:80]

    q_hats: list[torch.Tensor] = []
    predicted_validation: list[torch.Tensor] = []
    unweighted_condition: list[torch.Tensor] = []
    weighted_condition: list[torch.Tensor] = []
    weighted_minimum: list[torch.Tensor] = []
    for start in range(0, len(power_repeats), batch_size):
        stop = min(start + batch_size, len(power_repeats))
        batch_mean = mean_power[start:stop]
        graph_power = torch.zeros((stop - start, 96), dtype=batch_mean.dtype, device=device)
        graph_power[:, active_index] = batch_mean
        initial = _graph_synchronization_initialization_batch(graph_power, codebook, split="train")
        factor = _fit_factor_batch(
            train_commands,
            batch_mean[:, :64],
            rank=1,
            steps=steps,
            learning_rate=3e-2,
            seed=seed + start,
            initial_g=initial,
        )[:, 0]
        q_hats.append(rank_one_q_matrix(factor).detach().cpu())
        predicted_validation.append(
            _powers_from_factor_batch(factor[:, None], validation_commands).detach().cpu()
        )
        ones = torch.ones((stop - start, 64), dtype=batch_mean.dtype, device=device)
        _, condition_plain, _ = _jacobian_diagnostics_batch(factor, train_commands, ones)
        noise = power_repeats[start:stop, :64].std(dim=-1, unbiased=True)
        median_power = batch_mean[:, :64].median(dim=1).values.clamp_min(1e-12)
        positive_median = []
        for row in noise:
            positive = row[row > 0]
            positive_median.append(positive.median() if positive.numel() else row.new_zeros(()))
        floor = torch.maximum(median_power * 1e-6, torch.stack(positive_median) * 0.05)
        noise = torch.maximum(noise, floor[:, None])
        minimum, condition_weighted, _ = _jacobian_diagnostics_batch(factor, train_commands, noise)
        unweighted_condition.append(condition_plain.detach().cpu())
        weighted_condition.append(condition_weighted.detach().cpu())
        weighted_minimum.append(minimum.detach().cpu())

    q_hat = torch.cat(q_hats)
    predicted = torch.cat(predicted_validation)
    observed = observed_validation.detach().cpu()
    observed_num = (predicted - observed).square().sum(dim=1)
    observed_den = observed.square().sum(dim=1).clamp_min(1e-30)

    operator_num = operator_den = None
    if truth_g is not None:
        truth_q = rank_one_q_matrix(truth_g.detach().cpu())
        operator_num = (q_hat - truth_q).abs().square().sum(dim=(-2, -1))
        operator_den = truth_q.abs().square().sum(dim=(-2, -1)).clamp_min(1e-30)

    clean_num = clean_den = None
    if clean_validation_power is not None:
        clean = clean_validation_power.detach().cpu()
        if clean.shape != predicted.shape:
            raise ValueError("clean_validation_power must have shape [Q,16]")
        clean_num = (predicted - clean).square().sum(dim=1)
        clean_den = clean.square().sum(dim=1).clamp_min(1e-30)

    return GateBRecovery(
        q_hat=q_hat,
        predicted_validation_power=predicted,
        observed_validation_power=observed,
        operator_error_numerator=operator_num,
        operator_error_denominator=operator_den,
        clean_power_error_numerator=clean_num,
        clean_power_error_denominator=clean_den,
        observed_power_error_numerator=observed_num,
        observed_power_error_denominator=observed_den,
        jacobian_condition_unweighted=torch.cat(unweighted_condition),
        jacobian_condition_noise_weighted=torch.cat(weighted_condition),
        minimum_jacobian_singular_value_noise_weighted=torch.cat(weighted_minimum),
    )
