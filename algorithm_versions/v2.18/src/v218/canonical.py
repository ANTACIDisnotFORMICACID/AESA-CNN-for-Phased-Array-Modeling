"""Sealed canonical-to-V2.16 data adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from v210.contracts import ArrayGeometry
from v212.basis import FiltrationSSHTransform


FORBIDDEN_PATH_TOKENS = ("blind", "test_operator", "test_commands")


def assert_unsealed_path_not_used(path: str | Path) -> Path:
    target = Path(path).resolve()
    lowered = str(target).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"sealed path is forbidden in V2.18: {target.name}")
    return target


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    target = assert_unsealed_path_not_used(path)
    with np.load(target, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def validate_operator_cache(data: dict[str, np.ndarray], expected_count: int) -> None:
    required = {
        "canonical_indices", "direction", "g_ephi", "solid_angle_weights",
        "normalization_scale", "positions_m", "frequency_hz",
    }
    if missing := required - set(data):
        raise ValueError(f"operator cache is missing {sorted(missing)}")
    if data["direction"].shape != (expected_count, 3):
        raise ValueError("unexpected direction count")
    if data["g_ephi"].shape != (expected_count, 16):
        raise ValueError("complex operator must have shape [Q,16]")
    if data["solid_angle_weights"].shape != (expected_count,):
        raise ValueError("solid-angle weights must have shape [Q]")
    if len(np.unique(data["canonical_indices"])) != expected_count:
        raise ValueError("canonical indices are not unique")
    if not np.isfinite(data["direction"]).all() or not np.isfinite(data["g_ephi"]).all():
        raise ValueError("canonical cache contains non-finite values")
    np.testing.assert_allclose(np.linalg.norm(data["direction"], axis=1), 1.0, atol=2e-6)
    geometry = ArrayGeometry()
    np.testing.assert_allclose(data["positions_m"], geometry.positions_m.reshape(16, 3), atol=1e-12)
    if not np.isclose(float(data["frequency_hz"]), geometry.frequency_hz):
        raise ValueError("canonical frequency does not match V2.16")


def complex_from_iq(command_iq: np.ndarray) -> np.ndarray:
    value = np.asarray(command_iq, dtype=np.float32)
    if value.shape != (64, 2, 16):
        raise ValueError("frozen V2.16 command IQ must have shape [64,2,16]")
    result = value[:, 0] + 1j * value[:, 1]
    power = np.sum(np.abs(result) ** 2, axis=1)
    if not np.allclose(power, np.ones(64), atol=1e-5, rtol=1e-5):
        raise ValueError("training commands are not unit-power")
    return result.astype(np.complex64)


def amplitude_from_operator(operator: np.ndarray, commands: np.ndarray) -> np.ndarray:
    if operator.ndim != 2 or operator.shape[1] != 16:
        raise ValueError("operator must have shape [Q,16]")
    if commands.ndim != 2 or commands.shape[1] != 16:
        raise ValueError("commands must have shape [C,16]")
    return np.abs(operator @ commands.T).astype(np.float32)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_npz_compressed(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def prepare(
    *,
    train_cache_path: Path,
    validation_cache_path: Path,
    v216_exchange_path: Path,
    validation_commands_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the only packages accepted by V2.18 training."""
    inputs = [train_cache_path, validation_cache_path, v216_exchange_path, validation_commands_path]
    for path in inputs:
        assert_unsealed_path_not_used(path)
    fingerprints = {path.name: sha256(path) for path in inputs}
    receipt = output_dir / "PREPARATION.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if existing.get("source_fingerprints_sha256") != fingerprints:
            raise ValueError("existing prepared package belongs to different frozen inputs")
        required_outputs = ("train.npz", "validation.npz", "filtration_transform.npz")
        if not all((output_dir / name).is_file() for name in required_outputs):
            raise ValueError("existing preparation receipt is incomplete; preserve it for inspection")
        return existing
    train = load_npz(train_cache_path)
    validation = load_npz(validation_cache_path)
    exchange = load_npz(v216_exchange_path)
    val_commands = load_npz(validation_commands_path)
    validate_operator_cache(train, 45071)
    validate_operator_cache(validation, 9780)
    train_operator_rms = float(np.sqrt(np.mean(np.abs(train["g_ephi"]) ** 2)))
    if not np.isclose(train_operator_rms, 1.0, atol=2e-6):
        raise ValueError("canonical train operator is not normalized to unit RMS")
    train_commands = complex_from_iq(exchange["command_iq"])
    validation_commands = np.asarray(val_commands["weights"], dtype=np.complex64)
    if validation_commands.shape != (512, 16):
        raise ValueError("validation commands must have shape [512,16]")
    if not np.allclose(np.sum(np.abs(validation_commands) ** 2, axis=1), 1.0, atol=1e-5):
        raise ValueError("validation commands are not unit-power")
    train_amplitude = amplitude_from_operator(train["g_ephi"], train_commands)
    validation_amplitude = amplitude_from_operator(validation["g_ephi"], validation_commands)
    envelope_scale = float(np.median(np.abs(train["g_ephi"])))
    if not np.isfinite(envelope_scale) or envelope_scale <= 0:
        raise ValueError("invalid train-only envelope scale")
    transform = FiltrationSSHTransform.fit(
        torch.from_numpy(train["direction"].astype(np.float64)), degree=4
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz_compressed(
        output_dir / "train.npz",
        canonical_indices=train["canonical_indices"],
        direction=train["direction"].astype(np.float32),
        g_ephi=train["g_ephi"].astype(np.complex64),
        solid_angle_weights=train["solid_angle_weights"].astype(np.float64),
        commands=train_commands,
        amplitude=train_amplitude,
    )
    atomic_npz_compressed(
        output_dir / "validation.npz",
        canonical_indices=validation["canonical_indices"],
        direction=validation["direction"].astype(np.float32),
        g_ephi=validation["g_ephi"].astype(np.complex64),
        solid_angle_weights=validation["solid_angle_weights"].astype(np.float64),
        commands=validation_commands,
        command_family=np.asarray(val_commands.get("family", np.full(512, "unknown"))),
        amplitude=validation_amplitude,
    )
    transform.save(output_dir / "filtration_transform.npz")
    manifest = {
        "schema": "v218-canonical-adapter-1",
        "train_direction_count": 45071,
        "validation_direction_count": 9780,
        "train_command_count": 64,
        "validation_command_count": 512,
        "envelope_scale": envelope_scale,
        "canonical_operator_train_rms": train_operator_rms,
        "canonical_source_normalization_scale": float(train["normalization_scale"]),
        "filtration_degree": 4,
        "filtration_measure": transform.measure,
        "source_fingerprints_sha256": fingerprints,
        "blind_test_status": "sealed_not_opened",
        "amplitude_recomputed_from_complex_operator": True,
    }
    atomic_json(output_dir / "PREPARATION.json", manifest)
    return manifest


@dataclass(frozen=True)
class PreparedData:
    train: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    transform: FiltrationSSHTransform
    manifest: dict[str, object]


def load_prepared(path: str | Path) -> PreparedData:
    root = assert_unsealed_path_not_used(path)
    manifest = json.loads((root / "PREPARATION.json").read_text(encoding="utf-8"))
    if manifest.get("blind_test_status") != "sealed_not_opened":
        raise ValueError("prepared package did not preserve the confirmation seal")
    return PreparedData(
        train=load_npz(root / "train.npz"),
        validation=load_npz(root / "validation.npz"),
        transform=FiltrationSSHTransform.load(root / "filtration_transform.npz"),
        manifest=manifest,
    )
