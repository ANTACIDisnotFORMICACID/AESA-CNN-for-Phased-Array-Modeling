"""Controlled front-hemisphere CNN and four-point optimization campaign.

The experiment has two orthogonal axes:

* CNN axis: fixed A10 architecture and optimizer, 2x2 factorial losses.
* Beam axis: fixed exact HFSS operator and several phase-only solvers.

All training directions, four-point coordinates, and uncertainty neighbours
must satisfy q_x > 0. Reverse-hemisphere samples are rejected, not reweighted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import linprog
import torch
from torch import nn


BASE = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_ROOT = BASE / "artifacts" / "front_hemisphere_20260906" / "datasets"
CANONICAL_LEGACY_CHECKPOINT = (
    BASE / "artifacts" / "controlled_psd_20260906" / "runs" / "A10_seed21011" / "best.pt"
)
sys.path.insert(0, str(BASE / "src"))
from v210.models.quadratic_psd import (  # noqa: E402
    V210VectorBaseline,
    gains_from_factor,
    quadratic_form,
)


SEEDS = (21111, 21112, 21113)
CNN_RECIPES = ("F0", "FS", "FM", "FSM")
OPTIMIZERS = ("O0_legacy", "O1_q_eigen", "O2_coordinate", "O3_high_budget")
CONFIG = {
    "schema": "v210-front-hemisphere-factorial-1",
    "front_condition": "q_x > 0",
    "seeds": list(SEEDS),
    "cnn_recipes": list(CNN_RECIPES),
    "epochs": 2600,
    "steps_per_epoch": 4,
    "batch_size": 512,
    "commands_per_step": 64,
    "validation_directions": 2048,
    "validation_every": 40,
    "learning_rate": 1e-3,
    "minimum_learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "gradient_clip_norm": 5.0,
    "spectral_loss_weight": 0.05,
    "mined_command_loss_weight": 0.03,
    "optimizer_starts": 8,
    "optimizer_steps": 160,
    "high_budget_starts": 32,
    "high_budget_steps": 320,
    "coordinate_sweeps": 40,
    "validation_cases": 240,
    "holdout_cases": 600,
    "suite_strata": ["clustered", "moderate", "wide", "grazing"],
    "suite_seed": 2112026,
    "parameter_budget": 34973,
    "selection_nmse_guard_db": 0.30,
    "selection_strong_mae_guard_db": 0.02,
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_hashes() -> dict[str, str]:
    files = [
        Path(__file__).resolve(),
        BASE / "src/v210/models/quadratic_psd.py",
        BASE / "src/v210/models/blocks_2d.py",
        BASE / "src/v210/models/nominal_pattern.py",
        BASE / "src/v210/contracts.py",
    ]
    return {str(path.relative_to(BASE)): digest(path) for path in files}


def db(power: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(power.clamp_min(1e-5))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def dataset_path(root: Path, name: str) -> Path:
    folder = "dev" if name in {"train", "validation"} else "reserved"
    local = root / "datasets" / folder / f"{name}.npz"
    canonical = CANONICAL_DATASET_ROOT / folder / f"{name}.npz"
    if local.is_file():
        return local
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(f"dataset not found at {local} or {canonical}")


def dataset(root: Path, name: str) -> dict[str, np.ndarray]:
    return load_npz(dataset_path(root, name))


def legacy_checkpoint_path(root: Path) -> Path:
    protocol = read_json(root / "protocol.json")
    local = root / protocol["legacy_checkpoint"]["path"]
    if local.is_file():
        return local
    if CANONICAL_LEGACY_CHECKPOINT.is_file():
        return CANONICAL_LEGACY_CHECKPOINT
    raise FileNotFoundError(
        f"legacy checkpoint not found at {local} or {CANONICAL_LEGACY_CHECKPOINT}"
    )


def front_mask(query: np.ndarray) -> np.ndarray:
    return np.asarray(query)[..., 0] > 0.0


def assert_front(query: np.ndarray, label: str) -> None:
    if not bool(front_mask(query).all()):
        count = int((~front_mask(query)).sum())
        raise ValueError(f"{label} contains {count} reverse-hemisphere points")


def command_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    amplitude = (
        prediction.clamp_min(1e-12).sqrt()
        - target.clamp_min(1e-12).sqrt()
    ).square().mean() / target.mean().clamp_min(1e-8)
    power = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
    robust = nn.functional.huber_loss(db(prediction), db(target), delta=1.0)
    return amplitude + 0.10 * power + 0.03 * robust


def q_error(
    predicted_factor: torch.Tensor,
    true_factor: torch.Tensor,
    *,
    need_vector: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    predicted_q = quadratic_form(predicted_factor)
    true_q = quadratic_form(true_factor)
    delta = predicted_q - true_q
    denominator = torch.linalg.eigvalsh(true_q).real[:, -1].clamp_min(1e-8)
    if need_vector:
        eigenvalue, eigenvector = torch.linalg.eigh(delta.detach())
        index = eigenvalue.abs().argmax(dim=-1)
        batch = torch.arange(len(delta), device=delta.device)
        worst_vector = eigenvector[batch, :, index]
        spectrum = eigenvalue.abs().amax(dim=-1)
    else:
        eigenvalue = torch.linalg.eigvalsh(delta)
        worst_vector = None
        spectrum = eigenvalue.abs().amax(dim=-1)
    normalized = spectrum / denominator
    return normalized.square().mean(), worst_vector, predicted_q, true_q


def mined_phase_loss(
    predicted_q: torch.Tensor,
    true_q: torch.Tensor,
    eigenvector: torch.Tensor,
) -> torch.Tensor:
    weight = torch.exp(1j * torch.angle(eigenvector)) / 4.0
    predicted = torch.einsum("bn,bnm,bm->b", weight.conj(), predicted_q, weight).real
    target = torch.einsum("bn,bnm,bm->b", weight.conj(), true_q, weight).real
    return nn.functional.huber_loss(db(predicted), db(target), delta=1.0)


def _weighted_choice(rng: np.random.Generator, pool: np.ndarray, weights: np.ndarray) -> int:
    probability = weights[pool].astype(np.float64)
    probability /= probability.sum()
    return int(rng.choice(pool, p=probability))


def _angles(query: np.ndarray) -> np.ndarray:
    cosine = np.clip(query @ query.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def generate_suite(data: dict[str, np.ndarray], count: int, seed: int) -> dict[str, np.ndarray]:
    if count % len(CONFIG["suite_strata"]):
        raise ValueError("suite count must be divisible by number of strata")
    query = data["q"]
    available = np.flatnonzero(front_mask(query))
    theta_weight = np.sin(np.deg2rad(data["theta"])).clip(1e-4)
    grazing = available[(query[available, 0] < 0.25)]
    rng = np.random.default_rng(seed)
    source_indices = []
    strata = []
    seen = set()
    each = count // len(CONFIG["suite_strata"])

    def accept(indices: np.ndarray, stratum: str) -> bool:
        key = tuple(sorted(int(index) for index in indices))
        if len(set(key)) != 4 or key in seen:
            return False
        angle = _angles(query[indices])
        pair = angle[np.triu_indices(4, 1)]
        if stratum == "clustered" and pair.max() > 15.0:
            return False
        if stratum == "moderate" and pair.min() < 15.0:
            return False
        if stratum == "wide" and pair.min() < 45.0:
            return False
        if stratum == "grazing" and (query[indices, 0] >= 0.25).any():
            return False
        if stratum == "grazing" and pair.min() < 12.0:
            return False
        seen.add(key)
        source_indices.append(indices.astype(np.int64))
        strata.append(stratum)
        return True

    for stratum in CONFIG["suite_strata"]:
        made = 0
        attempts = 0
        while made < each and attempts < each * 20000:
            attempts += 1
            if stratum == "clustered":
                center = _weighted_choice(rng, available, theta_weight)
                cosine = query[available] @ query[center]
                near = available[cosine >= math.cos(math.radians(15.0))]
                if len(near) < 4:
                    continue
                probability = theta_weight[near].astype(np.float64)
                probability /= probability.sum()
                indices = rng.choice(near, 4, replace=False, p=probability)
            else:
                pool = grazing if stratum == "grazing" else available
                probability = theta_weight[pool].astype(np.float64)
                probability /= probability.sum()
                indices = rng.choice(pool, 4, replace=False, p=probability)
            made += int(accept(np.asarray(indices), stratum))
        if made != each:
            raise RuntimeError(f"could not generate {stratum}: {made}/{each}")

    source_indices_array = np.stack(source_indices)
    suite_query = query[source_indices_array]
    suite_factor = data["h"][source_indices_array]
    assert_front(suite_query, "suite")

    flat_query = query[available]
    neighbour_indices = np.empty((count, 4, 5), dtype=np.int64)
    neighbour_angles = np.empty((count, 4, 5), dtype=np.float32)
    for case in range(count):
        for point in range(4):
            source_index = source_indices_array[case, point]
            cosine = flat_query @ query[source_index]
            order = np.argpartition(-cosine, 5)[:5]
            order = order[np.argsort(-cosine[order])]
            neighbour_indices[case, point] = available[order]
            neighbour_angles[case, point] = np.rad2deg(
                np.arccos(np.clip(cosine[order], -1.0, 1.0))
            )
    neighbour_query = query[neighbour_indices]
    neighbour_factor = data["h"][neighbour_indices]
    assert_front(neighbour_query, "suite neighbours")
    return {
        "q": suite_query.astype(np.float32),
        "h": suite_factor.astype(np.complex64),
        "source_indices": source_indices_array,
        "stratum": np.asarray(strata, dtype="U16"),
        "neighbour_q": neighbour_query.astype(np.float32),
        "neighbour_h": neighbour_factor.astype(np.complex64),
        "neighbour_angle_deg": neighbour_angles,
    }


def prepare(root: Path, source_root: Path, legacy_checkpoint: Path) -> dict:
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["config"] != CONFIG:
            raise ValueError("existing protocol differs from frozen configuration")
        return protocol
    copied = {}
    front_counts = {}
    for folder, name in (
        ("dev", "train"),
        ("dev", "validation"),
        ("reserved", "calibration"),
        ("reserved", "holdout"),
    ):
        source = source_root / folder / f"{name}.npz"
        destination = CANONICAL_DATASET_ROOT / folder / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(source) != digest(destination):
                raise ValueError(f"canonical dataset differs from source: {destination}")
        else:
            shutil.copy2(source, destination)
        data = load_npz(destination)
        front_counts[name] = {
            "total_directions": int(len(data["q"])),
            "front_directions": int(front_mask(data["q"]).sum()),
            "reverse_directions_used": 0,
        }
        copied[str(destination.relative_to(BASE))] = {
            "sha256": digest(destination),
            "bytes": destination.stat().st_size,
            "source": str(source),
        }

    reference = CANONICAL_LEGACY_CHECKPOINT
    reference.parent.mkdir(parents=True, exist_ok=True)
    if not reference.exists():
        shutil.copy2(legacy_checkpoint, reference)
    elif digest(legacy_checkpoint) != digest(reference):
        raise ValueError(f"canonical legacy checkpoint differs from {legacy_checkpoint}")
    validation_suite = generate_suite(dataset(root, "validation"), CONFIG["validation_cases"], CONFIG["suite_seed"])
    holdout_suite = generate_suite(dataset(root, "holdout"), CONFIG["holdout_cases"], CONFIG["suite_seed"] + 1)
    np.savez_compressed(root / "validation_front_suite.npz", **validation_suite)
    np.savez_compressed(root / "holdout_front_suite.npz", **holdout_suite)
    model = V210VectorBaseline()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count > CONFIG["parameter_budget"]:
        raise ValueError("model exceeds parameter budget")
    train = dataset(root, "train")
    mask = front_mask(train["q"])
    scale = float(np.sqrt(np.mean(np.abs(train["h"][mask]) ** 2)))
    protocol = {
        "config": CONFIG,
        "source_code_hashes": source_hashes(),
        "dataset_files": copied,
        "front_counts": front_counts,
        "front_scale": scale,
        "parameter_count": count,
        "legacy_checkpoint": {
            "path": str(reference.relative_to(BASE)),
            "sha256": digest(reference),
            "source": str(legacy_checkpoint),
        },
        "suite_files": {
            "validation_front_suite.npz": digest(root / "validation_front_suite.npz"),
            "holdout_front_suite.npz": digest(root / "holdout_front_suite.npz"),
        },
        "factorial": {
            "F0": {"spectral": False, "mined": False},
            "FS": {"spectral": True, "mined": False},
            "FM": {"spectral": False, "mined": True},
            "FSM": {"spectral": True, "mined": True},
        },
        "isolation": (
            "CNN recipes share architecture, initialization by paired seed, data, updates, "
            "optimizer, and validation subset. Beam solvers use exact HFSS factors only. "
            "All q_x<=0 samples are excluded from training, suites, neighbours, selection, and reporting."
        ),
        "historical_boundary": (
            "The underlying HFSS state and source splits have been opened in earlier V2.9/V2.10 work; "
            "this is a controlled front-domain comparison, not an unseen physical-state blind test."
        ),
    }
    write_json(protocol_path, protocol)
    return protocol


def verify_protocol(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    if protocol["config"] != CONFIG:
        raise ValueError("configuration drift")
    if not protocol.get("source_code_hashes"):
        raise ValueError("frozen protocol is missing source provenance")
    for relative, metadata in protocol["dataset_files"].items():
        name = Path(relative).stem
        if digest(dataset_path(root, name)) != metadata["sha256"]:
            raise ValueError(f"dataset changed: {relative}")
    for relative, expected in protocol["suite_files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"suite changed: {relative}")
        assert_front(load_npz(root / relative)["q"], relative)
    return protocol


@torch.no_grad()
def predict(model: nn.Module, query: torch.Tensor, scale: float, batch: int = 1024) -> torch.Tensor:
    return torch.cat(
        [model(query[index : index + batch]) * scale for index in range(0, len(query), batch)]
    )


def forward_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    error = 10.0 * np.log10(np.maximum(predicted, 1e-30)) - 10.0 * np.log10(np.maximum(target, 1e-30))
    relative = 10.0 * np.log10(np.maximum(target / target.max(1, keepdims=True), 1e-30))

    def stat(mask: np.ndarray) -> dict:
        values = error[mask]
        return {
            "count": int(mask.sum()),
            "mae_db": float(np.mean(np.abs(values))),
            "bias_db": float(np.mean(values)),
            "p95_abs_db": float(np.percentile(np.abs(values), 95)),
        }

    return {
        "amplitude_nmse_db": float(10.0 * np.log10(np.sum((np.sqrt(predicted) - np.sqrt(target)) ** 2) / np.sum(target))),
        "power_nmse_db": float(10.0 * np.log10(np.sum((predicted - target) ** 2) / np.sum(target ** 2))),
        "bins": {
            "all": stat(np.ones_like(error, dtype=bool)),
            "strong": stat(relative >= -10.0),
            "medium": stat((relative < -10.0) & (relative >= -20.0)),
            "weak": stat((relative < -20.0) & (relative >= -30.0)),
            "deep": stat(relative < -30.0),
        },
    }


def validation_metrics(model: nn.Module, data: dict, scale: float, device: str) -> dict:
    mask = front_mask(data["q"])
    indices = np.flatnonzero(mask)
    rng = np.random.default_rng(21190)
    indices = rng.choice(indices, CONFIG["validation_directions"], replace=False)
    query = torch.tensor(data["q"][indices], device=device)
    true_factor = torch.tensor(data["h"][indices], device=device)
    command = torch.tensor(data["commands"], device=device)
    predicted_factor = predict(model, query, scale)
    predicted_gain = gains_from_factor(predicted_factor, command).cpu().numpy()
    true_gain = gains_from_factor(true_factor, command).cpu().numpy()
    result = forward_metrics(predicted_gain, true_gain)
    spectral, _, predicted_q, true_q = q_error(predicted_factor, true_factor, need_vector=False)
    result["spectral_relative_rms"] = float(torch.sqrt(spectral).cpu())
    result["q_frobenius_nmse_db"] = float(
        10.0 * torch.log10(
            (predicted_q - true_q).abs().square().sum()
            / true_q.abs().square().sum().clamp_min(1e-12)
        ).cpu()
    )
    return result


def train_one(root: Path, recipe: str, seed: int, device: str) -> dict:
    if (root / "cnn_selection_frozen.json").exists() or (root / "holdout_opened.json").exists():
        raise RuntimeError("training closed after freeze")
    protocol = verify_protocol(root)
    output = root / "runs" / f"{recipe}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        return read_json(output / "result.json")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = V210VectorBaseline().to(device)
    initial_hash = hashlib.sha256(
        b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in model.state_dict().values())
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=12,
        factor=0.5,
        threshold=0.01,
        threshold_mode="abs",
        min_lr=CONFIG["minimum_learning_rate"],
    )
    train = dataset(root, "train")
    validation = dataset(root, "validation")
    mask = front_mask(train["q"])
    query = torch.tensor(train["q"][mask], device=device)
    true_factor = torch.tensor(train["h"][mask], device=device)
    command = torch.tensor(train["commands"], device=device)
    assert bool((query[:, 0] > 0).all())
    scale = float(protocol["front_scale"])
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    best = math.inf
    best_epoch = 0
    history = []
    started = time.perf_counter()
    use_spectral = recipe in {"FS", "FSM"}
    use_mined = recipe in {"FM", "FSM"}
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        sums = {"total": 0.0, "command": 0.0, "spectral": 0.0, "mined": 0.0}
        for _ in range(CONFIG["steps_per_epoch"]):
            qi = torch.randint(len(query), (CONFIG["batch_size"],), generator=generator, device=device)
            ci = torch.randint(len(command), (CONFIG["commands_per_step"],), generator=generator, device=device)
            predicted_factor = model(query[qi]) * scale
            target_factor = true_factor[qi]
            predicted_gain = gains_from_factor(predicted_factor, command[ci])
            target_gain = gains_from_factor(target_factor, command[ci])
            base = command_loss(predicted_gain, target_gain)
            loss = base
            spectral_loss = torch.zeros((), device=device)
            mining_loss = torch.zeros((), device=device)
            if use_spectral or use_mined:
                spectral_loss, vector, predicted_q, true_q = q_error(
                    predicted_factor, target_factor, need_vector=use_mined
                )
                if use_spectral:
                    loss = loss + CONFIG["spectral_loss_weight"] * spectral_loss
                if use_mined:
                    mining_loss = mined_phase_loss(predicted_q, true_q, vector)
                    loss = loss + CONFIG["mined_command_loss_weight"] * mining_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            sums["total"] += float(loss.detach())
            sums["command"] += float(base.detach())
            sums["spectral"] += float(spectral_loss.detach())
            sums["mined"] += float(mining_loss.detach())
        if epoch % CONFIG["validation_every"]:
            continue
        model.eval()
        measured = validation_metrics(model, validation, scale, device)
        score = measured["amplitude_nmse_db"]
        scheduler.step(score)
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "recipe": recipe,
                    "seed": seed,
                    "epoch": epoch,
                    "scale": scale,
                    "initial_hash": initial_hash,
                },
                output / "best.pt",
            )
        history.append(
            {
                "epoch": epoch,
                "validation": measured,
                "best_amplitude_nmse_db": best,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": {key: value / CONFIG["steps_per_epoch"] for key, value in sums.items()},
            }
        )
        if epoch % 200 == 0 or epoch == CONFIG["epochs"]:
            write_json(output / "history.json", history)
            write_json(
                root / "progress.json",
                {
                    "recipe": recipe,
                    "seed": seed,
                    "epoch": epoch,
                    "best": best,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(recipe, seed, epoch, f"best={best:.4f}", f"seconds={time.perf_counter()-started:.1f}", flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    result = {
        "recipe": recipe,
        "seed": seed,
        "epochs": CONFIG["epochs"],
        "updates": CONFIG["epochs"] * CONFIG["steps_per_epoch"],
        "best_epoch": best_epoch,
        "initial_hash": initial_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "validation": validation_metrics(model, validation, scale, device),
        "seconds": time.perf_counter() - started,
        "checkpoint_sha256": digest(output / "best.pt"),
        "front_directions_only": True,
        "holdout_loaded": False,
    }
    write_json(output / "result.json", result)
    return result


def q_eigen_initial(factor: torch.Tensor, starts: int, seed: int) -> torch.Tensor:
    qmatrix = quadratic_form(factor)
    batch, targets = qmatrix.shape[:2]
    eigenvector = torch.linalg.eigh(qmatrix).eigenvectors[..., -1]
    candidates = [eigenvector[:, index] for index in range(min(targets, 4))]
    generator = torch.Generator(device=factor.device).manual_seed(seed)
    while len(candidates) < starts:
        if len(candidates) == 4:
            alpha = torch.ones((batch, targets), device=factor.device) / targets
        else:
            raw = torch.rand((batch, targets), generator=generator, device=factor.device)
            alpha = raw / raw.sum(-1, keepdim=True)
        combined = torch.einsum("bk,bknm->bnm", alpha.to(qmatrix.dtype), qmatrix)
        candidates.append(torch.linalg.eigh(combined).eigenvectors[..., -1])
    value = torch.stack(candidates[:starts], dim=1)
    return torch.exp(1j * torch.angle(value)) / 4.0


def optimize_softmin(
    factor: torch.Tensor,
    *,
    starts: int,
    steps: int,
    seed: int,
    initialization: str,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = factor.detach()
    batch, targets, rows, ports = factor.shape
    generator = torch.Generator(device=factor.device).manual_seed(seed)
    random_value = torch.randn((batch, starts, ports), generator=generator, device=factor.device)
    random_value = random_value + 1j * torch.randn(
        (batch, starts, ports), generator=generator, device=factor.device
    )
    weight = torch.exp(1j * torch.angle(random_value)) / 4.0
    if initialization == "legacy":
        row = 1 if rows > 1 else 0
        matched = factor[:, :, row, :].conj()
        count = min(targets, 4, starts)
        weight[:, :count] = torch.exp(1j * torch.angle(matched[:, :count])) / 4.0
    elif initialization == "q_eigen":
        weight = q_eigen_initial(factor, starts, seed)
    else:
        raise ValueError(initialization)
    if initial is not None:
        weight[:, -1] = initial.detach()
    phase = nn.Parameter(torch.angle(weight))
    optimizer = torch.optim.Adam([phase], lr=0.065)
    best = torch.full((batch,), -1e30, device=factor.device)
    best_weight = torch.zeros((batch, ports), dtype=torch.complex64, device=factor.device)
    for step in range(steps + 1):
        weight = torch.exp(1j * phase) / 4.0
        field = torch.einsum("bkrn,bsn->bskr", factor, weight)
        gain_db = db(field.abs().square().sum(-1) / 60.0)
        minimum = gain_db.min(-1).values
        value, index = minimum.max(1)
        improved = value > best
        best = torch.maximum(best, value.detach())
        best_weight[improved] = weight.detach()[torch.arange(batch, device=factor.device), index][improved]
        if step == steps:
            break
        temperature = 1.2 * (1.0 - step / steps) + 0.12
        loss = (temperature * torch.logsumexp(-gain_db / temperature, dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return best_weight, best


@torch.no_grad()
def coordinate_refine(
    factor: torch.Tensor,
    initial: torch.Tensor,
    sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmatrix = quadratic_form(factor)
    weight = initial.detach().clone()
    field = torch.einsum("bkrn,bn->bkr", factor, weight)
    best = db(field.abs().square().sum(-1) / 60.0).min(-1).values
    best_weight = weight.clone()
    batch = len(weight)
    for sweep in range(sweeps):
        gain = torch.einsum("bn,bknm,bm->bk", weight.conj(), qmatrix, weight).real.clamp_min(1e-8)
        gain_db = db(gain)
        temperature = 0.8 * (1.0 - sweep / max(sweeps, 1)) + 0.10
        alpha = torch.softmax(-gain_db / temperature, dim=-1) / gain
        matrix = torch.einsum("bk,bknm->bnm", alpha.to(qmatrix.dtype), qmatrix)
        for port in range(16):
            coupling = torch.einsum("bm,bm->b", matrix[:, port, :], weight)
            coupling = coupling - matrix[:, port, port] * weight[:, port]
            replacement = torch.exp(1j * torch.angle(coupling)) / 4.0
            weight = weight.clone()
            weight[:, port] = replacement
        field = torch.einsum("bkrn,bn->bkr", factor, weight)
        value = db(field.abs().square().sum(-1) / 60.0).min(-1).values
        improved = value > best
        best = torch.maximum(best, value)
        best_weight[improved] = weight[improved]
    return best_weight, best


def true_gain(factor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bkrn,bn->bkr", factor, weight).abs().square().sum(-1) / 60.0


def single_point_ceiling(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, targets, rows, ports = factor.shape
    flat = factor.reshape(batch * targets, 1, rows, ports)
    weight, _ = optimize_softmin(
        flat,
        starts=12,
        steps=160,
        seed=21301,
        initialization="q_eigen",
    )
    gain = true_gain(flat, weight)[:, 0]
    return weight.reshape(batch, targets, ports), gain.reshape(batch, targets)


def method_metrics(
    factor: torch.Tensor,
    weight: torch.Tensor,
    reference_minimum: torch.Tensor,
    solo_gain: torch.Tensor,
    strata: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    gain = true_gain(factor, weight)
    minimum = gain.min(-1).values
    reference = torch.maximum(reference_minimum, minimum)
    regret = (1.0 - minimum / reference).clamp_min(0.0)
    ratio = (gain / solo_gain).min(-1).values
    metrics = {
        "cases": int(len(factor)),
        "minimum_gain_db_p5_median": [
            float(torch.quantile(db(minimum), value)) for value in (0.05, 0.50)
        ],
        "normalized_feasibility_p5_median": [
            float(torch.quantile(ratio, value)) for value in (0.05, 0.50)
        ],
        "regret_median": float(torch.quantile(regret, 0.50)),
        "regret_p95": float(torch.quantile(regret, 0.95)),
        "regret_le_10pct": float((regret <= 0.10).float().mean()),
        "constraint_max_magnitude_error": float((weight.abs() - 0.25).abs().max()),
        "by_stratum": {},
    }
    ratio_np = ratio.cpu().numpy()
    regret_np = regret.cpu().numpy()
    minimum_np = db(minimum).cpu().numpy()
    for name in np.unique(strata):
        selected = strata == name
        metrics["by_stratum"][str(name)] = {
            "cases": int(selected.sum()),
            "regret_p95": float(np.percentile(regret_np[selected], 95)),
            "normalized_feasibility_p5": float(np.percentile(ratio_np[selected], 5)),
            "minimum_gain_db_p5": float(np.percentile(minimum_np[selected], 5)),
        }
    arrays = {
        "gain": gain.cpu().numpy(),
        "minimum": minimum.cpu().numpy(),
        "regret": regret_np,
        "normalized_feasibility": ratio_np,
        "weight": weight.cpu().numpy(),
    }
    return metrics, arrays


def exact_optimizer_benchmark(suite: dict, device: str) -> tuple[dict, dict[str, np.ndarray]]:
    assert_front(suite["q"], "optimizer suite")
    factor = torch.tensor(suite["h"], device=device)
    strata = suite["stratum"]
    solo_weight, solo_gain = single_point_ceiling(factor)
    outputs: dict[str, tuple[torch.Tensor, float]] = {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    o0, _ = optimize_softmin(
        factor,
        starts=CONFIG["optimizer_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21311,
        initialization="legacy",
    )
    torch.cuda.synchronize()
    outputs["O0_legacy"] = (o0, time.perf_counter() - started)
    started = time.perf_counter()
    o1, _ = optimize_softmin(
        factor,
        starts=CONFIG["optimizer_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21311,
        initialization="q_eigen",
    )
    torch.cuda.synchronize()
    outputs["O1_q_eigen"] = (o1, time.perf_counter() - started)
    started = time.perf_counter()
    o2, _ = coordinate_refine(factor, o1, CONFIG["coordinate_sweeps"])
    torch.cuda.synchronize()
    outputs["O2_coordinate"] = (o2, time.perf_counter() - started)
    started = time.perf_counter()
    o3, _ = optimize_softmin(
        factor,
        starts=CONFIG["high_budget_starts"],
        steps=CONFIG["high_budget_steps"],
        seed=21321,
        initialization="q_eigen",
        initial=o2,
    )
    torch.cuda.synchronize()
    outputs["O3_high_budget"] = (o3, time.perf_counter() - started)
    reference_gain = true_gain(factor, o3)
    reference_minimum = reference_gain.min(-1).values
    metrics = {}
    arrays = {
        "reference_minimum": reference_minimum.cpu().numpy(),
        "solo_gain": solo_gain.cpu().numpy(),
        "solo_weight": solo_weight.cpu().numpy(),
    }
    for name, (weight, seconds) in outputs.items():
        row, values = method_metrics(factor, weight, reference_minimum, solo_gain, strata)
        row["gpu_seconds"] = seconds
        metrics[name] = row
        arrays.update({f"{name}_{key}": value for key, value in values.items()})

    neighbour_factor = torch.tensor(suite["neighbour_h"], device=device).reshape(
        len(factor), -1, factor.shape[2], 16
    )
    robust_weight, _ = optimize_softmin(
        neighbour_factor,
        starts=CONFIG["optimizer_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21331,
        initialization="q_eigen",
    )
    robust_row, robust_values = method_metrics(
        factor, robust_weight, reference_minimum, solo_gain, strata
    )
    neighbour_gain = true_gain(neighbour_factor, robust_weight)
    robust_row["neighbour_worst_gain_db_p5"] = float(
        torch.quantile(db(neighbour_gain.min(-1).values), 0.05)
    )
    robust_row["neighbour_max_angle_deg"] = float(suite["neighbour_angle_deg"].max())
    metrics["O4_local_robust"] = robust_row
    arrays.update({f"O4_local_robust_{key}": value for key, value in robust_values.items()})
    arrays["O4_local_robust_neighbour_minimum"] = neighbour_gain.min(-1).values.cpu().numpy()

    # Exploratory average-power time sharing among four solo beams and O1.
    candidates = torch.cat((solo_weight, o1[:, None]), dim=1)
    gain_matrix = torch.einsum("bkrn,bmn->bmkr", factor, candidates).abs().square().sum(-1) / 60.0
    gain_numpy = gain_matrix.cpu().numpy()
    duty = np.zeros((len(factor), 5), dtype=np.float64)
    time_minimum = np.empty(len(factor), dtype=np.float64)
    for case in range(len(factor)):
        objective = np.array([0.0] * 5 + [-1.0])
        inequality = np.concatenate((-gain_numpy[case].T, np.ones((4, 1))), axis=1)
        result = linprog(
            objective,
            A_ub=inequality,
            b_ub=np.zeros(4),
            A_eq=np.array([[1.0] * 5 + [0.0]]),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * 5 + [(None, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(result.message)
        duty[case] = result.x[:5]
        time_minimum[case] = result.x[-1]
    joint_minimum = arrays["O1_q_eigen_minimum"]
    metrics["X1_time_sharing"] = {
        "scope": "average power over five phase-only states; not instantaneous four-point gain",
        "minimum_gain_db_p5_median": np.percentile(10 * np.log10(np.maximum(time_minimum, 1e-30)), [5, 50]).tolist(),
        "gain_over_O1_db_p5_median": np.percentile(
            10 * np.log10(np.maximum(time_minimum / joint_minimum, 1e-30)), [5, 50]
        ).tolist(),
        "mean_active_states": float(np.mean((duty > 1e-6).sum(axis=1))),
    }
    arrays["X1_time_sharing_minimum"] = time_minimum
    arrays["X1_time_sharing_duty"] = duty
    return metrics, arrays


def cnn_suite_metrics(
    model: nn.Module,
    scale: float,
    suite: dict,
    exact_reference: np.ndarray,
    solo_gain: np.ndarray,
    device: str,
    initialization: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    assert_front(suite["q"], "CNN suite")
    query = torch.tensor(suite["q"], device=device)
    true_factor = torch.tensor(suite["h"], device=device)
    predicted = predict(model, query.reshape(-1, 3), scale).reshape(
        len(query), 4, 3, 16
    )
    weight, predicted_minimum = optimize_softmin(
        predicted,
        starts=CONFIG["optimizer_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21401,
        initialization=initialization,
    )
    reference = torch.tensor(exact_reference, device=device)
    solo = torch.tensor(solo_gain, device=device)
    metrics, arrays = method_metrics(
        true_factor, weight, reference, solo, suite["stratum"]
    )
    actual_gain = torch.tensor(arrays["gain"], device=device)
    predicted_gain = true_gain(predicted, weight)
    error = db(predicted_gain) - db(actual_gain)
    bottleneck = actual_gain.argmin(-1)
    bottleneck_error = error[torch.arange(len(error), device=device), bottleneck]
    metrics.update(
        {
            "fourpoint_mae_db": float(error.abs().mean()),
            "bottleneck_bias_db": float(bottleneck_error.mean()),
            "bottleneck_over_p95_db": float(torch.quantile(bottleneck_error, 0.95)),
            "bottleneck_identity_accuracy": float(
                (predicted_gain.argmin(-1) == bottleneck).float().mean()
            ),
        }
    )
    arrays["prediction_error_db"] = error.cpu().numpy()
    arrays["bottleneck_error_db"] = bottleneck_error.cpu().numpy()
    arrays["predicted_minimum_db"] = predicted_minimum.cpu().numpy()
    return metrics, arrays


def validation_fourpoint(root: Path, recipe: str, seed: int, device: str) -> dict:
    output = root / "runs" / f"{recipe}_seed{seed}" / "validation_fourpoint.json"
    if output.exists():
        return read_json(output)
    suite = load_npz(root / "validation_front_suite.npz")
    exact = load_npz(root / "validation_optimizer_arrays.npz")
    checkpoint = torch.load(
        root / "runs" / f"{recipe}_seed{seed}" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model = V210VectorBaseline().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, _ = cnn_suite_metrics(
        model,
        checkpoint["scale"],
        suite,
        exact["reference_minimum"],
        exact["solo_gain"],
        device,
        "q_eigen",
    )
    write_json(output, metrics)
    return metrics


def train_campaign(root: Path, device: str) -> dict:
    verify_protocol(root)
    freeze_path = root / "cnn_selection_frozen.json"
    if freeze_path.exists():
        return read_json(freeze_path)
    suite = load_npz(root / "validation_front_suite.npz")
    if not (root / "validation_optimizer.json").exists():
        optimizer_metrics, optimizer_arrays = exact_optimizer_benchmark(suite, device)
        write_json(root / "validation_optimizer.json", optimizer_metrics)
        np.savez_compressed(root / "validation_optimizer_arrays.npz", **optimizer_arrays)
    rows = []
    for recipe in CNN_RECIPES:
        for seed in SEEDS:
            row = train_one(root, recipe, seed, device)
            row["validation_fourpoint"] = validation_fourpoint(root, recipe, seed, device)
            rows.append(row)
            write_json(root / "development_results.json", rows)
    paired_initialization = all(
        len({row["initial_hash"] for row in rows if row["seed"] == seed}) == 1
        for seed in SEEDS
    )
    if not paired_initialization:
        raise ValueError("paired recipes did not start from identical seed initialization")
    aggregate = {}
    for recipe in CNN_RECIPES:
        subset = [row for row in rows if row["recipe"] == recipe]
        aggregate[recipe] = {
            "validation_nmse_db_mean": float(np.mean([row["validation"]["amplitude_nmse_db"] for row in subset])),
            "validation_strong_mae_db_mean": float(np.mean([row["validation"]["bins"]["strong"]["mae_db"] for row in subset])),
            "validation_spectral_relative_rms_mean": float(np.mean([row["validation"]["spectral_relative_rms"] for row in subset])),
            "validation_regret_p95_mean": float(np.mean([row["validation_fourpoint"]["regret_p95"] for row in subset])),
            "validation_bottleneck_over_p95_db_mean": float(np.mean([row["validation_fourpoint"]["bottleneck_over_p95_db"] for row in subset])),
        }
    base = aggregate["F0"]
    eligible = [
        recipe
        for recipe in CNN_RECIPES
        if aggregate[recipe]["validation_nmse_db_mean"]
        <= base["validation_nmse_db_mean"] + CONFIG["selection_nmse_guard_db"]
        and aggregate[recipe]["validation_strong_mae_db_mean"]
        <= base["validation_strong_mae_db_mean"] + CONFIG["selection_strong_mae_guard_db"]
    ]
    selected = min(
        eligible,
        key=lambda recipe: (
            aggregate[recipe]["validation_regret_p95_mean"],
            aggregate[recipe]["validation_bottleneck_over_p95_db_mean"],
            aggregate[recipe]["validation_nmse_db_mean"],
        ),
    )
    candidate = [row for row in rows if row["recipe"] == selected]
    representative = min(
        candidate,
        key=lambda row: (
            row["validation_fourpoint"]["regret_p95"],
            row["validation"]["amplitude_nmse_db"],
        ),
    )["seed"]
    optimizer_validation = read_json(root / "validation_optimizer.json")
    allowed_optimizers = ["O0_legacy", "O1_q_eigen", "O2_coordinate"]
    selected_optimizer = min(
        allowed_optimizers,
        key=lambda name: optimizer_validation[name]["regret_p95"],
    )
    freeze = {
        "selected_cnn_recipe": selected,
        "representative_seed": representative,
        "selected_instantaneous_optimizer": selected_optimizer,
        "cnn_selection": "lowest validation four-point regret P95 under F0+0.3dB NMSE and F0+0.02dB strong-MAE guards",
        "optimizer_selection": "lowest exact-HFSS validation regret P95 among O0/O1/O2; O3 is reference budget",
        "aggregate": aggregate,
        "optimizer_validation": optimizer_validation,
        "paired_initialization_verified": paired_initialization,
        "all_runs_equal_updates": all(row["updates"] == rows[0]["updates"] for row in rows),
        "checkpoint_hashes": {
            f"{row['recipe']}_{row['seed']}": row["checkpoint_sha256"] for row in rows
        },
        "holdout_opened": False,
    }
    write_json(freeze_path, freeze)
    return freeze


def evaluate(root: Path, device: str) -> dict:
    verify_protocol(root)
    final_path = root / "final_report.json"
    if final_path.exists():
        return read_json(final_path)
    freeze = read_json(root / "cnn_selection_frozen.json")
    for key, expected in freeze["checkpoint_hashes"].items():
        recipe, seed = key.split("_")
        if digest(root / "runs" / f"{recipe}_seed{seed}" / "best.pt") != expected:
            raise ValueError(f"checkpoint changed: {key}")
    write_json(
        root / "holdout_opened.json",
        {
            "front_only": True,
            "selection": freeze,
            "purpose": "one-shot controlled same-state front-hemisphere evaluation",
        },
    )
    suite = load_npz(root / "holdout_front_suite.npz")
    optimizer_metrics, optimizer_arrays = exact_optimizer_benchmark(suite, device)
    write_json(root / "holdout_optimizer.json", optimizer_metrics)
    np.savez_compressed(root / "holdout_optimizer_arrays.npz", **optimizer_arrays)
    cnn_rows = []
    all_arrays = {}
    for recipe in CNN_RECIPES:
        for seed in SEEDS:
            checkpoint = torch.load(
                root / "runs" / f"{recipe}_seed{seed}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            model = V210VectorBaseline().to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            metrics, arrays = cnn_suite_metrics(
                model,
                checkpoint["scale"],
                suite,
                optimizer_arrays["reference_minimum"],
                optimizer_arrays["solo_gain"],
                device,
                "q_eigen",
            )
            key = f"{recipe}_{seed}"
            cnn_rows.append({"recipe": recipe, "seed": seed, **metrics})
            all_arrays.update({f"{key}_{name}": value for name, value in arrays.items() if name != "weight"})

    # Historical full-sphere-trained A10, evaluated on the same front-only suite.
    legacy_checkpoint = torch.load(
        legacy_checkpoint_path(root),
        map_location=device,
        weights_only=False,
    )
    legacy = V210VectorBaseline().to(device)
    legacy.load_state_dict(legacy_checkpoint["model"])
    legacy.eval()
    legacy_metrics, legacy_arrays = cnn_suite_metrics(
        legacy,
        legacy_checkpoint["scale"],
        suite,
        optimizer_arrays["reference_minimum"],
        optimizer_arrays["solo_gain"],
        device,
        "q_eigen",
    )
    all_arrays.update({f"legacy_{name}": value for name, value in legacy_arrays.items() if name != "weight"})

    cnn_aggregate = {}
    for recipe in CNN_RECIPES:
        subset = [row for row in cnn_rows if row["recipe"] == recipe]
        cnn_aggregate[recipe] = {
            "regret_median_mean": float(np.mean([row["regret_median"] for row in subset])),
            "regret_p95_mean": float(np.mean([row["regret_p95"] for row in subset])),
            "regret_le_10pct_mean": float(np.mean([row["regret_le_10pct"] for row in subset])),
            "fourpoint_mae_db_mean": float(np.mean([row["fourpoint_mae_db"] for row in subset])),
            "bottleneck_over_p95_db_mean": float(np.mean([row["bottleneck_over_p95_db"] for row in subset])),
            "bottleneck_identity_accuracy_mean": float(np.mean([row["bottleneck_identity_accuracy"] for row in subset])),
        }

    solo_gain = optimizer_arrays["solo_gain"]
    physical_ratio = np.min(
        np.maximum(optimizer_arrays["O3_high_budget_gain"], 1e-30)
        / np.maximum(solo_gain, 1e-30),
        axis=1,
    )
    physical = {
        "normalized_feasibility_p5_median_p95": np.percentile(physical_ratio, [5, 50, 95]).tolist(),
        "fraction_at_least_0_5": float(np.mean(physical_ratio >= 0.5)),
        "fraction_at_least_0_7": float(np.mean(physical_ratio >= 0.7)),
        "fraction_at_least_0_9": float(np.mean(physical_ratio >= 0.9)),
        "by_stratum": {},
    }
    for name in np.unique(suite["stratum"]):
        selected = suite["stratum"] == name
        physical["by_stratum"][str(name)] = {
            "p5_median": np.percentile(physical_ratio[selected], [5, 50]).tolist(),
            "fraction_at_least_0_7": float(np.mean(physical_ratio[selected] >= 0.7)),
        }

    report = {
        "domain": {
            "front_condition": "q_x > 0",
            "all_training_directions_front": True,
            "all_validation_points_front": bool(front_mask(load_npz(root / "validation_front_suite.npz")["q"]).all()),
            "all_holdout_points_front": bool(front_mask(suite["q"]).all()),
            "all_holdout_neighbours_front": bool(front_mask(suite["neighbour_q"]).all()),
            "reverse_samples_in_metrics": 0,
        },
        "selection": freeze,
        "cnn": {
            "aggregate": cnn_aggregate,
            "runs": cnn_rows,
            "legacy_full_sphere_trained_A10_on_front_suite": legacy_metrics,
        },
        "optimizer_exact_hfss": optimizer_metrics,
        "physical_feasibility": physical,
        "boundaries": [
            "Only q_x>0 is used; the horizon q_x=0 is excluded.",
            "The HFSS state is unchanged, so this is not external physical-state validation.",
            "O3 is a high-budget feasible numerical reference, not a certified global optimum.",
            "X1 time sharing is average power over multiple beam states, not instantaneous gain.",
            "Local robust neighbours come from the nearest same-split front directions, not a declared pointing-error distribution.",
        ],
    }
    write_json(final_path, report)
    np.savez_compressed(root / "cnn_comparison_arrays.npz", **all_arrays)
    return report


def selftest(root: Path, device: str) -> dict:
    protocol = verify_protocol(root)
    train = dataset(root, "train")
    validation_suite = load_npz(root / "validation_front_suite.npz")
    holdout_suite = load_npz(root / "holdout_front_suite.npz")
    assert_front(train["q"][front_mask(train["q"])], "front train subset")
    assert_front(validation_suite["q"], "validation suite")
    assert_front(validation_suite["neighbour_q"], "validation neighbours")
    assert_front(holdout_suite["q"], "holdout suite")
    assert_front(holdout_suite["neighbour_q"], "holdout neighbours")
    model = V210VectorBaseline().to(device)
    query = torch.tensor(validation_suite["q"][:4, 0], device=device)
    factor = model(query)
    command = torch.ones((2, 16), dtype=torch.complex64, device=device) / 4.0
    gain = gains_from_factor(factor, command)
    qmatrix = quadratic_form(factor)
    exact_factor = torch.tensor(validation_suite["h"][:4], device=device)
    weight, _ = optimize_softmin(
        exact_factor,
        starts=4,
        steps=4,
        seed=21901,
        initialization="q_eigen",
    )
    refined_weight, refined_minimum = coordinate_refine(exact_factor, weight, sweeps=2)
    result = {
        "front_train_count": int(protocol["front_counts"]["train"]["front_directions"]),
        "validation_cases": int(len(validation_suite["q"])),
        "holdout_cases": int(len(holdout_suite["q"])),
        "reverse_samples_used": 0,
        "model_finite": bool(torch.isfinite(gain).all()),
        "q_hermitian": bool(torch.allclose(qmatrix, qmatrix.mH, atol=1e-7)),
        "q_psd_min_eigenvalue": float(torch.linalg.eigvalsh(qmatrix).real.min().detach()),
        "phase_constraint_max_error": float((weight.abs() - 0.25).abs().max()),
        "coordinate_refine_finite": bool(torch.isfinite(refined_minimum).all()),
        "coordinate_phase_constraint_max_error": float(
            (refined_weight.abs() - 0.25).abs().max()
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    result["passed"] = (
        result["model_finite"]
        and result["q_hermitian"]
        and result["q_psd_min_eigenvalue"] >= -1e-6
        and result["phase_constraint_max_error"] <= 1e-6
        and result["coordinate_refine_finite"]
        and result["coordinate_phase_constraint_max_error"] <= 1e-6
        and result["parameter_count"] <= CONFIG["parameter_budget"]
    )
    write_json(root / "selftest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "selftest", "train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    if arguments.action in {"prepare", "all"}:
        if arguments.source_root is None or arguments.legacy_checkpoint is None:
            parser.error("prepare/all requires --source-root and --legacy-checkpoint")
        prepare(arguments.root, arguments.source_root, arguments.legacy_checkpoint)
    if arguments.action in {"selftest", "all"}:
        result = selftest(arguments.root, arguments.device)
        if not result["passed"]:
            raise SystemExit(result)
        print("SELFTEST PASSED", result, flush=True)
    if arguments.action in {"train", "all"}:
        print("CNN FROZEN", train_campaign(arguments.root, arguments.device), flush=True)
    if arguments.action in {"evaluate", "all"}:
        print("EVALUATED", evaluate(arguments.root, arguments.device), flush=True)


if __name__ == "__main__":
    main()
