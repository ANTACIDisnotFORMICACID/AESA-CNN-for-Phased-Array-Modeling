"""Controlled A10/B10/C10 experiment for the V2.10 quadratic-form study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from ..models.quadratic_psd import (
    build_experiment_model,
    gains_from_factor,
    quadratic_form,
)


SEEDS = (21011, 21012, 21013)
RECIPES = ("A10", "B10", "C10")
VERSION_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DATASET_ROOT = (
    VERSION_ROOT / "artifacts" / "front_hemisphere_20260906" / "datasets"
)
CONFIG = {
    "schema": "v210-controlled-psd-1",
    "seeds": list(SEEDS),
    "recipes": list(RECIPES),
    "epochs": 3600,
    "steps_per_epoch": 4,
    "batch_size": 512,
    "commands_per_step": 64,
    "validation_directions": 2048,
    "validation_every": 20,
    "learning_rate": 1e-3,
    "minimum_learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "gradient_clip_norm": 5.0,
    "q_loss_weight": 0.25,
    "trajectory_huber_weight": 0.08,
    "trajectory_cvar_weight": 0.04,
    "trajectory_cvar_fraction": 0.25,
    "gain_floor_db": -50.0,
    "optimizer_steps": 160,
    "model_starts": 8,
    "reference_starts": 32,
    "selection_nmse_guard_db": 0.3,
    "parameter_budget": 34973,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*.py"))
    }


def db(power: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(power.clamp_min(1e-5))


def objective_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    amplitude = (
        prediction.clamp_min(1e-12).sqrt()
        - target.clamp_min(1e-12).sqrt()
    ).square().mean() / target.mean().clamp_min(1e-8)
    power = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
    robust = nn.functional.huber_loss(db(prediction), db(target), delta=1.0)
    return amplitude + 0.10 * power + 0.03 * robust


def q_nmse_loss(predicted_factor: torch.Tensor, true_factor: torch.Tensor) -> torch.Tensor:
    predicted_q = quadratic_form(predicted_factor)
    true_q = quadratic_form(true_factor)
    return (predicted_q - true_q).abs().square().sum() / true_q.abs().square().sum().clamp_min(1e-12)


def dataset_path(root: Path, name: str) -> Path:
    folder = "dev" if name in {"train", "validation"} else "reserved"
    local = root / "datasets" / folder / f"{name}.npz"
    canonical = CANONICAL_DATASET_ROOT / folder / f"{name}.npz"
    if local.is_file():
        return local
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(f"dataset not found at {local} or {canonical}")


def load_dataset(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(dataset_path(root, name), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prepare(root: Path, source_root: Path) -> dict:
    """Copy the frozen V2.9 split once and record byte-level provenance."""

    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        protocol = _read_json(protocol_path)
        if protocol["config"] != CONFIG:
            raise ValueError("existing protocol differs from the current fixed configuration")
        return protocol
    copied = {}
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
            if _digest(source) != _digest(destination):
                raise ValueError(f"canonical dataset differs from source: {destination}")
        else:
            shutil.copy2(source, destination)
        copied[str(destination.relative_to(VERSION_ROOT))] = {
            "sha256": _digest(destination),
            "bytes": destination.stat().st_size,
            "source": str(source),
        }
    counts = {
        recipe: sum(parameter.numel() for parameter in build_experiment_model(recipe).parameters())
        for recipe in RECIPES
    }
    if max(counts.values()) > CONFIG["parameter_budget"]:
        raise ValueError("model exceeds the frozen V2.9 parameter budget")
    source_protocol = _read_json(source_root / "protocol.json")
    protocol = {
        "config": CONFIG,
        "dataset_files": copied,
        "source_code_hashes": _source_hashes(),
        "parameter_counts": counts,
        "v29_scale": float(source_protocol["scale"]),
        "v29_protocol_source": str(source_root / "protocol.json"),
        "v29_protocol_sha256": _digest(source_root / "protocol.json"),
        "data_policy": (
            "Exact byte copies of the already-opened V2.9 fixed-state splits. "
            "Training reads train/validation only; calibration/holdout are opened "
            "after checkpoint freeze. This is controlled retrospective comparison, "
            "not a new physical-state blind test."
        ),
        "model_contract": {
            "A10": "migrated V2.9 Cartesian transverse equivalent-field model",
            "B10": "rank-2 L(q), Q=L^H L/60, command loss plus Q loss",
            "C10": "B10 plus training-trajectory Huber and optimistic-tail CVaR surrogate",
        },
    }
    _write_json(protocol_path, protocol)
    return protocol


def verify_protocol(root: Path) -> dict:
    protocol = _read_json(root / "protocol.json")
    if protocol["config"] != CONFIG:
        raise ValueError("configuration drift after protocol creation")
    if not protocol.get("source_code_hashes"):
        raise ValueError("frozen protocol is missing source provenance")
    for relative, metadata in protocol["dataset_files"].items():
        name = Path(relative).stem
        if _digest(dataset_path(root, name)) != metadata["sha256"]:
            raise ValueError(f"dataset changed: {relative}")
    return protocol


@torch.no_grad()
def predict_factor(model: nn.Module, query: torch.Tensor, scale: float, batch: int = 1024) -> torch.Tensor:
    return torch.cat(
        [model(query[index : index + batch]) * scale for index in range(0, len(query), batch)]
    )


def optimize_phase(
    factor: torch.Tensor,
    *,
    starts: int,
    steps: int,
    seed: int,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-start feasible phase-only soft-min search for ``[B,4,R,16]``."""

    factor = factor.detach()
    generator = torch.Generator(device=factor.device).manual_seed(seed)
    batch = len(factor)
    z = torch.randn((batch, starts, 16), device=factor.device, generator=generator)
    z = z + 1j * torch.randn((batch, starts, 16), device=factor.device, generator=generator)
    matched = factor[:, :, 0, :].conj()
    z[:, : min(4, starts)] = matched[:, : min(4, starts)]
    if initial is not None:
        z[:, -1] = initial.detach()
    phase = nn.Parameter(torch.angle(z))
    optimizer = torch.optim.Adam([phase], lr=0.065)
    best = torch.full((batch,), -1e30, device=factor.device)
    best_weight = torch.zeros((batch, 16), dtype=torch.complex64, device=factor.device)
    for step in range(steps + 1):
        weight = torch.exp(1j * phase) / 4.0
        field = torch.einsum("bkrn,bsn->bskr", factor, weight)
        gain_db = db(field.abs().square().sum(-1) / 60.0)
        minimum = gain_db.min(-1).values
        value, index = minimum.max(1)
        improved = value > best
        best = torch.maximum(best, value)
        best_weight[improved] = weight.detach()[torch.arange(batch, device=factor.device), index][improved]
        if step == steps:
            break
        temperature = 1.2 * (1.0 - step / steps) + 0.12
        loss = (temperature * torch.logsumexp(-gain_db / temperature, dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return best_weight.detach(), best.detach()


def forward_metrics(predicted: np.ndarray, target: np.ndarray, theta: np.ndarray) -> dict:
    error = 10.0 * np.log10(np.maximum(predicted, 1e-30)) - 10.0 * np.log10(np.maximum(target, 1e-30))
    relative = 10.0 * np.log10(np.maximum(target / target.max(1, keepdims=True), 1e-30))
    solid = np.sin(np.deg2rad(theta))[None, :] * np.ones_like(error)

    def statistic(mask: np.ndarray) -> dict:
        values = error[mask]
        weights = solid[mask]
        return {
            "count": int(mask.sum()),
            "mae_db": float(np.mean(np.abs(values))),
            "bias_db": float(np.mean(values)),
            "p95_abs_db": float(np.percentile(np.abs(values), 95)),
            "solid_angle_mae_db": float(np.sum(np.abs(values) * weights) / weights.sum()),
        }

    return {
        "amplitude_nmse_db": float(10.0 * np.log10(np.sum((np.sqrt(predicted) - np.sqrt(target)) ** 2) / np.sum(target))),
        "power_nmse_db": float(10.0 * np.log10(np.sum((predicted - target) ** 2) / np.sum(target ** 2))),
        "bins": {
            "all": statistic(np.ones_like(error, dtype=bool)),
            "strong": statistic(relative >= -10.0),
            "medium": statistic((relative < -10.0) & (relative >= -20.0)),
            "weak": statistic((relative < -20.0) & (relative >= -30.0)),
            "deep": statistic(relative < -30.0),
        },
    }


def _validation_metrics(model: nn.Module, data: dict, scale: float, device: str) -> dict:
    rng = np.random.default_rng(21090)
    indices = rng.choice(len(data["q"]), CONFIG["validation_directions"], replace=False)
    query = torch.tensor(data["q"][indices], device=device)
    truth_factor = torch.tensor(data["h"][indices], device=device)
    command = torch.tensor(data["commands"], device=device)
    predicted_factor = predict_factor(model, query, scale)
    predicted = gains_from_factor(predicted_factor, command).cpu().numpy()
    target = gains_from_factor(truth_factor, command).cpu().numpy()
    result = forward_metrics(predicted, target, data["theta"][indices])
    predicted_q = quadratic_form(predicted_factor)
    true_q = quadratic_form(truth_factor)
    result["q_nmse_db"] = float(
        10.0 * torch.log10(
            (predicted_q - true_q).abs().square().sum()
            / true_q.abs().square().sum().clamp_min(1e-12)
        ).cpu()
    )
    return result


def train_one(root: Path, recipe: str, seed: int, device: str = "cuda") -> dict:
    if (root / "selection_frozen.json").exists() or (root / "holdout_opened.json").exists():
        raise RuntimeError("training is closed after checkpoint freeze")
    protocol = verify_protocol(root)
    output = root / "runs" / f"{recipe}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        return _read_json(result_path)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_experiment_model(recipe).to(device)
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
    train = load_dataset(root, "train")
    validation = load_dataset(root, "validation")
    query = torch.tensor(train["q"], device=device)
    true_factor = torch.tensor(train["h"], device=device)
    command = torch.tensor(train["commands"], device=device)
    scale = float(protocol["v29_scale"])
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    trajectory_query = torch.tensor(train["trajectory_q"].reshape(-1, 4, 3), device=device)
    trajectory_factor = torch.tensor(train["trajectory_h"].reshape(-1, 4, 3, 16), device=device)
    trajectory_weight = None
    if recipe == "C10":
        trajectory_weight, _ = optimize_phase(
            trajectory_factor, starts=CONFIG["model_starts"], steps=120, seed=seed + 2000
        )

    best_nmse = math.inf
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total = 0.0
        components = {"command": 0.0, "q": 0.0, "trajectory": 0.0, "cvar": 0.0}
        for _ in range(CONFIG["steps_per_epoch"]):
            query_index = torch.randint(
                len(query), (CONFIG["batch_size"],), generator=generator, device=device
            )
            command_index = torch.randint(
                len(command), (CONFIG["commands_per_step"],), generator=generator, device=device
            )
            prediction_factor = model(query[query_index]) * scale
            target_factor = true_factor[query_index]
            prediction_gain = gains_from_factor(prediction_factor, command[command_index])
            target_gain = gains_from_factor(target_factor, command[command_index])
            command_loss = objective_loss(prediction_gain, target_gain)
            loss = command_loss
            q_loss = torch.zeros((), device=device)
            trajectory_loss = torch.zeros((), device=device)
            cvar_loss = torch.zeros((), device=device)
            if recipe in {"B10", "C10"}:
                q_loss = q_nmse_loss(prediction_factor, target_factor)
                loss = loss + CONFIG["q_loss_weight"] * q_loss
            if recipe == "C10":
                trajectory_index = torch.randint(
                    len(trajectory_query), (16,), generator=generator, device=device
                )
                predicted_trajectory = model(
                    trajectory_query[trajectory_index].reshape(-1, 3)
                ).reshape(-1, 4, 2, 16) * scale
                weight = trajectory_weight[trajectory_index]
                predicted_hard = torch.einsum(
                    "bkrn,bn->bkr", predicted_trajectory, weight
                ).abs().square().sum(-1) / 60.0
                target_hard = torch.einsum(
                    "bkrn,bn->bkr", trajectory_factor[trajectory_index], weight
                ).abs().square().sum(-1) / 60.0
                error = db(predicted_hard) - db(target_hard)
                trajectory_loss = nn.functional.huber_loss(
                    db(predicted_hard), db(target_hard), delta=1.0
                )
                optimistic = torch.relu(error - 0.1).flatten()
                count = max(1, math.ceil(CONFIG["trajectory_cvar_fraction"] * optimistic.numel()))
                cvar_loss = torch.topk(optimistic, count).values.mean()
                loss = (
                    loss
                    + CONFIG["trajectory_huber_weight"] * trajectory_loss
                    + CONFIG["trajectory_cvar_weight"] * cvar_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss)
            components["command"] += float(command_loss)
            components["q"] += float(q_loss)
            components["trajectory"] += float(trajectory_loss)
            components["cvar"] += float(cvar_loss)

        if epoch % CONFIG["validation_every"] != 0:
            continue
        model.eval()
        metrics = _validation_metrics(model, validation, scale, device)
        score = metrics["amplitude_nmse_db"]
        scheduler.step(score)
        if score < best_nmse:
            best_nmse = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "recipe": recipe,
                    "seed": seed,
                    "epoch": epoch,
                    "scale": scale,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                },
                output / "best.pt",
            )
        history.append(
            {
                "epoch": epoch,
                "validation": metrics,
                "best_amplitude_nmse_db": best_nmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": total / CONFIG["steps_per_epoch"],
                "components": {
                    key: value / CONFIG["steps_per_epoch"] for key, value in components.items()
                },
            }
        )
        if epoch % 200 == 0 or epoch == CONFIG["epochs"]:
            _write_json(output / "history.json", history)
            _write_json(
                root / "progress.json",
                {
                    "recipe": recipe,
                    "seed": seed,
                    "epoch": epoch,
                    "best_amplitude_nmse_db": best_nmse,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(
                recipe,
                seed,
                epoch,
                f"best={best_nmse:.4f}",
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    validation_metrics = _validation_metrics(model, validation, scale, device)
    result = {
        "recipe": recipe,
        "seed": seed,
        "epochs": CONFIG["epochs"],
        "updates": CONFIG["epochs"] * CONFIG["steps_per_epoch"],
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.perf_counter() - started,
        "checkpoint_sha256": _digest(output / "best.pt"),
        "holdout_loaded": False,
    }
    _write_json(result_path, result)
    return result


def trajectory_metrics(
    model: nn.Module,
    data: dict,
    scale: float,
    *,
    device: str,
    reference_weight: torch.Tensor | None = None,
    reference_minimum: torch.Tensor | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    query = torch.tensor(data["trajectory_q"].reshape(-1, 4, 3), device=device)
    true_factor = torch.tensor(data["trajectory_h"].reshape(-1, 4, 3, 16), device=device)
    predicted_factor = predict_factor(model, query.reshape(-1, 3), scale).reshape(
        len(query), 4, -1, 16
    )
    weight, predicted_minimum = optimize_phase(
        predicted_factor,
        starts=CONFIG["model_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21101,
    )
    true_gain = torch.einsum("bkrn,bn->bkr", true_factor, weight).abs().square().sum(-1) / 60.0
    predicted_gain = torch.einsum("bkrn,bn->bkr", predicted_factor, weight).abs().square().sum(-1) / 60.0
    if reference_weight is None or reference_minimum is None:
        reference_weight, _ = optimize_phase(
            true_factor,
            starts=CONFIG["reference_starts"],
            steps=CONFIG["optimizer_steps"],
            seed=21101,
        )
        reference_gain = torch.einsum(
            "bkrn,bn->bkr", true_factor, reference_weight
        ).abs().square().sum(-1) / 60.0
        reference_minimum = reference_gain.min(-1).values
    true_minimum = true_gain.min(-1).values
    feasible_reference = torch.maximum(reference_minimum, true_minimum)
    regret = (1.0 - true_minimum / feasible_reference).clamp_min(0.0)
    error = db(predicted_gain) - db(true_gain)
    bottleneck = true_gain.argmin(-1)
    bottleneck_error = error[torch.arange(len(error), device=device), bottleneck]
    arrays = {
        "regret": regret.cpu().numpy(),
        "error_db": error.cpu().numpy(),
        "bottleneck_error_db": bottleneck_error.cpu().numpy(),
        "true_min_gain_db": db(true_minimum).cpu().numpy(),
        "predicted_min_gain_db": predicted_minimum.cpu().numpy(),
        "weights": weight.cpu().numpy(),
        "reference_weights": reference_weight.cpu().numpy(),
        "reference_minimum": reference_minimum.cpu().numpy(),
    }
    metrics = {
        "cases": int(len(regret)),
        "median_regret": float(torch.quantile(regret, 0.50)),
        "p95_regret": float(torch.quantile(regret, 0.95)),
        "success_regret_le_10pct": float((regret <= 0.10).float().mean()),
        "fourpoint_mae_db": float(error.abs().mean()),
        "bottleneck_bias_db": float(bottleneck_error.mean()),
        "bottleneck_over_p95_db": float(torch.quantile(bottleneck_error, 0.95)),
        "bottleneck_identity_accuracy": float(
            (predicted_gain.argmin(-1) == bottleneck).float().mean()
        ),
        "true_min_gain_db_quantiles": [
            float(torch.quantile(db(true_minimum), quantile))
            for quantile in (0.05, 0.50, 0.95)
        ],
        "constraint_max_magnitude_error": float((weight.abs() - 0.25).abs().max()),
    }
    return metrics, arrays


def _validation_reference(root: Path, data: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / "validation_reference.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            return (
                torch.tensor(stored["weights"], device=device),
                torch.tensor(stored["minimum"], device=device),
            )
    true_factor = torch.tensor(data["trajectory_h"].reshape(-1, 4, 3, 16), device=device)
    weight, _ = optimize_phase(
        true_factor,
        starts=CONFIG["reference_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21101,
    )
    gain = torch.einsum("bkrn,bn->bkr", true_factor, weight).abs().square().sum(-1) / 60.0
    minimum = gain.min(-1).values
    np.savez_compressed(path, weights=weight.cpu().numpy(), minimum=minimum.cpu().numpy())
    return weight, minimum


def validation_diagnostic(root: Path, recipe: str, seed: int, device: str) -> dict:
    path = root / "runs" / f"{recipe}_seed{seed}" / "validation_trajectory.json"
    if path.exists():
        return _read_json(path)
    data = load_dataset(root, "validation")
    reference_weight, reference_minimum = _validation_reference(root, data, device)
    checkpoint = torch.load(
        root / "runs" / f"{recipe}_seed{seed}" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model = build_experiment_model(recipe).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, _ = trajectory_metrics(
        model,
        data,
        checkpoint["scale"],
        device=device,
        reference_weight=reference_weight,
        reference_minimum=reference_minimum,
    )
    _write_json(path, metrics)
    return metrics


def train_campaign(root: Path, device: str = "cuda") -> dict:
    verify_protocol(root)
    if (root / "selection_frozen.json").exists():
        return _read_json(root / "selection_frozen.json")
    results = []
    for recipe in RECIPES:
        for seed in SEEDS:
            row = train_one(root, recipe, seed, device)
            row["validation_trajectory"] = validation_diagnostic(root, recipe, seed, device)
            results.append(row)
            _write_json(root / "development_results.json", results)

    aggregate = {}
    for recipe in RECIPES:
        rows = [row for row in results if row["recipe"] == recipe]
        aggregate[recipe] = {
            "mean_validation_nmse_db": float(np.mean([row["validation"]["amplitude_nmse_db"] for row in rows])),
            "mean_validation_p95_regret": float(np.mean([row["validation_trajectory"]["p95_regret"] for row in rows])),
            "mean_validation_bottleneck_over_p95_db": float(np.mean([row["validation_trajectory"]["bottleneck_over_p95_db"] for row in rows])),
        }
    guard = aggregate["A10"]["mean_validation_nmse_db"] + CONFIG["selection_nmse_guard_db"]
    eligible = [
        recipe for recipe in RECIPES if aggregate[recipe]["mean_validation_nmse_db"] <= guard
    ]
    selected = min(
        eligible,
        key=lambda recipe: (
            aggregate[recipe]["mean_validation_p95_regret"],
            aggregate[recipe]["mean_validation_nmse_db"],
        ),
    )
    selected_rows = [row for row in results if row["recipe"] == selected]
    representative = min(
        selected_rows,
        key=lambda row: (
            row["validation_trajectory"]["p95_regret"],
            row["validation"]["amplitude_nmse_db"],
        ),
    )["seed"]
    freeze = {
        "selected_recipe": selected,
        "representative_seed": representative,
        "selection_rule": "lowest mean validation phase-only regret P95 within A10+0.3 dB amplitude-NMSE guard",
        "aggregate": aggregate,
        "seeds": list(SEEDS),
        "all_runs_equal_updates": all(row["updates"] == results[0]["updates"] for row in results),
        "checkpoint_hashes": {
            f"{row['recipe']}_{row['seed']}": row["checkpoint_sha256"] for row in results
        },
        "holdout_opened": False,
    }
    _write_json(root / "selection_frozen.json", freeze)
    return freeze


def _paired_bootstrap(a: np.ndarray, b: np.ndarray, statistic, seed: int = 21099) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        index = rng.integers(0, len(a), len(a))
        values.append(float(statistic(a[index]) - statistic(b[index])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(root: Path, device: str = "cuda") -> dict:
    verify_protocol(root)
    freeze = _read_json(root / "selection_frozen.json")
    report_path = root / "final_report.json"
    if report_path.exists():
        return _read_json(report_path)
    for key, expected in freeze["checkpoint_hashes"].items():
        recipe, seed = key.split("_")
        if _digest(root / "runs" / f"{recipe}_seed{seed}" / "best.pt") != expected:
            raise ValueError(f"checkpoint changed after freeze: {key}")
    _write_json(
        root / "holdout_opened.json",
        {
            "selection": freeze["selected_recipe"],
            "purpose": "one-shot retrospective fixed-state comparison; training permanently closed",
        },
    )
    data = load_dataset(root, "holdout")
    query = torch.tensor(data["q"], device=device)
    true_factor = torch.tensor(data["h"], device=device)
    command = torch.tensor(data["commands"], device=device)
    target_gain = gains_from_factor(true_factor, command).cpu().numpy()

    true_trajectory = torch.tensor(data["trajectory_h"].reshape(-1, 4, 3, 16), device=device)
    reference_weight, _ = optimize_phase(
        true_trajectory,
        starts=CONFIG["reference_starts"],
        steps=CONFIG["optimizer_steps"],
        seed=21101,
    )
    reference_gain = torch.einsum(
        "bkrn,bn->bkr", true_trajectory, reference_weight
    ).abs().square().sum(-1) / 60.0
    reference_minimum = reference_gain.min(-1).values

    rows = []
    stored_arrays = {}
    for recipe in RECIPES:
        for seed in SEEDS:
            checkpoint_path = root / "runs" / f"{recipe}_seed{seed}" / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = build_experiment_model(recipe).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            predicted_factor = predict_factor(model, query, checkpoint["scale"])
            predicted_gain = gains_from_factor(predicted_factor, command).cpu().numpy()
            forward = forward_metrics(predicted_gain, target_gain, data["theta"])
            predicted_q = quadratic_form(predicted_factor)
            true_q = quadratic_form(true_factor)
            forward["q_nmse_db"] = float(
                10.0 * torch.log10(
                    (predicted_q - true_q).abs().square().sum()
                    / true_q.abs().square().sum().clamp_min(1e-12)
                ).cpu()
            )
            trajectory, arrays = trajectory_metrics(
                model,
                data,
                checkpoint["scale"],
                device=device,
                reference_weight=reference_weight,
                reference_minimum=reference_minimum,
            )
            key = f"{recipe}_{seed}"
            stored_arrays[key] = arrays
            rows.append(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "forward": forward,
                    "trajectory": trajectory,
                    "checkpoint_sha256": _digest(checkpoint_path),
                }
            )

    aggregate = {}
    for recipe in RECIPES:
        recipe_rows = [row for row in rows if row["recipe"] == recipe]
        aggregate[recipe] = {
            "amplitude_nmse_db_mean": float(np.mean([row["forward"]["amplitude_nmse_db"] for row in recipe_rows])),
            "amplitude_nmse_db_seed_std": float(np.std([row["forward"]["amplitude_nmse_db"] for row in recipe_rows], ddof=1)),
            "q_nmse_db_mean": float(np.mean([row["forward"]["q_nmse_db"] for row in recipe_rows])),
            "all_mae_db_mean": float(np.mean([row["forward"]["bins"]["all"]["mae_db"] for row in recipe_rows])),
            "strong_mae_db_mean": float(np.mean([row["forward"]["bins"]["strong"]["mae_db"] for row in recipe_rows])),
            "regret_median_mean": float(np.mean([row["trajectory"]["median_regret"] for row in recipe_rows])),
            "regret_p95_mean": float(np.mean([row["trajectory"]["p95_regret"] for row in recipe_rows])),
            "bottleneck_over_p95_db_mean": float(np.mean([row["trajectory"]["bottleneck_over_p95_db"] for row in recipe_rows])),
            "success_regret_le_10pct_mean": float(np.mean([row["trajectory"]["success_regret_le_10pct"] for row in recipe_rows])),
        }
    representative_seed = freeze["representative_seed"]
    selected_key = f"{freeze['selected_recipe']}_{representative_seed}"
    baseline_key = f"A10_{representative_seed}"
    selected_regret = stored_arrays[selected_key]["regret"]
    baseline_regret = stored_arrays[baseline_key]["regret"]
    comparison = {
        "selected_minus_A10_p95_regret_bootstrap95": _paired_bootstrap(
            selected_regret, baseline_regret, lambda value: np.percentile(value, 95)
        ),
        "selected_minus_A10_success_rate_bootstrap95": _paired_bootstrap(
            (selected_regret <= 0.10).astype(float),
            (baseline_regret <= 0.10).astype(float),
            np.mean,
            seed=21100,
        ),
        "bootstrap_units": "360 paired synthetic holdout trajectory frames for the frozen representative seed",
    }
    selected_summary = aggregate[freeze["selected_recipe"]]
    acceptance = {
        "strong_mae_le_0_25db": selected_summary["strong_mae_db_mean"] <= 0.25,
        "bottleneck_over_p95_le_0_5db": selected_summary["bottleneck_over_p95_db_mean"] <= 0.5,
        "seed_nmse_std_le_0_25db": selected_summary["amplitude_nmse_db_seed_std"] <= 0.25,
        "phase_regret_p95_le_10pct": selected_summary["regret_p95_mean"] <= 0.10,
    }
    report = {
        "selection": freeze,
        "aggregate": aggregate,
        "runs": rows,
        "paired_comparison": comparison,
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "boundaries": [
            "All four splits are byte-identical copies of the already-opened V2.9 one-state HFSS campaign.",
            "No new frequency, tolerance, temperature, radome, platform, manufacturing, or OTA state was available.",
            "Holdout was isolated from V2.10 training code but is not a historically unseen blind set.",
            "Link-margin evaluation is deferred because orbit, station, receiver, noise, and SNR contracts are absent.",
            "Multi-start references are feasible numerical lower bounds, not certified global optima.",
        ],
    }
    _write_json(report_path, report)
    np.savez_compressed(
        root / "comparison_arrays.npz",
        **{
            f"{key}_{name}": value
            for key, arrays in stored_arrays.items()
            for name, value in arrays.items()
            if name not in {"weights", "reference_weights"}
        },
    )
    return report


def selftest(root: Path, device: str = "cuda") -> dict:
    data = load_dataset(root, "train")
    query = torch.tensor(data["q"][:32], device=device)
    command = torch.tensor(data["commands"][:8], device=device)
    checks = {}
    for recipe in RECIPES:
        model = build_experiment_model(recipe).to(device)
        factor = model(query)
        gain = gains_from_factor(factor, command)
        q_matrix = quadratic_form(factor)
        quadratic = torch.einsum(
            "cn,qnm,cm->cq", command.conj(), q_matrix, command
        ).real / command.abs().square().sum(-1)[:, None]
        loss = objective_loss(gain, torch.ones_like(gain))
        loss.backward()
        checks[recipe] = {
            "finite": bool(torch.isfinite(gain).all()),
            "common_phase_invariance": bool(torch.allclose(gain, gains_from_factor(factor, 1j * command), atol=1e-7)),
            "quadratic_equivalence": bool(torch.allclose(gain, quadratic, atol=1e-6)),
            "nonzero_gradient": bool(model.branch.head.weight.grad.norm() > 0),
            "q_is_hermitian": bool(torch.allclose(q_matrix, q_matrix.mH, atol=1e-7)),
            "q_min_eigenvalue": float(torch.linalg.eigvalsh(q_matrix).real.min()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    weight, _ = optimize_phase(
        torch.tensor(data["trajectory_h"][:2, 0], device=device),
        starts=4,
        steps=4,
        seed=21001,
    )
    checks["phase_constraint_max_error"] = float((weight.abs() - 0.25).abs().max())
    checks["passed"] = all(
        row["finite"]
        and row["common_phase_invariance"]
        and row["quadratic_equivalence"]
        and row["nonzero_gradient"]
        and row["q_is_hermitian"]
        and row["q_min_eigenvalue"] >= -1e-6
        and row["parameter_count"] <= CONFIG["parameter_budget"]
        for row in (checks[recipe] for recipe in RECIPES)
    ) and checks["phase_constraint_max_error"] <= 1e-6
    _write_json(root / "selftest.json", checks)
    return checks
