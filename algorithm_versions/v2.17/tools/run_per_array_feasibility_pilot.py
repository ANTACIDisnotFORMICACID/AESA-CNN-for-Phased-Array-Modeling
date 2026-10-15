"""Run the preregistered validation-only V2.17 per-array feasibility pilot.

The pilot deliberately has no code path for blind-test data.  Every array owns
an independent model; only the physical modeling idea and evaluation protocol
are shared across arrays.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v217.per_array_models import (  # noqa: E402
    GeometryMatchedOperator,
    parameter_count,
)


ARRAYS = ("a1x4", "a1x8", "a2x4", "a4x4")
ARMS = ("control", "candidate")
SEEDS = (21721, 21722, 21723)


def reject_blind(path: Path) -> None:
    if "blind" in str(path).lower():
        raise ValueError(f"blind-test path is forbidden in this pilot: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    reject_blind(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} missing keys: {sorted(missing)}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    for key in required:
        if not np.isfinite(result[key]).all():
            raise ValueError(f"non-finite {key} in {path}")
    return result


def load_array(training_views: Path, splits: Path, array_id: str) -> dict[str, object]:
    train_path = training_views / array_id / "train_operator_cache.npz"
    validation_path = training_views / array_id / "validation_operator_cache.npz"
    command_path = splits / array_id / "validation_commands.npz"
    operator_required = {
        "direction", "g_ephi", "solid_angle_weights", "positions_m",
        "frequency_hz", "canonical_indices",
    }
    train = load_npz(train_path, operator_required)
    validation = load_npz(validation_path, operator_required)
    command = load_npz(command_path, {"weights"})
    weights = np.asarray(command["weights"], dtype=np.complex64)
    if weights.ndim != 2 or weights.shape[0] != 512:
        raise ValueError(f"{command_path} must contain 512 validation commands")
    if weights.shape[1] != train["g_ephi"].shape[1]:
        raise ValueError(f"command element count mismatch for {array_id}")
    if not np.allclose(np.sum(np.abs(weights) ** 2, axis=1), 1.0, atol=2e-6):
        raise ValueError(f"validation commands are not unit-power for {array_id}")
    return {
        "train": train,
        "validation": validation,
        "commands": weights,
        "read_receipt": [
            {"path": str(path.resolve()), "sha256": sha256(path), "role": role}
            for path, role in (
                (train_path, "train_operator_cache"),
                (validation_path, "validation_operator_cache"),
                (command_path, "validation_commands"),
            )
        ],
    }


def deterministic_subset(count: int, requested: int, array_id: str) -> np.ndarray:
    if requested <= 0 or requested >= count:
        return np.arange(count, dtype=np.int64)
    seed = 21719 + ARRAYS.index(array_id)
    return np.sort(np.random.default_rng(seed).choice(count, requested, replace=False))


def weighted_complex_nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    numerator = float(np.sum(weights[:, None] * np.abs(reference - prediction) ** 2))
    denominator = float(np.sum(weights[:, None] * np.abs(reference) ** 2))
    return float(10.0 * np.log10(max(numerator / max(denominator, 1e-30), 1e-30)))


def q_nmse_db(reference: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(reference), 1024):
        stop = min(start + 1024, len(reference))
        ref = reference[start:stop]
        pred = prediction[start:stop]
        q_ref = np.einsum("qi,qj->qij", np.conj(ref), ref)
        q_pred = np.einsum("qi,qj->qij", np.conj(pred), pred)
        local = weights[start:stop, None, None]
        numerator += float(np.sum(local * np.abs(q_ref - q_pred) ** 2))
        denominator += float(np.sum(local * np.abs(q_ref) ** 2))
    return float(10.0 * np.log10(max(numerator / max(denominator, 1e-30), 1e-30)))


def predict_operator(
    model: GeometryMatchedOperator, direction: np.ndarray, device: torch.device, chunk_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(direction), chunk_size):
            local = torch.as_tensor(direction[start:start + chunk_size], dtype=torch.float32, device=device)
            value = model.operator(local[None])[0].cpu().numpy().astype(np.complex64)
            predictions.append(value)
    return np.concatenate(predictions, axis=0)


def amplitude_metrics(reference_operator: np.ndarray, prediction_operator: np.ndarray, commands: np.ndarray) -> dict[str, object]:
    target = np.abs(reference_operator @ commands.T)
    prediction = np.abs(prediction_operator @ commands.T)
    error2 = np.square(prediction - target)
    target2 = np.square(target)
    nmse = float(10.0 * np.log10(max(float(np.sum(error2)) / max(float(np.sum(target2)), 1e-30), 1e-30)))
    quantiles = np.quantile(target, [0.25, 0.5, 0.75])
    bucket_index = np.digitize(target, quantiles, right=True)
    buckets = []
    labels = ("weakest", "low_mid", "high_mid", "strongest")
    for index, label in enumerate(labels):
        mask = bucket_index == index
        ratio = float(np.sum(error2[mask])) / max(float(np.sum(target2[mask])), 1e-30)
        buckets.append({
            "label": label,
            "count": int(np.sum(mask)),
            "nmse_db": float(10.0 * np.log10(max(ratio, 1e-30))),
            "mae": float(np.mean(np.abs(prediction[mask] - target[mask]))),
        })
    return {
        "linear_nmse_db": nmse,
        "mae": float(np.mean(np.abs(prediction - target))),
        "target_quartiles": [float(value) for value in quantiles],
        "strength_buckets": buckets,
    }


def validation_metrics(
    model: GeometryMatchedOperator,
    validation: dict[str, np.ndarray],
    commands: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, object], np.ndarray]:
    reference = np.asarray(validation["g_ephi"], dtype=np.complex64)
    prediction = predict_operator(model, validation["direction"], device, chunk_size)
    solid_angle = np.asarray(validation["solid_angle_weights"], dtype=np.float64)
    amplitude = amplitude_metrics(reference, prediction, commands)
    metrics = {
        "amplitude": amplitude,
        "complex_operator_nmse_db": weighted_complex_nmse_db(reference, prediction, solid_angle),
        "Q_nmse_db": q_nmse_db(reference, prediction, solid_angle),
        "finite": bool(np.isfinite(prediction).all()),
    }
    return metrics, prediction


def benchmark_latency(
    model: GeometryMatchedOperator, direction: np.ndarray, device: torch.device, repeats: int,
) -> dict[str, float]:
    local = torch.as_tensor(direction[: min(2048, len(direction))], dtype=torch.float32, device=device)
    model.eval()
    values = []
    with torch.no_grad():
        model.operator(local[None])
        for _ in range(repeats):
            started = time.perf_counter()
            model.operator(local[None])
            values.append(time.perf_counter() - started)
    return {
        "queries": int(len(local)),
        "median_seconds": float(np.median(values)),
        "microseconds_per_query": float(1e6 * np.median(values) / len(local)),
    }


def train_run(
    *, array_id: str, arm: str, seed: int, package: dict[str, object], output_dir: Path,
    epochs: int, batch_size: int, training_queries: int, learning_rate: float,
    late_learning_rate: float, learning_rate_drop_epoch: int, validation_interval: int,
    device: torch.device, chunk_size: int, latency_repeats: int,
) -> dict[str, object]:
    run_dir = output_dir / f"{array_id}_{arm}_seed{seed}"
    result_path = run_dir / "result.json"
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("blind_test_status") != "not_read_or_generated":
            raise ValueError(f"existing run violates blind-test seal: {run_dir}")
        print(json.dumps({"resume": "skip_complete", "array": array_id, "arm": arm, "seed": seed}), flush=True)
        return saved

    train = package["train"]
    validation = package["validation"]
    commands = package["commands"]
    subset = deterministic_subset(len(train["direction"]), training_queries, array_id)
    direction = torch.as_tensor(train["direction"][subset], dtype=torch.float32)
    target = torch.as_tensor(train["g_ephi"][subset], dtype=torch.complex64)
    weights = torch.as_tensor(train["solid_angle_weights"][subset], dtype=torch.float32)
    positions = torch.as_tensor(train["positions_m"], dtype=torch.float32)
    frequency = float(train["frequency_hz"])
    torch.manual_seed(seed)
    model = GeometryMatchedOperator(
        array_id, arm, positions_m=positions, frequency_hz=frequency,
        initialization_seed=seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_nmse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    initial_metrics, _ = validation_metrics(model, validation, commands, device, chunk_size)
    history.append({"epoch": 0, "validation_amplitude_nmse_db": initial_metrics["amplitude"]["linear_nmse_db"]})
    print(json.dumps({"start": True, "array": array_id, "arm": arm, "seed": seed, "initial_nmse_db": history[-1]["validation_amplitude_nmse_db"]}), flush=True)

    for epoch in range(1, epochs + 1):
        if epoch == learning_rate_drop_epoch:
            for group in optimizer.param_groups:
                group["lr"] = late_learning_rate
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(direction), generator=generator)
        model.train()
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            local_direction = direction[selected].to(device)
            local_target = target[selected].to(device)
            local_weight = weights[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.operator(local_direction[None])[0]
            numerator = torch.sum(local_weight[:, None] * torch.abs(prediction - local_target) ** 2)
            denominator = torch.sum(local_weight[:, None] * torch.abs(local_target) ** 2).clamp_min(1e-12)
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss for {array_id}/{arm}/{seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(selected)
            seen += len(selected)
        record: dict[str, object] = {
            "epoch": epoch,
            "train_complex_nmse_linear": loss_sum / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % validation_interval == 0 or epoch == epochs:
            metrics, _ = validation_metrics(model, validation, commands, device, chunk_size)
            current = float(metrics["amplitude"]["linear_nmse_db"])
            record["validation_amplitude_nmse_db"] = current
            if current < best_nmse:
                best_nmse = current
                best_epoch = epoch
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            print(json.dumps({"array": array_id, "arm": arm, "seed": seed, **record}), flush=True)
        history.append(record)

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    final_metrics, _ = validation_metrics(model, validation, commands, device, chunk_size)
    latency = benchmark_latency(model, validation["direction"], device, latency_repeats)
    checkpoint = {
        "schema": "v217-per-array-feasibility-checkpoint-1",
        "array_id": array_id,
        "arm": arm,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_state": best_state,
        "metadata": model.metadata(),
        "parameter_count": parameter_count(model),
        "checkpoint_selection": "validation_amplitude_linear_nmse_only",
        "blind_test_status": "not_read_or_generated",
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "schema": "v217-per-array-feasibility-result-1",
        "array_id": array_id,
        "arm": arm,
        "seed": seed,
        "trained_from_scratch": True,
        "training_query_count": int(len(subset)),
        "training_subset_seed": 21719 + ARRAYS.index(array_id),
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(model),
        "validation_metrics": final_metrics,
        "latency": latency,
        "training_seconds": time.perf_counter() - started,
        "read_receipt": package["read_receipt"],
        "validation_used_for": "checkpoint selection and frozen feasibility thresholds only",
        "blind_test_status": "not_read_or_generated",
        "model_scope": "independent per-array fit; no parameter transfer",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, **result}, ensure_ascii=False), flush=True)
    del model, optimizer
    gc.collect()
    return result


def summarise(results: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    grouped: dict[str, dict[str, dict[int, dict[str, object]]]] = {
        array_id: {arm: {} for arm in ARMS} for array_id in ARRAYS
    }
    for result in results:
        grouped[result["array_id"]][result["arm"]][int(result["seed"])] = result
    per_array = {}
    pass_count = 0
    for array_id in ARRAYS:
        control = grouped[array_id]["control"]
        candidate = grouped[array_id]["candidate"]
        if set(control) != set(SEEDS) or set(candidate) != set(SEEDS):
            per_array[array_id] = {"status": "INCOMPLETE"}
            continue
        paired = []
        for seed in SEEDS:
            c = control[seed]
            d = candidate[seed]
            c_metrics = c["validation_metrics"]
            d_metrics = d["validation_metrics"]
            c_buckets = {row["label"]: row["nmse_db"] for row in c_metrics["amplitude"]["strength_buckets"]}
            d_buckets = {row["label"]: row["nmse_db"] for row in d_metrics["amplitude"]["strength_buckets"]}
            paired.append({
                "seed": seed,
                "amplitude_improvement_db": c_metrics["amplitude"]["linear_nmse_db"] - d_metrics["amplitude"]["linear_nmse_db"],
                "complex_operator_improvement_db": c_metrics["complex_operator_nmse_db"] - d_metrics["complex_operator_nmse_db"],
                "Q_improvement_db": c_metrics["Q_nmse_db"] - d_metrics["Q_nmse_db"],
                "weakest_bucket_degradation_db": d_buckets["weakest"] - c_buckets["weakest"],
                "strongest_bucket_degradation_db": d_buckets["strongest"] - c_buckets["strongest"],
            })
        averages = {
            key: float(np.mean([row[key] for row in paired]))
            for key in (
                "amplitude_improvement_db", "complex_operator_improvement_db", "Q_improvement_db",
                "weakest_bucket_degradation_db", "strongest_bucket_degradation_db",
            )
        }
        positive_seeds = sum(row["amplitude_improvement_db"] > 0 for row in paired)
        control_parameters = control[SEEDS[0]]["parameter_count"]
        candidate_parameters = candidate[SEEDS[0]]["parameter_count"]
        parameter_delta = (candidate_parameters - control_parameters) / control_parameters
        finite = all(
            grouped[array_id][arm][seed]["validation_metrics"]["finite"]
            for arm in ARMS for seed in SEEDS
        )
        gates = {
            "mean_amplitude_improvement_at_least_0_30_db": averages["amplitude_improvement_db"] >= 0.30,
            "at_least_2_of_3_positive_seeds": positive_seeds >= 2,
            "mean_complex_operator_improvement_positive": averages["complex_operator_improvement_db"] > 0,
            "mean_Q_improvement_positive": averages["Q_improvement_db"] > 0,
            "parameter_delta_within_0_5_percent": abs(parameter_delta) <= 0.005,
            "weakest_bucket_mean_degradation_at_most_0_20_db": averages["weakest_bucket_degradation_db"] <= 0.20,
            "strongest_bucket_mean_degradation_at_most_0_20_db": averages["strongest_bucket_degradation_db"] <= 0.20,
            "all_outputs_finite": finite,
        }
        passed = all(gates.values())
        pass_count += int(passed)
        per_array[array_id] = {
            "status": "PASS" if passed else "FAIL",
            "paired_seed_results": paired,
            "three_seed_means": averages,
            "positive_seed_count": positive_seeds,
            "control_parameter_count": control_parameters,
            "candidate_parameter_count": candidate_parameters,
            "parameter_delta_fraction": parameter_delta,
            "gates": gates,
        }
    universality = "SUPPORTED_4_OF_4" if pass_count == 4 else (
        f"PARTIAL_{pass_count}_OF_4" if pass_count >= 2 else "NOT_SUPPORTED"
    )
    summary = {
        "schema": "v217-per-array-feasibility-summary-1",
        "scope": "validation-only feasibility pilot",
        "model_scope": "four independently trained geometry-matched models; shared idea only",
        "arrays": per_array,
        "arrays_passed": pass_count,
        "universality": universality,
        "thresholds_frozen_before_training": True,
        "blind_test_status": "not_read_or_generated",
        "formal_model_promotion_authority": False,
    }
    (output_dir / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-views", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arrays", nargs="+", choices=ARRAYS, default=list(ARRAYS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--training-queries", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--late-learning-rate", type=float, default=5e-4)
    parser.add_argument("--learning-rate-drop-epoch", type=int, default=31)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"only preregistered seeds {SEEDS} are permitted")
    if args.epochs <= 0 or args.validation_interval <= 0:
        raise ValueError("epochs and validation interval must be positive")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "v217-per-array-feasibility-config-1",
        "arrays": list(ARRAYS),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_queries_per_array": args.training_queries,
        "training_subset": "fixed train-only random subset; seed 21719 plus array ordinal",
        "learning_rate": args.learning_rate,
        "late_learning_rate": args.late_learning_rate,
        "learning_rate_drop_epoch": args.learning_rate_drop_epoch,
        "validation_interval": args.validation_interval,
        "checkpoint_selection": "validation_amplitude_linear_nmse_only",
        "device": str(device),
        "model_scope": "independent geometry-matched model per array; shared physical idea only",
        "blind_test_status": "no loader or output path exists",
    }
    config_path = args.output_dir / "RUN_CONFIG.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != run_config:
        raise ValueError("existing RUN_CONFIG differs from the frozen pilot protocol")
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    packages = {array_id: load_array(args.training_views, args.splits, array_id) for array_id in args.arrays}
    results = []
    for array_id in args.arrays:
        for seed in args.seeds:
            for arm in args.arms:
                results.append(train_run(
                    array_id=array_id, arm=arm, seed=seed, package=packages[array_id],
                    output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
                    training_queries=args.training_queries, learning_rate=args.learning_rate,
                    late_learning_rate=args.late_learning_rate,
                    learning_rate_drop_epoch=args.learning_rate_drop_epoch,
                    validation_interval=args.validation_interval, device=device,
                    chunk_size=args.chunk_size, latency_repeats=args.latency_repeats,
                ))
    all_results = []
    for path in sorted(args.output_dir.glob("*_seed*/result.json")):
        all_results.append(json.loads(path.read_text(encoding="utf-8")))
    summary = summarise(all_results, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
