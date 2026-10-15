"""Reproducible GPU training with dev-only inputs and complete restart state."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..models import model_from_config
from ..training.amplitude import amplitude_only_loss, balanced_command_family_weights
from .protocol import MODEL_CONFIG, PROTOCOL, assert_protocol, digest, read_json, write_json


def atomic_torch_save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def window_improvement(history: list[dict], window: int) -> float:
    if len(history) <= window:
        return float("inf")
    return history[-window-1]["best_nmse_db"] - history[-1]["best_nmse_db"]


def converged(history: list[dict], minimum_epoch: int = 400, window: int = 150,
              threshold_db: float = 0.05) -> bool:
    return len(history) >= minimum_epoch and window_improvement(history, window) < threshold_db


def metrics(prediction: torch.Tensor, target: torch.Tensor, weights=None) -> dict:
    prediction, target = prediction.float().cpu(), target.float().cpu()
    losses = amplitude_only_loss(prediction, target, noise_floor_amplitude=0,
                                 command_weights=weights, db_loss_weight=0.03, power_loss_weight=0.10)
    nmse = (prediction-target).square().sum() / target.square().sum().clamp_min(1e-12)
    power_nmse = (prediction.square()-target.square()).square().sum() / target.pow(4).sum().clamp_min(1e-12)
    p = prediction - prediction.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    corr = (p*t).sum(dim=1) / (p.norm(dim=1)*t.norm(dim=1)).clamp_min(1e-12)
    return {
        "global_amplitude_nmse_db": float(10*torch.log10(nmse.clamp_min(1e-20))),
        "global_power_nmse_db": float(10*torch.log10(power_nmse.clamp_min(1e-20))),
        "amplitude_db_mae": float(losses.amplitude_db_mae),
        "correlation": float(corr.mean()),
        "absolute_error_p95": float(torch.quantile((prediction-target).abs(), 0.95)),
        "validation_loss": float(losses.total),
    }


@torch.no_grad()
def predict(model, commands: torch.Tensor, directions: torch.Tensor, batch: int = 2048):
    model.eval()
    device = next(model.parameters()).device
    commands = commands.to(device)
    result = []
    for start in range(0, len(directions), batch):
        result.append(model.forward_command_grid(commands, directions[start:start+batch].to(device)).cpu())
    return torch.cat(result, dim=1)


def run_training(root: Path, phase: str, recipe: str, seed: int, epochs: int,
                 device: str = "cuda", *, test_options: dict | None = None) -> dict:
    """Only dev/train.npz and dev/validation.npz are loaded by the trainer."""
    options = {**PROTOCOL, **(test_options or {})}
    if test_options is None:
        assert_protocol(root)
        if epochs != PROTOCOL["rounds"][phase]:
            raise ValueError("epoch cap differs from approved phase budget")
        if seed not in PROTOCOL["model_seeds"] or recipe not in PROTOCOL["recipes"]:
            raise ValueError("unapproved seed/recipe")
        if (root / "blind_opened.json").exists() or (root / "frozen_selection.json").exists():
            raise RuntimeError("training is permanently closed after final freeze/blind opening")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("approved campaign requires CUDA")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed + 10000)
    run_dir = root / "runs" / phase / f"{recipe}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "result.json"
    protocol_hash = digest(root / "protocol.json")
    if done_path.exists():
        result = read_json(done_path)
        if result["protocol_sha256"] != protocol_hash or digest(run_dir/"best.pt") != result["checkpoint_sha256"]:
            raise ValueError("completed run was modified")
        return result
    arrays = {}
    for name in ("train", "validation"):
        with np.load(root / "dev" / (name+".npz"), allow_pickle=False) as archive:
            arrays[name] = {k: archive[k] for k in ("direction", "amplitude", "command_iq", "family")}
    training, validation = arrays["train"], arrays["validation"]
    q = torch.tensor(training["direction"], dtype=torch.float32, device=device)
    w = torch.tensor(training["command_iq"], dtype=torch.float32, device=device)
    y = torch.tensor(training["amplitude"], dtype=torch.float32, device=device)
    vq = torch.tensor(validation["direction"], dtype=torch.float32, device=device)
    vw = torch.tensor(validation["command_iq"], dtype=torch.float32, device=device)
    vy = torch.tensor(validation["amplitude"], dtype=torch.float32)
    weights = balanced_command_family_weights(training["family"]).to(device)
    vweights = balanced_command_family_weights(validation["family"])
    model = model_from_config(MODEL_CONFIG).to(device)
    if sum(p.numel() for p in model.parameters()) != 34973:
        raise ValueError("frozen CNN structure changed")
    lr = options["recipes"][recipe]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=options["weight_decay"])
    schedule = options["recipes"][recipe]["schedule"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=options["scheduler_factor"],
        patience=options["scheduler_patience"], threshold=1e-4, threshold_mode="abs",
        min_lr=options["minimum_learning_rate"])
    history = []
    best_score, best_epoch = float("inf"), 0
    best_state = None
    elapsed_before = 0.0
    latest = run_dir / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["protocol_sha256"] != protocol_hash or checkpoint["epochs_cap"] != epochs:
            raise ValueError("resume protocol/budget mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint["history"]
        best_state = checkpoint["best_model_state"]
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        generator.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        elapsed_before = checkpoint["elapsed_seconds"]
    started = time.perf_counter()
    train_loss_config = {k: options["loss"][k] for k in ("db_loss_weight", "power_loss_weight")}
    regularization_config = {k: options["loss"][k] for k in (
        "residual_energy_weight", "coupling_l2_weight", "reciprocity_weight")}
    print(f"START {phase}/{recipe}/{seed} cap={epochs} resume={len(history)}", flush=True)

    def pack() -> dict:
        return {
            "schema_version": "v2.9-convergence-checkpoint-1",
            "model_config": MODEL_CONFIG, "model_state": cpu_state(model),
            "best_model_state": best_state, "best_score": best_score, "best_epoch": best_epoch,
            "history": history, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "sampler_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "seed": seed, "sampling_seed": seed+10000, "phase": phase, "recipe": recipe,
            "epochs_cap": epochs, "protocol_sha256": protocol_hash,
            "elapsed_seconds": elapsed_before + time.perf_counter()-started,
            "input_contract": {"measured_phase_used": False, "s_parameters_used": False,
                               "blind_labels_used": False, "observable": "Ephi amplitude",
                               "complex_convention": "transmit_h_transpose_w"},
        }

    stopped = False
    for epoch in range(len(history)+1, epochs+1):
        model.train()
        if schedule == "cosine":
            warm = min(options["warmup_epochs"], max(1, epochs-1))
            if epoch <= warm:
                current_lr = lr*(0.1+0.9*epoch/warm)
            else:
                fraction = (epoch-warm)/(epochs-warm)
                current_lr = options["minimum_learning_rate"] + (lr-options["minimum_learning_rate"])*(1+math.cos(math.pi*fraction))/2
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        indices = torch.randperm(len(q), generator=generator)[:options["queries_per_epoch"]]
        total_loss = 0.0
        samples = 0
        for start in range(0, len(indices), options["batch_size"]):
            ids = indices[start:start+options["batch_size"]].to(device)
            prediction, terms = model.forward_command_grid_with_terms(w, q[ids])
            loss = amplitude_only_loss(prediction, y[:, ids], noise_floor_amplitude=0,
                                       command_weights=weights, **train_loss_config).total
            reg, _ = model.structured_regularization(terms, **regularization_config)
            objective = loss + reg
            if not torch.isfinite(objective):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(objective.detach())*len(ids)
            samples += len(ids)
        measured = metrics(predict(model, vw, vq, options["validation_batch_size"]), vy, vweights)
        score = measured["global_amplitude_nmse_db"]
        if not math.isfinite(score):
            raise FloatingPointError("non-finite validation metric")
        if schedule == "plateau":
            scheduler.step(score)
        if score < best_score:
            best_score, best_epoch, best_state = score, epoch, cpu_state(model)
            atomic_torch_save({"model_config": MODEL_CONFIG, "model_state": best_state,
                               "epoch": epoch, "validation": measured, "protocol_sha256": protocol_hash,
                               "seed": seed, "recipe": recipe}, run_dir/"best.pt")
        history.append({"epoch": epoch, "train_loss": total_loss/samples,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "best_nmse_db": best_score, **measured})
        stopped = converged(history, options["minimum_stop_epoch"], options["plateau_window"], options["plateau_db"])
        if epoch % options["checkpoint_every"] == 0 or stopped or epoch == epochs:
            state = pack()
            atomic_torch_save(state, latest)
            if epoch == 200:
                atomic_torch_save(state, run_dir/"epoch200.pt")
            write_json(run_dir/"history.json", history)
            write_json(root/"progress.json", {
                "phase": phase, "recipe": recipe, "seed": seed, "epoch": epoch, "cap": epochs,
                "best_validation_nmse_db": best_score, "best_epoch": best_epoch,
                "elapsed_seconds": state["elapsed_seconds"], "blind_opened": False})
            print(f"{phase}/{recipe}/{seed} epoch={epoch}/{epochs} best={best_score:.4f} lr={optimizer.param_groups[0]['lr']:.2e} seconds={state['elapsed_seconds']:.1f}", flush=True)
        if stopped:
            break
    best_meta = torch.load(run_dir/"best.pt", map_location="cpu", weights_only=False)
    cap_improvement = window_improvement(history, options["cap_window"])
    is_converged = stopped or (len(history) >= options["minimum_stop_epoch"] and cap_improvement <= options["plateau_db"])
    result = {
        "phase": phase, "recipe": recipe, "seed": seed,
        "requested_epochs": epochs, "executed_epochs": len(history), "best_epoch": best_epoch,
        "converged": is_converged, "stop_reason": "validation_plateau" if stopped else "epoch_cap",
        "last_200_best_improvement_db": cap_improvement if math.isfinite(cap_improvement) else None,
        "validation": best_meta["validation"], "protocol_sha256": protocol_hash,
        "checkpoint_sha256": digest(run_dir/"best.pt"),
        "elapsed_seconds": elapsed_before+time.perf_counter()-started,
        "device": device, "torch_version": torch.__version__,
        "parameter_count": 34973, "blind_labels_loaded": False,
    }
    write_json(done_path, result)
    del model, optimizer, q, w, y, vq, vw
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def ranking(results: list[dict]) -> list[str]:
    recipes = sorted({r["recipe"] for r in results})
    def key(recipe):
        rows = [r["validation"] for r in results if r["recipe"] == recipe]
        return tuple(float(np.mean([r[k] for r in rows])) for k in (
            "global_amplitude_nmse_db", "global_power_nmse_db", "amplitude_db_mae"))
    return sorted(recipes, key=key)


def campaign(root: Path, device: str = "cuda") -> None:
    protocol = assert_protocol(root)
    if (root/"blind_opened.json").exists():
        raise RuntimeError("cannot train after either blind has been opened")
    if (root/"frozen_selection.json").exists():
        print("Already frozen; no more training allowed.", flush=True)
        return
    screen = [run_training(root, "screen", r, 29001, 300, device) for r in ("R1", "R2", "R3")]
    finalists = ranking(screen)[:2]
    write_json(root/"screen_selection.json", {"rank": ranking(screen), "survivors": finalists, "source": "validation_only"})
    confirm = [run_training(root, "confirm", r, seed, 800, device)
               for r in finalists for seed in (29001, 29002)]
    winner = ranking(confirm)[0]
    write_json(root/"recipe_selection.json", {"rank": ranking(confirm), "winner": winner, "source": "validation_only"})
    final = [run_training(root, "final", winner, seed, 2000, device) for seed in (29001, 29002, 29003)]
    selected = min(final, key=lambda r: (r["validation"]["global_amplitude_nmse_db"],
                                        r["validation"]["global_power_nmse_db"],
                                        r["validation"]["amplitude_db_mae"]))
    checkpoints = {}
    for row in final:
        run = root/"runs"/"final"/f"{winner}_seed{row['seed']}"
        checkpoints[str(row["seed"])] = {name: digest(run/name) for name in ("best.pt", "epoch200.pt")}
    write_json(root/"frozen_selection.json", {
        "protocol_sha256": digest(root/"protocol.json"), "winner": winner,
        "selected_seed": selected["seed"], "selection_source": "validation_only",
        "final_runs": final, "checkpoint_hashes": checkpoints,
        "all_final_runs_complete": True, "all_converged": all(r["converged"] for r in final),
        "test_access_before_freeze": False, "source_code_hashes": protocol["source_code_hashes"],
    })
    print(f"FROZEN winner={winner} selected_seed={selected['seed']}; blind evaluation is now permitted.", flush=True)
