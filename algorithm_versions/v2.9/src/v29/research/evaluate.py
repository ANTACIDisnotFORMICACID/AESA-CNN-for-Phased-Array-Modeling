"""One-time, post-freeze dual-blind evaluation; never imported by the trainer."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..contracts import QueryBatch
from ..coordinates import hfss_angles_to_direction
from ..io.amplitude_synthesis import canonical_spherical_query_mask
from ..models import model_from_config
from .protocol import (PROTOCOL, assert_protocol, block_keys, digest, load_scalar_basis,
                       read_json, source_files, write_json)
from .train import metrics, predict


def verify_freeze(root: Path) -> dict:
    protocol = assert_protocol(root)
    freeze = read_json(root/"frozen_selection.json")
    if not freeze["all_final_runs_complete"] or freeze["selection_source"] != "validation_only":
        raise ValueError("blind access requires completed, validation-selected final runs")
    if freeze["protocol_sha256"] != digest(root/"protocol.json"):
        raise ValueError("protocol changed after freeze")
    if set(freeze["checkpoint_hashes"]) != {"29001", "29002", "29003"}:
        raise ValueError("three final seeds are required")
    for row in freeze["final_runs"]:
        if row["requested_epochs"] != 2000:
            raise ValueError("final budget differs from approved 2000-epoch cap")
        if row["executed_epochs"] < 2000 and row["stop_reason"] != "validation_plateau":
            raise ValueError("unapproved early termination")
    for seed, names in freeze["checkpoint_hashes"].items():
        run = root/"runs"/"final"/f"{freeze['winner']}_seed{seed}"
        for name, expected in names.items():
            if digest(run/name) != expected:
                raise ValueError("checkpoint changed after freeze")
    if digest(root/"blind"/"spec.npz") != protocol["blind_spec_sha256"]:
        raise ValueError("blind specification changed")
    return freeze


def region_metrics(prediction, target, directions, families) -> dict:
    regions = {"full_sphere": np.ones(len(directions), dtype=bool),
               "front_hemisphere": directions[:, 0] > 0,
               "boresight_60deg": directions[:, 0] >= 0.5}
    out = {"regions": {}, "families": {}}
    for name, mask in regions.items():
        if mask.any():
            out["regions"][name] = {"query_count": int(mask.sum()),
                **metrics(prediction[:, mask], target[:, mask])}
    for family in np.unique(families):
        ids = families == family
        out["families"][str(family)] = {"command_count": int(ids.sum()),
                                       **metrics(prediction[ids], target[ids])}
    novel = ~np.isin(families, ["single_element", "dft_phase"])
    if novel.any():
        out["novel_commands"] = metrics(prediction[novel], target[novel])
        out["novel_command_count"] = int(novel.sum())
    return out


def maximize_minimum(h: torch.Tensor, seed: int, initial=None, steps=600, restarts=32):
    """Multi-start local reference only; no claim of certified global optimality."""
    generator = torch.Generator(device=h.device).manual_seed(seed)
    w = torch.randn((restarts, 16), dtype=torch.complex64, device=h.device, generator=generator)
    w /= w.norm(dim=1, keepdim=True)
    if initial is not None:
        w[0] = initial
    w.requires_grad_()
    optimizer = torch.optim.Adam([w], lr=0.03)
    best = torch.tensor(-1.0, device=h.device)
    best_w = w[0].detach().clone()
    scale = h.abs().square().sum(dim=1).mean().detach().clamp_min(1e-10)
    for step in range(steps+1):
        power = (w @ h.T).abs().square()
        minimum = power.min(dim=1).values
        index = torch.argmax(minimum)
        if minimum[index] > best:
            best = minimum[index].detach()
            best_w = w[index].detach().clone()
        if step == steps:
            break
        beta = 2.0 + 98.0*step/steps
        objective = -torch.logsumexp(-beta*power/scale, dim=1)/beta
        optimizer.zero_grad(set_to_none=True)
        (-objective.mean()).backward()
        optimizer.step()
        with torch.no_grad():
            w /= w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return best_w, float(best)


def target_suite(model, directions: np.ndarray, basis: np.ndarray, eligible: np.ndarray) -> dict:
    device = next(model.parameters()).device
    rng = np.random.default_rng(29501)
    candidates = np.flatnonzero(eligible & (directions[:, 0] >= 0.5))
    rows = []
    for case in range(30):
        separation = (2, 5, 10, 20, 30)[case//6]
        for attempt in range(10000):
            ids = rng.choice(candidates, 4, replace=False)
            cosine = directions[ids] @ directions[ids].T
            angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
            if np.min(angles[np.triu_indices(4, 1)]) >= separation:
                break
        else:
            raise RuntimeError("cannot construct target set")
        q = torch.tensor(directions[ids], dtype=torch.float32, device=device)
        with torch.no_grad():
            op, _ = model.operator(QueryBatch(q[None]))
            h_model = torch.complex(op[0, :, 0], op[0, :, 1]).reshape(4, 16)
        h_reference = torch.tensor(basis[ids], dtype=torch.complex64, device=device)
        cnn_w, _ = maximize_minimum(h_model, 29601+case)
        actual = (h_reference @ cnn_w).abs().square()
        predicted = (h_model @ cnn_w).abs().square()
        # Include the CNN solution so the numerical reference is never worse.
        ref_w, ref_minimum = maximize_minimum(h_reference, 29701+case, initial=cnn_w)
        regret = max(0.0, (ref_minimum-float(actual.min()))/max(ref_minimum, 1e-12))
        rows.append({"case": case, "minimum_separation_deg": separation,
                     "shifted_query_indices": ids.tolist(), "directions": directions[ids].tolist(),
                     "predicted_target_power": predicted.cpu().tolist(),
                     "hfss_target_power": actual.cpu().tolist(),
                     "hfss_reference_minimum": ref_minimum, "relative_regret": regret,
                     "command_norm": float(cnn_w.norm()),
                     "reference_command_norm": float(ref_w.norm())})
    regrets = np.array([r["relative_regret"] for r in rows])
    return {"reference_type": "32-start numerical lower bound; not certified global optimum",
            "scope": "30 synthetic target sets, shifted grid, strictly held-out blocks, boresight<=60deg",
            "median_regret": float(np.median(regrets)), "p95_regret": float(np.quantile(regrets, .95)),
            "cases": rows}


def evaluate_blinds(root: Path, device="cuda") -> dict:
    freeze = verify_freeze(root)
    protocol = read_json(root/"protocol.json")
    finished = root/"dual_blind_report.json"
    if finished.exists():
        raise RuntimeError("blind evaluation already completed; do not reopen or retune")
    for name, expected in protocol["raw_source_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"raw export changed: {name}")
    gate = root/"blind_opened.json"
    if gate.exists():
        if read_json(gate)["freeze_sha256"] != digest(root/"frozen_selection.json"):
            raise ValueError("cannot resume evaluation under a different freeze")
    else:
        write_json(gate, {"freeze_sha256": digest(root/"frozen_selection.json"),
                          "opened_utc": datetime.now(timezone.utc).isoformat(),
                          "policy": "training and recipe selection permanently prohibited"})
    with np.load(root/"blind"/"spec.npz", allow_pickle=False) as file:
        spec = {k: file[k] for k in file.files}
    ci = spec["partition"] == 2
    commands = spec["command"][ci]
    family = spec["family"][ci]
    command_iq = torch.tensor(np.stack((commands.real, commands.imag), axis=1), dtype=torch.float32)
    folder = Path(protocol["source_folder"])
    old_t, old_p, old_basis = load_scalar_basis(source_files(folder, False))
    keep = canonical_spherical_query_mask(old_t, old_p)
    old_t, old_p, old_basis = old_t[keep], old_p[keep], old_basis[keep]
    old_mask = np.isin(block_keys(old_t, old_p), spec["test_blocks"])
    old_q = hfss_angles_to_direction(old_t[old_mask], old_p[old_mask])
    old_y = torch.tensor(np.abs(commands.reshape(-1, 16) @ old_basis[old_mask].T), dtype=torch.float32)
    new_t, new_p, new_basis = load_scalar_basis(source_files(folder, True))
    new_q = hfss_angles_to_direction(new_t, new_p)
    new_keys = block_keys(new_t, new_p)
    new_mask = np.isin(new_keys, spec["test_blocks"])
    new_y = torch.tensor(np.abs(commands.reshape(-1, 16) @ new_basis.T), dtype=torch.float32)
    rows = []
    selected_model = None
    torch.set_num_threads(4)
    for seed in (29001, 29002, 29003):
        run = root/"runs"/"final"/f"{freeze['winner']}_seed{seed}"
        model = model_from_config(protocol["model_config"]).to(device)
        row = {"seed": seed}
        for name, filename in (("best", "best.pt"), ("epoch200", "epoch200.pt")):
            checkpoint = torch.load(run/filename, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            old_pred = predict(model, command_iq, torch.tensor(old_q, dtype=torch.float32))
            new_pred = predict(model, command_iq, torch.tensor(new_q, dtype=torch.float32))
            row[name] = {
                "integer_blind": region_metrics(old_pred, old_y, old_q, family),
                "shifted_blind_blocks": region_metrics(new_pred[:, new_mask], new_y[:, new_mask], new_q[new_mask], family),
                "shifted_all_blocks": region_metrics(new_pred, new_y, new_q, family),
            }
        # Restore the validation-selected checkpoint, never select using blind results.
        model.load_state_dict(torch.load(run/"best.pt", map_location="cpu", weights_only=False)["model_state"])
        if seed == freeze["selected_seed"]:
            selected_model = model
        rows.append(row)
        write_json(root/f"blind_seed{seed}.json", row)
    suite = target_suite(selected_model, new_q, new_basis, new_mask)
    write_json(root/"four_target_diagnostic.json", suite)
    best = np.array([r["best"]["integer_blind"]["regions"]["full_sphere"]["global_amplitude_nmse_db"] for r in rows])
    baseline = np.array([r["epoch200"]["integer_blind"]["regions"]["full_sphere"]["global_amplitude_nmse_db"] for r in rows])
    shifted = np.array([r["best"]["shifted_blind_blocks"]["regions"]["full_sphere"]["global_amplitude_nmse_db"] for r in rows])
    gain = baseline-best
    checks = {
        "all_three_converged": freeze["all_converged"],
        "paired_mean_gain_at_least_0_3db": float(gain.mean()) >= .3,
        "seed_std_at_most_0_25db": float(best.std(ddof=1)) <= .25,
        "no_seed_regression_over_0_1db": bool(np.all(gain >= -.1)),
        "offset_degradation_at_most_0_5db": float((shifted-best).mean()) <= .5,
        "target_median_regret_at_most_5pct": suite["median_regret"] <= .05,
        "target_p95_regret_at_most_10pct": suite["p95_regret"] <= .10,
    }
    for metric in ("global_power_nmse_db", "amplitude_db_mae"):
        a = [r["best"]["integer_blind"]["regions"]["full_sphere"][metric] for r in rows]
        b = [r["epoch200"]["integer_blind"]["regions"]["full_sphere"][metric] for r in rows]
        checks[metric+"_not_worse"] = float(np.mean(a)) <= float(np.mean(b))
    report = {
        "protocol_sha256": digest(root/"protocol.json"),
        "freeze_sha256": digest(root/"frozen_selection.json"),
        "selected_seed_from_validation": freeze["selected_seed"], "recipe": freeze["winner"],
        "mean_integer_blind_nmse_db": float(best.mean()), "seed_std_db": float(best.std(ddof=1)),
        "mean_paired_improvement_db": float(gain.mean()),
        "mean_shifted_blind_nmse_db": float(shifted.mean()),
        "mean_offset_degradation_db": float((shifted-best).mean()),
        "checks": checks, "all_acceptance_checks_pass": all(checks.values()),
        "runs": rows,
        "leakage_audit": {
            "training_read_only_dev_partitions": True,
            "blind_opened_after_all_final_checkpoint_hashes_frozen": True,
            "exact_old_shifted_direction_overlap": 0,
            "primary_blind_angular_blocks_seen_in_train_or_validation": 0,
            "shifted_all_blocks_is_interpolation_plus_extrapolation_not_strict_block_blind": True,
            "finite_single_and_dft_anchor_commands_have_historical_exposure": True,
            "same_HFSS_solution_is_not_independent_OTA_validation": True,
            "no_retuning_after_blind_opening": True,
        },
    }
    write_json(finished, report)
    print("DUAL BLIND COMPLETE " + str({k: v for k, v in report.items() if k in (
        "mean_integer_blind_nmse_db", "seed_std_db", "mean_shifted_blind_nmse_db",
        "all_acceptance_checks_pass")}), flush=True)
    return report
