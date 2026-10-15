"""Exact-HFSS four-strong-point baseline on the forward hemisphere.

This experiment intentionally excludes every CNN checkpoint and prediction.  It
uses only the embedded HFSS factors, phase-only weights, and q_x > 0 directions.
The legacy high-budget target max-min solution is the fixed control.  New
methods optimize target-to-hemisphere-peak ratio with an active set of exact
HFSS directions, optionally preserving the control target minimum within a
specified dB budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import front_hemisphere_campaign as campaign  # noqa: E402
from v210.models.quadratic_psd import quadratic_form  # noqa: E402


DEFAULT_SOURCE = BASE / "artifacts" / "front_hemisphere_20260906"
DEFAULT_OUTPUT = BASE / "artifacts" / "hfss_four_strong_20260906"

METHODS = {
    "H1_ratio_only": None,
    "H2_preserve_0p5db": 0.5,
    "H3_preserve_1db": 1.0,
    "H4_preserve_2db": 2.0,
}


def db(power: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(power.clamp_min(1e-12))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_front_grid(source: Path) -> tuple[np.ndarray, np.ndarray]:
    query_parts = []
    factor_parts = []
    for name in ("train", "validation", "calibration", "holdout"):
        data = campaign.dataset(source, name)
        selected = campaign.front_mask(data["q"])
        query_parts.append(data["q"][selected])
        factor_parts.append(data["h"][selected])
    query = np.concatenate(query_parts).astype(np.float32)
    factor = np.concatenate(factor_parts).astype(np.complex64)
    campaign.assert_front(query, "combined HFSS front grid")
    if len(query) != 32400:
        raise ValueError(f"expected 32,400 forward directions, got {len(query)}")
    if len(np.unique(np.round(query, 7), axis=0)) != len(query):
        raise ValueError("combined forward grid contains duplicate directions")
    return query, factor


def subset_by_stratum(
    suite: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    limit_per_stratum: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    if limit_per_stratum is None:
        index = np.arange(len(suite["q"]))
    else:
        chosen = []
        for name in np.unique(suite["stratum"]):
            current = np.flatnonzero(suite["stratum"] == name)
            chosen.extend(current[:limit_per_stratum].tolist())
        index = np.asarray(sorted(chosen), dtype=np.int64)

    def take(value: np.ndarray) -> np.ndarray:
        return value[index] if len(value) == len(suite["q"]) else value

    return (
        {key: take(value) for key, value in suite.items()},
        {key: take(value) for key, value in exact.items()},
        index,
    )


@torch.no_grad()
def scan_front(
    front_factor: torch.Tensor,
    weight: torch.Tensor,
    *,
    topk: int,
    case_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact full-grid top-k gain and indices for [case,start,port] weights."""

    if weight.ndim == 2:
        weight = weight[:, None, :]
    values = []
    indices = []
    for start in range(0, len(weight), case_batch):
        current = weight[start : start + case_batch]
        field = torch.einsum("drn,bsn->bsdr", front_factor, current)
        gain = field.abs().square().sum(dim=-1) / 60.0
        value, index = torch.topk(gain, k=topk, dim=-1)
        values.append(value)
        indices.append(index)
    return torch.cat(values), torch.cat(indices)


def make_starts(
    target_factor: torch.Tensor,
    baseline: torch.Tensor,
    seed: int,
    starts: int,
) -> torch.Tensor:
    if starts < 2:
        raise ValueError("starts must be at least 2")
    generator = torch.Generator(device=target_factor.device).manual_seed(seed)
    qmatrix = quadratic_form(target_factor)
    candidates = [baseline]
    for position in range(1, starts):
        if position == 1:
            alpha = torch.full(
                (len(baseline), target_factor.shape[1]),
                1.0 / target_factor.shape[1],
                device=baseline.device,
            )
            combined = torch.einsum("bk,bknm->bnm", alpha.to(qmatrix.dtype), qmatrix)
            vector = torch.linalg.eigh(combined).eigenvectors[..., -1]
            candidates.append(torch.exp(1j * torch.angle(vector)) / 4.0)
        elif position % 3 == 2:
            sigma = 0.30 + 0.12 * (position // 3)
            noise = torch.randn(baseline.shape, generator=generator, device=baseline.device)
            candidates.append(
                torch.exp(1j * (torch.angle(baseline) + sigma * noise)) / 4.0
            )
        else:
            raw = torch.rand(
                (len(baseline), target_factor.shape[1]),
                generator=generator,
                device=baseline.device,
            )
            alpha = raw / raw.sum(dim=-1, keepdim=True)
            combined = torch.einsum("bk,bknm->bnm", alpha.to(qmatrix.dtype), qmatrix)
            vector = torch.linalg.eigh(combined).eigenvectors[..., -1]
            candidates.append(torch.exp(1j * torch.angle(vector)) / 4.0)
    return torch.stack(candidates, dim=1)


@torch.no_grad()
def candidate_state(
    target_factor: torch.Tensor,
    front_factor: torch.Tensor,
    weight: torch.Tensor,
    *,
    topk: int,
    case_batch: int,
) -> dict[str, torch.Tensor]:
    target_field = torch.einsum("bkrn,bsn->bskr", target_factor, weight)
    target_gain = target_field.abs().square().sum(dim=-1) / 60.0
    target_min_db = db(target_gain).min(dim=-1).values
    peak_gain, peak_index = scan_front(
        front_factor, weight, topk=topk, case_batch=case_batch
    )
    peak_db = db(peak_gain[..., 0])
    margin_db = target_min_db - peak_db
    return {
        "target_gain": target_gain,
        "target_min_db": target_min_db,
        "peak_gain": peak_gain,
        "peak_index": peak_index,
        "peak_db": peak_db,
        "margin_db": margin_db,
    }


def update_best(
    best: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    weight: torch.Tensor,
    floor_db: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if floor_db is None:
        eligible = torch.ones_like(state["margin_db"], dtype=torch.bool)
    else:
        eligible = state["target_min_db"] >= floor_db[:, None] - 1e-4
    score = torch.where(
        eligible,
        state["margin_db"],
        torch.full_like(state["margin_db"], -1e30),
    )
    value, index = score.max(dim=1)
    improved = value > best["margin_db"]
    row = torch.arange(len(weight), device=weight.device)
    selected_weight = weight[row, index]
    selected_target = state["target_min_db"][row, index]
    selected_peak = state["peak_gain"][row, index, 0]
    selected_peak_index = state["peak_index"][row, index, 0]
    best["weight"][improved] = selected_weight[improved]
    best["target_min_db"][improved] = selected_target[improved]
    best["peak_gain"][improved] = selected_peak[improved]
    best["peak_index"][improved] = selected_peak_index[improved]
    best["margin_db"] = torch.maximum(best["margin_db"], value)
    return best


def optimize_ratio_active_set(
    target_factor: torch.Tensor,
    front_factor: torch.Tensor,
    baseline: torch.Tensor,
    *,
    preserve_db: float | None,
    seed: int,
    starts: int,
    outer_steps: int,
    inner_steps: int,
    active_topk: int,
    case_batch: int,
) -> tuple[torch.Tensor, dict]:
    """Optimize exact target/hemisphere margin with cutting-plane peak mining."""

    baseline_target = campaign.true_gain(target_factor, baseline)
    baseline_min_db = db(baseline_target).min(dim=1).values
    floor_db = None if preserve_db is None else baseline_min_db - preserve_db
    weight = make_starts(target_factor, baseline, seed, starts)
    initial = candidate_state(
        target_factor,
        front_factor,
        weight,
        topk=active_topk,
        case_batch=case_batch,
    )
    best = {
        "weight": baseline.detach().clone(),
        "target_min_db": baseline_min_db.detach().clone(),
        "peak_gain": initial["peak_gain"][:, 0, 0].detach().clone(),
        "peak_index": initial["peak_index"][:, 0, 0].detach().clone(),
        "margin_db": initial["margin_db"][:, 0].detach().clone(),
    }
    best = update_best(best, initial, weight, floor_db)
    history = []

    for outer in range(outer_steps):
        state = candidate_state(
            target_factor,
            front_factor,
            weight,
            topk=active_topk,
            case_batch=case_batch,
        )
        active_factor = front_factor[state["peak_index"]]
        phase = torch.nn.Parameter(torch.angle(weight))
        optimizer = torch.optim.Adam([phase], lr=0.045)
        for step in range(inner_steps):
            current = torch.exp(1j * phase) / 4.0
            target_field = torch.einsum("bkrn,bsn->bskr", target_factor, current)
            target_gain_db = db(target_field.abs().square().sum(dim=-1) / 60.0)
            active_field = torch.einsum("bsarn,bsn->bsar", active_factor, current)
            active_gain_db = db(active_field.abs().square().sum(dim=-1) / 60.0)
            target_temperature = 0.35 - 0.20 * step / max(inner_steps - 1, 1)
            peak_temperature = 0.30 - 0.15 * step / max(inner_steps - 1, 1)
            target_softmin = -target_temperature * torch.logsumexp(
                -target_gain_db / target_temperature, dim=-1
            )
            peak_softmax = peak_temperature * torch.logsumexp(
                active_gain_db / peak_temperature, dim=-1
            )
            objective = peak_softmax - target_softmin
            if floor_db is not None:
                violation = torch.relu(floor_db[:, None] - target_softmin)
                objective = objective + 8.0 * violation.square()
            loss = objective.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([phase], 10.0)
            optimizer.step()
        weight = (torch.exp(1j * phase.detach()) / 4.0).detach()
        state = candidate_state(
            target_factor,
            front_factor,
            weight,
            topk=active_topk,
            case_batch=case_batch,
        )
        best = update_best(best, state, weight, floor_db)
        history.append(
            {
                "outer": outer + 1,
                "best_margin_db_median": float(best["margin_db"].median()),
                "best_target_min_db_median": float(best["target_min_db"].median()),
            }
        )
    return best["weight"], {"history": history}


def spectral_solo_upper_bound(
    suite: dict[str, np.ndarray], exact: dict[str, np.ndarray]
) -> np.ndarray:
    """Valid upper bound on min_i G_i/G_i^solo for any unit-norm complex weight."""

    factor = suite["h"].astype(np.complex128)
    qmatrix = np.einsum("bkrn,bkrm->bknm", factor.conj(), factor) / 60.0
    solo = np.maximum(exact["solo_gain"].astype(np.float64), 1e-30)
    normalized_q = qmatrix / solo[:, :, None, None]
    output = np.empty(len(factor), dtype=np.float64)

    def softmax(value: np.ndarray) -> np.ndarray:
        shifted = value - value.max()
        probability = np.exp(shifted)
        return probability / probability.sum()

    for case in range(len(factor)):
        def bound(value: np.ndarray) -> float:
            alpha = softmax(value)
            matrix = np.einsum("k,knm->nm", alpha, normalized_q[case])
            return float(np.linalg.eigvalsh(matrix)[-1].real)

        equal = bound(np.zeros(4))
        result = minimize(
            bound,
            np.zeros(4),
            method="BFGS",
            options={"maxiter": 100, "gtol": 1e-8},
        )
        output[case] = min(equal, float(result.fun) if np.isfinite(result.fun) else equal)
    return output


def bootstrap_difference(
    candidate: np.ndarray, baseline: np.ndarray, seed: int
) -> dict[str, float]:
    difference = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(difference, size=(2000, len(difference)), replace=True).mean(axis=1)
    return {
        "paired_fraction_difference": float(difference.mean()),
        "bootstrap_95ci": np.percentile(sampled, [2.5, 97.5]).tolist(),
    }


@torch.no_grad()
def evaluate_method(
    name: str,
    target_factor: torch.Tensor,
    front_query: torch.Tensor,
    front_factor: torch.Tensor,
    weight: torch.Tensor,
    baseline_min_db: torch.Tensor,
    solo_gain: torch.Tensor,
    strata: np.ndarray,
    *,
    case_batch: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    target_gain = campaign.true_gain(target_factor, weight)
    minimum_gain = target_gain.min(dim=1).values
    minimum_db = db(minimum_gain)
    peak_gain, peak_index = scan_front(
        front_factor, weight, topk=1, case_batch=case_batch
    )
    peak_gain = peak_gain[:, 0, 0]
    peak_index = peak_index[:, 0, 0]
    rho = minimum_gain / peak_gain.clamp_min(1e-30)
    solo_utilization = (target_gain / solo_gain.clamp_min(1e-30)).min(dim=1).values
    target_loss_db = baseline_min_db - minimum_db
    peak_query = front_query[peak_index]
    target_query = torch.tensor(suite_global["q"], device=weight.device)
    target_to_peak = torch.rad2deg(
        torch.arccos(torch.einsum("bik,bk->bi", target_query, peak_query).clamp(-1, 1))
    )
    spatial = rho >= 0.5
    high = solo_utilization >= 0.5
    strict = spatial & high
    result = {
        "name": name,
        "cases": int(len(weight)),
        "minimum_gain_db_p5_median_p95": torch.quantile(
            minimum_db, torch.tensor([0.05, 0.5, 0.95], device=weight.device)
        ).cpu().tolist(),
        "minimum_rho_p5_median_p95": torch.quantile(
            rho, torch.tensor([0.05, 0.5, 0.95], device=weight.device)
        ).cpu().tolist(),
        "solo_utilization_p5_median_p95": torch.quantile(
            solo_utilization,
            torch.tensor([0.05, 0.5, 0.95], device=weight.device),
        ).cpu().tolist(),
        "target_loss_vs_H0_db_p5_median_p95": torch.quantile(
            target_loss_db, torch.tensor([0.05, 0.5, 0.95], device=weight.device)
        ).cpu().tolist(),
        "fraction_spatial_strong": {
            "rho_ge_0_1": float((rho >= 0.1).float().mean()),
            "rho_ge_0_2": float((rho >= 0.2).float().mean()),
            "rho_ge_0_5": float(spatial.float().mean()),
        },
        "fraction_solo_utilization_ge_0_5": float(high.float().mean()),
        "fraction_strict_four_strong": float(strict.float().mean()),
        "nearest_target_to_peak_deg_median": float(target_to_peak.min(dim=1).values.median()),
        "farthest_target_to_peak_deg_median": float(target_to_peak.max(dim=1).values.median()),
        "constant_modulus_max_error": float((weight.abs() - 0.25).abs().max()),
        "by_stratum": {},
    }
    rho_np = rho.cpu().numpy()
    util_np = solo_utilization.cpu().numpy()
    strict_np = strict.cpu().numpy()
    minimum_np = minimum_db.cpu().numpy()
    loss_np = target_loss_db.cpu().numpy()
    for stratum in np.unique(strata):
        selected = strata == stratum
        result["by_stratum"][str(stratum)] = {
            "cases": int(selected.sum()),
            "minimum_rho_median": float(np.median(rho_np[selected])),
            "minimum_gain_db_median": float(np.median(minimum_np[selected])),
            "target_loss_db_median": float(np.median(loss_np[selected])),
            "fraction_rho_ge_0_5": float(np.mean(rho_np[selected] >= 0.5)),
            "fraction_solo_utilization_ge_0_5": float(np.mean(util_np[selected] >= 0.5)),
            "fraction_strict_four_strong": float(np.mean(strict_np[selected])),
        }
    arrays = {
        "weight": weight.cpu().numpy(),
        "target_gain": target_gain.cpu().numpy(),
        "minimum_gain_db": minimum_np,
        "hemisphere_peak_gain": peak_gain.cpu().numpy(),
        "peak_index": peak_index.cpu().numpy(),
        "minimum_rho": rho_np,
        "minimum_solo_utilization": util_np,
        "target_loss_db": loss_np,
        "spatial_strong": spatial.cpu().numpy(),
        "strict_four_strong": strict_np,
        "nearest_target_to_peak_deg": target_to_peak.min(dim=1).values.cpu().numpy(),
        "farthest_target_to_peak_deg": target_to_peak.max(dim=1).values.cpu().numpy(),
    }
    return result, arrays


def cause_breakdown(
    arrays: dict[str, dict[str, np.ndarray]], spectral_bound: np.ndarray
) -> dict:
    baseline = arrays["H0_target_maxmin"]
    ratio = arrays["H1_ratio_only"]
    preserve = arrays.get("H3_preserve_1db")
    if preserve is None:
        preserve = next(value for key, value in arrays.items() if key.startswith("H2_"))
    certified_physical = spectral_bound < 0.5
    objective_resolved = (
        (~certified_physical)
        & preserve["strict_four_strong"]
        & (preserve["target_loss_db"] <= 1.0 + 1e-4)
    )
    sacrifice_only = (
        (~certified_physical)
        & (~objective_resolved)
        & ratio["spatial_strong"]
        & (ratio["target_loss_db"] > 1.0)
    )
    unresolved = ~(certified_physical | objective_resolved | sacrifice_only)
    return {
        "definitions": {
            "certified_physical_high_value_limit": "spectral upper bound on min_i G_i/G_i_solo is below 0.5; impossible even with arbitrary complex amplitudes",
            "old_objective_mismatch_resolved": "1 dB-preserving spatial optimizer attains rho>=0.5 and solo utilization>=0.5",
            "spatial_balance_requires_sacrifice": "ratio-only attains rho>=0.5 but loses more than 1 dB target minimum",
            "unresolved_phase_only_or_nonconvex_gap": "not certified and not attained; needs SDP/amplitude relaxation or more global phase search",
        },
        "fractions": {
            "certified_physical_high_value_limit": float(certified_physical.mean()),
            "old_objective_mismatch_resolved": float(objective_resolved.mean()),
            "spatial_balance_requires_sacrifice": float(sacrifice_only.mean()),
            "unresolved_phase_only_or_nonconvex_gap": float(unresolved.mean()),
        },
        "counts": {
            "certified_physical_high_value_limit": int(certified_physical.sum()),
            "old_objective_mismatch_resolved": int(objective_resolved.sum()),
            "spatial_balance_requires_sacrifice": int(sacrifice_only.sum()),
            "unresolved_phase_only_or_nonconvex_gap": int(unresolved.sum()),
        },
    }


suite_global: dict[str, np.ndarray]


def run(args: argparse.Namespace) -> None:
    global suite_global
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite_path = source / f"{args.suite}_front_suite.npz"
    exact_path = source / f"{args.suite}_optimizer_arrays.npz"
    suite = campaign.load_npz(suite_path)
    exact = campaign.load_npz(exact_path)
    suite, exact, source_index = subset_by_stratum(
        suite, exact, args.limit_per_stratum
    )
    suite_global = suite
    campaign.assert_front(suite["q"], f"{args.suite} four-point suite")
    front_query_np, front_factor_np = load_front_grid(source)
    device = torch.device(args.device)
    front_query = torch.tensor(front_query_np, device=device)
    front_factor = torch.tensor(front_factor_np, device=device)
    target_factor = torch.tensor(suite["h"], device=device)
    baseline = torch.tensor(exact["O3_high_budget_weight"], device=device)
    solo_gain = torch.tensor(exact["solo_gain"], device=device)
    baseline_target = campaign.true_gain(target_factor, baseline)
    baseline_min_db = db(baseline_target).min(dim=1).values

    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    protocol = {
        "purpose": "exact-HFSS four-strong-point baseline with CNN influence excluded",
        "suite": args.suite,
        "cases": int(len(suite["q"])),
        "source_indices": source_index.tolist(),
        "forward_grid_directions": int(len(front_query_np)),
        "reverse_directions_used": 0,
        "continuous_hemisphere_certificate": False,
        "array_and_weight_control": "fixed 4x4, 47 GHz, phase-only |w_n|=1/4",
        "control": "existing exact-HFSS O3 target-only high-budget max-min weights",
        "cnn_files_loaded": [],
        "definitions": {
            "spatial_strength": "rho=min_i G(q_i,w)/max_{q_x>0} G(q,w)",
            "solo_utilization": "u=min_i G(q_i,w)/G_i_solo",
            "strict_four_strong": "rho>=0.5 and u>=0.5 (both within 3.01 dB)",
        },
        "methods": {name: {"preserve_db": METHODS[name]} for name in methods},
        "optimizer": {
            "starts": args.starts,
            "outer_steps": args.outer_steps,
            "inner_steps": args.inner_steps,
            "active_topk": args.active_topk,
            "case_batch": args.case_batch,
        },
        "source_hashes": {
            suite_path.name: sha256(suite_path),
            exact_path.name: sha256(exact_path),
            "script": sha256(Path(__file__)),
        },
    }
    write_json(output / f"{args.suite}_protocol.json", protocol)
    print(json.dumps({"stage": "loaded", "suite": args.suite, "cases": len(suite["q"])}), flush=True)

    metrics: dict[str, dict] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    row, value = evaluate_method(
        "H0_target_maxmin",
        target_factor,
        front_query,
        front_factor,
        baseline,
        baseline_min_db,
        solo_gain,
        suite["stratum"],
        case_batch=args.case_batch,
    )
    metrics["H0_target_maxmin"] = row
    arrays["H0_target_maxmin"] = value
    print(json.dumps({"stage": "method", "name": "H0_target_maxmin", "rho_median": row["minimum_rho_p5_median_p95"][1]}), flush=True)

    for position, name in enumerate(methods):
        started = time.perf_counter()
        weight, diagnostic = optimize_ratio_active_set(
            target_factor,
            front_factor,
            baseline,
            preserve_db=METHODS[name],
            seed=args.seed + position * 1009,
            starts=args.starts,
            outer_steps=args.outer_steps,
            inner_steps=args.inner_steps,
            active_topk=args.active_topk,
            case_batch=args.case_batch,
        )
        row, value = evaluate_method(
            name,
            target_factor,
            front_query,
            front_factor,
            weight,
            baseline_min_db,
            solo_gain,
            suite["stratum"],
            case_batch=args.case_batch,
        )
        row["runtime_seconds"] = time.perf_counter() - started
        row["optimizer_diagnostic"] = diagnostic
        metrics[name] = row
        arrays[name] = value
        print(json.dumps({"stage": "method", "name": name, "rho_median": row["minimum_rho_p5_median_p95"][1], "strict": row["fraction_strict_four_strong"], "seconds": row["runtime_seconds"]}), flush=True)

    spectral_bound = spectral_solo_upper_bound(suite, exact)
    baseline_spatial = arrays["H0_target_maxmin"]["spatial_strong"]
    comparisons = {}
    for name in methods:
        comparisons[name] = {
            "spatial_strong_vs_H0": bootstrap_difference(
                arrays[name]["spatial_strong"], baseline_spatial, args.seed + 8001
            ),
            "strict_four_strong_vs_H0": bootstrap_difference(
                arrays[name]["strict_four_strong"],
                arrays["H0_target_maxmin"]["strict_four_strong"],
                args.seed + 9001,
            ),
        }
    result = {
        "protocol": protocol,
        "methods": metrics,
        "comparisons": comparisons,
        "spectral_solo_upper_bound": {
            "p5_median_p95": np.percentile(spectral_bound, [5, 50, 95]).tolist(),
            "fraction_certified_below_0_5": float(np.mean(spectral_bound < 0.5)),
            "statement": "valid for arbitrary unit-norm complex weights, so also for phase-only weights",
        },
        "cause_breakdown": cause_breakdown(arrays, spectral_bound),
    }

    query = suite["q"].astype(np.float64)
    cosine = np.einsum("bik,bjk->bij", query, query).clip(-1, 1)
    angle = np.rad2deg(np.arccos(cosine))
    upper = np.triu_indices(4, 1)
    pair = angle[:, upper[0], upper[1]]
    boresight = np.rad2deg(np.arccos(query[:, :, 0].clip(-1, 1)))
    final_name = "H3_preserve_1db" if "H3_preserve_1db" in arrays else methods[-1]
    final_rho = arrays[final_name]["minimum_rho"]
    result["position_relationship"] = {
        "method": final_name,
        "spearman_with_minimum_rho": {
            "minimum_pair_angle_deg": float(spearmanr(pair.min(axis=1), final_rho).statistic),
            "mean_pair_angle_deg": float(spearmanr(pair.mean(axis=1), final_rho).statistic),
            "maximum_pair_angle_deg": float(spearmanr(pair.max(axis=1), final_rho).statistic),
            "mean_boresight_angle_deg": float(spearmanr(boresight.mean(axis=1), final_rho).statistic),
            "farthest_target_to_peak_deg": float(spearmanr(arrays[final_name]["farthest_target_to_peak_deg"], final_rho).statistic),
        },
    }
    write_json(output / f"{args.suite}_results.json", result)
    flat = {
        "source_index": source_index,
        "spectral_solo_upper_bound": spectral_bound,
        "q": suite["q"],
        "stratum": suite["stratum"],
    }
    for method, values in arrays.items():
        flat.update({f"{method}_{key}": value for key, value in values.items()})
    np.savez_compressed(output / f"{args.suite}_arrays.npz", **flat)
    print(json.dumps({"stage": "complete", "output": str(output / f'{args.suite}_results.json')}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", choices=("validation", "holdout"), default="validation")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-stratum", type=int)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--inner-steps", type=int, default=120)
    parser.add_argument("--active-topk", type=int, default=24)
    parser.add_argument("--case-batch", type=int, default=12)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=21601)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
