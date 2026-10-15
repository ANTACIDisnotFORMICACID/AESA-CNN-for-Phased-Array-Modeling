"""Frozen-CNN four-point optimizer study.

This script never updates CNN parameters.  It pre-registers a deterministic
coordinate set and optimizer configurations, computes all candidate weights
from the frozen CNN, and only then opens HFSS labels for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
BASE = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import beam_campaign as campaign


SEED = 2960501
N_PER_STRATUM = 60
STRATA = ("moderate", "distributed", "wide")
METHODS = {
    "baseline_softmin": {"starts": 8, "steps": 160, "kind": "baseline"},
    "eigen_multibeam": {"starts": 8, "steps": 160, "kind": "eigen"},
    "flat_top_2deg": {"starts": 8, "steps": 180, "kind": "flat"},
    "peak_ratio": {"starts": 8, "steps": 180, "kind": "ratio", "scan_directions": 192},
    "relocate_5deg": {"starts": 8, "steps": 160, "kind": "relocate", "rounds": 2},
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def q_to_angles(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.degrees(np.arccos(np.clip(q[..., 2], -1, 1)))
    phi = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
    return theta, phi


def angular_matrix(q: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(q @ q.T, -1, 1)))


def generate_cases(qfull: np.ndarray, theta: np.ndarray, phi: np.ndarray):
    """Draw 180 four-point sets from the +x cone, using only geometry."""
    off_axis = np.degrees(np.arccos(np.clip(qfull[:, 0], -1, 1)))
    pool = np.flatnonzero((off_axis <= 60) & (theta >= 7.5) & (theta <= 172.5))
    rng = np.random.default_rng(SEED)
    cases, labels = [], []
    rules = {
        "moderate": lambda mn, mx: mn >= 8 and 25 <= mx <= 65,
        "distributed": lambda mn, mx: mn >= 18 and 45 <= mx <= 95,
        "wide": lambda mn, mx: mn >= 25 and mx >= 80,
    }
    for label in STRATA:
        attempts = 0
        while sum(x == label for x in labels) < N_PER_STRATUM:
            attempts += 1
            if attempts > 2_000_000:
                raise RuntimeError(f"could not generate stratum {label}")
            idx = rng.choice(pool, 4, replace=False)
            a = angular_matrix(qfull[idx])
            values = a[np.triu_indices(4, 1)]
            if rules[label](float(values.min()), float(values.max())):
                cases.append(idx)
                labels.append(label)
    return np.asarray(cases, dtype=np.int64), np.asarray(labels)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    z = 1 - 2 * (i + .5) / n
    az = (math.pi * (3 - math.sqrt(5)) * i) % (2 * math.pi)
    r = np.sqrt(np.maximum(0, 1 - z * z))
    return np.stack([r * np.cos(az), r * np.sin(az), z], axis=1).astype(np.float32)


def eigen_seeds(h: torch.Tensor) -> torch.Tensor:
    """Four single-target eigenbeams plus four coherent multibeam sums."""
    qmat = torch.einsum("bkpn,bkpm->bknm", h.conj(), h) / 60
    v = torch.linalg.eigh(qmat).eigenvectors[..., -1]
    anchor = torch.angle(v[..., :1])
    v = v * torch.exp(-1j * anchor)
    signs = torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        device=h.device,
        dtype=h.real.dtype,
    )
    multi = torch.einsum("sk,bkn->bsn", signs.to(v.dtype), v)
    return torch.cat([v, multi], dim=1)


def run_optimizer(
    h: torch.Tensor,
    steps: int,
    seed: int,
    objective_h: torch.Tensor | None = None,
    scan_h: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant-modulus projected Adam with deterministic physics seeds."""
    torch.manual_seed(seed)
    z = eigen_seeds(h)
    param = nn.Parameter(torch.angle(z))
    opt = torch.optim.Adam([param], lr=.065)
    best_score = torch.full((len(h),), -1e30, device=h.device)
    best_min = torch.full((len(h),), -1e30, device=h.device)
    best_w = torch.zeros((len(h), 16), dtype=torch.complex64, device=h.device)
    obj_h = h if objective_h is None else objective_h
    for step in range(steps + 1):
        w = torch.exp(1j * param) / 4
        target_g = torch.einsum("bkpn,bsn->bskp", h, w).abs().square().sum(-1) / 60
        obj_g = torch.einsum("bkpn,bsn->bskp", obj_h, w).abs().square().sum(-1) / 60
        target_db = campaign.tdb(target_g)
        obj_db = campaign.tdb(obj_g)
        hard_min = target_db.min(-1).values
        tau = 1.2 * (1 - step / max(steps, 1)) + .12
        smooth_min = -tau * torch.logsumexp(-obj_db / tau, dim=-1)
        score = smooth_min
        if scan_h is not None:
            scan_g = torch.einsum("mpn,bsn->bsmp", scan_h, w).abs().square().sum(-1) / 60
            scan_db = campaign.tdb(scan_g)
            peak = .5 * torch.logsumexp(scan_db / .5, dim=-1)
            violation = torch.relu(peak - hard_min - 10)
            score = smooth_min - .15 * violation.square()
        selected_score, selected = score.max(1)
        selected_min = hard_min[torch.arange(len(h), device=h.device), selected]
        improve = selected_score > best_score
        best_score[improve] = selected_score[improve].detach()
        best_min[improve] = selected_min[improve].detach()
        best_w[improve] = w.detach()[torch.arange(len(h), device=h.device), selected][improve]
        if step == steps:
            break
        loss = -score.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best_w, best_min


def flat_directions(q: np.ndarray, radius_deg: float = 2.) -> np.ndarray:
    result = []
    for case in q:
        rows = []
        for point in case:
            ref = np.array([0., 0., 1.]) if abs(point[2]) < .9 else np.array([0., 1., 0.])
            u = np.cross(point, ref); u /= np.linalg.norm(u)
            v = np.cross(point, u)
            local = [point]
            for az in np.arange(0, 360, 90):
                d = math.cos(math.radians(az)) * u + math.sin(math.radians(az)) * v
                local.append(math.cos(math.radians(radius_deg)) * point + math.sin(math.radians(radius_deg)) * d)
            rows.extend(local)
        result.append(rows)
    return np.asarray(result, dtype=np.float32)


def relocation_candidates(indices: np.ndarray, qfull: np.ndarray, theta: np.ndarray, phi: np.ndarray):
    candidates = []
    cos_limit = math.cos(math.radians(5.05))
    for case in indices:
        case_rows = []
        for idx in case:
            ti = int((theta[idx] - .5) + .1)
            pi = int((phi[idx] + 179.5) + .1)
            row = []
            for dt in range(-5, 6):
                for dp in range(-5, 6):
                    nt, np_ = ti + dt, (pi + dp) % 360
                    if 0 <= nt < 180:
                        j = nt * 360 + np_
                        if qfull[idx] @ qfull[j] >= cos_limit and qfull[j, 0] > 0:
                            row.append(j)
            row = sorted(set(row))
            while len(row) < 121:
                row.append(idx)
            case_rows.append(row[:121])
        candidates.append(case_rows)
    return np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def choose_relocated(hcand: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    gain = torch.einsum("bkcpn,bn->bkcp", hcand, w).abs().square().sum(-1) / 60
    return gain.argmax(-1)


def optimize_relocated(model, scale, original_h, cand_q, rounds, steps):
    hp_cand = campaign.predict(model, cand_q.reshape(-1, 3), scale).reshape(*cand_q.shape[:-1], 3, 16)
    w, _ = run_optimizer(original_h, steps, SEED + 50)
    selected = torch.zeros(cand_q.shape[:2], dtype=torch.long, device="cuda")
    for round_id in range(rounds):
        selected = choose_relocated(hp_cand, w)
        moved_h = hp_cand[
            torch.arange(len(hp_cand), device="cuda")[:, None],
            torch.arange(4, device="cuda")[None, :],
            selected,
        ]
        w, _ = run_optimizer(moved_h, steps, SEED + 51 + round_id)
    return w, selected, moved_h


def evaluate_method(
    w: np.ndarray,
    target_idx: np.ndarray,
    original_idx: np.ndarray,
    htrue_full: np.ndarray,
    hpred_target: np.ndarray,
    qfull: np.ndarray,
    labels: np.ndarray,
):
    target_h = htrue_full[target_idx]
    true = np.abs(np.einsum("bkpn,bn->bkp", target_h, w)) ** 2
    true = true.sum(-1) / 60
    pred = np.abs(np.einsum("bkpn,bn->bkp", hpred_target, w)) ** 2
    pred = pred.sum(-1) / 60
    global_peak, forward_peak, local_peak = [], [], np.zeros_like(true)
    front = qfull[:, 0] > 0
    cos5 = math.cos(math.radians(5))
    for lo in range(0, len(w), 12):
        hi = min(lo + 12, len(w))
        gain = np.abs(np.einsum("qpn,bn->bqp", htrue_full, w[lo:hi])) ** 2
        gain = gain.sum(-1) / 60
        global_peak.extend(gain.max(1))
        forward_peak.extend(gain[:, front].max(1))
        for bi in range(lo, hi):
            for k in range(4):
                mask = qfull @ qfull[target_idx[bi, k]] >= cos5
                local_peak[bi, k] = gain[bi - lo, mask].max()
    global_peak = np.asarray(global_peak)
    forward_peak = np.asarray(forward_peak)
    true_db = campaign.db(true)
    pred_db = campaign.db(pred)
    min_db = true_db.min(1)
    rel_global = true_db - campaign.db(global_peak)[:, None]
    rel_front = true_db - campaign.db(forward_peak)[:, None]
    local_gap = campaign.db(local_peak) - true_db
    bottleneck = true.argmin(1)
    be = (pred_db - true_db)[np.arange(len(w)), bottleneck]
    result = {
        "cases": len(w),
        "all_four_within_global_peak_minus10_rate": float(np.mean(rel_global.min(1) >= -10)),
        "all_four_within_forward_peak_minus10_rate": float(np.mean(rel_front.min(1) >= -10)),
        "all_four_within_1dB_of_local_peak_rate": float(np.mean(local_gap.max(1) <= 1)),
        "min_gain_dBi_p5_median_p95": np.percentile(min_db, [5, 50, 95]).tolist(),
        "min_relative_global_dB_p5_median_p95": np.percentile(rel_global.min(1), [5, 50, 95]).tolist(),
        "fourpoint_mae_dB": float(np.mean(np.abs(pred_db - true_db))),
        "bottleneck_overestimate_p95_dB": float(np.percentile(be, 95)),
        "bottleneck_overestimate_rate": float(np.mean(be > 0)),
        "local_peak_gap_dB_p50_p95": np.percentile(local_gap, [50, 95]).tolist(),
        "by_stratum": {},
    }
    for label in STRATA:
        mask = labels == label
        result["by_stratum"][label] = {
            "cases": int(mask.sum()),
            "all_strong_global_rate": float(np.mean(rel_global[mask].min(1) >= -10)),
            "min_gain_dBi_median": float(np.median(min_db[mask])),
            "min_gain_dBi_p5": float(np.percentile(min_db[mask], 5)),
            "local_peak_1dB_rate": float(np.mean(local_gap[mask].max(1) <= 1)),
        }
    original_true = np.abs(np.einsum("bkpn,bn->bkp", htrue_full[original_idx], w)) ** 2
    original_true = original_true.sum(-1) / 60
    result["original_coordinate_min_gain_dBi_p5_median"] = np.percentile(campaign.db(original_true.min(1)), [5, 50]).tolist()
    return result, true_db, rel_global, local_gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE / "artifacts" / "four_point_optimizer_20260905")
    parser.add_argument("--campaign-root", type=Path, default=BASE / "artifacts" / "vector_beam_20260905")
    args = parser.parse_args()
    out = args.root.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / "RESULTS.md").exists():
        print("STUDY ALREADY COMPLETE", out)
        return
    theta, phi, qfull, hspherical = campaign.ffd("exportelement_new")
    indices, labels = generate_cases(qfull, theta, phi)
    coords = []
    for case_id, (row, label) in enumerate(zip(indices, labels)):
        for point_id, idx in enumerate(row):
            coords.append({"case": case_id, "stratum": str(label), "point": point_id,
                           "theta_deg": float(theta[idx]), "phi_deg": float(phi[idx]),
                           "off_axis_deg": float(np.degrees(np.arccos(qfull[idx, 0])))})
    with (out / "coordinates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coords[0])); writer.writeheader(); writer.writerows(coords)
    protocol = {
        "status": "frozen_before_HFSS_evaluation",
        "seed": SEED,
        "coordinate_rule": "180 deterministic random four-point cases in +x cone <=60deg; 60 per geometry stratum",
        "strata": {"moderate": "min separation >=8deg, max 25..65deg",
                    "distributed": "min >=18deg, max 45..95deg",
                    "wide": "min >=25deg, max >=80deg"},
        "methods": METHODS,
        "constraint": "phase-only, |w_n|=1/4",
        "strong_definition": "all four true HFSS gains within 10dB of the same command's full-sphere peak",
        "cnn": "frozen C/seed29513; no parameter update",
        "boundary": "same historically exposed single HFSS state; optimizer study, not independent electromagnetic blind test",
    }
    dump(out / "protocol.json", protocol)
    htrue_full = campaign.cartesian(theta, phi, hspherical).astype(np.complex64)
    model = campaign.CompactVectorCNN().cuda()
    checkpoint = torch.load(args.campaign_root / "runs" / "C_29513" / "best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()
    if any(p.requires_grad for p in model.parameters()):
        for p in model.parameters(): p.requires_grad_(False)
    scale = campaign.read(args.campaign_root / "protocol.json")["scale"]
    q = qfull[indices].astype(np.float32)
    hp = campaign.predict(model, torch.tensor(q.reshape(-1, 3), device="cuda"), scale).reshape(len(q), 4, 3, 16)
    flat_q = flat_directions(q)
    flat_h = campaign.predict(model, torch.tensor(flat_q.reshape(-1, 3), device="cuda"), scale).reshape(len(q), 20, 3, 16)
    scan_q = fibonacci_sphere(METHODS["peak_ratio"]["scan_directions"])
    scan_h = campaign.predict(model, torch.tensor(scan_q, device="cuda"), scale)
    relocate_idx = relocation_candidates(indices, qfull, theta, phi)
    cand_q = torch.tensor(qfull[relocate_idx].astype(np.float32), device="cuda")
    weights, method_target_idx, method_hpred, elapsed = {}, {}, {}, {}
    for name in ("baseline_softmin", "eigen_multibeam", "flat_top_2deg", "peak_ratio"):
        start = time.perf_counter(); cfg = METHODS[name]
        if name == "baseline_softmin":
            w, _ = campaign.optimize(hp, mode="phase", starts=cfg["starts"], steps=cfg["steps"], seed=SEED + 10)
        elif name == "flat_top_2deg":
            w, _ = run_optimizer(hp, cfg["steps"], SEED + 20, objective_h=flat_h)
        elif name == "peak_ratio":
            # Limit memory without changing the predefined objective.
            chunks = []
            for lo in range(0, len(hp), 30):
                chunks.append(run_optimizer(hp[lo:lo+30], cfg["steps"], SEED + 30 + lo, scan_h=scan_h)[0])
            w = torch.cat(chunks)
        else:
            w, _ = run_optimizer(hp, cfg["steps"], SEED + 15)
        torch.cuda.synchronize(); elapsed[name] = time.perf_counter() - start
        weights[name] = w.cpu().numpy(); method_target_idx[name] = indices; method_hpred[name] = hp.cpu().numpy()
        print("OPTIMIZED", name, f"{elapsed[name]:.2f}s", flush=True)
    start = time.perf_counter()
    wr, selected, moved_hp = optimize_relocated(model, scale, hp, cand_q, METHODS["relocate_5deg"]["rounds"], METHODS["relocate_5deg"]["steps"])
    torch.cuda.synchronize(); elapsed["relocate_5deg"] = time.perf_counter() - start
    selected_np = selected.cpu().numpy()
    moved_idx = np.take_along_axis(relocate_idx, selected_np[:, :, None], axis=2).squeeze(2)
    weights["relocate_5deg"] = wr.cpu().numpy(); method_target_idx["relocate_5deg"] = moved_idx; method_hpred["relocate_5deg"] = moved_hp.cpu().numpy()
    print("OPTIMIZED relocate_5deg", f"{elapsed['relocate_5deg']:.2f}s", flush=True)
    # HFSS reference is opened only after all CNN-derived weights are frozen above.
    results, arrays = {}, {}
    for name in METHODS:
        result, true_db, rel_global, local_gap = evaluate_method(
            weights[name], method_target_idx[name], indices, htrue_full, method_hpred[name], qfull, labels)
        result["optimizer_seconds"] = elapsed[name]
        if name == "relocate_5deg":
            displacement = np.degrees(np.arccos(np.clip(np.sum(qfull[indices] * qfull[moved_idx], axis=-1), -1, 1)))
            result["coordinate_displacement_deg_p50_p95_max"] = np.percentile(displacement, [50, 95, 100]).tolist()
            result["warning"] = "coordinates changed; not a solution for fixed ground stations"
        results[name] = result
        arrays[name + "_weights"] = weights[name]
        arrays[name + "_target_indices"] = method_target_idx[name]
        arrays[name + "_true_gain_dBi"] = true_db
        arrays[name + "_relative_global_dB"] = rel_global
        arrays[name + "_local_peak_gap_dB"] = local_gap
        print("EVALUATED", name, result["all_four_within_global_peak_minus10_rate"], flush=True)
    np.savez_compressed(out / "study_arrays.npz", original_indices=indices, strata=labels, **arrays)
    protocol["status"] = "complete_no_posthoc_tuning"
    dump(out / "protocol.json", protocol)
    dump(out / "results.json", {"protocol": protocol, "results": results})
    rows = []
    for name, r in results.items():
        rows.append([name, 100*r["all_four_within_global_peak_minus10_rate"],
                     r["min_gain_dBi_p5_median_p95"][0], r["min_gain_dBi_p5_median_p95"][1],
                     100*r["all_four_within_1dB_of_local_peak_rate"], r["fourpoint_mae_dB"],
                     r["bottleneck_overestimate_p95_dB"]])
    lines = ["# 冻结CNN的随机四点优化实验", "",
             "CNN保持C/seed29513冻结；所有权值先由CNN生成，随后才用HFSS嵌入式方向图统一评估。",
             "180组四点均位于+x方向60°锥内，每种几何60组。该实验仍共享同一HFSS状态，不是新的电磁状态盲测。", "",
             "|方法|四点全在全空间峰值-10dB内|最小增益P5(dBi)|最小增益中位(dBi)|四点均距局部峰<=1dB|四点MAE(dB)|瓶颈高估P95(dB)|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row[0]}|{row[1]:.1f}%|{row[2]:.2f}|{row[3]:.2f}|{row[4]:.1f}%|{row[5]:.3f}|{row[6]:.3f}|")
    lines += ["", "## 解释", "",
              "- baseline_softmin为现有方法；eigen_multibeam以四个Q_k主特征向量及其相干组合初始化。",
              "- flat_top_2deg同时抬高每点及其2°邻域，目标是鲁棒平台而非尖峰。",
              "- peak_ratio在max-min之外惩罚192方向近似全局峰与最弱目标相差超过10dB。",
              "- relocate_5deg允许目标坐标在5°内移动，只用于地面站选址/服务区规划；不能代表固定站点性能。",
              "- strong采用HFSS真值和全空间峰值定义。局部峰采用目标5°邻域内HFSS最大值。", "",
              "逐方法、逐几何分层统计见results.json；全部坐标见coordinates.csv。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STUDY COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
