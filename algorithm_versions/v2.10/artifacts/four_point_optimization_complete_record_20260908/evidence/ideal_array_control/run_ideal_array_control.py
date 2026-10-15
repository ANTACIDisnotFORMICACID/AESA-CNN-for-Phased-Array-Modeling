"""Paired ideal-array control for the frozen V2.10 four-target HFSS campaign.

The target directions, phase-only constraint, O3 initialization/optimization
budget, gain convention, and evaluation metrics are held fixed.  Only the
3x16 HFSS active-element factor is replaced by an ideal 4x4 array factor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
VERSION_ROOT = HERE.parents[3]
SOURCE = HERE.parent / "hfss_o3_sdp_cross_validation"
FRONT = VERSION_ROOT / "artifacts" / "front_hemisphere_20260906"
FULL_DATA = FRONT / "datasets"

FREQUENCY_HZ = 47.0e9
C0 = 299_792_458.0
LAMBDA_M = C0 / FREQUENCY_HZ
DY_M = 6.8e-3
DZ_M = 15.0e-3
STARTS_O1 = 8
STEPS_O1 = 160
COORDINATE_SWEEPS = 40
STARTS_O3 = 32
STEPS_O3 = 320
SEED_O1 = 21311
SEED_O3 = 21321
LEARNING_RATE = 0.065


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db(value: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(value.clamp_min(1e-30))


def quadratic_form(factor: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...rn,...rm->...nm", factor.conj(), factor) / 60.0


def true_gain(factor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bkrn,bn->bkr", factor, weight).abs().square().sum(-1) / 60.0


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
    return torch.exp(1j * torch.angle(torch.stack(candidates[:starts], dim=1))) / 4.0


def optimize_softmin(
    factor: torch.Tensor,
    *,
    starts: int,
    steps: int,
    seed: int,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = factor.detach()
    batch, targets, _, ports = factor.shape
    weight = q_eigen_initial(factor, starts, seed)
    if initial is not None:
        weight[:, -1] = initial.detach()
    phase = nn.Parameter(torch.angle(weight))
    optimizer = torch.optim.Adam([phase], lr=LEARNING_RATE)
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
        selected = weight.detach()[torch.arange(batch), index]
        best_weight[improved] = selected[improved]
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
    factor: torch.Tensor, initial: torch.Tensor, sweeps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    qmatrix = quadratic_form(factor)
    weight = initial.detach().clone()
    best = db(true_gain(factor, weight)).min(-1).values
    best_weight = weight.clone()
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
        value = db(true_gain(factor, weight)).min(-1).values
        improved = value > best
        best = torch.maximum(best, value)
        best_weight[improved] = weight[improved]
    return best_weight, best


def array_positions() -> np.ndarray:
    y = (np.arange(4, dtype=np.float64) - 1.5) * DY_M
    z = (np.arange(4, dtype=np.float64) - 1.5) * DZ_M
    return np.asarray([(0.0, yy, zz) for zz in z for yy in y], dtype=np.float64)


def ideal_factor(q: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    positions = array_positions()
    phase = np.exp(-1j * (2.0 * np.pi / LAMBDA_M) * np.einsum("...d,nd->...n", q, positions))
    factor = np.zeros(q.shape[:-1] + (3, 16), dtype=np.complex64)
    factor[..., 0, :] = amplitude[..., None] * phase
    return factor


def boresight_rms_amplitude() -> tuple[float, dict]:
    best_qx = -np.inf
    best = None
    best_path = None
    for folder, name in (("dev", "train"), ("dev", "validation"), ("reserved", "calibration"), ("reserved", "holdout")):
        path = FULL_DATA / folder / f"{name}.npz"
        with np.load(path, allow_pickle=False) as data:
            index = int(np.argmax(data["q"][:, 0]))
            if float(data["q"][index, 0]) > best_qx:
                best_qx = float(data["q"][index, 0])
                best = (data["q"][index].copy(), data["h"][index].copy())
                best_path = path
    q, h = best
    amplitude = float(np.sqrt(np.mean(np.sum(np.abs(h) ** 2, axis=0))))
    return amplitude, {
        "source": str(best_path),
        "q": q.astype(float).tolist(),
        "mean_embedded_element_field_norm": amplitude,
    }


def optimize_model(factor_np: np.ndarray, batch_size: int = 48) -> dict[str, np.ndarray]:
    weights = []
    gains = []
    minimum = []
    started = time.perf_counter()
    for begin in range(0, len(factor_np), batch_size):
        factor = torch.tensor(factor_np[begin : begin + batch_size], dtype=torch.complex64)
        o1, _ = optimize_softmin(
            factor, starts=STARTS_O1, steps=STEPS_O1, seed=SEED_O1
        )
        o2, _ = coordinate_refine(factor, o1, COORDINATE_SWEEPS)
        o3, _ = optimize_softmin(
            factor,
            starts=STARTS_O3,
            steps=STEPS_O3,
            seed=SEED_O3,
            initial=o2,
        )
        gain = true_gain(factor, o3)
        weights.append(o3.numpy())
        gains.append(gain.numpy())
        minimum.append(gain.min(-1).values.numpy())
        print(f"optimized {min(begin + batch_size, len(factor_np))}/{len(factor_np)}", flush=True)
    return {
        "weight": np.concatenate(weights),
        "gain": np.concatenate(gains),
        "minimum": np.concatenate(minimum),
        "runtime_seconds": np.asarray(time.perf_counter() - started),
    }


def bootstrap_ci(values: np.ndarray, seed: int, iterations: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    medians = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 500):
        count = min(500, iterations - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        medians[start : start + count] = np.median(values[sample], axis=1)
    return np.percentile(medians, [2.5, 97.5]).tolist()


def summarize(values: np.ndarray) -> dict:
    return {
        "minimum": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    amplitude_boresight, boresight_record = boresight_rms_amplitude()
    rows = []
    raw = {}
    protocol_inputs = []

    for split in ("validation", "holdout"):
        suite_path = FRONT / f"{split}_front_suite.npz"
        o3_path = SOURCE / "raw" / "o3_results" / f"{split}_o3_results.npz"
        protocol_inputs.extend([suite_path, o3_path])
        with np.load(suite_path, allow_pickle=False) as suite, np.load(o3_path, allow_pickle=False) as o3:
            q = suite["q"].astype(np.float64)
            hfss_h = suite["h"].astype(np.complex64)
            stratum = suite["stratum"].astype(str)
            hfss_gain = o3["target_gain"].astype(np.float64)
            hfss_min_db = o3["minimum_gain_db"].astype(np.float64)
            saved_weight = o3["weight"].astype(np.complex64)
            recomputed = np.sum(np.abs(np.einsum("bkrn,bn->bkr", hfss_h, saved_weight)) ** 2, axis=-1) / 60.0
            if np.max(np.abs(recomputed - hfss_gain)) > 2e-5:
                raise RuntimeError("saved HFSS O3 result failed recomputation")

            rms_envelope = np.sqrt(np.mean(np.sum(np.abs(hfss_h) ** 2, axis=-2), axis=-1))
            common_h = ideal_factor(q, rms_envelope)
            isotropic_h = ideal_factor(q, np.full(q.shape[:-1], amplitude_boresight))

            for model_name, factor in (("common_element_envelope", common_h), ("isotropic_boresight_calibrated", isotropic_h)):
                print(f"{split}: {model_name}", flush=True)
                result = optimize_model(factor)
                minimum_db = 10.0 * np.log10(np.maximum(result["minimum"], 1e-30))
                gain_db = 10.0 * np.log10(np.maximum(result["gain"], 1e-30))
                solo = 16.0 * (rms_envelope if model_name == "common_element_envelope" else amplitude_boresight) ** 2 / 60.0
                utilization_db = 10.0 * np.log10(np.maximum(np.min(result["gain"] / solo, axis=1), 1e-30))
                raw[f"{split}_{model_name}_weight"] = result["weight"]
                raw[f"{split}_{model_name}_gain"] = result["gain"]
                raw[f"{split}_{model_name}_minimum_db"] = minimum_db
                raw[f"{split}_{model_name}_utilization_db"] = utilization_db
                raw[f"{split}_{model_name}_runtime_seconds"] = result["runtime_seconds"]
                for case in range(len(q)):
                    row = {
                        "split": split,
                        "case_index": case,
                        "source_index": int(o3["source_index"][case]),
                        "stratum": stratum[case],
                        "model": model_name,
                        "hfss_gmin_db": float(hfss_min_db[case]),
                        "ideal_gmin_db": float(minimum_db[case]),
                        "delta_ideal_minus_hfss_db": float(minimum_db[case] - hfss_min_db[case]),
                        "ideal_min_solo_utilization_db": float(utilization_db[case]),
                        "hfss_min_solo_utilization_db": float(10.0 * np.log10(max(float(o3["solo_utilization"][case]), 1e-30))),
                    }
                    for point in range(4):
                        row[f"q{point+1}_x"] = float(q[case, point, 0])
                        row[f"q{point+1}_y"] = float(q[case, point, 1])
                        row[f"q{point+1}_z"] = float(q[case, point, 2])
                        row[f"hfss_g{point+1}_db"] = float(10.0 * np.log10(max(hfss_gain[case, point], 1e-30)))
                        row[f"ideal_g{point+1}_db"] = float(gain_db[case, point])
                    rows.append(row)

    with (HERE / "paired_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(HERE / "ideal_o3_raw_results.npz", **raw)

    summary = {
        "contract": {
            "frequency_hz": FREQUENCY_HZ,
            "wavelength_mm": LAMBDA_M * 1e3,
            "array": "4x4 rectangular, boresight +x, columns +y, rows +z",
            "dy_mm": DY_M * 1e3,
            "dz_mm": DZ_M * 1e3,
            "dy_lambda": DY_M / LAMBDA_M,
            "dz_lambda": DZ_M / LAMBDA_M,
            "element_amplitude": 0.25,
            "total_weight_norm_squared": 1.0,
            "gain_formula": "sum_r |H_r w|^2 / 60",
            "optimizer": "O3: q-eigen O1 8x160 Adam + 40 coordinate sweeps + 32x320 Adam soft-min, O2 included as final start",
            "cnn_used": False,
        },
        "boresight_calibration": boresight_record,
        "inputs": {str(path): sha256(path) for path in protocol_inputs},
        "results": {},
    }
    for split in ("validation", "holdout"):
        summary["results"][split] = {}
        for model_name in ("common_element_envelope", "isotropic_boresight_calibrated"):
            selected = [row for row in rows if row["split"] == split and row["model"] == model_name]
            hfss = np.asarray([row["hfss_gmin_db"] for row in selected])
            ideal = np.asarray([row["ideal_gmin_db"] for row in selected])
            delta = ideal - hfss
            result = {
                "cases": len(selected),
                "hfss_gmin_db": summarize(hfss),
                "ideal_gmin_db": summarize(ideal),
                "paired_delta_db": summarize(delta),
                "paired_median_delta_95pct_bootstrap_ci_db": bootstrap_ci(delta, 20260907),
                "fraction_ideal_higher": float(np.mean(delta > 0.0)),
                "fraction_ideal_higher_by_at_least_1db": float(np.mean(delta >= 1.0)),
                "fraction_ideal_higher_by_at_least_3db": float(np.mean(delta >= 3.0)),
                "wilcoxon_two_sided_p": float(wilcoxon(delta, alternative="two-sided").pvalue),
                "by_stratum": {},
            }
            for group in ("clustered", "moderate", "wide", "grazing"):
                group_delta = np.asarray([row["delta_ideal_minus_hfss_db"] for row in selected if row["stratum"] == group])
                result["by_stratum"][group] = {
                    "cases": len(group_delta),
                    "paired_delta_db": summarize(group_delta),
                    "fraction_ideal_higher": float(np.mean(group_delta > 0.0)),
                }
            summary["results"][split][model_name] = result

    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {
        "purpose": "paired control isolating ideal array geometry from the stored HFSS active-element operator",
        "models": {
            "common_element_envelope": "At each target direction, use RMS HFSS per-port field norm as one common element envelope, then apply only the ideal geometric phase. Removes port-dependent embedded-pattern/coupling structure while retaining direction-dependent element strength.",
            "isotropic_boresight_calibrated": "Use a direction-independent element field norm equal to the nearest-boresight RMS HFSS per-port field norm, then apply ideal geometric phase. Removes element scan loss as well as coupling/asymmetry.",
        },
        "causal_boundary": "A difference between HFSS and common-element-envelope models measures all port-dependent full-wave nonidealities together; it cannot identify mutual coupling alone without a no-coupling full-wave model or network de-embedding.",
        "source_code_sha256": sha256(Path(__file__)),
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
