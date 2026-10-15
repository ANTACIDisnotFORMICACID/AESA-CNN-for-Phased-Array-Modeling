"""Post-freeze diagnostics for the front-hemisphere campaign.

This script is intentionally separate from the frozen development protocol.
It may only be run after ``final_report.json`` exists.  It adds full front-only
forward metrics, a CNN/optimizer cross check, and nominal-versus-neighbour
robustness diagnostics without changing model or solver selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import torch

import front_hemisphere_campaign as campaign
from v210.models.quadratic_psd import V210VectorBaseline, gains_from_factor


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_model(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = V210VectorBaseline().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def full_forward_metrics(
    model: torch.nn.Module,
    scale: float,
    data: dict[str, np.ndarray],
    device: str,
) -> dict:
    selected = np.flatnonzero(campaign.front_mask(data["q"]))
    query = torch.tensor(data["q"][selected], device=device)
    true_factor = torch.tensor(data["h"][selected], device=device)
    command = torch.tensor(data["commands"], device=device)
    predicted_factor = campaign.predict(model, query, scale)
    predicted_gain = gains_from_factor(predicted_factor, command).cpu().numpy()
    measured_gain = gains_from_factor(true_factor, command).cpu().numpy()
    result = campaign.forward_metrics(predicted_gain, measured_gain)
    spectral, _, predicted_q, true_q = campaign.q_error(
        predicted_factor, true_factor, need_vector=False
    )
    result.update(
        {
            "directions": int(len(selected)),
            "commands": int(len(data["commands"])),
            "reverse_directions_used": 0,
            "spectral_relative_rms": float(torch.sqrt(spectral).cpu()),
            "q_frobenius_nmse_db": float(
                10.0
                * torch.log10(
                    (predicted_q - true_q).abs().square().sum()
                    / true_q.abs().square().sum().clamp_min(1e-12)
                ).cpu()
            ),
        }
    )
    return result


def cnn_optimizer_cross(
    root: Path,
    final: dict,
    suite: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    device: str,
) -> dict:
    true_factor = torch.tensor(suite["h"], device=device)
    query = torch.tensor(suite["q"], device=device)
    reference = torch.tensor(exact["reference_minimum"], device=device)
    solo = torch.tensor(exact["solo_gain"], device=device)
    rows = []
    selected_recipe = final["selection"]["selected_cnn_recipe"]
    selected_seed = int(final["selection"]["representative_seed"])
    selected_high = None

    for recipe in campaign.CNN_RECIPES:
        for seed in campaign.SEEDS:
            model, checkpoint = load_model(
                root / "runs" / f"{recipe}_seed{seed}" / "best.pt", device
            )
            predicted = campaign.predict(
                model, query.reshape(-1, 3), checkpoint["scale"]
            ).reshape(len(query), 4, 3, 16)
            o0, _ = campaign.optimize_softmin(
                predicted,
                starts=campaign.CONFIG["optimizer_starts"],
                steps=campaign.CONFIG["optimizer_steps"],
                seed=21401,
                initialization="legacy",
            )
            metrics, _ = campaign.method_metrics(
                true_factor, o0, reference, solo, suite["stratum"]
            )
            metrics.update({"recipe": recipe, "seed": seed, "optimizer": "O0_legacy"})
            rows.append(metrics)
            if recipe == selected_recipe and seed == selected_seed:
                high, _ = campaign.optimize_softmin(
                    predicted,
                    starts=campaign.CONFIG["high_budget_starts"],
                    steps=campaign.CONFIG["high_budget_steps"],
                    seed=21421,
                    initialization="q_eigen",
                    initial=o0,
                )
                selected_high, _ = campaign.method_metrics(
                    true_factor, high, reference, solo, suite["stratum"]
                )
                selected_high.update(
                    {
                        "recipe": recipe,
                        "seed": seed,
                        "optimizer": "O3_high_budget_on_CNN",
                    }
                )

    legacy_model, legacy_checkpoint = load_model(
        campaign.legacy_checkpoint_path(root), device
    )
    legacy_predicted = campaign.predict(
        legacy_model, query.reshape(-1, 3), legacy_checkpoint["scale"]
    ).reshape(len(query), 4, 3, 16)
    legacy_o0, _ = campaign.optimize_softmin(
        legacy_predicted,
        starts=campaign.CONFIG["optimizer_starts"],
        steps=campaign.CONFIG["optimizer_steps"],
        seed=21401,
        initialization="legacy",
    )
    legacy_o0_metrics, _ = campaign.method_metrics(
        true_factor, legacy_o0, reference, solo, suite["stratum"]
    )
    legacy_high, _ = campaign.optimize_softmin(
        legacy_predicted,
        starts=campaign.CONFIG["high_budget_starts"],
        steps=campaign.CONFIG["high_budget_steps"],
        seed=21421,
        initialization="q_eigen",
        initial=legacy_o0,
    )
    legacy_high_metrics, _ = campaign.method_metrics(
        true_factor, legacy_high, reference, solo, suite["stratum"]
    )

    aggregate = {}
    q_eigen_rows = final["cnn"]["runs"]
    for recipe in campaign.CNN_RECIPES:
        o0_subset = [row for row in rows if row["recipe"] == recipe]
        o1_subset = [row for row in q_eigen_rows if row["recipe"] == recipe]
        aggregate[recipe] = {}
        for name, subset in (("O0_legacy", o0_subset), ("O1_q_eigen", o1_subset)):
            aggregate[recipe][name] = {
                "regret_median_mean": float(np.mean([row["regret_median"] for row in subset])),
                "regret_p95_mean": float(np.mean([row["regret_p95"] for row in subset])),
                "regret_le_10pct_mean": float(np.mean([row["regret_le_10pct"] for row in subset])),
            }
    return {
        "scope": "post-freeze holdout diagnostic; not used for selection",
        "rows_O0": rows,
        "aggregate_O0_vs_O1": aggregate,
        "selected_CNN_high_budget": selected_high,
        "legacy_A10_O0": legacy_o0_metrics,
        "legacy_A10_high_budget": legacy_high_metrics,
    }


@torch.no_grad()
def neighbour_robustness(
    suite: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    device: str,
) -> dict:
    nominal_factor = torch.tensor(suite["h"], device=device)
    neighbour_factor = torch.tensor(suite["neighbour_h"], device=device).reshape(
        len(nominal_factor), -1, nominal_factor.shape[2], 16
    )
    output = {}
    for method in (
        "O0_legacy",
        "O1_q_eigen",
        "O2_coordinate",
        "O3_high_budget",
        "O4_local_robust",
    ):
        weight = torch.tensor(exact[f"{method}_weight"], device=device)
        nominal_db = campaign.db(campaign.true_gain(nominal_factor, weight).min(-1).values)
        neighbour_db = campaign.db(campaign.true_gain(neighbour_factor, weight).min(-1).values)
        drop = nominal_db - neighbour_db
        output[method] = {
            "nominal_minimum_db_p5_median": torch.quantile(
                nominal_db, torch.tensor([0.05, 0.5], device=device)
            ).cpu().tolist(),
            "neighbour_minimum_db_p5_median": torch.quantile(
                neighbour_db, torch.tensor([0.05, 0.5], device=device)
            ).cpu().tolist(),
            "nominal_to_neighbour_drop_db_median_p95": torch.quantile(
                drop, torch.tensor([0.5, 0.95], device=device)
            ).cpu().tolist(),
        }
    return output


def spectral_feasibility_upper_bound(
    suite: dict[str, np.ndarray], exact: dict[str, np.ndarray]
) -> dict:
    """Bound normalized max-min gain using convex combinations of target Q.

    For every nonnegative alpha summing to one,
    min_i w^H(Q_i/s_i)w <= lambda_max(sum_i alpha_i Q_i/s_i)
    for any unit-norm w.  The bound therefore also applies to the smaller
    constant-modulus feasible set.  Numerical solo gains s_i make the test
    conservative: an underestimated denominator only raises the bound.
    """

    factor = suite["h"].astype(np.complex128)
    qmatrix = np.einsum("bkrn,bkrm->bknm", factor.conj(), factor) / 60.0
    solo = np.maximum(exact["solo_gain"].astype(np.float64), 1e-30)
    normalized_q = qmatrix / solo[:, :, None, None]
    equal = np.empty(len(factor), dtype=np.float64)
    optimized = np.empty(len(factor), dtype=np.float64)
    alpha_value = np.empty((len(factor), 4), dtype=np.float64)

    def softmax(value: np.ndarray) -> np.ndarray:
        shifted = value - value.max()
        probability = np.exp(shifted)
        return probability / probability.sum()

    for case in range(len(factor)):
        def bound_from_alpha(alpha: np.ndarray) -> float:
            matrix = np.einsum("k,knm->nm", alpha, normalized_q[case])
            return float(np.linalg.eigvalsh(matrix)[-1].real)

        uniform = np.full(4, 0.25)
        equal[case] = bound_from_alpha(uniform)
        result = minimize(
            lambda value: bound_from_alpha(softmax(value)),
            np.zeros(4),
            method="BFGS",
            options={"maxiter": 100, "gtol": 1e-8},
        )
        candidate = softmax(result.x) if np.isfinite(result.fun) else uniform
        candidate_bound = bound_from_alpha(candidate)
        if candidate_bound > equal[case]:
            candidate = uniform
            candidate_bound = equal[case]
        alpha_value[case] = candidate
        optimized[case] = candidate_bound

    achieved = np.min(
        np.maximum(exact["O3_high_budget_gain"], 1e-30) / solo,
        axis=1,
    )
    result = {
        "statement": (
            "Each reported value is a valid spectral upper bound for any unit-norm complex "
            "weight, hence also for phase-only constant-modulus weights. Optimization of alpha "
            "tightens the bound but is not required for validity."
        ),
        "optimized_upper_bound_p5_median_p95": np.percentile(
            optimized, [5, 50, 95]
        ).tolist(),
        "equal_weight_upper_bound_p5_median_p95": np.percentile(
            equal, [5, 50, 95]
        ).tolist(),
        "achieved_to_upper_bound_ratio_p5_median": np.percentile(
            achieved / np.maximum(optimized, 1e-30), [5, 50]
        ).tolist(),
        "fraction_certified_below_0_5": float(np.mean(optimized < 0.5)),
        "fraction_certified_below_0_7": float(np.mean(optimized < 0.7)),
        "fraction_certified_below_0_9": float(np.mean(optimized < 0.9)),
        "by_stratum": {},
        "optimized_upper_bound": optimized.tolist(),
        "alpha": alpha_value.tolist(),
    }
    for name in np.unique(suite["stratum"]):
        selected = suite["stratum"] == name
        result["by_stratum"][str(name)] = {
            "upper_bound_p5_median": np.percentile(
                optimized[selected], [5, 50]
            ).tolist(),
            "fraction_certified_below_0_7": float(
                np.mean(optimized[selected] < 0.7)
            ),
        }
    return result


def coordinate_relationship(
    suite: dict[str, np.ndarray], exact: dict[str, np.ndarray]
) -> dict:
    query = suite["q"].astype(np.float64)
    cosine = np.einsum("bik,bjk->bij", query, query).clip(-1.0, 1.0)
    angles = np.rad2deg(np.arccos(cosine))
    upper = np.triu_indices(4, 1)
    pair = angles[:, upper[0], upper[1]]
    boresight = np.rad2deg(np.arccos(query[:, :, 0].clip(-1.0, 1.0)))
    gain = exact["O3_high_budget_gain"]
    solo = np.maximum(exact["solo_gain"], 1e-30)
    absolute_db = 10.0 * np.log10(np.maximum(gain.min(axis=1), 1e-30))
    normalized = np.min(gain / solo, axis=1)
    features = {
        "minimum_pair_angle_deg": pair.min(axis=1),
        "mean_pair_angle_deg": pair.mean(axis=1),
        "maximum_pair_angle_deg": pair.max(axis=1),
        "mean_boresight_angle_deg": boresight.mean(axis=1),
        "maximum_boresight_angle_deg": boresight.max(axis=1),
        "mean_qx": query[:, :, 0].mean(axis=1),
        "minimum_qx": query[:, :, 0].min(axis=1),
    }
    correlations = {
        name: {
            "absolute_minimum_gain_db": float(spearmanr(value, absolute_db).statistic),
            "normalized_feasibility": float(spearmanr(value, normalized).statistic),
        }
        for name, value in features.items()
    }

    def bins(value: np.ndarray, edges: list[float]) -> list[dict]:
        rows = []
        for lower, upper_edge in zip(edges[:-1], edges[1:]):
            selected = (value >= lower) & (value < upper_edge)
            if not selected.any():
                continue
            rows.append(
                {
                    "lower": lower,
                    "upper": upper_edge,
                    "cases": int(selected.sum()),
                    "absolute_minimum_gain_db_p5_median": np.percentile(
                        absolute_db[selected], [5, 50]
                    ).tolist(),
                    "normalized_feasibility_p5_median": np.percentile(
                        normalized[selected], [5, 50]
                    ).tolist(),
                }
            )
        return rows

    return {
        "interpretation": (
            "Angular separation is expected to relate more directly to normalized feasibility; "
            "absolute gain also contains the direction-dependent single-point ceiling."
        ),
        "spearman": correlations,
        "by_maximum_pair_angle_deg": bins(pair.max(axis=1), [0, 15, 45, 75, 105, 181]),
        "by_mean_boresight_angle_deg": bins(
            boresight.mean(axis=1), [0, 30, 50, 70, 90.001]
        ),
    }


def strong_field_diagnostic(
    root: Path,
    final: dict,
    suite: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    device: str,
) -> dict:
    """Check whether optimized targets are truly within their single-point strong region."""

    solo = torch.tensor(exact["solo_gain"], device=device)
    exact_gain = torch.tensor(exact["O3_high_budget_gain"], device=device)
    exact_ratio = exact_gain / solo.clamp_min(1e-30)
    selected_recipe = final["selection"]["selected_cnn_recipe"]
    selected_seed = int(final["selection"]["representative_seed"])
    model, checkpoint = load_model(
        root / "runs" / f"{selected_recipe}_seed{selected_seed}" / "best.pt",
        device,
    )
    query = torch.tensor(suite["q"], device=device)
    true_factor = torch.tensor(suite["h"], device=device)
    predicted_factor = campaign.predict(
        model, query.reshape(-1, 3), checkpoint["scale"]
    ).reshape(len(query), 4, 3, 16)
    weight, _ = campaign.optimize_softmin(
        predicted_factor,
        starts=campaign.CONFIG["optimizer_starts"],
        steps=campaign.CONFIG["optimizer_steps"],
        seed=21401,
        initialization="legacy",
    )
    _, predicted_solo = campaign.single_point_ceiling(predicted_factor)
    predicted_gain = campaign.true_gain(predicted_factor, weight)
    measured_gain = campaign.true_gain(true_factor, weight)
    predicted_ratio = predicted_gain / predicted_solo.clamp_min(1e-30)
    measured_ratio = measured_gain / solo.clamp_min(1e-30)
    error = campaign.db(predicted_gain) - campaign.db(measured_gain)
    point_strong = measured_ratio >= 0.1
    predicted_point_strong = predicted_ratio >= 0.1
    all_strong = point_strong.all(dim=1)
    predicted_all_strong = predicted_point_strong.all(dim=1)
    reference_minimum = torch.tensor(exact["reference_minimum"], device=device)
    regret = (
        1.0 - measured_gain.min(dim=1).values / reference_minimum.clamp_min(1e-30)
    ).clamp_min(0.0)

    def error_stat(mask: torch.Tensor) -> dict:
        value = error[mask]
        if value.numel() == 0:
            return {"count": 0}
        return {
            "count": int(value.numel()),
            "mae_db": float(value.abs().mean()),
            "bias_db": float(value.mean()),
            "p95_abs_db": float(torch.quantile(value.abs(), 0.95)),
        }

    def case_stat(mask: torch.Tensor) -> dict:
        if not bool(mask.any()):
            return {"cases": 0}
        selected_error = error[mask]
        minimum_error = (
            campaign.db(predicted_gain[mask].min(dim=1).values)
            - campaign.db(measured_gain[mask].min(dim=1).values)
        )
        return {
            "cases": int(mask.sum()),
            "point_mae_db": float(selected_error.abs().mean()),
            "worst_point_abs_error_p95_db": float(
                torch.quantile(selected_error.abs().max(dim=1).values, 0.95)
            ),
            "minimum_gain_error_mae_db": float(minimum_error.abs().mean()),
            "minimum_gain_error_p95_abs_db": float(
                torch.quantile(minimum_error.abs(), 0.95)
            ),
            "optimizer_regret_p95": float(torch.quantile(regret[mask], 0.95)),
        }

    thresholds = {}
    for threshold in (0.1, 0.2, 0.5, 0.7, 0.9):
        selected_true = measured_ratio.min(dim=1).values >= threshold
        thresholds[str(threshold)] = {
            "exact_O3_fraction_all_four": float(
                (exact_ratio.min(dim=1).values >= threshold).float().mean()
            ),
            "selected_CNN_O0_fraction_all_four_true": float(
                selected_true.float().mean()
            ),
            "selected_CNN_O0_fraction_all_four_predicted": float(
                (predicted_ratio.min(dim=1).values >= threshold).float().mean()
            ),
            "selected_CNN_O0_true_case_metrics": case_stat(selected_true),
        }
    true_positive = predicted_all_strong & all_strong
    return {
        "definition": (
            "A point is strong when its realized gain is at least 0.1 of its own "
            "single-point numerical ceiling, equivalent to within 10 dB."
        ),
        "selected_model": f"{selected_recipe}/{selected_seed}",
        "thresholds": thresholds,
        "point_error_when_true_strong": error_stat(point_strong),
        "point_error_when_true_not_strong": error_stat(~point_strong),
        "cases_all_four_true_strong": case_stat(all_strong),
        "cases_not_all_four_true_strong": case_stat(~all_strong),
        "all_four_strong_gate": {
            "predicted_positive_cases": int(predicted_all_strong.sum()),
            "true_positive_cases": int(true_positive.sum()),
            "precision": float(
                true_positive.sum() / predicted_all_strong.sum().clamp_min(1)
            ),
            "recall": float(true_positive.sum() / all_strong.sum().clamp_min(1)),
        },
    }


@torch.no_grad()
def hemisphere_maximum(
    front_factor: torch.Tensor,
    front_query: torch.Tensor,
    weight: torch.Tensor,
    batch: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = []
    peak_query = []
    for start in range(0, len(weight), batch):
        current = weight[start : start + batch]
        field = torch.einsum("drn,bn->bdr", front_factor, current)
        gain = field.abs().square().sum(-1) / 60.0
        value, index = gain.max(dim=1)
        maximum.append(value)
        peak_query.append(front_query[index])
    return torch.cat(maximum), torch.cat(peak_query)


def hemisphere_relative_diagnostic(
    root: Path,
    final: dict,
    suite: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    device: str,
) -> dict:
    """Measure target strength relative to each optimized pattern's front-hemisphere peak."""

    front_query_parts = []
    front_factor_parts = []
    for name in ("train", "validation", "calibration", "holdout"):
        data = campaign.dataset(root, name)
        selected = campaign.front_mask(data["q"])
        front_query_parts.append(data["q"][selected])
        front_factor_parts.append(data["h"][selected])
    front_query_numpy = np.concatenate(front_query_parts)
    front_factor_numpy = np.concatenate(front_factor_parts)
    if len(front_query_numpy) != 32400:
        raise ValueError("combined front grid must contain 32,400 directions")
    if len(np.unique(np.round(front_query_numpy, 7), axis=0)) != len(front_query_numpy):
        raise ValueError("combined front grid contains duplicate directions")
    campaign.assert_front(front_query_numpy, "combined one-degree front grid")
    front_query = torch.tensor(front_query_numpy, device=device)
    front_factor = torch.tensor(front_factor_numpy, device=device)
    target_query = torch.tensor(suite["q"], device=device)
    target_factor = torch.tensor(suite["h"], device=device)

    selected_recipe = final["selection"]["selected_cnn_recipe"]
    selected_seed = int(final["selection"]["representative_seed"])
    model, checkpoint = load_model(
        root / "runs" / f"{selected_recipe}_seed{selected_seed}" / "best.pt",
        device,
    )
    predicted_factor = campaign.predict(
        model, target_query.reshape(-1, 3), checkpoint["scale"]
    ).reshape(len(target_query), 4, 3, 16)
    selected_weight, _ = campaign.optimize_softmin(
        predicted_factor,
        starts=campaign.CONFIG["optimizer_starts"],
        steps=campaign.CONFIG["optimizer_steps"],
        seed=21401,
        initialization="legacy",
    )
    methods = {
        "O0_exact": torch.tensor(exact["O0_legacy_weight"], device=device),
        "O3_exact": torch.tensor(exact["O3_high_budget_weight"], device=device),
        f"{selected_recipe}_{selected_seed}_O0": selected_weight,
    }
    result = {
        "definition": (
            "rho_i(w)=G(q_i,w)/max_{q_x>0}G(q,w), where the denominator is scanned "
            "on the combined 32,400-direction one-degree HFSS front grid."
        ),
        "grid": {
            "directions": int(len(front_query_numpy)),
            "unique_directions": int(
                len(np.unique(np.round(front_query_numpy, 7), axis=0))
            ),
            "reverse_directions_used": 0,
            "continuous_hemisphere_certificate": False,
        },
        "methods": {},
    }

    pair_cosine = np.einsum("bik,bjk->bij", suite["q"], suite["q"]).clip(-1, 1)
    pair_angle = np.rad2deg(np.arccos(pair_cosine))
    upper = np.triu_indices(4, 1)
    pair = pair_angle[:, upper[0], upper[1]]
    boresight = np.rad2deg(np.arccos(suite["q"][:, :, 0].clip(-1, 1)))

    for name, weight in methods.items():
        target_gain = campaign.true_gain(target_factor, weight)
        maximum, peak_query = hemisphere_maximum(
            front_factor, front_query, weight
        )
        rho = target_gain / maximum[:, None].clamp_min(1e-30)
        minimum_rho = rho.min(dim=1).values
        target_to_peak = torch.rad2deg(
            torch.arccos(
                torch.einsum("bik,bk->bi", target_query, peak_query).clamp(-1.0, 1.0)
            )
        )
        minimum_gain_db = campaign.db(target_gain.min(dim=1).values)
        method = {
            "minimum_rho_p5_median_p95": torch.quantile(
                minimum_rho, torch.tensor([0.05, 0.5, 0.95], device=device)
            ).cpu().tolist(),
            "minimum_gain_db_p5_median": torch.quantile(
                minimum_gain_db, torch.tensor([0.05, 0.5], device=device)
            ).cpu().tolist(),
            "hemisphere_peak_gain_db_p5_median": torch.quantile(
                campaign.db(maximum), torch.tensor([0.05, 0.5], device=device)
            ).cpu().tolist(),
            "nearest_target_to_peak_deg_p5_median": torch.quantile(
                target_to_peak.min(dim=1).values,
                torch.tensor([0.05, 0.5], device=device),
            ).cpu().tolist(),
            "farthest_target_to_peak_deg_median_p95": torch.quantile(
                target_to_peak.max(dim=1).values,
                torch.tensor([0.5, 0.95], device=device),
            ).cpu().tolist(),
            "thresholds": {},
            "by_stratum": {},
        }
        for threshold in (0.1, 0.2, 0.5, 0.7, 0.9):
            method["thresholds"][str(threshold)] = {
                "fraction_all_four": float(
                    (minimum_rho >= threshold).float().mean()
                ),
                "cases": int((minimum_rho >= threshold).sum()),
            }
        minimum_rho_numpy = minimum_rho.cpu().numpy()
        target_to_peak_numpy = target_to_peak.cpu().numpy()
        features = {
            "minimum_pair_angle_deg": pair.min(axis=1),
            "mean_pair_angle_deg": pair.mean(axis=1),
            "maximum_pair_angle_deg": pair.max(axis=1),
            "mean_boresight_angle_deg": boresight.mean(axis=1),
            "nearest_target_to_peak_deg": target_to_peak_numpy.min(axis=1),
            "farthest_target_to_peak_deg": target_to_peak_numpy.max(axis=1),
        }
        method["spearman_with_minimum_rho"] = {
            feature: float(spearmanr(value, minimum_rho_numpy).statistic)
            for feature, value in features.items()
        }
        for stratum in np.unique(suite["stratum"]):
            selected = suite["stratum"] == stratum
            values = minimum_rho_numpy[selected]
            method["by_stratum"][str(stratum)] = {
                "cases": int(selected.sum()),
                "minimum_rho_p5_median": np.percentile(values, [5, 50]).tolist(),
                "fraction_all_four_above_0_1": float(np.mean(values >= 0.1)),
                "fraction_all_four_above_0_5": float(np.mean(values >= 0.5)),
            }
        if name == f"{selected_recipe}_{selected_seed}_O0":
            predicted_gain = campaign.true_gain(predicted_factor, weight)
            error = campaign.db(predicted_gain) - campaign.db(target_gain)
            conditional = {}
            for threshold in (0.1, 0.2, 0.5):
                selected = minimum_rho >= threshold
                if not bool(selected.any()):
                    conditional[str(threshold)] = {"cases": 0}
                    continue
                selected_error = error[selected]
                conditional[str(threshold)] = {
                    "cases": int(selected.sum()),
                    "point_mae_db": float(selected_error.abs().mean()),
                    "worst_point_abs_error_p95_db": float(
                        torch.quantile(selected_error.abs().max(dim=1).values, 0.95)
                    ),
                    "minimum_gain_error_p95_abs_db": float(
                        torch.quantile(
                            (
                                campaign.db(predicted_gain[selected].min(dim=1).values)
                                - campaign.db(target_gain[selected].min(dim=1).values)
                            ).abs(),
                            0.95,
                        )
                    ),
                }
            method["CNN_error_conditioned_on_true_rho"] = conditional
        result["methods"][name] = method
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    campaign.verify_protocol(root)
    final = campaign.read_json(root / "final_report.json")
    if not (root / "holdout_opened.json").exists():
        raise RuntimeError("holdout has not been opened by the frozen campaign")

    holdout = campaign.dataset(root, "holdout")
    forward = {"runs": [], "aggregate": {}}
    for recipe in campaign.CNN_RECIPES:
        for seed in campaign.SEEDS:
            model, checkpoint = load_model(
                root / "runs" / f"{recipe}_seed{seed}" / "best.pt", arguments.device
            )
            metrics = full_forward_metrics(model, checkpoint["scale"], holdout, arguments.device)
            metrics.update({"recipe": recipe, "seed": seed})
            forward["runs"].append(metrics)
    legacy, legacy_checkpoint = load_model(
        campaign.legacy_checkpoint_path(root), arguments.device
    )
    forward["legacy_A10"] = full_forward_metrics(
        legacy, legacy_checkpoint["scale"], holdout, arguments.device
    )
    for recipe in campaign.CNN_RECIPES:
        subset = [row for row in forward["runs"] if row["recipe"] == recipe]
        forward["aggregate"][recipe] = {
            "amplitude_nmse_db_mean": float(np.mean([row["amplitude_nmse_db"] for row in subset])),
            "amplitude_nmse_db_std": float(np.std([row["amplitude_nmse_db"] for row in subset])),
            "all_mae_db_mean": float(np.mean([row["bins"]["all"]["mae_db"] for row in subset])),
            "strong_mae_db_mean": float(np.mean([row["bins"]["strong"]["mae_db"] for row in subset])),
            "spectral_relative_rms_mean": float(np.mean([row["spectral_relative_rms"] for row in subset])),
        }

    suite = campaign.load_npz(root / "holdout_front_suite.npz")
    exact = campaign.load_npz(root / "holdout_optimizer_arrays.npz")
    result = {
        "scope": "post-freeze diagnostics on already-opened front-only holdout",
        "front_condition": "q_x > 0",
        "reverse_samples_used": 0,
        "forward": forward,
        "cnn_optimizer_cross": cnn_optimizer_cross(
            root, final, suite, exact, arguments.device
        ),
        "neighbour_robustness": neighbour_robustness(
            suite, exact, arguments.device
        ),
        "spectral_feasibility_upper_bound": spectral_feasibility_upper_bound(
            suite, exact
        ),
        "coordinate_relationship": coordinate_relationship(suite, exact),
        "strong_field_diagnostic": strong_field_diagnostic(
            root, final, suite, exact, arguments.device
        ),
        "hemisphere_relative_diagnostic": hemisphere_relative_diagnostic(
            root, final, suite, exact, arguments.device
        ),
    }
    write_json(root / "post_analysis.json", result)
    print("POST ANALYSIS COMPLETE", root / "post_analysis.json", flush=True)


if __name__ == "__main__":
    main()
