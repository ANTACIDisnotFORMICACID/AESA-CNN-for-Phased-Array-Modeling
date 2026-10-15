from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEPS = HERE / "sdp_deps"
sys.path.insert(0, str(DEPS))

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


V210 = HERE.parents[4]
SOURCE = V210 / "artifacts/front_hemisphere_20260906"
FOUR_STRONG = V210 / "artifacts/hfss_four_strong_highsearch_20260906"
OUTPUT = HERE / "absolute_maxmin_sdp_20260906"
PORTS = 16
TARGETS = 4
POWER_SCALE = 60.0
MODULUS_SQUARED = 1.0 / PORTS


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def db(value: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(value, 1e-30))


def quantiles(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.percentile(value, [5, 50, 95])]


def absolute_trace_upper_bound(qmatrix: np.ndarray, initial: np.ndarray) -> float:
    """Upper bound after dropping constant modulus and retaining ||w||_2=1."""

    def objective(alpha: np.ndarray) -> float:
        matrix = np.einsum("k,knm->nm", alpha, qmatrix)
        return float(np.linalg.eigvalsh(matrix)[-1].real)

    starts = [np.full(TARGETS, 1.0 / TARGETS), initial, *np.eye(TARGETS)]
    values = [objective(starts[0])]
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * TARGETS,
            constraints={"type": "eq", "fun": lambda alpha: alpha.sum() - 1.0},
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if np.isfinite(result.fun):
            values.append(float(result.fun))
    # Every evaluated simplex point gives a valid upper bound. Minimization only
    # tightens it, so an optimizer status failure cannot manufacture infeasibility.
    return min(values)


class ConstantModulusSdp:
    def __init__(self) -> None:
        self.q = [cp.Parameter((PORTS, PORTS), hermitian=True) for _ in range(TARGETS)]
        self.w = cp.Variable((PORTS, PORTS), hermitian=True)
        self.t = cp.Variable()
        self.psd_constraint = self.w >> 0
        self.diag_constraint = cp.real(cp.diag(self.w)) == MODULUS_SQUARED
        self.target_constraints = [
            cp.real(cp.trace(self.q[index] @ self.w)) >= self.t
            for index in range(TARGETS)
        ]
        self.problem = cp.Problem(
            cp.Maximize(self.t),
            [self.psd_constraint, self.diag_constraint, *self.target_constraints],
        )

    def solve(self, qmatrix: np.ndarray) -> dict[str, object]:
        for parameter, value in zip(self.q, qmatrix, strict=True):
            parameter.value = (value + value.conj().T) / 2.0

        value = self.problem.solve(
            solver="SCS",
            eps=1e-7,
            max_iters=50_000,
            warm_start=True,
            verbose=False,
        )
        if self.problem.status not in {"optimal", "optimal_inaccurate"}:
            value = self.problem.solve(solver="CLARABEL", warm_start=True, verbose=False)
        if self.w.value is None:
            raise RuntimeError(f"SDP failed with status {self.problem.status}")

        lifted = np.asarray(self.w.value, dtype=np.complex128)
        lifted = (lifted + lifted.conj().T) / 2.0
        eigenvalue, eigenvector = np.linalg.eigh(lifted)
        order = np.argsort(eigenvalue)[::-1]
        eigenvalue = eigenvalue[order]
        eigenvector = eigenvector[:, order]
        positive = np.maximum(eigenvalue, 0.0)
        normalized = positive / max(positive.sum(), 1e-30)

        alpha = np.asarray(
            [constraint.dual_value for constraint in self.target_constraints],
            dtype=np.float64,
        ).reshape(-1)
        alpha = np.maximum(alpha, 0.0)
        if not np.isfinite(alpha).all() or alpha.sum() <= 0:
            alpha = np.full(TARGETS, 1.0 / TARGETS)
        else:
            alpha /= alpha.sum()

        diagonal = np.asarray(self.diag_constraint.dual_value, dtype=np.float64).reshape(-1)
        combined = np.einsum("k,knm->nm", alpha, qmatrix)
        slack = np.diag(diagonal) - combined
        minimum_slack_eigenvalue = float(np.linalg.eigvalsh(slack)[0].real)
        safety = 1e-9 * max(1.0, float(np.linalg.norm(combined, ord=2)))
        repair = max(0.0, -minimum_slack_eigenvalue) + safety
        certified_upper = float(diagonal.mean() + repair)

        target_value = np.asarray(
            [np.trace(matrix @ lifted).real for matrix in qmatrix], dtype=np.float64
        )
        return {
            "status": self.problem.status,
            "raw_objective": float(value),
            "certified_upper": certified_upper,
            "dual_alpha": alpha,
            "dual_diagonal": diagonal,
            "dual_repair": repair,
            "minimum_slack_eigenvalue": minimum_slack_eigenvalue,
            "lifted": lifted,
            "lifted_target_minimum": float(target_value.min()),
            "diag_error": float(np.max(np.abs(np.diag(lifted).real - MODULUS_SQUARED))),
            "minimum_lifted_eigenvalue": float(eigenvalue[-1]),
            "eigenvalue": eigenvalue,
            "leading_eigenvector": eigenvector[:, 0],
            "leading_share": float(normalized[0]),
            "effective_rank": float(1.0 / max(np.square(normalized).sum(), 1e-30)),
            "rank_99": int(np.searchsorted(np.cumsum(normalized), 0.99) + 1),
        }


def gains(qmatrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.einsum("sn,knm,sm->sk", weight.conj(), qmatrix, weight).real


def phase_candidates(
    qmatrix: np.ndarray,
    lifted: np.ndarray,
    leading_vector: np.ndarray,
    existing_weight: list[tuple[str, np.ndarray]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, np.ndarray, str]:
    eigenvalue, eigenvector = np.linalg.eigh(lifted)
    positive = np.maximum(eigenvalue, 0.0)
    factor = eigenvector @ np.diag(np.sqrt(positive))
    normal = (
        rng.standard_normal((samples, PORTS))
        + 1j * rng.standard_normal((samples, PORTS))
    ) / math.sqrt(2.0)
    randomized = normal @ factor.T
    randomized = np.exp(1j * np.angle(randomized)) / math.sqrt(PORTS)
    principal = np.exp(1j * np.angle(leading_vector))[None, :] / math.sqrt(PORTS)

    names = [name for name, _ in existing_weight]
    known = np.stack([weight for _, weight in existing_weight], axis=0)
    candidate = np.concatenate((known, principal, randomized), axis=0)
    value = gains(qmatrix, candidate).min(axis=1)
    index = int(np.argmax(value))
    if index < len(names):
        source = names[index]
    elif index == len(names):
        source = "sdp_principal_phase"
    else:
        source = "sdp_randomized_phase"
    return float(value[index]), candidate[index], source


def classify(gap_db: float) -> str:
    if gap_db <= 0.3:
        return "optimizer_closed_le_0p3db"
    if gap_db <= 1.0:
        return "limited_headroom_0p3_to_1db"
    return "targeted_solver_worthwhile_gt_1db"


def run_suite(suite_name: str, samples: int, seed: int) -> dict[str, object]:
    suite_path = SOURCE / f"{suite_name}_front_suite.npz"
    exact_path = SOURCE / f"{suite_name}_optimizer_arrays.npz"
    strong_path = FOUR_STRONG / f"{suite_name}_arrays.npz"
    suite = np.load(suite_path, allow_pickle=False)
    exact = np.load(exact_path, allow_pickle=False)
    strong = np.load(strong_path, allow_pickle=False)

    h = suite["h"].astype(np.complex128)
    qmatrix_all = np.einsum("bkrn,bkrm->bknm", h.conj(), h) / POWER_SCALE
    o3 = exact["O3_high_budget_minimum"].astype(np.float64)
    case_count = len(h)

    result: dict[str, list[object]] = {
        "status": [],
        "sdp_upper": [],
        "trace_upper": [],
        "sdp_gap_db": [],
        "trace_gap_db": [],
        "leading_share": [],
        "effective_rank": [],
        "rank_99": [],
        "dual_repair": [],
        "diag_error": [],
        "minimum_lifted_eigenvalue": [],
        "candidate_minimum": [],
        "candidate_gain_over_o3_db": [],
        "candidate_source": [],
        "candidate_weight": [],
        "dual_alpha": [],
        "dual_diagonal": [],
        "lifted_matrix": [],
        "sdp_eigenvalue": [],
        "classification": [],
    }
    exact_weight_keys = sorted(
        key for key in exact.files if key.endswith("_weight") and exact[key].ndim == 2
    )
    strong_weight_keys = sorted(key for key in strong.files if key.endswith("_weight"))
    solver = ConstantModulusSdp()
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    for case, qmatrix in enumerate(qmatrix_all):
        solved = solver.solve(qmatrix)
        existing = [(key, exact[key][case]) for key in exact_weight_keys]
        existing.extend((key, strong[key][case]) for key in strong_weight_keys)
        candidate_value, candidate_weight, candidate_source = phase_candidates(
            qmatrix,
            solved["lifted"],
            solved["leading_eigenvector"],
            existing,
            rng,
            samples,
        )
        sdp_upper = max(float(solved["certified_upper"]), float(o3[case]))
        trace_upper = max(
            absolute_trace_upper_bound(qmatrix, solved["dual_alpha"]),
            sdp_upper,
        )
        sdp_gap = float(db(sdp_upper / o3[case]))
        trace_gap = float(db(trace_upper / o3[case]))
        candidate_gain = float(db(candidate_value / o3[case]))

        result["status"].append(str(solved["status"]))
        result["sdp_upper"].append(sdp_upper)
        result["trace_upper"].append(trace_upper)
        result["sdp_gap_db"].append(sdp_gap)
        result["trace_gap_db"].append(trace_gap)
        result["leading_share"].append(float(solved["leading_share"]))
        result["effective_rank"].append(float(solved["effective_rank"]))
        result["rank_99"].append(int(solved["rank_99"]))
        result["dual_repair"].append(float(solved["dual_repair"]))
        result["diag_error"].append(float(solved["diag_error"]))
        result["minimum_lifted_eigenvalue"].append(
            float(solved["minimum_lifted_eigenvalue"])
        )
        result["candidate_minimum"].append(candidate_value)
        result["candidate_gain_over_o3_db"].append(candidate_gain)
        result["candidate_source"].append(candidate_source)
        result["candidate_weight"].append(candidate_weight)
        result["dual_alpha"].append(solved["dual_alpha"])
        result["dual_diagonal"].append(solved["dual_diagonal"])
        result["lifted_matrix"].append(solved["lifted"])
        result["sdp_eigenvalue"].append(solved["eigenvalue"])
        result["classification"].append(classify(sdp_gap))

        if (case + 1) % 25 == 0 or case + 1 == case_count:
            print(
                json.dumps(
                    {
                        "suite": suite_name,
                        "complete": case + 1,
                        "cases": case_count,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )

    arrays = {key: np.asarray(value) for key, value in result.items()}
    sdp_gap = arrays["sdp_gap_db"].astype(np.float64)
    candidate_gain = arrays["candidate_gain_over_o3_db"].astype(np.float64)
    classes, counts = np.unique(arrays["classification"], return_counts=True)
    summary = {
        "suite": suite_name,
        "cases": case_count,
        "objective": "maximize absolute four-target minimum gain under |w_n|=1/4",
        "relaxation": "W PSD, diag(W)=1/16, rank-one constraint removed",
        "solver": "CVXPY 1.9.2 with SCS eps=1e-7; dual PSD slack repaired upward",
        "randomized_phase_samples_per_case": samples,
        "o3_minimum_gain_db_p5_median_p95": quantiles(db(o3)),
        "constant_modulus_sdp_upper_db_p5_median_p95": quantiles(
            db(arrays["sdp_upper"].astype(np.float64))
        ),
        "constant_modulus_sdp_gap_db_p5_median_p95": quantiles(sdp_gap),
        "unrestricted_trace_gap_db_p5_median_p95": quantiles(
            arrays["trace_gap_db"].astype(np.float64)
        ),
        "leading_eigenvalue_share_p5_median_p95": quantiles(
            arrays["leading_share"].astype(np.float64)
        ),
        "effective_rank_p5_median_p95": quantiles(
            arrays["effective_rank"].astype(np.float64)
        ),
        "rank_99_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(arrays["rank_99"].astype(np.int64), return_counts=True),
                strict=True,
            )
        },
        "classification_counts": {
            str(key): int(value) for key, value in zip(classes, counts, strict=True)
        },
        "fraction_gap_le_0p3db": float(np.mean(sdp_gap <= 0.3)),
        "fraction_gap_le_0p5db": float(np.mean(sdp_gap <= 0.5)),
        "fraction_gap_le_1db": float(np.mean(sdp_gap <= 1.0)),
        "fraction_gap_gt_1db": float(np.mean(sdp_gap > 1.0)),
        "candidate_gain_over_o3_db_p5_median_p95": quantiles(candidate_gain),
        "candidate_improved_over_0p01db": int(np.sum(candidate_gain > 0.01)),
        "candidate_improved_over_0p1db": int(np.sum(candidate_gain > 0.1)),
        "maximum_candidate_improvement_db": float(candidate_gain.max()),
        "maximum_dual_repair": float(arrays["dual_repair"].astype(float).max()),
        "maximum_diagonal_error": float(arrays["diag_error"].astype(float).max()),
        "minimum_lifted_eigenvalue": float(
            arrays["minimum_lifted_eigenvalue"].astype(float).min()
        ),
        "input_sha256": {
            suite_path.name: digest(suite_path),
            exact_path.name: digest(exact_path),
            strong_path.name: digest(strong_path),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / f"{suite_name}_absolute_maxmin_sdp_arrays.npz",
        source_index=strong["source_index"],
        q=suite["q"],
        stratum=suite["stratum"],
        o3_minimum=o3,
        **arrays,
    )
    (OUTPUT / f"{suite_name}_absolute_maxmin_sdp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    order = np.argsort(-sdp_gap)
    with (OUTPUT / f"{suite_name}_ranked_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rank",
                "case_index",
                "source_index",
                "stratum",
                "o3_minimum_db",
                "constant_modulus_sdp_upper_db",
                "sdp_gap_db",
                "unrestricted_trace_gap_db",
                "leading_eigenvalue_share",
                "effective_rank",
                "rank_99",
                "classification",
                "candidate_gain_over_o3_db",
                "candidate_source",
            ]
        )
        for rank, case in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    int(case),
                    int(strong["source_index"][case]),
                    str(suite["stratum"][case]),
                    float(db(o3[case])),
                    float(db(arrays["sdp_upper"][case])),
                    float(sdp_gap[case]),
                    float(arrays["trace_gap_db"][case]),
                    float(arrays["leading_share"][case]),
                    float(arrays["effective_rank"][case]),
                    int(arrays["rank_99"][case]),
                    str(arrays["classification"][case]),
                    float(candidate_gain[case]),
                    str(arrays["candidate_source"][case]),
                ]
            )
    return summary


def write_report(summaries: list[dict[str, object]]) -> None:
    holdout = next(item for item in summaries if item["suite"] == "holdout")
    counts = holdout["classification_counts"]
    lines = [
        "# 四点绝对 max-min 恒模 SDP 证书",
        "",
        "本报告直接回答固定 4x4、47 GHz、单一 phase-only 权值下，继续更换优化器还有多少绝对最小增益空间。",
        "这里不使用 `rho` 或单点归一化 `u`；目标是 `max min_i G_i(w)`。",
        "",
        "## 留出集结论",
        "",
        f"- O3 最小增益 P5/中位/P95：`{holdout['o3_minimum_gain_db_p5_median_p95']}` dB。",
        f"- 恒模 SDP 上界 P5/中位/P95：`{holdout['constant_modulus_sdp_upper_db_p5_median_p95']}` dB。",
        f"- O3 到恒模 SDP 上界差距 P5/中位/P95：`{holdout['constant_modulus_sdp_gap_db_p5_median_p95']}` dB。",
        f"- 差距不超过 0.3/0.5/1.0 dB 的比例：`{holdout['fraction_gap_le_0p3db']:.2%}` / `{holdout['fraction_gap_le_0p5db']:.2%}` / `{holdout['fraction_gap_le_1db']:.2%}`。",
        f"- 真正值得进一步全局 phase-only 搜索（差距 >1 dB）：`{counts.get('targeted_solver_worthwhile_gt_1db', 0)}` / `{holdout['cases']}`。",
        f"- SDP 主特征值占比 P5/中位/P95：`{holdout['leading_eigenvalue_share_p5_median_p95']}`；高秩表示上界可能需要多状态协方差，不能由单一权值直接实现。",
        f"- SDP 相位投影与随机化相对 O3 的最大实际改进：`{holdout['maximum_candidate_improvement_db']:.4f}` dB。",
        "",
        "## 判定规则",
        "",
        "- `gap <= 0.3 dB`：优化器基本闭合，继续投入收益很小。",
        "- `0.3 < gap <= 1 dB`：存在有限余量，但不足以改变四点能力的定性结论。",
        "- `gap > 1 dB`：进入定向流形优化、SDP 随机化和必要时分支定界。",
        "- SDP 高秩：即使 SDP 上界高，也不能把差距全部解释为当前优化器不足；它可能是单状态秩一限制。",
        "",
        "逐样本结果见 `holdout_ranked_cases.csv` 和 `holdout_absolute_maxmin_sdp_arrays.npz`。",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", default="validation,holdout", help="comma-separated suite names"
    )
    parser.add_argument("--randomized-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=21701)
    args = parser.parse_args()
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    summaries = [
        run_suite(name, args.randomized_samples, args.seed + 1009 * index)
        for index, name in enumerate(suites)
    ]
    if "holdout" in suites:
        write_report(summaries)
    print(json.dumps({"status": "ABSOLUTE_MAXMIN_SDP_OK", "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
