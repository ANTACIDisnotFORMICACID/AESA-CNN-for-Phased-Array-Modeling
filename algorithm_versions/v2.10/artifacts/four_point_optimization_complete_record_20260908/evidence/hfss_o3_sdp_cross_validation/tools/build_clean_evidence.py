"""Build O3-only raw results, readable tables, figures, and source hashes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[3]
FRONT = SOURCE / "artifacts" / "front_hemisphere_20260906"
STRONG = SOURCE / "artifacts" / "hfss_four_strong_highsearch_20260906"
OLD_SDP = SOURCE / "four_point_optimization_limit_evidence_20260906" / "raw" / "sdp_ceiling_and_refinement"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def db(value: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(np.asarray(value, dtype=float), 1e-30))


def export_split(split: str) -> list[dict[str, object]]:
    suite_path = FRONT / f"{split}_front_suite.npz"
    optimizer_path = FRONT / f"{split}_optimizer_arrays.npz"
    strong_path = STRONG / f"{split}_arrays.npz"
    sdp_path = OLD_SDP / f"{split}_absolute_maxmin_sdp_arrays.npz"
    suite = np.load(suite_path, allow_pickle=False)
    optimizer = np.load(optimizer_path, allow_pickle=False)
    strong = np.load(strong_path, allow_pickle=False)
    sdp = np.load(sdp_path, allow_pickle=False)

    o3_out = ROOT / "raw" / "o3_results" / f"{split}_o3_results.npz"
    np.savez_compressed(
        o3_out,
        method=np.array("H0_O3_32x320_Adam_phase_only"),
        cnn_files_loaded=np.array([], dtype="U1"),
        source_index=strong["source_index"],
        q=strong["q"],
        stratum=strong["stratum"],
        weight=strong["H0_target_maxmin_weight"],
        target_gain=strong["H0_target_maxmin_target_gain"],
        minimum_gain_db=strong["H0_target_maxmin_minimum_gain_db"],
        hemisphere_peak_gain=strong["H0_target_maxmin_hemisphere_peak_gain"],
        rho=strong["H0_target_maxmin_minimum_rho"],
        solo_utilization=strong["H0_target_maxmin_minimum_solo_utilization"],
        target_loss_db=strong["H0_target_maxmin_target_loss_db"],
        spectral_solo_upper_bound=strong["spectral_solo_upper_bound"],
        solo_gain=optimizer["solo_gain"],
        source_sha256=np.array(
            [sha256(suite_path), sha256(optimizer_path), sha256(strong_path)]
        ),
    )
    upper_out = ROOT / "raw" / "sdp_upper_bound" / f"{split}_sdp_upper_bound.npz"
    np.savez_compressed(
        upper_out,
        source_index=sdp["source_index"],
        stratum=sdp["stratum"],
        o3_minimum=sdp["o3_minimum"],
        constant_modulus_sdp_upper=sdp["sdp_upper"],
        arbitrary_amplitude_trace_upper=sdp["trace_upper"],
        constant_modulus_gap_db=sdp["sdp_gap_db"],
        trace_gap_db=sdp["trace_gap_db"],
        leading_share=sdp["leading_share"],
        effective_rank=sdp["effective_rank"],
        rank_99=sdp["rank_99"],
        status=sdp["status"],
        dual_repair=sdp["dual_repair"],
        diagonal_error=sdp["diag_error"],
        minimum_lifted_eigenvalue=sdp["minimum_lifted_eigenvalue"],
        source_sha256=np.array(sha256(sdp_path)),
    )

    gains_db = db(strong["H0_target_maxmin_target_gain"])
    weights = strong["H0_target_maxmin_weight"]
    rows: list[dict[str, object]] = []
    for i in range(len(weights)):
        row: dict[str, object] = {
            "case_index": i,
            "source_index": int(strong["source_index"][i]),
            "group": str(strong["stratum"][i]),
            "method": "H0_O3_32x320_Adam_phase_only",
            "g1_db": float(gains_db[i, 0]),
            "g2_db": float(gains_db[i, 1]),
            "g3_db": float(gains_db[i, 2]),
            "g4_db": float(gains_db[i, 3]),
            "gmin_db": float(strong["H0_target_maxmin_minimum_gain_db"][i]),
            "rho": float(strong["H0_target_maxmin_minimum_rho"][i]),
            "solo_utilization": float(
                strong["H0_target_maxmin_minimum_solo_utilization"][i]
            ),
            "sdp_upper_db": float(db(sdp["sdp_upper"][i])),
            "sdp_gap_db": float(sdp["sdp_gap_db"][i]),
        }
        for t in range(4):
            for axis, label in enumerate("xyz"):
                row[f"q{t+1}_{label}"] = float(strong["q"][i, t, axis])
            row[f"solo_g{t+1}_db"] = float(db(optimizer["solo_gain"][i, t]))
        for port in range(16):
            row[f"w{port+1}_magnitude"] = float(abs(weights[i, port]))
            row[f"w{port+1}_phase_deg"] = float(
                np.degrees(np.angle(weights[i, port]))
            )
        rows.append(row)
    with (ROOT / "tables" / f"o3_{split}_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_statistics(all_rows: dict[str, list[dict[str, object]]]) -> None:
    thresholds: list[dict[str, object]] = []
    strata: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for split, rows in all_rows.items():
        values = np.array([row["gmin_db"] for row in rows], dtype=float)
        gaps = np.array([row["sdp_gap_db"] for row in rows], dtype=float)
        quant = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
        summary.append(
            {
                "split": split,
                "method": "H0_O3_32x320_Adam_phase_only",
                "cases": len(rows),
                "gmin_min_db": quant[0],
                "gmin_p5_db": quant[1],
                "gmin_p25_db": quant[2],
                "gmin_median_db": quant[3],
                "gmin_p75_db": quant[4],
                "gmin_p95_db": quant[5],
                "gmin_max_db": quant[6],
                "sdp_gap_median_db": np.median(gaps),
                "sdp_gap_p95_db": np.percentile(gaps, 95),
            }
        )
        for threshold in (0, 1, 3, 5, 8, 10, 12):
            thresholds.append(
                {
                    "split": split,
                    "method": "H0_O3_32x320_Adam_phase_only",
                    "threshold_db": threshold,
                    "passed": int(np.sum(values >= threshold)),
                    "cases": len(values),
                    "pass_rate": float(np.mean(values >= threshold)),
                }
            )
        for group in ("clustered", "moderate", "wide", "grazing"):
            chosen = [row for row in rows if row["group"] == group]
            group_values = np.array([row["gmin_db"] for row in chosen], dtype=float)
            strata.append(
                {
                    "split": split,
                    "group": group,
                    "method": "H0_O3_32x320_Adam_phase_only",
                    "cases": len(chosen),
                    "gmin_p5_db": np.percentile(group_values, 5),
                    "gmin_median_db": np.median(group_values),
                    "gmin_p95_db": np.percentile(group_values, 95),
                    "pass_5db": np.mean(group_values >= 5),
                    "pass_8db": np.mean(group_values >= 8),
                    "pass_10db": np.mean(group_values >= 10),
                }
            )
    write_csv(ROOT / "RESULTS.csv", summary)
    write_csv(ROOT / "tables" / "threshold_rates.csv", thresholds)
    write_csv(ROOT / "tables" / "stratum_statistics.csv", strata)


def build_figures(rows: list[dict[str, object]]) -> None:
    labels = ["clustered", "moderate", "wide", "grazing"]
    display = ["集中\n四点较近", "中等分散", "宽分布", "掠射\n接近阵列侧面"]
    values = [
        [row["gmin_db"] for row in rows if row["group"] == label] for label in labels
    ]
    plt.figure(figsize=(9, 4.8))
    plot = plt.boxplot(values, tick_labels=display, showfliers=False, patch_artist=True)
    for patch, color in zip(plot["boxes"], ["#3b82f6", "#60a5fa", "#f59e0b", "#ef4444"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    plt.axhline(5, color="#374151", linestyle="--", linewidth=1, label="5 dB参考线")
    plt.ylabel("四点中的最低增益 Gmin（dB）")
    plt.title("四类目标均使用同一H0/O3算法（600组独立测试）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "o3_gmin_by_group.png", dpi=180)
    plt.close()

    thresholds = np.array([0, 1, 3, 5, 8, 10, 12])
    all_values = np.array([row["gmin_db"] for row in rows], dtype=float)
    rates = np.array([np.mean(all_values >= t) * 100 for t in thresholds])
    plt.figure(figsize=(8.5, 4.6))
    bars = plt.bar(thresholds.astype(str), rates, color="#2563eb")
    for bar, value in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)
    plt.ylim(0, 106)
    plt.xlabel("要求四点最低增益达到的门槛（dB）")
    plt.ylabel("达到门槛的案例比例（%）")
    plt.title("同一H0/O3算法下的四点最低增益达标率")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "o3_threshold_success_rate.png", dpi=180)
    plt.close()


def main() -> None:
    for path in (
        ROOT / "raw" / "o3_results",
        ROOT / "raw" / "sdp_upper_bound",
        ROOT / "tables",
        ROOT / "figures",
    ):
        path.mkdir(parents=True, exist_ok=True)
    all_rows = {split: export_split(split) for split in ("validation", "holdout")}
    build_statistics(all_rows)
    build_figures(all_rows["holdout"])
    print("CLEAN_O3_EVIDENCE_BUILT")


if __name__ == "__main__":
    main()
