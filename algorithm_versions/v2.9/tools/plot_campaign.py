"""Render completed campaign evidence; never feeds results back into training."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
colors = {"R1": "#1665a8", "R2": "#e87924", "R3": "#23945a"}
for ax, phase in zip(axes, ("screen", "confirm", "final")):
    for path in sorted((root/"runs"/phase).glob("*/history.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        recipe, seed = path.parent.name.split("_seed")
        linestyle = {"29001": "-", "29002": "--", "29003": ":"}[seed]
        ax.plot([r["epoch"] for r in history], [r["best_nmse_db"] for r in history],
                label=recipe+" / "+seed, color=colors[recipe], linestyle=linestyle)
    ax.set_title(phase.capitalize()+" - validation only")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best amplitude NMSE (dB; lower is better)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("V2.9 fixed architecture: 300 / 800 / up to 2000 epochs", fontsize=14)
fig.savefig(root/"convergence_curves.png", dpi=170)
plt.close(fig)
report_path = root/"dual_blind_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["runs"]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    positions = np.arange(3)
    for shift, label, snapshot, split, color in (
        (-.26, "Epoch 200 - integer blind", "epoch200", "integer_blind", "#c0c6cd"),
        (0, "Selected - integer blind", "best", "integer_blind", "#1665a8"),
        (.26, "Selected - shifted blind blocks", "best", "shifted_blind_blocks", "#23945a")):
        values = [r[snapshot][split]["regions"]["full_sphere"]["global_amplitude_nmse_db"] for r in rows]
        bars = ax.bar(positions+shift, values, width=.25, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(positions, [str(r["seed"]) for r in rows])
    ax.set_xlabel("Model seed")
    ax.set_ylabel("Amplitude NMSE (dB; lower is better)")
    ax.set_title("Dual-blind results after configuration / checkpoint freeze")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(root/"dual_blind_comparison.png", dpi=170)
    plt.close(fig)
print("PLOTS_WRITTEN")
