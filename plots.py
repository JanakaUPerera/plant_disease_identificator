"""Figures shared by all experiments (Question 3)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLOURS = {"train": "#2a78d6", "val_lab": "#eb6834", "val_wild": "#1baf7a"}
MARKERS = {"train": "o", "val_lab": "s", "val_wild": "D"}
RANDOM_BENCHMARK = 0.285  # random guessing within the crop, val_wild disease macro-F1 (Question 1)


def plot_history(run_id, results_dir="results", save=True):
    """Learning curves of one run: losses (left) and disease macro-F1 (right) per epoch."""
    h = pd.read_csv(Path(results_dir) / "runs" / run_id / "history.csv")
    best = h["val_wild_macro_f1_16_disease"].idxmax()
    e_best = h.at[best, "epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for name, col in [("train", "train_loss"), ("val_lab", "val_lab_loss"), ("val_wild", "val_wild_loss")]:
        ax1.plot(h["epoch"], h[col], color=COLOURS[name], marker=MARKERS[name], ms=5, lw=2, label=name)
    ax1.set_ylabel("cross-entropy loss")

    for name in ("val_lab", "val_wild"):
        ax2.plot(h["epoch"], h[f"{name}_macro_f1_16_disease"], color=COLOURS[name], marker=MARKERS[name],
                 ms=5, lw=2, label=name)
    ax2.axhline(RANDOM_BENCHMARK, color="#8a8984", lw=1, ls="--")
    ax2.text(h["epoch"].max(), RANDOM_BENCHMARK, "random guess (val_wild)", color="#52514e", fontsize=8,
             ha="right", va="top")
    ax2.axvline(e_best, color="#8a8984", lw=1, ls=":")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("disease macro-F1")

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(color="#ebeae5", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    lab, wild = h.at[best, "val_lab_macro_f1_16_disease"], h.at[best, "val_wild_macro_f1_16_disease"]
    fig.suptitle(f"{run_id}: best val_wild disease macro-F1 {wild:.3f} at epoch {e_best} "
                 f"(val_lab {lab:.3f}, lab-to-wild gap {lab - wild:.2f})", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if save:
        Path("figures").mkdir(exist_ok=True)
        fig.savefig(Path("figures") / f"q3_curves_{run_id}.png", dpi=200)
    return fig