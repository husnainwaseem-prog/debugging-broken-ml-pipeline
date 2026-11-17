"""One chart for the demo: broken vs fixed on the three metrics that
exist for both runs. The broken run has no ROC-AUC (its test set holds
a single class, so ROC-AUC is undefined) — that is left out of the
chart on purpose and noted in the caption, not plotted as zero.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_before_after_plot(broken_metrics: dict, fixed_metrics: dict, out_path) -> pathlib.Path:
    labels = ["Accuracy", "Balanced acc.", "F1"]
    broken_vals = [
        broken_metrics["accuracy"],
        broken_metrics["balanced_accuracy"],
        broken_metrics["f1"],
    ]
    fixed_vals = [
        fixed_metrics["accuracy"],
        fixed_metrics["balanced_accuracy"],
        fixed_metrics["f1"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, broken_vals, width, label="Broken (reported)", color="#c0392b")
    ax.bar(x + width / 2, fixed_vals, width, label="Fixed (honest)", color="#2e7d32")
    ax.set_ylabel("Score")
    ax.set_title("Broken vs fixed pipeline — same data, honest metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    for bars in (ax.containers):
        ax.bar_label(bars, fmt="%.2f", padding=2)
    fig.tight_layout()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def default_plot_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "results" / "plots" / "before_after.png"
