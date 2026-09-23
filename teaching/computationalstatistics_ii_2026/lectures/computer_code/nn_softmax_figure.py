"""Draw a worked softmax example; run manually before rendering Quarto."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn-softmax-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def generate():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none"})
    colours = ["#0f4494", "#7d3c98", "#be2832"]
    ink = "#232754"
    scores = np.array([2.0, 1.0, 0.0])
    weights = np.exp(scores)
    probabilities = weights / weights.sum()
    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    ax.set(xlim=(-0.8, 8.55), ylim=(-0.7, 4.3))
    ax.axis("off")
    for x, label in [(1.25, "Score"), (3.65, "Positive value"), (6.95, "Probability")]:
        ax.text(x, 3.82, label, color=ink, weight="bold", ha="center", fontsize=15)
    ax.text(3.65, 3.35, "Exponentiate", color=ink, ha="center", fontsize=13)
    ax.text(6.95, 3.35, "Divide by the total", color=ink, ha="center", fontsize=13)

    for k, (s, weight, probability, colour) in enumerate(
            zip(scores, weights, probabilities, colours), 1):
        y = 3.5 - 1.04 * k
        ax.text(-0.65, y, f"Class {k}", color=colour, ha="left", va="center", fontsize=15)
        ax.add_patch(FancyBboxPatch((0.87, y - 0.30), 0.76, 0.60,
                                   boxstyle="round,pad=0.03,rounding_size=0.07",
                                   facecolor="white", edgecolor=colour, lw=1.8))
        ax.text(1.25, y, f"{s:.0f}", color=colour, ha="center", va="center", fontsize=19)
        for start, end in [(1.9, 2.9), (4.35, 5.2)]:
            ax.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="-|>",
                                        mutation_scale=13, color="#94a3b8", lw=1.4))
        ax.text(3.65, y, f"{weight:.2f}", color=colour, ha="center", va="center", fontsize=18)
        ax.barh(y, 2.1, left=5.5, height=0.34, color="#edf0f5")
        ax.barh(y, 2.1 * probability, left=5.5, height=0.34, color=colour)
        ax.text(7.82, y, f"{probability:.3f}", color=colour, ha="left", va="center", fontsize=16)

    ax.plot([2.95, 4.35], [-0.1, -0.1], color="#cbd5e1", lw=1)
    ax.text(3.65, -0.46, f"Total ≈ {weights.sum():.2f}", color=ink,
            ha="center", va="center", fontsize=14)
    ax.text(6.9, -0.46, "Sum = 1", color=ink, ha="center", va="center", fontsize=14)
    out = Path(__file__).resolve().parents[1] / "images" / "nn-softmax.svg"
    fig.savefig(out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    generate()
