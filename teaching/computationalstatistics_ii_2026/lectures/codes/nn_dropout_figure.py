"""Generate the dropout diagram manually; Quarto never executes this script.

The displayed mask is one possible realisation, not an expected prediction.
Run from any working directory. An optional --preview path writes a PNG copy.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn-dropout-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


BLUE = "#0f4494"
PURPLE = "#7d3c98"
RED = "#be2832"
GREY = "#8793a3"
INK = "#263445"


def arrow(ax, start, end, colour, *, dashed=False, width=2, shrink=21):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            shrinkA=shrink,
            shrinkB=21,
            linewidth=width,
            color=colour,
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    )


def node(ax, xy, label, colour, *, fill="white", size=17, radius=0.31):
    ax.add_patch(Circle(xy, radius, facecolor=fill, edgecolor=colour, lw=2, zorder=3))
    ax.text(*xy, label, fontsize=size, color=colour, ha="center", va="center", zorder=4)


def panel(ax, training):
    ax.set(xlim=(0, 5.8), ylim=(0, 4.2), aspect="equal")
    ax.axis("off")
    ax.text(2.9, 3.99, "Training" if training else "Prediction", fontsize=23,
            color=BLUE, ha="center", va="center", weight="bold")
    ax.text(2.9, 3.60,
            r"Retain independently with probability $q$" if training
            else "Use every hidden activation",
            fontsize=14, color=INK, ha="center", va="center")

    hidden_x, sum_x = 1.05, 4.0
    sum_xy = (sum_x, 1.72)
    heights = (2.72, 1.72, 0.72)
    for j, height in enumerate(heights, start=1):
        hidden_xy = (hidden_x, height)
        masked = training and j == 2
        colour = GREY if masked else PURPLE
        arrow(ax, hidden_xy, sum_xy, colour, dashed=masked, width=1.7 if masked else 2)
        label_x = 2.36
        label_y = height + (1.72 - height) * (label_x - hidden_x) / (sum_x - hidden_x)
        ax.text(label_x, label_y + (0.16 if j != 3 else -0.18), rf"$v_{j}$",
                fontsize=17, color=colour, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=2)
        label = rf"$h_{j}/q$" if training and not masked else rf"$h_{j}$"
        node(ax, hidden_xy, label, colour, fill="#f2f4f7" if masked else "white",
             size=16 if training and not masked else 18, radius=0.34)
        if masked:
            ax.plot([hidden_x - 0.23, hidden_x + 0.23], [height - 0.23, height + 0.23],
                    color=GREY, lw=2.6, zorder=5)
            ax.plot([hidden_x - 0.23, hidden_x + 0.23], [height + 0.23, height - 0.23],
                    color=GREY, lw=2.6, zorder=5)
            ax.text(0.37, height, "$0$", color=GREY, fontsize=18, va="center", ha="center")

    constant_xy = (3.61, 2.99)
    arrow(ax, constant_xy, sum_xy, RED, dashed=True)
    const_x, const_y = constant_xy
    ax.add_patch(Polygon([(const_x, const_y + 0.32), (const_x + 0.32, const_y),
                          (const_x, const_y - 0.32), (const_x - 0.32, const_y)],
                         closed=True, facecolor="#fff1f2", edgecolor=RED, lw=2,
                         zorder=3))
    ax.text(*constant_xy, "1", fontsize=18, color=RED, ha="center", va="center", zorder=4)
    ax.text(4.12, 2.45, "$c$", color=RED, fontsize=19, ha="center", va="center")
    node(ax, sum_xy, r"$\Sigma$", BLUE, size=25, radius=0.32)
    ax.text(sum_x, 1.20, "Affine sum", color=BLUE, fontsize=13, ha="center")
    ax.add_patch(FancyArrowPatch((4.35, 1.72), (5.1, 1.72), arrowstyle="-|>",
                                mutation_scale=14, linewidth=2, color=BLUE))
    ax.text(5.32, 1.72, r"$\widetilde s$" if training else "$s$",
            color=BLUE, fontsize=25, ha="center", va="center")
    caption = "One possible mask: (1, 0, 1)" if training else "No masking or rescaling"
    ax.text(2.9, 0.13, caption, fontsize=14, color=INK, ha="center", va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="Optional PNG preview path.")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.025, top=0.99, wspace=0.045)
    panel(axes[0], training=True)
    panel(axes[1], training=False)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.13, 0.89], transform=fig.transFigure,
                              color="#d9dfe8", linewidth=1))
    output = Path(__file__).resolve().parents[1] / "images" / "nn-dropout.svg"
    fig.savefig(output, transparent=True, metadata={
        "Title": "Dropout during training and prediction",
        "Description": "Training uses one random mask, retaining activations h1/q and h3/q "
                       "and masking h2. Prediction uses all three unscaled activations. "
                       "Both panels combine them with purple output weights and a red "
                       "bias weight c on a fixed constant input of one.",
    })
    if args.preview:
        fig.savefig(args.preview, dpi=160, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
