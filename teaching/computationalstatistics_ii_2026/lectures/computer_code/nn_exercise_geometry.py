"""Draw geometric targets for the neural-network exercises.

Run manually from any working directory; Quarto never executes this file.
Use --preview-dir /tmp/nn-exercise-geometry for optional PNG previews.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn-exercise-geometry-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon


BLUE = "#0f4494"
PURPLE = "#7d3c98"
RED = "#be2832"
INK = "#263445"
GUIDE = "#cdd5e0"
OUT = Path(__file__).resolve().parents[1] / "images"


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca8b8")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK, labelsize=16, width=0.9, length=4, pad=8)
    ax.set_xlabel("$x$", fontsize=19, color=INK, labelpad=8)
    ax.set_ylim(-0.10, 1.25)


def save(fig, filename, description, preview_dir):
    fig.savefig(OUT / filename, transparent=True, metadata={
        "Title": filename.removesuffix(".svg").replace("-", " "),
        "Description": description,
    })
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / filename.replace(".svg", ".png"),
                    facecolor="white", dpi=150)
    plt.close(fig)
    print(OUT / filename)


def relu_target(preview_dir):
    fig, ax = plt.subplots(figsize=(6, 4.8))
    fig.subplots_adjust(left=0.11, right=0.975, bottom=0.17, top=0.84)
    axes_style(ax)
    ax.set_xlim(-0.72, 2.72)
    ax.set_xticks([0, 1, 2], ["$0$", "$1$", "$2$"])
    ax.set_yticks([0, 1], ["$0$", "$1$"])
    for x in (0, 1, 2):
        ax.vlines(x, 0, 1.10, color=GUIDE, linestyle="--", lw=1.1, zorder=0)
    ax.plot([-0.72, 0, 1, 2, 2.72], [0, 0, 1, 0, 0], color=BLUE, lw=3.7,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter([0, 1, 2], [0, 1, 0], s=36, color=BLUE, zorder=4)
    for x, y, text in [(-0.38, 0.18, "slope 0"), (0.40, 0.70, "+1"),
                        (1.60, 0.70, "−1"), (2.39, 0.18, "slope 0")]:
        ax.text(x, y, text, ha="center", va="center", color=PURPLE, fontsize=16,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_title("Target shape", fontsize=22, weight="bold", color=BLUE, pad=16)
    save(fig, "nn-relu-design.svg",
         "A triangular target is zero for x at most zero, rises with slope one "
         "to height one at x equals one, falls with slope minus one to zero at "
         "x equals two, and remains zero thereafter. The four slopes are labelled.",
         preview_dir)


def step_and_ramp(preview_dir):
    delta = 0.4  # Illustrative geometry; the displayed labels are symbolic.
    fig, ax = plt.subplots(figsize=(6, 4.8))
    fig.subplots_adjust(left=0.115, right=0.97, bottom=0.18, top=0.76)
    axes_style(ax)
    ax.set_xlim(-1.07, 1.07)
    ax.set_ylim(-0.10, 1.14)
    ax.set_xticks([-1, -delta, 0, delta, 1],
                  ["$-1$", "$-\\delta$", "$0$", "$\\delta$", "$1$"])
    ax.set_yticks([0, 0.5, 1], ["$0$", "$\\frac{1}{2}$", "$1$"])
    for x in (-delta, 0, delta):
        ax.vlines(x, 0, 1.05, color=GUIDE, linestyle="--", lw=1.0, zorder=0)
    for vertices in ([(-delta, 0), (0, 0.5), (0, 0)],
                     [(0, 0.5), (delta, 1), (0, 1)]):
        ax.add_patch(Polygon(vertices, closed=True, facecolor=RED, alpha=0.17,
                             edgecolor="none", zorder=1))

    # Draw the common tails wide enough to retain both colours under the ramp.
    ax.plot([-1, 0], [0, 0], color=BLUE, lw=4.4, zorder=2)
    ax.plot([0, 1], [1, 1], color=BLUE, lw=4.4, zorder=2)
    ax.plot([-1, -delta, delta, 1], [0, 0, 1, 1], color=PURPLE, lw=2.7,
            linestyle=(0, (5, 3)), solid_joinstyle="round", zorder=3)
    # The target takes value one at zero: lower endpoint open, upper closed.
    ax.plot(0, 0, marker="o", ms=9, markerfacecolor="white", markeredgecolor=BLUE,
            markeredgewidth=2.2, linestyle="none", zorder=5)
    ax.plot(0, 1, marker="o", ms=8.5, markerfacecolor=BLUE, markeredgecolor=BLUE,
            linestyle="none", zorder=5)
    ax.plot(0, 0.5, marker="o", ms=5.5, color=PURPLE, linestyle="none", zorder=4)

    fig.suptitle("Step and continuous ramp", y=0.965, fontsize=21,
                 weight="bold", color=BLUE)
    handles = [Line2D([], [], color=BLUE, lw=3.5, label="Target $f$"),
               Line2D([], [], color=PURPLE, lw=2.7, linestyle=(0, (5, 3)),
                      label="Ramp $g_\\delta$"),
               Patch(facecolor=RED, alpha=0.17, edgecolor="none", label="Absolute error")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.875),
               ncol=3, frameon=False, fontsize=12, handlelength=1.7,
               handletextpad=0.5, columnspacing=1.1)
    save(fig, "nn-step-ramp.svg",
         "The blue target is the indicator of x at least zero, with an open "
         "endpoint at zero height and a closed endpoint at height one. A purple "
         "continuous ramp agrees outside minus delta to delta and crosses height "
         "one half at zero. Two pale red triangles indicate absolute error. "
         "The tick labels use symbolic delta; the drawing illustrates delta equal to 0.4.",
         preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    relu_target(args.preview_dir)
    step_and_ramp(args.preview_dir)


if __name__ == "__main__":
    main()
