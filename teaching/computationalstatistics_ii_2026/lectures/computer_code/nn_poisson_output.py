"""Plot output maps for Poisson regression; run manually before rendering Quarto."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn-poisson-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none"})
    blue, purple, red = "#0f4494", "#7d3c98", "#be2832"
    s = np.linspace(-2, 1.5, 500)
    fig, ax = plt.subplots(figsize=(6.4, 5.1), layout="constrained")
    ax.axhspan(-2.2, 0, color=red, alpha=0.07)
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.axvline(0, color="#cbd5e1", lw=1)
    ax.plot(s, np.exp(s), color=blue, lw=3, label=r"Exponential: $e^s$")
    ax.plot(s, np.logaddexp(0, s), color=purple, lw=3,
            label=r"Softplus: $\log(1+e^s)$")
    ax.plot(s, s, color="#64748b", lw=2, ls="--", label=r"Identity: $s$")
    ax.text(0.1, -1.65, "Negative means\nare invalid", ha="center", va="center",
            color=red, fontsize=15)
    ax.set(xlim=(-2.05, 1.55), ylim=(-2.2, 4.8),
           xlabel="Network score $s$", ylabel=r"Poisson mean $\lambda$")
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#94a3b8")
    ax.tick_params(colors="#232754")
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    out = Path(__file__).resolve().parents[1] / "images" / "nn-poisson-output.svg"
    fig.savefig(out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    generate()
