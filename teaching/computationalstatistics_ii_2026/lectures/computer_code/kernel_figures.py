"""Generate the kernel lecture figures offline: python3 kernel_figures.py.

No Python is executed by Quarto. All examples use a fixed design and fixed seed.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kernel-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve

OUT = Path(__file__).resolve().parents[1] / "images"
BLUE, RED, PURPLE, INK = "#0f4494", "#be2832", "#7d3c98", "#232754"


def rbf(x, z, bandwidth):
    return np.exp(-0.5 * ((np.asarray(x)[:, None] - np.asarray(z)[None, :]) / bandwidth) ** 2)


def fit_kernel(gram, y, lam):
    """Loss convention: mean squared error + lam * squared RKHS norm."""
    factor = cho_factor(gram + len(y) * lam * np.eye(len(y)), lower=True)
    alpha = cho_solve(factor, y)
    return alpha, gram @ alpha


def finish(fig, name):
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)


def completeness_figure():
    """The same Cauchy sequence in a complete space and an incomplete one."""
    OUT.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 16,
                         "svg.fonttype": "none", "text.color": INK}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 3.2), layout="constrained")
        points = 1 / (np.arange(1, 28) + 1)
        for ax, space, included in zip(axes, [r"$\mathbb{R}$", r"$(0,1)$"], [True, False]):
            ax.plot([-0.07 if included else 0, 1.05 if included else 1],
                    [0, 0], color="#94a3b8", lw=2, zorder=1)
            ax.scatter(points, np.zeros_like(points), s=27, color=BLUE, zorder=3)
            ax.scatter([0], [0], s=100, facecolors=PURPLE if included else "white",
                       edgecolors=PURPLE if included else RED, lw=2.2, zorder=4)
            if not included:
                ax.scatter([1], [0], s=100, facecolors="white", edgecolors=RED,
                           lw=2.2, zorder=4)
            ax.annotate("", xy=(0.08, 0.14), xytext=(0.47, 0.14),
                        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": BLUE})
            ax.text(-0.12, 0, space, ha="right", va="center", fontsize=20)
            ax.text(0, -0.18, "0", ha="center")
            ax.text(0.5, -0.18, "1/2", ha="center")
            ax.text(1, -0.18, "1", ha="center")
            ax.text(0.64, 0.16, "Limit included" if included else "Limit excluded",
                    color=PURPLE if included else RED, ha="center", va="center")
            ax.set(xlim=(-0.23, 1.1), ylim=(-0.35, 0.33))
            ax.axis("off")
        finish(fig, "kernel-completeness.svg")


def generate():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK,
                         "text.color": INK, "axes.titleweight": "bold"})
    completeness_figure()
    grid = np.linspace(0, 1, 1001)

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for m, colour in zip([4, 12, 40], [BLUE, PURPLE, RED]):
        ax.plot(grid, np.maximum(1 - m * np.abs(grid - 0.5), 0),
                color=colour, lw=2.5, label=f"m = {m}")
    ax.set(xlabel="t", ylabel=r"$g_m(t)$", ylim=(-0.04, 1.12))
    ax.axvline(0.5, color="#94a3b8", ls=":")
    ax.legend(frameon=False, loc="upper left")
    finish(fig, "kernel-evaluation.svg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for s, colour in zip([0.2, 0.5, 0.8], [BLUE, PURPLE, RED]):
        axes[0].plot(grid, np.minimum(grid, s), color=colour, lw=2.6, label=f"s = {s}")
    axes[0].set(xlabel="t", ylabel=r"$K_s(t)=\min(s,t)$", title="Kernel sections")
    axes[0].legend(frameon=False)
    x = np.array([0.12, 0.29, 0.45, 0.64, 0.83])
    y = np.array([0.55, 0.85, 0.35, 0.70, 0.40])
    gram = np.minimum.outer(x, x)
    for lam, colour, label in [(0, BLUE, "Minimum-norm interpolant"),
                              (0.02, PURPLE, r"Regularised: $\lambda=0.02$")]:
        alpha = np.linalg.solve(gram, y) if lam == 0 else fit_kernel(gram, y, lam)[0]
        axes[1].plot(grid, np.minimum.outer(grid, x) @ alpha,
                     color=colour, lw=2.6, label=label)
    axes[1].scatter(x, y, color=RED, zorder=3, s=35)
    axes[1].scatter([0], [0], marker="s", color=INK, s=35, zorder=3)
    axes[1].set(xlabel="t", ylabel="f(t)", title="Piecewise linear fits", ylim=(-0.04, 1))
    axes[1].legend(frameon=False, fontsize=11, loc="upper left")
    finish(fig, "kernel-sobolev.svg")

    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    origin = np.array([0., 0.]); parallel = np.array([2.8, 0.9]); f = np.array([2.1, 3.1])
    perp = f - parallel
    # A schematic Euclidean slice of the Hilbert-space orthogonal decomposition.
    ax.fill([-0.4, 3.8, 4.5, 0.3], [-0.9, 0.45, 1.35, 0.0], color=BLUE, alpha=0.08)
    ax.plot([-0.4, 4.5], [-0.4 * 0.9/2.8, 4.5 * 0.9/2.8], color=BLUE, lw=1.5)
    for start, end, colour, label in [(origin, f, INK, r"$f$"),
                                      (origin, parallel, BLUE, r"$f_{\parallel}$"),
                                      (parallel, f, PURPLE, r"$f_{\perp}$")]:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": colour, "lw": 2.8})
        mid = (start+end)/2
        ax.text(mid[0]+0.12, mid[1]+0.08, label, color=colour, fontsize=22)
    u = parallel / np.linalg.norm(parallel) * 0.2
    v = perp / np.linalg.norm(perp) * 0.2
    corner = np.array([parallel-u, parallel-u+v, parallel+v])
    ax.plot(corner[:, 0], corner[:, 1], color=INK, lw=1)
    ax.text(2.9, 0.1, r"$S_n=\mathrm{span}\{K_{x_1},\ldots,K_{x_n}\}$", color=BLUE, fontsize=15)
    ax.text(0.1, -0.65, "Same fitted values; a smaller norm after projection.", fontsize=14)
    ax.set(xlim=(-0.5, 5.2), ylim=(-1, 3.5)); ax.axis("off")
    finish(fig, "kernel-projection.svg")

    rng = np.random.default_rng(2026)
    x = np.linspace(0.04, 0.96, 22)
    truth = lambda t: np.sin(2*np.pi*t) + 0.3*np.cos(6*np.pi*t)
    y = truth(x) + rng.normal(0, 0.2, len(x))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), layout="constrained", sharey=True)
    for ax, bandwidth, lam in zip(axes, [0.04, 0.16, 0.16], [0.0001, 0.0001, 0.08]):
        gram = rbf(x, x, bandwidth)
        alpha, _ = fit_kernel(gram, y, lam)
        ax.plot(grid, truth(grid), color="#64748b", ls="--", lw=1.5, label="True mean")
        ax.plot(grid, rbf(grid, x, bandwidth) @ alpha, color=BLUE, lw=2.4, label="KRR")
        ax.scatter(x, y, color=RED, s=22, zorder=3, label="Training data")
        ax.set(xlabel="x", title=rf"$\ell={bandwidth},\ \lambda={lam:g}$", ylim=(-1.6, 1.6))
    axes[0].set_ylabel("Response"); axes[0].legend(frameon=False, fontsize=10)
    finish(fig, "kernel-bandwidth.svg")

    gram = rbf(x, x, 0.16)
    mu = np.linalg.eigvalsh(gram)[::-1].clip(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    axes[0].semilogy(np.arange(1, len(x)+1), mu.clip(1e-15), "o-", color=BLUE)
    axes[0].set(xlabel="Ordered eigenvalue index j", ylabel=r"$\mu_j$", title="Gram spectrum")
    for lam, colour in zip([1e-4, 1e-2, 1e-1], [BLUE, PURPLE, RED]):
        weights = mu/(mu+len(x)*lam)
        axes[1].plot(np.arange(1, len(x)+1), weights, "o-", color=colour,
                     label=rf"$\lambda={lam:g}$; df = {weights.sum():.1f}")
    axes[1].set(xlabel="Ordered eigenvalue index j", ylabel=r"$\mu_j/(\mu_j+n\lambda)$",
                title="Spectral shrinkage", ylim=(-0.03, 1.05))
    axes[1].legend(frameon=False, fontsize=11)
    finish(fig, "kernel-spectrum.svg")


if __name__ == "__main__":
    generate()
