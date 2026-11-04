"""Draw the GP regression example offline; Quarto never runs this script.

Run: python3 lectures/computer_code/kernel_gp_figures.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kernel-gp-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

OUT = Path(__file__).resolve().parents[1] / "images"
BLUE, RED, PURPLE, INK = "#0f4494", "#be2832", "#7d3c98", "#232754"


def kernel(x, z, bandwidth=0.18, signal_sd=1.0):
    x, z = np.asarray(x), np.asarray(z)
    return signal_sd**2 * np.exp(-0.5 * ((x[:, None] - z[None, :]) / bandwidth)**2)


def posterior(x, y, grid, noise_sd=0.18, bandwidth=0.18, signal_sd=1.0):
    """Return exact zero-mean GP conditional mean and covariance on a grid."""
    gram = kernel(x, x, bandwidth, signal_sd)
    cross = kernel(x, grid, bandwidth, signal_sd)
    factor = cho_factor(gram + noise_sd**2 * np.eye(len(x)), lower=True)
    alpha = cho_solve(factor, y)
    projected = solve_triangular(factor[0], cross, lower=True)
    covariance = kernel(grid, grid, bandwidth, signal_sd) - projected.T @ projected
    return cross.T @ alpha, (covariance + covariance.T) / 2


def gaussian_draws(covariance, rng, count=3):
    """Handle a positive semidefinite covariance without adding model noise."""
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-10:
        raise ValueError("Covariance has a negative eigenvalue beyond round-off.")
    root = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    return root @ rng.standard_normal((len(values), count))


def generate():
    rng = np.random.default_rng(20261103)
    grid = np.linspace(-0.15, 1.15, 251)
    x = np.array([0.08, 0.16, 0.24, 0.35, 0.69, 0.77, 0.84, 0.91])
    noise_sd = 0.18
    truth = lambda t: np.sin(2*np.pi*t) + 0.2*np.cos(6*np.pi*t)
    y = truth(x) + rng.normal(0, noise_sd, len(x))
    prior_draws = gaussian_draws(kernel(grid, grid), rng)
    mean, covariance = posterior(x, y, grid, noise_sd=noise_sd)
    latent_sd = np.sqrt(np.maximum(np.diag(covariance), 0))
    observation_sd = np.sqrt(latent_sd**2 + noise_sd**2)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK,
                         "text.color": INK, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained", sharey=True)
    for j, colour in enumerate([BLUE, PURPLE, "#64748b"]):
        axes[0].plot(grid, prior_draws[:, j], color=colour, lw=2, label=f"Prior draw {j+1}")
    axes[0].axhline(0, color=INK, lw=1.4, ls="--", label="Prior mean")
    axes[0].set_title("Before observing data")
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    axes[0].set_ylabel("Function value / response")

    axes[1].fill_between(grid, mean - 1.96*observation_sd, mean + 1.96*observation_sd,
                         color=RED, alpha=0.12, label="95% new response")
    axes[1].fill_between(grid, mean - 1.96*latent_sd, mean + 1.96*latent_sd,
                         color=BLUE, alpha=0.22, label="95% latent function")
    axes[1].plot(grid, mean, color=BLUE, lw=2.4, label="Posterior mean = KRR")
    axes[1].scatter(x, y, color=RED, edgecolor="white", lw=0.6, s=35, zorder=4,
                    label="Observations")
    axes[1].set_title("After observing data")
    axes[1].legend(frameon=False, fontsize=10, loc="lower left")
    for ax in axes:
        ax.set(xlabel="x", xlim=(-0.15, 1.15), ylim=(-2.9, 2.9))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "kernel-gp-regression.svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    generate()
