"""Generate the static figures for the first neural-network session.

Run manually from any working directory; Quarto does not execute this file.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'images'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 14, 'svg.fonttype': 'none'})
blue, red, purple = '#0f4494', '#be2832', '#7d3c98'
z = np.linspace(-3, 3, 301)
fig, ax = plt.subplots(figsize=(7, 4.7), layout='constrained')
for values, colour, label in [(np.maximum(z, 0), blue, 'ReLU'),
                               (1/(1+np.exp(-z)), red, 'Sigmoid'),
                               (np.tanh(z), purple, 'tanh')]:
    ax.plot(z, values, color=colour, lw=2.7, label=label)
ax.axhline(0, color='#94a3b8', lw=0.8)
ax.axvline(0, color='#94a3b8', lw=0.8)
ax.set(xlabel='Pre-activation z', ylabel='Activation', xlim=(-3, 3), ylim=(-1.15, 3.15))
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper left')
fig.savefig(OUT / 'nn-activations.svg', transparent=True)
plt.close(fig)

# Reuse the same shape convention across all network diagrams.
from nn_network_diagrams import generate
generate()
