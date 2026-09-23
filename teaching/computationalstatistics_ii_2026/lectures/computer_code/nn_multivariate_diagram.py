"""Draw the one-hidden-layer network with a vector-valued response.

Run manually with Python; Quarto uses the resulting static SVG.
The shared drawing helpers keep activation shapes and colours consistent.
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn-multivariate-matplotlib")
LECTURES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LECTURES / "codes"))

from nn_network_diagrams import (
    BLUE, PURPLE, RED, activation, canvas, edge, save, text, value_node,
)


def generate():
    fig, ax = canvas((9.2, 6.0), (-0.65, 8.3, -3.0, 2.95))
    inputs = [(0, 0.7), (0, -0.9)]
    hidden = [(3, 1.1), (3, -0.15), (3, -1.4)]
    outputs = [(6.5, 1.1), (6.5, -0.15), (6.5, -1.9)]
    indices = ["1", "2", "k"]
    hidden_constant, output_constant = (0, 2.35), (3, 2.35)

    for source in inputs:
        for target in hidden:
            edge(ax, source, target, colour=BLUE)
    for source in hidden:
        for target in outputs:
            edge(ax, source, target, colour=PURPLE)

    for j, target in enumerate(hidden, 1):
        edge(ax, hidden_constant, target, rf"$b_{j}$", RED, True,
             offset=(-0.38, 0), shrink=23)
    for r, target in zip(indices, outputs):
        edge(ax, output_constant, target, colour=RED, dashed=True)
        # Place the intercept labels near their destination to separate the rays.
        fraction = 0.77
        x = output_constant[0] + fraction * (target[0] - output_constant[0])
        y = output_constant[1] + fraction * (target[1] - output_constant[1])
        text(ax, x, y + 0.12, rf"$c_{r}$", 14, RED,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    for i, point in enumerate(inputs, 1):
        value_node(ax, point, rf"$x_{i}$")
    for j, point in enumerate(hidden, 1):
        activation(ax, point, "relu", size=0.66)
        text(ax, point[0], point[1] - 0.5, rf"$h_{j}$", 14)
    for r, point in zip(indices, outputs):
        value_node(ax, point, r"$\Sigma$")
        edge(ax, point, (7.3, point[1]), colour=PURPLE, shrink=15)
        text(ax, 7.65, point[1], rf"$s_{r}$", 20, PURPLE)
    text(ax, 6.5, -1.05, r"$\vdots$", 21)
    text(ax, 7.65, -1.05, r"$\vdots$", 21, PURPLE)
    for point in (hidden_constant, output_constant):
        value_node(ax, point, "1", True, r=0.26)

    text(ax, 1.35, -2.65, r"Input weights $w_{ji}$", 14, BLUE)
    text(ax, 4.9, -2.65, r"Output weights $v_{rj}$", 14, PURPLE)
    save(fig, "nn-one-hidden-layer-multivariate.svg")


if __name__ == "__main__":
    generate()
