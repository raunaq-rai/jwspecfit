"""Render logo.svg to PNG at standard sizes.

Reproduces the SVG primitives in matplotlib (filled hex, baseline, Gaussian
polyline) and exports PNG fallbacks at 16, 32, 64, 128, 256 px with
transparent background.  Run from the repo root:

    python docs/_static/logos/_render_pngs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = Path(__file__).parent
SIZES = (16, 32, 64, 128, 256)

GOLD = "#C9A227"
GOLD_EDGE = "#5A4615"
DARK = "#2C2416"

# Hexagon vertices (flat-top, matching logo.svg)
HEX = np.array([
    (78, 41.4), (178, 41.4), (228, 128),
    (178, 214.6), (78, 214.6), (28, 128),
])

# Gaussian: baseline y=178, peak y=73, centre x=128, sigma=30
xs = np.arange(46, 210.5, 2.0)
ys = 178 - 105 * np.exp(-((xs - 128) / 30.0) ** 2)


def render(size_px: int) -> None:
    """Render the logo at *size_px* x *size_px* and write logo-<size>.png."""
    dpi = 100
    fig_inches = size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)

    # Stroke widths scale linearly with output size (SVG used 3.5 / 2.5 / 6.5
    # in a 256-unit viewBox; here we mirror the same ratios at requested px).
    sw = size_px / 256.0
    ax.add_patch(Polygon(
        HEX, closed=True, facecolor=GOLD, edgecolor=GOLD_EDGE,
        linewidth=3.5 * sw, joinstyle="round",
    ))
    ax.plot(
        [46, 210], [178, 178],
        color=DARK, linewidth=2.5 * sw, alpha=0.4, solid_capstyle="butt",
    )
    ax.plot(
        xs, ys, color=DARK, linewidth=6.5 * sw,
        solid_capstyle="round", solid_joinstyle="round",
    )

    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out = HERE / f"logo-{size_px}.png"
    fig.savefig(out, dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    for s in SIZES:
        render(s)
