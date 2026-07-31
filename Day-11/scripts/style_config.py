"""
Shared visual style for Day 11 scripts.
----------------------------------------
Import this once at the top of any script (`import style_config`) to get a
consistent, elegant look across every plot - same fonts, same palette,
same grid style - instead of each script having its own random defaults.
"""

import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = ["#2563EB", "#F59E0B", "#059669", "#DB2777", "#7C3AED"]

# Diverging palette for correlation heatmaps
HEATMAP_CMAP = "rocket_r"

# Sequential palette for continuous scales (e.g. variance bars)
SEQUENTIAL_CMAP = "crest"

sns.set_theme(
    style="whitegrid",
    palette=PALETTE,
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 1.0,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.labelcolor": "#2B2B2B",
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
        "font.size": 11,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#DDDDDD",
        "legend.fontsize": 9.5,
        "savefig.facecolor": "white",
    },
)


def style_axes_spines(ax):
    """Removes the top/right border and softens the remaining ones for a
    cleaner, more modern look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def add_title(ax, title, subtitle=None):
    """Axis-level title with an optional lighter subtitle placed just above
    the plot (in AXES coordinates, so it can never collide with anything
    else - unlike the old version, which used fixed figure coordinates)."""
    ax.set_title(title, fontsize=13.5, fontweight="bold", color="#1A1A1A", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0.5, 1.045, subtitle, transform=ax.transAxes,
                 ha="center", va="bottom", fontsize=9.5, color="#8A8A8A")


def add_figure_title(fig, title, subtitle=None, y=1.02):
    """Figure-level title for multi-panel figures (pairplot, 2x2 grids, etc).
    Title and subtitle are spaced with enough gap that they never overlap,
    regardless of figure size."""
    fig.suptitle(title, fontsize=16.5, fontweight="bold", color="#1A1A1A", y=y)
    if subtitle:
        fig.text(0.5, y - 0.03, subtitle, ha="center", fontsize=10.5, color="#8A8A8A")