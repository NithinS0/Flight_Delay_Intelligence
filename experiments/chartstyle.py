"""Shared matplotlib style for presentation charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CHART_DIR

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 220, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.labelsize": 11, "legend.frameon": False,
})
BLUE, CYAN, AMBER, GREEN, RED, SLATE, PURPLE = "#2563eb", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#64748b", "#7c3aed"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
