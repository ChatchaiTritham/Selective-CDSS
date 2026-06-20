#!/usr/bin/env python3
"""
pubviz.py -- canonical Top-Tier visualization utilities shared, byte-identical,
by every figure script across the ChatchaiTritham PhD reproducibility repos.

Mirrors _management/FIGURE_STYLE.md. One look across all papers:
  * Okabe-Ito colour-blind-safe palette (consistent series order)
  * Times serif fonts, STIX math; spines off; light grid; constrained layout
  * 300 dpi PNG + vector PDF, tight bbox
  * helpers: results_dir / load_results / require_results / save_fig
  * schematic primitives: add_box / arrow

Location-independent: it finds results/ by walking up from the current working
directory (scripts are run from the repo root). Vendor this file UNCHANGED into
each repo; an identity gate asserts every copy matches this source.

Data must always be loaded from results/ at run time -- never hardcode numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Color-blind-safe (Okabe-Ito) -- use in this order.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#000000"]

# Semantic aliases drawn from the same palette (keeps DAGs/bars/schematics consistent).
C_TREATMENT = PALETTE[0]   # blue
C_OUTCOME = PALETTE[1]     # vermillion
C_CONFOUNDER = PALETTE[4]  # orange
C_MEDIATOR = PALETTE[2]    # bluish green
C_NEUTRAL = "#555555"      # grey -- naive / reference / baseline


def apply_pub_style():
    """Apply the canonical publication rcParams (idempotent)."""
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor containing results/ or .git (falls back to cwd)."""
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "results").is_dir() or (d / ".git").exists():
            return d
    return Path.cwd().resolve()


def results_dir(start: Path | None = None) -> Path:
    return repo_root(start) / "results"


def load_results(name: str, start: Path | None = None):
    """Load a results/ artifact by file name. .json -> obj; .csv -> list[dict]."""
    p = results_dir(start) / name
    if not p.exists():
        raise FileNotFoundError(
            f"missing results/{name} -- run the repo's run_all.py/run_eval.py "
            f"(seed 42) to regenerate results/ before plotting.")
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    import csv
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def require_results(*names: str, start: Path | None = None) -> None:
    """Fail early with a clear hint if any results/ artifact is absent."""
    missing = [n for n in names if not (results_dir(start) / n).exists()]
    if missing:
        raise FileNotFoundError(
            "missing results artifacts: " + ", ".join(missing) +
            " -- run the repo's reproduction script (seed 42) first.")


def save_fig(fig, basename: str, out_dir="figures"):
    """Save matched vector .pdf + 300-dpi .png (tight bbox, white bg)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{basename}.{ext}", bbox_inches="tight", facecolor="white")
    print(f"saved: {basename}.pdf / .png  ->  {out}")


def add_box(ax, xy, width, height, label, facecolor="#eef2f7",
            edgecolor=PALETTE[0], textcolor="#1a1a1a", size=9):
    """Rounded schematic box with centred label (for architecture diagrams)."""
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.1, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center",
            fontsize=size, color=textcolor, zorder=3)
    return box


def arrow(ax, start, end, color="#555555", lw=1.35):
    """Schematic arrow that shrinks off the box edges (no head-in-text overlap)."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 lw=lw, color=color, shrinkA=4, shrinkB=4, zorder=1))
