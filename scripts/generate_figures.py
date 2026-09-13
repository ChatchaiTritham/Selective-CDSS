#!/usr/bin/env python3
"""
generate_figures.py -- importable PLOTTING module for Selective-CDSS.

Separation of concerns: this module NEVER recomputes the science. It reads the
arrays persisted by run_eval.py into results/real_results.json (seed 42) and
draws the two manuscript figures via the shared, byte-identical pubviz style.

Figures
  fig_risk_coverage : selective FNR vs coverage at moderate MCAR, with a 95%
                      bootstrap CI band, the softmax-response (SR) and
                      split-conformal baselines overlaid, the safe (<=5% FNR)
                      zone shaded, and the LTT operating point marked.
  fig_ablation      : two vertically-stacked, x-aligned panels sharing the
                      MCAR-severity axis -- (top) baseline vs LTT-retained FNR
                      with the 5% target line; (bottom) abstention rate.
                      Replaces the old dual-axis chart.

Run standalone (no retraining):  python scripts/generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Repo root is the parent of scripts/; make `import pubviz` resolve when this is
# executed directly as a file.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pubviz import apply_pub_style, save_fig, PALETTE, load_results  # noqa: E402

C_LTT = PALETTE[0]        # blue  -- LTT / retained
C_BASE = PALETTE[1]       # vermillion -- baseline / no abstention
C_SR = PALETTE[2]         # green -- softmax-response baseline
C_CP = PALETTE[4]         # orange -- split-conformal baseline
C_TARGET = PALETTE[6]     # black -- 5% target line
C_ABSTAIN = PALETTE[5]    # light blue -- abstention bars
SAFE_GREEN = "#009E73"

# Springer Nature sn-jnl single-column layout: \textwidth measures 372 pt.
# Drawing at exactly that width and including at \textwidth means matplotlib
# never has its type rescaled by LaTeX.
COL_IN = 372.0 / 72.0
FS = 8.5


def _arr(seq):
    """JSON list (with possible None) -> float array with NaN for None."""
    return np.array([np.nan if v is None else float(v) for v in seq], dtype=float)


def _style_axes(ax, grid="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS)


def fig_risk_coverage(fd, out_dirs):
    """Selective FNR against coverage at moderate MCAR (test fold)."""
    import matplotlib.pyplot as plt
    m = fd["moderate_mcar"]
    cov = _arr(m["coverage_grid"])
    ltt = _arr(m["ltt_curve"])
    sr = _arr(m["sr_curve"])
    lo, hi = _arr(m["sr_ci_lo"]), _arr(m["sr_ci_hi"])
    safe = float(m["safe_fnr"])
    ltt_op = m["ltt_op"]
    cp_op = m["split_conformal_op"]
    ci_pct = int(round(float(m.get("ci_level", 0.95)) * 100))

    fig, ax = plt.subplots(figsize=(COL_IN, COL_IN * 0.50))
    ax.axhspan(0, 100 * safe, color=SAFE_GREEN, alpha=0.10, linewidth=0, zorder=0,
               label=f"At or below target ({safe:.0%})")
    band = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(cov[band], 100 * lo[band], 100 * hi[band], color=C_SR, alpha=0.18,
                    linewidth=0, zorder=1, label=f"Softmax response, {ci_pct}% bootstrap CI")
    ax.plot(cov, 100 * sr, color=C_SR, linewidth=1.4, zorder=2, label="Softmax response (confidence ranking)")
    ax.plot(cov, 100 * ltt, color=C_LTT, linewidth=1.4, zorder=3,
            label=f"Threshold-band sweep (AURC {float(m['aurc']):.3f})")
    ax.axhline(100 * safe, color=C_TARGET, linestyle="--", linewidth=1.0, zorder=4)
    if cp_op and np.isfinite(cp_op[1]):
        ax.scatter([cp_op[0]], [100 * cp_op[1]], color=C_CP, marker="^", s=40, zorder=5,
                   edgecolor="black", linewidth=0.4, label="Split conformal (singleton sets)")
    ax.scatter([ltt_op[0]], [100 * ltt_op[1]], color=C_LTT, marker="D", s=40, zorder=6,
               edgecolor="black", linewidth=0.4, label=f"Calibrated operating point (coverage {ltt_op[0]:.2f})")
    ax.set_xlabel("Coverage (fraction of test cases decided)", fontsize=FS + 0.5)
    ax.set_ylabel("False-negative rate among\ndecided cases (%)", fontsize=FS + 0.5)
    ax.set_xlim(0, 1.0)
    ymax = np.nanmax([np.nanmax(sr), np.nanmax(ltt), np.nanmax(hi)])
    ax.set_ylim(0, 100 * max(0.12, float(ymax) * 1.05))
    _style_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=FS - 0.5, frameon=False,
              borderaxespad=0.0, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    for d in out_dirs:
        save_fig(fig, "fig_risk_coverage", out_dir=d)
    plt.close(fig)


def fig_ablation(fd, results, out_dirs):
    """(a) FNR by MCAR severity, matched vs clean calibration; (b) abstention; (c) ten splits."""
    import matplotlib.pyplot as plt
    a = fd["ablation"]
    short = {"clean": "Clean", "mild": "Mild", "moderate": "Mod.", "severe": "Sev."}
    labs = [short.get(s, s.capitalize()) for s in a["levels"]]
    bf = 100 * _arr(a["base_fnr"])
    rf = 100 * _arr(a["retained_fnr"])
    ab = 100 * _arr(a["abstention"])
    tgt = 100 * float(a["target_fnr"])
    x = np.arange(len(labs))

    mism = {r["level"]: r for r in results["calibration_mismatch_mcar"]}
    mm_f = np.array([np.nan] + [100 * mism[k]["retained_fnr_test"] for k in ("mild", "moderate", "severe")])
    mm_a = np.array([np.nan] + [100 * mism[k]["abstention"] for k in ("mild", "moderate", "severe")])
    reps = results["repeated_splits_moderate_mcar"]["runs"]

    fig, axes = plt.subplots(1, 3, figsize=(COL_IN, COL_IN * 0.40),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 0.75]})
    ax = axes[0]
    ax.plot(x, bf, color=C_BASE, marker="o", markersize=4, linewidth=1.3, label="No abstention")
    ax.plot(x, rf, color=C_LTT, marker="s", markersize=4, linewidth=1.3, label="Calibrated on matched data")
    ax.plot(x, mm_f, color=C_LTT, marker="s", markersize=4, markerfacecolor="white", linestyle="--",
            linewidth=1.1, label="Calibrated on clean data")
    ax.axhline(tgt, color=C_TARGET, linestyle=":", linewidth=1.0)
    ax.text(x[-1], tgt + 0.8, "target", ha="right", va="bottom", fontsize=FS - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=FS - 0.5)
    ax.set_ylabel("False-negative rate (%)", fontsize=FS + 0.5)
    ax.set_ylim(0, 30)
    ax.set_title("(a) FNR of decided cases", fontsize=FS, loc="left")
    ax.legend(loc="upper left", fontsize=FS - 1, frameon=False, handlelength=1.8)
    _style_axes(ax)

    ax = axes[1]
    w = 0.38
    ax.bar(x - w / 2, ab, w, color=C_ABSTAIN, edgecolor="black", linewidth=0.4, label="Matched")
    ax.bar(x + w / 2, np.nan_to_num(mm_a), w, color="white", edgecolor=C_LTT, hatch="////",
           linewidth=0.6, label="Clean")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=FS - 0.5)
    ax.set_ylabel("Abstention (%)", fontsize=FS + 0.5)
    ax.set_ylim(0, 50)
    ax.set_title("(b) Cases deferred", fontsize=FS, loc="left")
    ax.legend(loc="upper left", fontsize=FS - 1, frameon=False, title="Calibration", title_fontsize=FS - 1)
    _style_axes(ax, "y")

    ax = axes[2]
    vals = np.array([100 * r["retained_fnr_test"] for r in reps])
    jitter = np.linspace(-0.12, 0.12, len(vals))
    ax.scatter(jitter, vals, s=16, color=C_LTT, edgecolor="black", linewidth=0.3, zorder=3)
    ax.hlines(vals.mean(), -0.25, 0.25, color="black", linewidth=1.2, zorder=4)
    ax.axhline(tgt, color=C_TARGET, linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, 10)
    ax.set_ylabel("False-negative rate (%)", fontsize=FS + 0.5)
    ax.set_title(f"(c) {len(vals)} random splits", fontsize=FS, loc="left")
    _style_axes(ax, "y")

    fig.tight_layout(pad=0.3, w_pad=0.9)
    for d in out_dirs:
        save_fig(fig, "fig_ablation", out_dir=d)
    plt.close(fig)


def generate_all(start=None, out_dirs=None):
    """Read results/ (seed 42) and render both figures.

    start    : directory to resolve results/ from (defaults to repo root / cwd).
    out_dirs : iterable of directories to write the matched pdf+png into;
               defaults to figures/ and the repo root (README expects both).
    """
    import matplotlib
    matplotlib.use("Agg")
    apply_pub_style()

    results = load_results("real_results.json", start=start)
    fd = results.get("figure_data")
    if fd is None:
        raise KeyError(
            "results/real_results.json has no 'figure_data' block -- re-run "
            "run_eval.py (seed 42) to persist the figure arrays.")

    base = Path(start) if start else _REPO_ROOT
    if out_dirs is None:
        out_dirs = [base / "figures", base]
    out_dirs = [Path(d) for d in out_dirs]

    fig_risk_coverage(fd, out_dirs)
    fig_ablation(fd, results, out_dirs)
    print("[OK] figures (risk_coverage + ablation) ->", ", ".join(map(str, out_dirs)))


if __name__ == "__main__":
    generate_all()
