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


def _arr(seq):
    """JSON list (with possible None) -> float array with NaN for None."""
    return np.array([np.nan if v is None else float(v) for v in seq], dtype=float)


def fig_risk_coverage(fd, out_dirs):
    """Risk-coverage with bootstrap CI, baselines, safe zone, LTT op point."""
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

    fig, ax = plt.subplots(figsize=(3.7, 2.9))

    # Safe (<= target FNR) zone, shaded behind everything.
    ax.axhspan(0, safe, color=SAFE_GREEN, alpha=0.10, zorder=0,
               label=f"Safe zone (FNR $\\leq$ {safe:.0%})")

    # 95% bootstrap CI band around the confidence-ranked (SR) selective curve.
    band = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(cov[band], lo[band], hi[band], color=C_SR, alpha=0.18,
                    linewidth=0, zorder=1, label=f"{ci_pct}% bootstrap CI")

    # Baseline selective curves.
    ax.plot(cov, sr, color=C_SR, linestyle="-", marker="", zorder=2,
            label="Softmax-response baseline")
    ax.plot(cov, ltt, color=C_LTT, linestyle="-", marker="", zorder=3,
            label=f"Acceptance-threshold sweep (AURC = {float(m['aurc']):.3f})")

    # Split-conformal single operating point.
    if cp_op and np.isfinite(cp_op[1]):
        ax.scatter([cp_op[0]], [cp_op[1]], color=C_CP, marker="^", s=44,
                   zorder=5, edgecolor="white", linewidth=0.6,
                   label="Split-conformal (singletons)")

    # Target line + LTT operating point.
    ax.axhline(safe, color=C_TARGET, linestyle="--", linewidth=1.3, zorder=4,
               label=f"Target FNR = {safe:.2f}")
    ax.scatter([ltt_op[0]], [ltt_op[1]], color=C_LTT, marker="D", s=46,
               zorder=6, edgecolor="white", linewidth=0.7,
               label=f"LTT operating point (cov = {ltt_op[0]:.2f})")

    ax.set_xlabel("Coverage (fraction of cases retained)")
    ax.set_ylabel("Selective false-negative rate (test)")
    ax.set_title("Risk-controlled selective prediction\n(moderate MCAR degradation)")
    ax.set_xlim(0, 1.0)
    ymax = np.nanmax([np.nanmax(sr), np.nanmax(ltt), np.nanmax(hi)])
    ax.set_ylim(0, max(0.12, float(ymax) * 1.05))
    ax.legend(loc="upper right", fontsize=7)
    for d in out_dirs:
        save_fig(fig, "fig_risk_coverage", out_dir=d)
    plt.close(fig)


def fig_ablation(fd, out_dirs):
    """Two stacked, x-aligned panels (FNR on top, abstention below)."""
    import matplotlib.pyplot as plt
    a = fd["ablation"]
    labs = [s.capitalize() for s in a["levels"]]
    bf = _arr(a["base_fnr"]); rf = _arr(a["retained_fnr"])
    ab = _arr(a["abstention"]); tgt = float(a["target_fnr"])
    x = np.arange(len(labs))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.7, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})

    # Top panel: baseline vs LTT-retained FNR + target line.
    ax_top.plot(x, bf, color=C_BASE, marker="o", linestyle="-",
                label="Baseline FNR (no abstention)")
    ax_top.plot(x, rf, color=C_LTT, marker="s", linestyle="-",
                label="LTT retained FNR (test)")
    ax_top.axhline(tgt, color=C_TARGET, linestyle=":", linewidth=1.4,
                   label=f"Target FNR = {tgt:.2f}")
    ax_top.set_ylabel("False-negative rate")
    ax_top.set_ylim(bottom=0)
    ax_top.set_title("Safety holds as data quality degrades")
    ax_top.legend(loc="upper left", fontsize=8)

    # Bottom panel: abstention rate (the price paid).
    ax_bot.bar(x, ab, width=0.55, color=C_ABSTAIN, alpha=0.85,
               label="Abstention rate")
    ax_bot.set_ylabel("Abstention rate")
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labs)
    ax_bot.set_xlabel("Data-quality degradation (MCAR severity)")
    ax_bot.legend(loc="upper left", fontsize=8)

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
    fig_ablation(fd, out_dirs)
    print("[OK] figures (risk_coverage + ablation) ->", ", ".join(map(str, out_dirs)))


if __name__ == "__main__":
    generate_all()
