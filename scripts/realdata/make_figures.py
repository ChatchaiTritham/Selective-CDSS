"""Figures for the real-data study (Okabe-Ito palette, vector PDF, no in-plot titles).

Usage: python -I make_figures.py <results_dir> <fig_dir>
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OI = {"ltt_selective": "#0072B2", "ltt_overall": "#56B4E9", "empirical": "#D55E00",
      "sgr": "#009E73", "scrc_i": "#CC79A7", "crc": "#E69F00"}
LAB = {"ltt_selective": "LTT-selective (ours)", "ltt_overall": "LTT-overall",
       "empirical": "Empirical", "sgr": "SGR", "scrc_i": "SCRC-I", "crc": "CRC"}
SETS = ["within_A", "within_B", "cross_AtoB", "cross_BtoA", "recal_AtoB", "recal_BtoA"]
SLAB = ["A→A", "B→B", "A→B", "B→A", "A→B\nrecal.", "B→A\nrecal."]
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def fig_violation(S, out, alpha):
    rules = ["ltt_selective", "empirical", "sgr", "crc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, mname in zip(axes, ("LR", "GB")):
        x = np.arange(len(SETS)); w = 0.2
        for j, r in enumerate(rules):
            g = S[(S.model == mname) & (S.alpha == alpha) & (S.rule == r) & (S.n_cal == "all")].set_index("setting").loc[SETS]
            y = g.viol_rate.values
            xx = x + (j - 1.5) * w
            ax.errorbar(xx, y, yerr=[y - g.viol_ci_lo.values, g.viol_ci_hi.values - y],
                        fmt="none", color=OI[r], capsize=1.5, lw=0.8)
            ref = g.refusals.values >= 25
            ax.plot(xx[~ref], y[~ref], "o", ms=3, color=OI[r], label=LAB[r])
            ax.plot(xx[ref], y[ref], "x", ms=4, color=OI[r])
        ax.axhline(0.10, color="0.3", ls="--", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(SLAB)
        ax.text(0.01, 0.97, mname, transform=ax.transAxes, va="top", fontweight="bold")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(f"Frequency of rFNR > {alpha:g} (50 runs)")
    axes[1].plot([], [], "x", color="0.3", label="refused in ≥25/50 runs")
    axes[1].legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(out / f"fig_rd_violation_a{int(alpha*100):02d}.pdf"); plt.close(fig)


def fig_rfnr(R, out, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, mname in zip(axes, ("LR", "GB")):
        for j, r in enumerate(["ltt_selective", "empirical"]):
            data = [R[(R.model == mname) & (R.alpha == alpha) & (R.rule == r) & (R.setting == s)
                      & (R.n_cal == "all") & (~R.refused)].retained_fnr.values for s in SETS]
            pos = np.arange(len(SETS)) + (j - 0.5) * 0.35
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                            flierprops=dict(ms=2, mec=OI[r]), medianprops=dict(color="k", lw=0.8))
            for b in bp["boxes"]:
                b.set(facecolor=OI[r], alpha=0.6, lw=0.6)
        ax.axhline(alpha, color="0.3", ls="--", lw=0.7)
        ax.set_xticks(np.arange(len(SETS))); ax.set_xticklabels(SLAB)
        ax.text(0.01, 0.97, mname, transform=ax.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel("Test retained FNR")
    h = [plt.Rectangle((0, 0), 1, 1, fc=OI[r], alpha=0.6) for r in ("ltt_selective", "empirical")]
    axes[1].legend(h, [LAB["ltt_selective"], LAB["empirical"]], frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(out / f"fig_rd_rfnr_a{int(alpha*100):02d}.pdf"); plt.close(fig)


def fig_sweep(S, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    sw = S[(S.setting == "sweep_A")].copy(); sw["nc"] = sw.n_cal.astype(int)
    full = S[(S.setting == "within_A") & (S.alpha == 0.05) & (S.n_cal == "all")]
    for mname, ls in (("LR", "-"), ("GB", ":")):
        for r in ("ltt_selective", "empirical"):
            g = sw[(sw.model == mname) & (sw.rule == r)].sort_values("nc")
            f = full[(full.model == mname) & (full.rule == r)]
            xs = np.r_[g.nc.values, 3991]
            axes[0].plot(xs, np.r_[g.refusals.values, f.refusals.values] / 50, ls, marker="o", ms=3,
                         color=OI[r], label=f"{LAB[r]}, {mname}", lw=0.9)
            axes[1].plot(xs, np.r_[g.viol_rate.values, f.viol_rate.values], ls, marker="o", ms=3,
                         color=OI[r], lw=0.9)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Calibration patients (site A; rightmost = full fold)")
    axes[0].set_ylabel("Refusal frequency"); axes[1].set_ylabel("Frequency of rFNR > 0.05")
    axes[1].axhline(0.10, color="0.3", ls="--", lw=0.7)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_rd_calsweep.pdf"); plt.close(fig)


def main():
    res, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(res / "summary.csv", dtype={"n_cal": str})
    R = pd.read_csv(res / "runs.csv", dtype={"n_cal": str})
    for a in (0.05, 0.10):
        fig_violation(S, out, a); fig_rfnr(R, out, a)
    fig_sweep(S, out)


if __name__ == "__main__":
    main()
