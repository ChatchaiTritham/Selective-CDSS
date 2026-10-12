"""Regenerate both ICT Express figures from persisted REAL results (no science
recomputed): results/real_results.json (run_eval.py, seed 42) and
results/addons.json (scripts/run_ictx_addons.py), results/r2.json (scripts/run_ictx_r2.py). Vector PDF, Okabe-Ito
palette, sized to one 3p two-column column (222 pt), the journal-sample layout.
Run from the package root:  python figures/src/make_figures.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "results/real_results.json").read_text())
A = json.loads((ROOT / "results/addons.json").read_text())
B = json.loads((ROOT / "results/r2.json").read_text())["E1_rules"]  # scripts/run_ictx_r2.py
BLUE, VERM, GREEN, PINK, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555"
W = 222 / 72.27  # 3p two-column columnwidth (222 pt = 78.3 mm)
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False})


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


# ---- Fig. 2: risk-coverage at moderate MCAR --------------------------------
f = R["figure_data"]["moderate_mcar"]
c = np.array(f["coverage_grid"]) * 100
nz = lambda a: np.array([np.nan if v is None else v for v in a]) * 100
fig, ax = plt.subplots(figsize=(W, 0.70 * W), constrained_layout=True)
ax.fill_between(c, nz(f["sr_ci_lo"]), nz(f["sr_ci_hi"]), color=VERM, alpha=0.15, lw=0,
                label="Confidence ranking, 95% bootstrap band")
ax.plot(c, nz(f["sr_curve"]), color=VERM, ls="--", label="Confidence ranking (softmax response)")
ax.plot(c, nz(f["ltt_curve"]), color=BLUE, label="Score-threshold sweep")
ax.axhline(5, color=GREY, lw=0.8, ls=":")
ax.text(6, 5.6, "target FNR = 5%", fontsize=7, color=GREY)
e, ce = A["mcar"]["moderate"]["empirical"], A["mcar"]["moderate"]["certified"]
for d, col, mk, lab in [(e, BLUE, "o", "Empirical LTT"), (ce, GREEN, "s", r"Overall-cert. LTT"),
                        (B["moderate"]["selective_certified"], ORANGE, "D", r"Selective-cert. LTT")]:
    lo, hi = d["ci95"]
    ax.errorbar(d["coverage"] * 100, d["retained_fnr"] * 100,
                yerr=[[100 * (d["retained_fnr"] - lo)], [100 * (hi - d["retained_fnr"])]],
                fmt=mk, color=col, ms=5, capsize=2, lw=0.9, zorder=5, label=lab)
cc, cf = f["split_conformal_op"]
ax.plot(cc * 100, cf * 100, "^", color=PINK, ms=5, zorder=5, label="Split conformal (singletons)")
ax.set_xlabel("Coverage (% of test inputs decided)")
ax.set_ylabel("Retained FNR (%)")
ax.set_xlim(0, 101); ax.set_ylim(0, 20)
fig.legend(loc="outside lower center", ncol=2, columnspacing=0.8, handlelength=1.6, frameon=False)
fig.savefig(ROOT / "fig_risk_coverage.pdf", bbox_inches="tight", pad_inches=0.02)

# ---- Fig. 3: MCAR severity sweep -------------------------------------------
lv = ["clean", "mild", "moderate", "severe"]
x = np.arange(4)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 0.70 * W), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
def series(key):
    v, lo, hi = [], [], []
    for l in lv:
        a = A["mcar"][l]
        if key == "base":
            k, n = a["base_fn"], a["n_test_pos"]
        elif key == "selective_certified":
            k, n = B[l][key]["retained_fn"], B[l][key]["retained_pos"]
        else:
            k, n = a[key]["retained_fn"], a[key]["retained_pos"]
        p = k / n; l_, h_ = cp(k, n); v.append(100 * p); lo.append(100 * (p - l_)); hi.append(100 * (h_ - p))
    return np.array(v), [lo, hi]
for key, col, mk, off, lab in [("base", VERM, "D", -0.18, "Full automation (threshold 0.5)"),
                               ("empirical", BLUE, "o", -0.06, "Empirical LTT, retained"),
                               ("certified", GREEN, "s", 0.06, "Overall-cert. LTT, retained"),
                               ("selective_certified", ORANGE, "^", 0.18, "Selective-cert. LTT, retained")]:
    v, err = series(key)
    a1.errorbar(x + off, v, yerr=err, fmt=mk + "-", color=col, capsize=2, lw=0.9, label=lab)
a1.axhline(5, color=GREY, lw=0.8, ls=":")
a1.set_ylabel("Test FNR (%)")
a1.set_ylim(0, 40); a1.legend(loc="upper left", frameon=False)
for key, col, off, lab, hat in [("empirical", BLUE, -0.25, "Empirical", ""), ("certified", GREEN, 0.0, "Overall-cert.", "///"),
                               ("selective_certified", ORANGE, 0.25, "Selective-cert.", "...")]:
    ab = [100 * (1 - B[l][key]["coverage"]) for l in lv]
    fp = [100 * B[l][key]["retained_fpr"] for l in lv]
    a2.bar(x + off, ab, width=0.24, color=col, alpha=0.85, hatch=hat, edgecolor="white", lw=0, label=lab)
    a2.plot(x + off, fp, "_", color="k", ms=7, mew=1.2)
a2.plot([], [], "_", color="k", ms=7, mew=1.2, label="retained FPR")
a2.set_ylabel("Abst. / FPR (%)"); a2.set_ylim(0, 140); a2.set_yticks([0, 50, 100])
a2.legend(loc="upper center", ncol=4, frameon=False, fontsize=7, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
a2.set_xticks(x); a2.set_xticklabels(["Clean", "Mild\n(20%, 0.3)", "Moderate\n(35%, 0.6)", "Severe\n(50%, 0.9)"])
a2.set_xlabel(r"MCAR degradation level (mask rate, noise $\sigma$)")
fig.savefig(ROOT / "fig_ablation.pdf", bbox_inches="tight", pad_inches=0.02)
print("ok")

# ---- Fig. 1 (r6): true retained FNR over 50 independent runs -----------------
_r6 = ROOT / "results/r6.json"
if _r6.exists():
    Q = json.loads(_r6.read_text())
    ST = ["moderate", "severe", "mar", "mnar"]; LAB = ["MCAR\nmoderate", "MCAR\nsevere", "MAR", "MNAR"]
    SER = [("full/empirical", BLUE, "Empirical, full cal."), ("full/selective_certified", ORANGE, "Selective-cert., full cal."),
           ("half/empirical", BLUE, "Empirical, half cal."), ("half/selective_certified", ORANGE, "Selective-cert., half cal.")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 0.95 * W), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1.5, 1]})
    for i, st in enumerate(ST):
        for j, (k, col, lab) in enumerate(SER):
            v = [100 * x for x in Q["H3_per_run"][st][k] if x is not None]
            pos = i + (j - 1.5) * 0.2
            bp = a1.boxplot(v, positions=[pos], widths=0.16, patch_artist=True, showfliers=False,
                            medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            bp["boxes"][0].set(facecolor=col, alpha=0.9 if j < 2 else 0.35, lw=0.6, hatch="" if j < 2 else "///")
            a1.plot(np.full(len(v), pos) + np.random.RandomState(j).uniform(-0.05, 0.05, len(v)), v, ".",
                    color="k", ms=1.5, alpha=0.5)
    a1.axhline(5, color=GREY, lw=0.8, ls=":")
    a1.set_xticks(range(4)); a1.set_xticklabels([]); a1.set_ylabel("True retained FNR (%)"); a1.set_ylim(0, 12)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(facecolor=c_, alpha=al, hatch=h, label=l) for (_, c_, l), al, h in
                       zip(SER, [0.9, 0.9, 0.35, 0.35], ["", "", "///", "///"])],
              loc="upper left", ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.6)
    for i, st in enumerate(ST):
        v = [100 * x for x in Q["H3_acs_per_run"][st]]
        a2.boxplot(v, positions=[i], widths=0.35, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=VERM, alpha=0.6, lw=0.6), medianprops=dict(color="k", lw=0.8))
        a2.plot(np.full(len(v), i) + np.random.RandomState(9).uniform(-0.1, 0.1, len(v)), v, ".", color="k", ms=1.5, alpha=0.5)
        a2.text(i, 33, f"{sum(x > 5 for x in v)}/{len(v)} runs > 5%" if i == 0 else f"{sum(x > 5 for x in v)}/{len(v)}", ha="center", fontsize=6.5)
    a2.axhline(5, color=GREY, lw=0.8, ls=":")
    a2.set_xticks(range(4)); a2.set_xticklabels(LAB); a2.set_ylim(0, 36)
    a2.set_ylabel("ACS retained\nFNR (%)")
    fig.savefig(ROOT / "fig_runs.pdf", bbox_inches="tight", pad_inches=0.02)
