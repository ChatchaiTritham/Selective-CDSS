# ICT Express analyses

These scripts produce every number in the ICT Express paper "Risk-Controlled Selective Prediction for AI Decision Services under Degraded Data Quality", together with its supplementary tables and figures. They import `run_eval.py` from the repository root unchanged, so the cohort, splits and degradation operators are the same.

| Script | Output | Content |
|---|---|---|
| `scripts/run_ictx_addons.py` | `results/addons.json` | Certified LTT rule and Clopper–Pearson intervals |
| `scripts/run_ictx_r2.py` | `results/r2.json` | Selective-certified rule; SGR and Mondrian baselines; band sweep; 50 test-fold re-splits; WDBC |
| `scripts/run_ictx_r2_runtime.py` | `results/r2_runtime.json` | Earlier runtime measurement (superseded by r4) |
| `scripts/run_ictx_r4.py` | `results/r4.json` | Pool of 9,000 fresh simulated patients used to measure the true retained FNR over 50 re-splits (full and half calibration fold); subgroups; calibration shift; SGR with binary search and with fixed sequence; confidence ranking at τ̂; split-conformal FPR; δ sensitivity; runtime |
| `scripts/make_supplementary.py` | `supplementary.tex` | Supplementary Tables S1–S9 |
| `figures/src/make_figures.py` | `fig_ablation.pdf`, `fig_risk_coverage.pdf` | Figures |

Run from this folder. `<repo>` is the repository root (`../..`):

```
python scripts/run_ictx_r2.py ../.. results/r2.json
python scripts/run_ictx_r4.py ../.. results/r4.json
python scripts/make_supplementary.py
python figures/src/make_figures.py
```

All seeds are fixed, so every number except wall-clock timing is deterministic. Tested with Python 3.11.15, numpy 2.4.6 and scikit-learn 1.9.0.
