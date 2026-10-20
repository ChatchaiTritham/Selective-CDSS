# ICT Express analyses

These scripts produce every number in the ICT Express paper "Risk-Controlled Selective Prediction for AI Decision Services under Degraded Data Quality", together with its supplementary tables and figures. They import `run_eval.py` from the repository root unchanged, so the cohort, splits and degradation operators are the same.

| Script | Output | Content |
|---|---|---|
| `scripts/run_ictx_addons.py` | `results/addons.json` | Certified LTT rule and Clopper–Pearson intervals |
| `scripts/run_ictx_r2.py` | `results/r2.json` | Selective-certified rule; SGR and Mondrian baselines; band sweep; 50 test-fold re-splits; WDBC |
| `scripts/run_ictx_r2_runtime.py` | `results/r2_runtime.json` | Earlier runtime measurement (superseded by r4) |
| `scripts/run_ictx_r4.py` | `results/r4.json` | Pool of 9,000 fresh simulated patients used to measure the true retained FNR over 50 re-splits (full and half calibration fold); subgroups; calibration shift; SGR with binary search and with fixed sequence; confidence ranking at τ̂; split-conformal FPR; δ sensitivity; runtime |
| `scripts/run_ictx_r5.py` | `results/r5.json` | Independent-run design used in the paper: a fresh cohort per run, training-median imputation, the pool degraded anew per run, per-run subgroups and condition-wise (Mondrian) selective calibration (supersedes the r4 pool analysis) |
| `scripts/run_ictx_r6.py` | `results/r6.json` | Same independent runs as r5 (verified identical) plus per-run workload, system FNR, calibration shift over 50 runs and per-run values for Fig. 1 |
| `scripts/run_ictx_r16_shiftgate.py` | `results/r16_shiftgate_samefill.json` (`same_fill`), `results/r16_shiftgate.json` | Label-free shift gate on the r6 cohorts: domain-classifier Mann-Whitney test on 285 unlabelled pool inputs; refusal on detected shift (paper Sec. 3.5, Table S20) |
| `scripts/run_ictx_r18_monitor.py` | `results/r18_monitor.json` | Anytime-valid retained-FNR monitor (mixture e-process, `scripts/realdata/monitor.py`) on the r16 runs (paper Sec. 3.5, Table S22) |
| `scripts/make_supplementary.py` | `supplementary.tex` | Supplementary Tables S1–S10 |
| `scripts/verify_refs.py` | — | Crossref lookup used to check the reference list |
| `figures/src/make_figures.py` | `fig_runs.pdf` (Fig. 1); also `fig_ablation.pdf` and `fig_risk_coverage.pdf` (earlier drafts, not used) | Figures |

Run from this folder. `<repo>` is the repository root (`../..`):

```
python ../../run_eval.py            # testbed, results/real_results.json
python ../../run_eval_secondary.py  # IoT check, results/real_results_secondary.json
python scripts/run_ictx_addons.py ../.. results/addons.json
python scripts/run_ictx_r2.py ../.. results/r2.json
python scripts/run_ictx_r4.py ../.. results/r4.json
python scripts/run_ictx_r5.py ../.. results/r5.json
python scripts/run_ictx_r6.py ../.. results/r6.json results/r5.json
python scripts/run_ictx_r16_shiftgate.py ../.. results/r16_shiftgate.json
python scripts/run_ictx_r16_shiftgate.py ../.. results/r16_shiftgate_samefill.json same_fill
python scripts/run_ictx_r18_monitor.py ../.. results/r18_monitor.json
python scripts/make_supplementary.py
python figures/src/make_figures.py
```

All seeds are fixed, so every number except wall-clock timing is deterministic. Tested with Python 3.11.15, numpy 2.4.6 and scikit-learn 1.9.0.
