# Selective-CDSS — Distribution-Free Safety Guarantees for AI Decision Services under Degraded Data Quality

Reproducibility code for the manuscript:

> **Distribution-Free Safety Guarantees for AI Decision Services under Degraded
> Data Quality via Risk-Controlled Selective Prediction**
> (target: *ICT Express*, Elsevier/KICS)

The method wraps any probabilistic clinical classifier in a **Learn-Then-Test
(LTT) conformal risk controller** that abstains near the decision boundary so the
**false-negative rate (FNR) on retained cases stays below a target** (here 5%),
even as input data quality degrades. Evaluation runs on a controlled digital-twin
testbed (sepsis / ARDS / cardiac) from the `basics_cdss` library.

## What is in this repository

| Path | Contents |
|------|----------|
| `run_eval.py` | Single seeded driver (`SEED=42`) — builds the cohort, fits the models, runs every experiment, writes results + figures |
| `results/real_results.json` | Committed reference output (the numbers reported in the manuscript) |
| `src/basics_cdss/` | Vendored simulation + metrics library, so the repo is self-contained |
| `figures/` | Pre-generated `fig_risk_coverage` and `fig_ablation` (PDF + PNG) |

This is **not** a skeleton: every number in the manuscript is produced by
`run_eval.py`. No values are hardcoded in figure scripts.

## Reproduce (≈1 minute, CPU only)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # installs deps + the vendored basics_cdss package
python run_eval.py
```

This writes `real_results.json` and `fig_risk_coverage.{pdf,png}` /
`fig_ablation.{pdf,png}` to the repository root. Compare against the committed
`results/real_results.json` — the run is deterministic (`seed=42`), so the files
match bit-stably across machines.

## Headline results (`results/real_results.json`)

Cohort: 1,140 simulated trajectories (380 per disease), 48 features, prevalence
0.34, split 570 / 285 / 285 (train / calibration / test). Target FNR = 0.05,
abstention band `|p − τ| < 0.10`.

| Condition | Baseline FNR (no abstention) | LTT retained FNR (test) | Abstention | Risk controlled |
|-----------|------------------------------|--------------------------|------------|-----------------|
| Clean | 0.031 | 0.011 | 0.018 | ✅ |
| MCAR (rate 0.35, σ 0.6) | 0.103 | 0.022 | 0.351 | ✅ |
| MAR | 0.227 | 0.000 | 0.344 | ✅ |
| MNAR | 0.216 | 0.011 | 0.375 | ✅ |
| Severe MCAR (0.50, σ 0.9) | 0.258 | 0.012 | 0.372 | ✅ |

Baseline comparison at moderate MCAR (target FNR 0.05): only LTT *targets* FNR
with a certified guarantee — full automation 0.103, softmax-response 0.052,
split-conformal 0.048, **LTT 0.022** (coverage 0.65).

## Method coverage

`run_eval.py` produces, all on the test fold under one consistent abstention
policy: (i) clean reference, (ii) three missingness mechanisms (MCAR / MAR /
MNAR), (iii) an MCAR severity sweep, (iv) a model-agnostic check (logistic
regression), (v) a baseline comparison (full automation, softmax-response,
split-conformal, LTT), and (vi) harm-by-risk-tier on the retained set.

## Citation

```bibtex
@article{tritham_selective_cdss,
  title  = {Distribution-Free Safety Guarantees for AI Decision Services under
            Degraded Data Quality via Risk-Controlled Selective Prediction},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Licensed under the MIT License (see `LICENSE`). The vendored `basics_cdss`
package is part of the companion project
[BASICS-CDSS](https://github.com/ChatchaiTritham/BASICS-CDSS).
