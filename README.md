# Risk-Controlled Selective Prediction for Clinical Decision Services (Selective-CDSS)

> Wraps any probabilistic clinical classifier in an abstention layer that holds the false-negative rate of the decisions it keeps below a chosen ceiling, even when the incoming data is messy.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seed--42-success)

## Overview

A clinical model that scores well on clean data can quietly become unsafe once the inputs it sees in the field start to degrade — sensors drop out, values go missing, readings get noisy. The error that hurts patients most, a missed positive, has no built-in ceiling: as data quality slips, the false-negative rate (FNR) drifts upward with nothing to stop it. We treat that drift as the problem to solve rather than something to average away.

The approach here is deliberately model-agnostic. Instead of retraining or recalibrating, we add a thin wrapper around the classifier's probability output. A Learn-Then-Test (LTT) step searches a calibration fold for the most permissive decision threshold whose FNR still meets a stated target; cases that fall too close to that threshold are not predicted at all but handed back to a clinician. Every reported quantity — FNR, coverage, and harm — is read off the same retained set on a held-out test fold, so the safety claim and the cost of achieving it are measured under one consistent policy.

This repository is self-contained. The simulation and metrics library (`basics_cdss`) is vendored, and a single seeded driver rebuilds the cohort, fits the models, and regenerates every number and figure that appears in the companion manuscript. Nothing is copied by hand from one place to another.

## Key results

All figures below come straight from `run_eval.py` (seed 42, synthetic digital-twin cohort, in-distribution only — no human subjects).

- On clean test data the base model is strong (accuracy 98.6%, baseline FNR 3.1%); the wrapper trims retained FNR to 1.1% while abstaining on under 2% of cases.
- Push the data harder and the unguarded baseline FNR climbs to 25.8% under severe MCAR and to 21.6% under value-dependent (MNAR) missingness — both well past the 5% target.
- Across clean, three missingness mechanisms, and a severity sweep, the wrapped service keeps retained-decision test FNR at or below 2.2%, paying for it with abstention that grows from ~2% to ~37%.
- At moderate MCAR, full automation lets 10 positives through; the wrapper misses 4 among retained cases and defers the rest.
- Swapping in a weaker base model (logistic regression) drove baseline FNR to 38.1%, and the LTT step declined to certify the target rather than return an unsafe threshold — the intended failure mode.

## Repository structure

```
Selective-CDSS/
├── run_eval.py              # COMPUTE entry: cohort → models → all experiments → results/ (numbers + figure arrays)
├── scripts/
│   └── generate_figures.py  # PLOTTING module: reads results/ and draws the figures (no recompute)
├── pubviz.py                # shared, byte-identical Top-Tier figure style (palette / rcParams / save_fig)
├── results/
│   └── real_results.json    # committed reference output (every manuscript number + persisted figure arrays)
├── figures/
│   ├── fig_risk_coverage.{pdf,png}
│   └── fig_ablation.{pdf,png}
├── src/basics_cdss/         # vendored simulation + metrics library (disease models, conformal, coverage-risk, harm)
├── requirements.txt
├── setup.py                 # installs deps + the vendored package
└── LICENSE
```

Compute and plotting are separated. `run_eval.py` trains the models and persists
**every number and every figure array** (coverage-risk curves, the bootstrap CI
band, the baseline curves, and the ablation series) to `results/real_results.json`.
`scripts/generate_figures.py` only **reads** that file and renders the figures via
`from pubviz import apply_pub_style, save_fig, PALETTE, load_results`, so the
figures can be restyled or redrawn at any time **without** retraining.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # installs deps + the vendored basics_cdss package
```

## Reproducing the results

```bash
python run_eval.py                  # COMPUTE: fixed seed 42; ~1 min on a standard CPU workstation
python scripts/generate_figures.py  # PLOT ONLY: redraw figures from results/ (no retraining)
```

`run_eval.py` writes `real_results.json` (a root mirror plus the committed copy in `results/`) and, for convenience, calls the figure module at the end so a single command reproduces everything. The figures themselves are produced by `scripts/generate_figures.py`, which reads `results/real_results.json` and writes `fig_risk_coverage.{pdf,png}` and `fig_ablation.{pdf,png}` into both `figures/` and the repository root. Because the cohort, the train/calibration/test split, the random forest, the degradation operators, and the figure bootstrap are all seeded at 42, the JSON metrics and persisted figure arrays reproduce exactly run to run on the same machine. The PNG/PDF images depend on your matplotlib build, so the plotted data matches while the rendered bytes need not.

## Results and figures

- `figures/fig_risk_coverage.png` — Selective FNR versus coverage under moderate MCAR. The acceptance-threshold and softmax-response (SR) curves are overlaid, a 95% bootstrap confidence band surrounds the SR curve, the safe (FNR ≤ 5%) region is shaded, the split-conformal singleton operating point is marked, and the LTT operating point sits inside the safe zone while still retaining about two-thirds of inputs.
- `figures/fig_ablation.png` — Two vertically-stacked panels sharing the MCAR-severity axis (clean → severe). The top panel contrasts baseline FNR (no abstention) against LTT-retained FNR with the 5% target line; the bottom panel shows the abstention rate. The take-away is the gap: the baseline line rises and crosses the 5% target while the retained line stays near the floor, with abstention in the lower panel climbing to absorb the degradation. (This replaces the earlier dual-axis chart — stacked aligned panels avoid the dual-axis scale ambiguity.)

Both figures are drawn from arrays persisted to `results/real_results.json` by `run_eval.py` (the coverage-risk and baseline curves, the bootstrap band, and the per-level ablation series), not from any table of pre-typed values. The plotting module recomputes nothing: it reads those arrays and renders them, so there are no hardcoded results in the figure code.

## Data

The cohort is fully synthetic: 1,140 simulated patient trajectories (380 each for sepsis, ARDS, and cardiac models from the vendored `basics_cdss` digital-twin library), 48 summary features, prevalence 0.34, split 570 / 285 / 285 for train / calibration / test. Degradation is applied with documented operators — a library noise operator plus MCAR, MAR, and MNAR missingness masks — so no real patient records are involved and no ethics approval is required.

## Citation

```bibtex
@article{tritham_selective_cdss,
  title  = {Distribution-Free Safety Guarantees for AI Decision Services under
            Degraded Data Quality via Risk-Controlled Selective Prediction},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review (ICT Express)}
}
```

## License

Released under the MIT License (see `LICENSE`). The vendored `basics_cdss` package belongs to the companion project [BASICS-CDSS](https://github.com/ChatchaiTritham/BASICS-CDSS).

## Contact

**Chatchai Tritham** — Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand. Email: chatchait66@nu.ac.th · ORCID: 0000-0001-7899-228X
**Chakkrit Snae Namahoot** — same affiliation. Email: chakkrits@nu.ac.th · ORCID: 0000-0003-4660-4590

## Portfolio relationship

| Repository | Role |
|---|---|
| BASICS-CDSS | Beyond-accuracy evaluation methodology |
| TRI-X | Framework-level package |
| ORASR | Routing and safety-action component |
| DRAS-5 | Dynamic risk-state component |
| SAFE-Gate | Safety-gated ensemble framework |
| SynDX | Synthetic validation and explainability evidence |
| SURgul | SRGL/governance reproducibility component |
| TRI-X-CDSS | Integration and implementation package |
| Selective-CDSS | Risk-controlled selective-prediction (abstention) component |
| Causal-CDSS | Causal-inference evaluation component |
| Beyond-Accuracy | Simulation-based safety/calibration evaluation framework |
