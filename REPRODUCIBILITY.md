# Reproducibility

All numbers in the companion manuscript are produced by `run_eval.py` at a fixed
seed (42) on a fully synthetic digital-twin cohort. No human-subject or external
data is involved.

## One command

```bash
git clone https://github.com/ChatchaiTritham/Selective-CDSS.git
cd Selective-CDSS
pip install -e .
python run_eval.py            # COMPUTE: ~1 min on a standard CPU workstation
```

`run_eval.py` rebuilds the cohort, fits the models, runs every experiment, and
writes `results/real_results.json` (plus a root mirror) and the two figures. The
plotting step (`scripts/generate_figures.py`) only *reads* that JSON and never
recomputes the science.

## Determinism

The cohort simulation, the train/calibration/test split (570/285/285), the
RandomForest (300 trees, depth 12), the degradation operators (Gaussian noise +
MCAR/MAR/MNAR masks), and the figure bootstrap are all seeded at 42, so the JSON
metrics and persisted figure arrays reproduce exactly run-to-run on the same
machine. Rendered PNG/PDF bytes depend on your matplotlib build; the plotted data
matches even when the image bytes do not.

## Headline numbers (committed `results/real_results.json`, seed 42)

| Setting | Base FNR | Retained FNR (test) | Abstention | Certified |
|---|---|---|---|---|
| Clean | 3.1% | 1.1% | 1.8% | yes |
| MCAR moderate | 10.3% | 2.2% | 35.1% | yes |
| MCAR severe | 25.8% | 1.2% | 37.2% | yes |
| MAR moderate | 22.7% | 0.0% | 34.4% | yes |
| MNAR moderate | 21.6% | 1.1% | 37.5% | yes |
| LogReg (weak base) | 38.1% | — | — | **no** (declined) |

Target FNR = 5%; abstention band half-width m = 0.10. Across clean, three
missingness mechanisms, and the severity sweep, retained-decision test FNR stays
at or below 2.2%. With a weak base model the LTT step declines to certify rather
than return an unsafe threshold.

## Tests

```bash
pip install pytest
python -m pytest -q                 # fast regression-lock of the committed JSON
python -m pytest -q -m slow         # live re-run of run_eval.py vs committed JSON
```

The default run locks the committed numbers; the `slow` test re-executes the full
pipeline and asserts the regenerated metrics match within tolerance.
