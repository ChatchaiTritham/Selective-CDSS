# Real-data evaluation protocol: PhysioNet/CinC Challenge 2019 (pre-specified)

Written on 2026-10-14, before any model was fitted or any result was inspected.
Every change made after results were seen is logged in the "Deviations" section at the end.

## Data
- PhysioNet/CinC Challenge 2019 "Early Prediction of Sepsis from Clinical Data", v1.0.0 (CC BY 4.0).
  Public, de-identified training sets A (hospital system A) and B (hospital system B), one `.psv` file per ICU stay.
- Obtained from the official PhysioNet AWS Open Data mirror (`s3://physionet-open/challenge-2019/1.0.0/`)
  by `download_data.py`. Every file is MD5-verified against its S3 ETag, and the SHA-256 manifest is written next to the data.
- The data are stored outside the repository (`D:/PhD-NU/data/physionet_sepsis2019/`) and are never committed.

## Prediction task (patient level, no future information)
- Observation window: the first 6 ICU hours (`ICULOS` 1–6).
- Inclusion: the stay lasts at least 7 hours, and `SepsisLabel` is 0 for every row in the window.
  This excludes patients who are already septic, or labelled as such, during the window.
- Label: y = 1 if `SepsisLabel` = 1 at any hour after the window. Challenge labels are Sepsis-3 onset shifted 6 h earlier.
- Features, all computed from the window only:
  - for the 8 vitals: last, mean, min and max;
  - for the 26 labs: the last value;
  - for each of the 34 variables: an "observed in window" indicator;
  - Age, Gender, Unit1, Unit2 (with NaN mapped to its own indicator), HospAdmTime, and the window length.
- Natural missingness is kept:
  - logistic regression gets median imputation fitted on the training fold, plus the indicators;
  - gradient boosting (HistGradientBoosting) uses native NaN handling plus the indicators.

## Base models (refit in every run, on that run's training fold only)
- LR: StandardScaler + LogisticRegression(C=1, class_weight="balanced", max_iter=2000).
- GB: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=run seed).
- Class weighting spreads the scores over [0, 1], so the fixed band (half-width m = 0.10) and the 0.5 cut are meaningful.
  None of the rules requires calibrated probabilities.

## Rules (all reuse the Selective-CDSS implementations in `scripts/ictx/scripts/run_ictx_r4.py`, alpha/delta as parameters)
1. **LTT-selective (ours)**: band |p − τ| ≥ m is retained.
   - The τ grid 0.01…0.99 is tested in ascending order (fixed sequence).
   - Each τ gets an exact binomial p-value for retained-positive FNR ≤ α.
   - Testing stops at the first non-rejection.
   - If no τ certifies, the rule refuses. Refusal floor: n_r ≥ ⌈ln δ / ln(1−α)⌉.
2. **LTT-overall**: the same test on all calibration positives. Applied with the same band at deployment (as in r4).
3. **Empirical (uncorrected)**: the largest τ with calibration FNR ≤ α, applied with the band.
4. **SGR** (Geifman and El-Yaniv 2017, binary search at δ/⌈log2 K⌉).
   - Uses the confidence cut max(p, 1−p) and the decision p ≥ 0.5.
   - Its test targets retained FNR, as in r4.
5. **SCRC-I** (Xu, Guo and Wei 2025, arXiv:2512.12844, Prop. 4), adapted to the class-conditional target.
   - Applied to the calibration positives, which are i.i.d. from X|Y=1.
   - Selection score g = max(p, 1−p), with target selection rate ξ = 0.8 on the cut grid of rule 4.
   - DKW lower bound on the selection rate at δ/2.
   - Uniform Hoeffding bound over the Λ1×Λ2 grid at δ/2.
   - Picks the smallest-risk feasible τ: the largest τ such that N̂ + ε ≤ α·ξ_LCB.
   - Refuses if no τ is feasible.
6. **CRC** (Angelopoulos et al. 2024): no abstention.
   - The largest τ with (n·FNR̂(τ) + 1)/(n + 1) ≤ α over the calibration positives.
   - The guarantee holds in expectation, not with high probability. Violation frequency is reported for comparison only.

Parameters: α ∈ {0.05, 0.10}, δ = 0.10, m = 0.10.

## Settings (50 runs each; each run draws a fresh, stratified, patient-level split with seed = 1000 + run)
- Per site, every run splits patients 40/20/40 into train/calibration/test.
- (i) **Within-site**: A→A and B→B. Model, calibration and test all come from the same site.
- (ii) **Cross-site**: A→B and B→A. Model and calibration come from the source site. The test fold is the target site's test fold.
- (ii-b) **Cross-site with target recalibration**: the source model is recalibrated on the target site's calibration fold.
- (iii) **Calibration-size sweep** (site A, both models, α = 0.05).
  - The calibration fold is subsampled to n_cal ∈ {100, 250, 500, 1000, 2000, all} patients.
  - The smaller sizes fall below the refusal floor of 45 retained positives.

## Outcomes, per run and rule
- On the held-out test patients:
  - true retained FNR (rFNR), with a 95% Clopper–Pearson CI;
  - coverage (retained fraction) and abstention (deferral) rate;
  - deferred patients per 1,000 (workload) and retained FPR.
- Violation: test rFNR > α.
  - The test fold is finite, so a stricter version is also reported: the 95% CP lower bound of test rFNR > α.
- Across runs:
  - violation frequency, with a 95% Clopper–Pearson CI (refused runs count as non-violations; a refusal sends every patient to a clinician);
  - the frequency among non-refused runs;
  - the number of refusals;
  - mean and SD of coverage and rFNR.

## Pre-specified expectations (to be judged honestly)
- E1: within-site, the violation frequency of the LTT-selective rule is ≤ δ = 0.10, up to binomial noise.
- E2: the empirical rule violates more often than δ.
- E3: cross-site shift may break the guarantee, because exchangeability fails. This is to be reported whatever the result.
- E4: below the floor, LTT-selective refuses.

## Deviations

- D1 (2026-10-14, before any result): feature columns with no observed value at one site
  (EtCO2 last/mean/min/max: never recorded in set A) are dropped because gradient boosting
  cannot bin an all-missing column; the EtCO2_obs indicator is kept. Detected by a crash, not by results.
- D2 (2026-10-14, after results): `extras.py` adds one descriptive quantity that was not pre-specified.
  - It counts "degenerate" certifications, where τ̂ ≤ m. In that case no retained patient can be predicted negative, so rFNR = 0 by construction.
  - It was added because GB certifications looked suspiciously perfect.
  - No pre-specified outcome was changed.

## Reproduce
```
python -I scripts/realdata/download_data.py <DATA>        # 20,336 + 20,000 .psv, MANIFEST_sha256.txt
python -I scripts/realdata/build_features.py <DATA> <DATA>/features_w6.csv.gz
python -I -W ignore scripts/realdata/run_realdata.py <DATA>/features_w6.csv.gz results/realdata 50
python -I scripts/realdata/extras.py results/realdata
python -I scripts/realdata/make_figures.py results/realdata results/realdata/figures
```
Data manifest SHA-256 (sorted path/sha list): 82ba256812d9c7d8913159921811c114d8353fb4651631ad6a0c201ae48e9105.
