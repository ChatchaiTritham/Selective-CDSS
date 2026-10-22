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

## Addendum A1 (2026-10-20, written before running `run_shift.py`; post hoc relative to the main study)
The main study showed cross-site failure (E3). A1 asks whether the failure can be caught or corrected
WITHOUT target-site labels. It is exploratory: the main results above were already known.
- Data, splits, seeds (1000-1049), model (LR only), alpha = 0.05, delta = 0.10, m = 0.10 as in the main study.
- Per run and setting (A->A, B->B, A->B, B->A): the target test fold is split at random (seed) into an
  unlabelled half U (used only for shift handling, labels never read) and an evaluation half E.
- Shift gate: a domain classifier (median-impute + standardise + logistic regression) separates source
  calibration inputs from U with 5-fold cross-fitted scores; one-sided Mann-Whitney test of AUC > 0.5.
  If p <= delta the rule REFUSES (shift detected), else the ordinary selective rule is used.
- Weighted rule: importance weights w(x) = c(x)/(1-c(x)) * n_cal/n_U from the cross-fitted domain
  probabilities on source calibration points, clipped at 20. Weighted retained miss fraction f and Kish
  effective size n_eff = (sum w)^2 / sum w^2 over retained positives; p-value = Pr[Bin(n_eff, alpha) <= f*n_eff]
  via the regularised incomplete beta; same fixed-sequence scan. Approximate: assumes covariate shift
  (p(y|x) shared) and estimated weights, so no finite-sample guarantee is claimed.
- Outcomes on E: violations (retained FNR > alpha), refusals, mean retained FNR, coverage, retained FPR.
- Expectations: A1-1 the gate rarely refuses within site (about delta) and almost always across sites;
  A1-2 the weighted rule lowers cross-site violations relative to the unweighted rule, at some cost.
  Reported whatever the result.

## Addendum A2 (2026-10-20, written before running the monitor; post hoc relative to A1)
A1 showed that the label-free gate misses gradual drift. A2 adds a label-based anytime-valid monitor
for the deployed threshold, in the spirit of Podkopaev & Ramdas (ICLR 2022).
- Deployed threshold: the ungated selective-certified threshold of each run/setting (as in A1).
- Stream: retained positives of the evaluation half E in random order (seed); X_i = 1 if missed.
- e-process: E_t = mean over lambda in {1, 2, 5, 10, 15} of prod_{i<=t} (1 + lambda (X_i - alpha)),
  alpha = 0.05; alarm when E_t >= 1/delta = 10. Under H0: rFNR <= alpha, Pr[ever alarm] <= delta (Ville).
- Outcomes: alarm frequency and number of labelled retained positives until alarm, separately for runs
  whose retained FNR on E exceeds alpha and runs that do not.
- Expectation: alarms in most exceeding runs; alarms in at most about delta of non-exceeding runs.
  Reported whatever the result. The same monitor is applied in simulation
  (scripts/ictx/scripts/run_ictx_r18_monitor.py).

## Addendum A3 (2026-10-22, written before running `run_eicu.py`): second real dataset
Question: do the refusal floor, the label-free gate (A1) and the anytime-valid monitor (A2) behave on a
second, independent multi-hospital ICU source? Exploratory; A1/A2 results were known.
- Data: eICU Collaborative Research Database Demo v2.0.1 (PhysioNet, open access, ODbL;
  doi:10.13026/4mxk-na84). Tables patient, hospital, apacheApsVar (SHA-256 checked against SHA256SUMS.txt).
  apachePatientResult (APACHE predicted mortality) is NOT used, to avoid leakage.
- Cohort: first ICU stay of each hospital stay; known hospital discharge status; hospital region known.
  Outcome: in-hospital death (hospitaldischargestatus == Expired).
- Features: apacheApsVar first-24-h physiology (value -1 = missing -> NaN), intubated, vent, dialysis,
  GCS components, age (">89" -> 90), gender; missing indicators; median imputation inside the pipeline.
- Sites: source S = Midwest + West, target T = South + Northeast (regions with unknown label excluded).
- Model: class-weighted logistic regression (same pipeline as run_realdata.py), trained on 40% of S.
- Runs: 50 (seeds 1000-1049); S split 40/20/40 train/calibration/test (stratified); T split in halves:
  U (unlabelled, gate only) and E (evaluation); within-site check uses S-test halves the same way.
- Rules: selective-certified LTT at alpha in {0.05, 0.10}, delta = 0.10, m = 0.10; uncorrected rule
  (empirical, banded) at the same alpha.
- Gate (A1 definition) on S-calibration inputs vs U: refusal rate within S and across S->T.
- Monitor (A2 definition, alpha of the rule) on E for every deployed threshold (uncorrected always;
  certified when it certifies): alarm and labels-to-alarm, split by whether retained FNR on E > alpha.
- Expectations: alpha = 0.05 mostly refuses (calibration positives near or below the floor of 45);
  gate refusals within S about delta; any detection across S->T reported as found; monitor false
  alarms rare where retained FNR <= alpha. Reported whatever the result; small sample, descriptive.

## Addendum A4 (2026-10-22, written before running the A4 scripts): finite-sample-valid gate
A3 showed the A1 gate (5-fold cross-fitted scores + Mann-Whitney) refusing 17/50 exchangeable eICU runs
at nominal delta = 0.10: cross-fitted scores are dependent, so the test is anti-conservative at small n.
A4 replaces it by a sample-split classifier two-sample test (Lopez-Paz & Oquab, ICLR 2017;
Kim, Ramdas, Singh & Wasserman, Ann. Stat. 2021), valid in finite samples under exchangeability:
- Pool calibration inputs (label 0) and unlabelled deployment inputs U (label 1); split the pool 50/50 at
  random, stratified by label (seed of the run).
- Fit the same logistic domain classifier (median impute + standardise + LR) on the first half only.
- Statistic: AUROC of its scores on the second half (U vs calibration).
- p-value: permutation of the labels within the second half, B = 999 draws (seed 7000 + run),
  p = (1 + #{perm AUROC >= observed}) / (B + 1). Refuse if p <= delta.
- Everything else (data, runs, seeds, splits, U/E halves, thresholds, monitor) is unchanged from
  A1 (PhysioNet), r16/r18 (simulation, same_fill) and A3 (eICU), so A4 differs only in the gate.
- Outcomes: gate refusals in exchangeable settings (target: about delta), detection in shifted settings,
  failures left undetected by gate, and by gate + monitor, per dataset.
- Expectation: exchangeable refusal rates near 0.10 on all three datasets; some loss of power relative
  to A1 because only half of the pooled inputs is scored. Reported whatever the result.
Scripts: scripts/realdata/gate_split.py (test), run_shift_a4.py (PhysioNet), run_eicu.py --gate split
(eICU), scripts/ictx/scripts/run_ictx_r22_gate_a4.py (simulation).

## Reproduce
```
python -I scripts/realdata/download_data.py <DATA>        # 20,336 + 20,000 .psv, MANIFEST_sha256.txt
python -I scripts/realdata/build_features.py <DATA> <DATA>/features_w6.csv.gz
python -I -W ignore scripts/realdata/run_realdata.py <DATA>/features_w6.csv.gz results/realdata 50
python -I scripts/realdata/extras.py results/realdata
python -I scripts/realdata/make_figures.py results/realdata results/realdata/figures
python -I -W ignore scripts/realdata/run_shift.py <DATA>/features_w6.csv.gz results/realdata_shift 50
python -I -W ignore scripts/realdata/run_monitor.py <DATA>/features_w6.csv.gz results/realdata_monitor 50
python -I -W ignore scripts/realdata/run_eicu.py <EICU_DEMO_DIR> results/eicu 50
```
Data manifest SHA-256 (sorted path/sha list): 82ba256812d9c7d8913159921811c114d8353fb4651631ad6a0c201ae48e9105.
