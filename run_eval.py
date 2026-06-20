"""
ICT Express letter — REAL evaluation (v2: fixes CRITICAL reviewer risks)
Risk-controlled selective prediction (abstention) for AI services under degraded
data quality, via Learn-Then-Test (LTT) conformal risk control. Validated on a
controlled digital-twin testbed (sepsis/ARDS/cardiac).

FIXES vs v1:
  C2  ONE consistent abstention policy everywhere: abstain iff |p - tau| < margin
      (the same band the LTT routine uses); FNR, coverage and harm are all
      computed on the SAME retained set, on the TEST fold (not the calibration
      empirical risk).
  C1  Three missingness mechanisms (MCAR / MAR / MNAR), not MCAR only.

All numbers are REAL outputs of the basics_cdss library + standard, explicitly
documented missingness operators. No mock data.

Architecture (separation of concerns):
  * This module is the COMPUTE entry. It rebuilds the cohort, fits the models,
    runs every experiment, and persists ALL numbers AND the arrays the figures
    need (coverage-risk curves, bootstrap CI band, baseline curves, ablation
    series) to results/real_results.json.
  * Plotting lives in the importable module scripts/generate_figures.py, which
    READS results/ and draws the figures. It never recomputes science.
  * For convenience run_eval.py calls that figure module at the very end, but
    the figures can be restyled/redrawn at any time WITHOUT retraining by just
    running:  python scripts/generate_figures.py

Run:  python run_eval.py
Out:  results/real_results.json (+ real_results.json mirror at repo root)
      figures/fig_risk_coverage.{pdf,png}  figures/fig_ablation.{pdf,png}
      (mirrored to repo root for the README)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from basics_cdss.temporal.disease_models import (
    SepsisModel, RespiratoryDistressModel, CardiacEventModel,
)
from basics_cdss.clinical_metrics.conformal_prediction import (
    risk_control_conformal, split_conformal_classification,
)
from basics_cdss.metrics.coverage_risk import (
    selective_prediction_metrics, coverage_risk_curve, area_under_risk_coverage_curve,
)
from basics_cdss.metrics.harm import harm_by_risk_tier
from basics_cdss.scenario.perturbations import NoiseOperator, PerturbationConfig

SEED = 42
MARGIN = 0.10           # abstention band half-width (matches LTT routine)
TARGET_FNR = 0.05
rng = np.random.RandomState(SEED)
OUT = Path(__file__).parent

# ---- Top-Tier figure style (canonical _management/FIGURE_STYLE.md) ----------
# Color-blind-safe Okabe-Ito palette; use in this order so series colors stay
# consistent across every figure and every repo.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_pub_style():
    """Publication-grade matplotlib rcParams. Call once before plotting."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })

BASELINES = {
    "sepsis": (SepsisModel, {
        "blood_pressure_sys": (118, 18), "heart_rate": (92, 18), "lactate": (2.2, 1.1),
        "respiratory_rate": (19, 4), "temperature": (37.6, 0.9), "white_blood_cell_count": (11.5, 4.0),
    }, "_infection_severity"),
    "ards": (RespiratoryDistressModel, {
        "heart_rate": (96, 18), "oxygen_saturation": (93, 4),
        "pf_ratio": (290, 90), "respiratory_rate": (23, 5),
    }, "_lung_injury"),
    "cardiac": (CardiacEventModel, {
        "blood_pressure_dia": (80, 12), "blood_pressure_sys": (134, 20), "chest_pain_score": (3.0, 2.0),
        "heart_rate": (88, 18), "st_elevation": (0.10, 0.10), "troponin": (0.06, 0.12),
    }, "_ischemia_severity"),
}
N_PER, T_STEPS, DT = 380, 24, 0.25


def make_patient(model, spec, lat):
    state = {k: float(rng.normal(m, s)) for k, (m, s) in spec.items()}
    traj = []
    for _ in range(T_STEPS):
        state = model.evolve(state, DT, None, rng)
        traj.append({k: state[k] for k in spec})
    df = pd.DataFrame(traj)
    feat = {}
    for k in spec:
        c = df[k].values
        feat[f"{k}__last"], feat[f"{k}__mean"] = c[-1], c.mean()
        feat[f"{k}__max"], feat[f"{k}__slope"] = c.max(), (c[-1] - c[0]) / T_STEPS
    return feat, float(state.get(lat, 0.0))


def build_cohort():
    rows, sev = [], []
    for _, (cls, spec, lat) in BASELINES.items():
        m = cls()
        for _ in range(N_PER):
            f, s = make_patient(m, spec, lat); rows.append(f); sev.append(s)
    X = pd.DataFrame(rows); sev = np.array(sev)
    y = (sev >= np.quantile(sev, 0.66)).astype(int)
    cols = list(X.columns)
    Xmat = X[cols].apply(lambda c: c.fillna(c.median())).values
    return Xmat, y, cols


def fnr_at(y_true, y_proba, tau):
    yp = (y_proba >= tau).astype(int)
    pos = (y_true == 1).sum()
    return 0.0 if pos == 0 else ((y_true == 1) & (yp == 0)).sum() / pos


def degrade(Xmat, cols, medians, mechanism, rate, sigma, seed=SEED):
    """REAL Gaussian noise (library NoiseOperator) + documented missingness.
       MCAR: mask independent of data. MAR: mask prob depends on an OBSERVED
       feature (heart_rate__mean). MNAR: mask prob depends on the value itself
       (extreme/high values more likely missing -> 'sick patients have gaps')."""
    r = np.random.RandomState(seed)
    cfg = PerturbationConfig(p_mask=0.0, noise_sigma=sigma, continuous_features=list(cols))
    noise = NoiseOperator(cfg, seed=seed)
    out = Xmat.copy().astype(float)
    # reference observed feature for MAR
    ref_j = cols.index("heart_rate__mean") if "heart_rate__mean" in cols else 0
    ref = Xmat[:, ref_j]
    ref_hi = ref > np.median(ref)
    for i in range(out.shape[0]):
        d = {c: float(out[i, j]) for j, c in enumerate(cols)}
        d, _ = noise.apply(d)                          # REAL noise operator
        row = np.array([d[c] for c in cols])
        if mechanism == "mcar":
            mask = r.rand(len(cols)) < rate
        elif mechanism == "mar":
            p = rate * (1.5 if ref_hi[i] else 0.5)     # depends on observed ref
            mask = r.rand(len(cols)) < p
        elif mechanism == "mnar":
            # per-feature: higher (standardized) value -> higher mask prob
            z = (row - np.array([medians[c] for c in cols]))
            p = np.clip(rate * (1.0 + 0.8 * np.sign(z)), 0, 1)
            mask = r.rand(len(cols)) < p
        else:
            mask = np.zeros(len(cols), bool)
        for j, c in enumerate(cols):
            out[i, j] = medians[c] if mask[j] else row[j]
    return out


def policy(p, tau, margin=MARGIN):
    """ONE selective policy used everywhere: abstain iff |p - tau| < margin."""
    abstain = np.abs(p - tau) < margin
    retained = ~abstain
    pred = (p >= tau).astype(int)
    return retained, pred


def evaluate(clf, X_cal, y_cal, X_te, y_te):
    rc = risk_control_conformal(clf, X_cal, y_cal, X_te, fnr_at, target_risk=TARGET_FNR)
    tau = rc.threshold
    p_te = clf.predict_proba(X_te)[:, 1]
    base_pred = (p_te >= 0.5).astype(int)
    base_fnr = fnr_at(y_te, p_te, 0.5)
    base_acc = float((base_pred == y_te).mean())
    retained, pred = policy(p_te, tau)
    cov = float(retained.mean())
    # retained TEST FNR (consistent policy, test fold)
    ret_fnr = fnr_at(y_te[retained], p_te[retained], tau) if retained.sum() else 0.0
    sp = selective_prediction_metrics(y_te, p_te, target_coverage=0.8, target_risk=0.10)
    return dict(tau=float(tau), base_fnr=float(base_fnr), base_acc=base_acc,
                cal_empirical_risk=float(rc.empirical_risk),
                risk_controlled=bool(rc.risk_controlled),
                coverage=cov, abstention=float(1 - cov),
                retained_fnr_test=float(ret_fnr), aurc=float(sp.aurc),
                p_te=p_te, retained=retained, base_pred=base_pred)


def tiers_from(p):
    e = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    return np.array([f"R{i+1}" for i in np.digitize(p, e)])


def main():
    print("[*] cohort...")
    X, y, cols = build_cohort()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
    X_cal0, X_te0, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    medians = {c: float(np.median(X_tr[:, j])) for j, c in enumerate(cols)}
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    print(f"    cohort={len(y)} feat={X.shape[1]} prev={y.mean():.3f} "
          f"train={len(y_tr)} cal={len(y_cal)} test={len(y_te)}")

    results = {"cohort": {"n": int(len(y)), "n_features": int(X.shape[1]),
                          "prevalence": float(y.mean()), "n_per_disease": N_PER,
                          "seed": SEED, "margin": MARGIN, "target_fnr": TARGET_FNR},
               "split": {"train": int(len(y_tr)), "cal": int(len(y_cal)), "test": int(len(y_te))}}

    # ---- clean reference ----
    clean = evaluate(clf, X_cal0, y_cal, X_te0, y_te)
    results["clean"] = {k: clean[k] for k in
                        ["tau", "base_fnr", "base_acc", "coverage", "abstention",
                         "retained_fnr_test", "cal_empirical_risk", "risk_controlled", "aurc"]}
    print(f"    clean: baseFNR={clean['base_fnr']:.3f} retFNR={clean['retained_fnr_test']:.3f} "
          f"cov={clean['coverage']:.3f}")

    # ---- C1: mechanism comparison (MCAR/MAR/MNAR) at moderate rate ----
    print("[*] mechanisms (MCAR/MAR/MNAR @ rate0.35,sigma0.6)...")
    mech = {}
    for mn in ["mcar", "mar", "mnar"]:
        Xc = degrade(X_cal0, cols, medians, mn, 0.35, 0.6)
        Xt = degrade(X_te0, cols, medians, mn, 0.35, 0.6)
        e = evaluate(clf, Xc, y_cal, Xt, y_te)
        mech[mn] = {k: e[k] for k in ["tau", "base_fnr", "base_acc", "coverage",
                                       "abstention", "retained_fnr_test", "risk_controlled", "aurc"]}
        print(f"    {mn.upper()}: baseFNR={e['base_fnr']:.3f} retFNR_test={e['retained_fnr_test']:.3f} "
              f"abstain={e['abstention']:.3f} ctrl={e['risk_controlled']}")
    results["mechanisms"] = mech

    # ---- ablation: MCAR severity sweep (consistent policy, test FNR) ----
    print("[*] degradation sweep (MCAR)...")
    abl = []
    for name, rate, sig in [("clean", 0.0, 0.0), ("mild", 0.20, 0.3),
                            ("moderate", 0.35, 0.6), ("severe", 0.50, 0.9)]:
        if rate == 0.0:
            Xc, Xt = X_cal0, X_te0
        else:
            Xc = degrade(X_cal0, cols, medians, "mcar", rate, sig)
            Xt = degrade(X_te0, cols, medians, "mcar", rate, sig)
        e = evaluate(clf, Xc, y_cal, Xt, y_te)
        abl.append({"level": name, "rate": rate, "noise_sigma": sig,
                    "base_fnr": e["base_fnr"], "retained_fnr_test": e["retained_fnr_test"],
                    "abstention": e["abstention"], "aurc": e["aurc"],
                    "risk_controlled": e["risk_controlled"], "tau": e["tau"]})
        print(f"    {name}: baseFNR={e['base_fnr']:.3f} retFNR_test={e['retained_fnr_test']:.3f} "
              f"abstain={e['abstention']:.3f} ctrl={e['risk_controlled']}")
    results["ablation_mcar"] = abl

    # ---- model-agnostic check: logistic regression at moderate MCAR ----
    lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    Xc = degrade(X_cal0, cols, medians, "mcar", 0.35, 0.6)
    Xt = degrade(X_te0, cols, medians, "mcar", 0.35, 0.6)
    e_lr = evaluate(lr, Xc, y_cal, Xt, y_te)
    results["logreg_moderate_mcar"] = {k: e_lr[k] for k in
                                       ["base_fnr", "retained_fnr_test", "abstention", "risk_controlled"]}
    print(f"    LogReg moderate: baseFNR={e_lr['base_fnr']:.3f} retFNR={e_lr['retained_fnr_test']:.3f} "
          f"ctrl={e_lr['risk_controlled']}")

    # ---- M2: baseline comparison at moderate MCAR (why LTT?) ----------------
    # Same degraded cal/test; fixed target FNR=0.05. Three selective methods +
    # full automation, each reporting retained-FNR(test), coverage, and whether
    # it *targets* FNR with a guarantee.
    print("[*] baselines @ moderate MCAR...")
    Xc = degrade(X_cal0, cols, medians, "mcar", 0.35, 0.6)
    Xt = degrade(X_te0, cols, medians, "mcar", 0.35, 0.6)
    p_cal = clf.predict_proba(Xc)[:, 1]
    p_test = clf.predict_proba(Xt)[:, 1]

    # (a) full automation
    full_pred = (p_test >= 0.5).astype(int)
    base = {"method": "Full automation", "retained_fnr": float(fnr_at(y_te, p_test, 0.5)),
            "coverage": 1.0, "targets_fnr": False}

    # (b) softmax-response (SR) selective: abstain least-confident to MATCH LTT coverage
    rc_m = risk_control_conformal(clf, Xc, y_cal, Xt, fnr_at, target_risk=TARGET_FNR)
    ret_ltt, pred_ltt = policy(p_test, rc_m.threshold)
    ltt_cov = float(ret_ltt.mean())
    conf = np.maximum(p_test, 1 - p_test)
    n_keep = int(round(ltt_cov * len(p_test)))
    keep_idx = np.argsort(-conf)[:n_keep]               # most-confident n_keep
    sr_ret = np.zeros(len(p_test), bool); sr_ret[keep_idx] = True
    sr_pred = (p_test >= 0.5).astype(int)
    sr = {"method": "Softmax-response (matched coverage)",
          "retained_fnr": float(fnr_at(y_te[sr_ret], p_test[sr_ret], 0.5)) if sr_ret.sum() else 0.0,
          "coverage": float(sr_ret.mean()), "targets_fnr": False}

    # (c) split-conformal sets (alpha=0.05): abstain non-singletons
    cp = split_conformal_classification(clf, X_tr, y_tr, Xc, y_cal, Xt, alpha=0.05)
    sizes = np.asarray(cp.set_sizes)
    cp_ret = sizes == 1
    # singleton predicted label = argmax proba
    cp_pred = (p_test >= 0.5).astype(int)
    cp = {"method": "Split-conformal (alpha=0.05)",
          "retained_fnr": float(fnr_at(y_te[cp_ret], p_test[cp_ret], 0.5)) if cp_ret.sum() else 0.0,
          "coverage": float(cp_ret.mean()), "targets_fnr": False}

    # (d) LTT (ours)
    ltt = {"method": "LTT risk control (ours)",
           "retained_fnr": float(fnr_at(y_te[ret_ltt], p_test[ret_ltt], rc_m.threshold)) if ret_ltt.sum() else 0.0,
           "coverage": ltt_cov, "targets_fnr": True, "certified": bool(rc_m.risk_controlled)}
    results["baselines_moderate_mcar"] = {"target_fnr": TARGET_FNR,
                                           "methods": [base, sr, cp, ltt]}
    for m in [base, sr, cp, ltt]:
        print(f"    {m['method']:38s} retFNR={m['retained_fnr']:.3f} cov={m['coverage']:.3f} targetsFNR={m['targets_fnr']}")

    # ---- harm on the SAME retained set (moderate MCAR, RF) ----
    Xc = degrade(X_cal0, cols, medians, "mcar", 0.35, 0.6)
    Xt = degrade(X_te0, cols, medians, "mcar", 0.35, 0.6)
    e = evaluate(clf, Xc, y_cal, Xt, y_te)
    p_te, retained, base_pred = e["p_te"], e["retained"], e["base_pred"]
    tiers = tiers_from(p_te)
    harm_full = harm_by_risk_tier(y_te, base_pred, tiers)
    harm_sel = harm_by_risk_tier(y_te[retained], base_pred[retained], tiers[retained])
    fn_full = int(((y_te == 1) & (base_pred == 0)).sum())
    fn_ret = int(((y_te[retained] == 1) & (base_pred[retained] == 0)).sum())
    results["harm_moderate_mcar"] = {
        "policy": "abstain iff |p-tau|<0.10; harm on retained only",
        "full_automation": {k: float(v) for k, v in harm_full.items()},
        "selective_retained": {k: float(v) for k, v in harm_sel.items()},
        "missed_pos_full": fn_full, "missed_pos_retained": fn_ret,
        "n_abstained": int((~retained).sum()),
        "deferred_pos": int((y_te[~retained] == 1).sum())}

    # ---- figure_data: persist EVERY array the figures need (seed-42 stable) ---
    # Separation of concerns: compute the curves HERE so generate_figures.py can
    # restyle/redraw without ever recomputing the science. Same moderate-MCAR
    # test fold (p_te, y_te) used by the harm block above.
    results["figure_data"] = build_figure_data(
        y_te=y_te, p_te=p_te, retained=retained,
        cp_ret=cp_ret, ltt_coverage=ltt_cov,
        ltt_op=(float(e["coverage"]), float(e["retained_fnr_test"])),
        ablation=abl, target_fnr=TARGET_FNR)

    payload = json.dumps({k: v for k, v in results.items()},
                         indent=2, default=lambda o: float(o))
    # Root mirror (what the determinism test reads via monkeypatched OUT) ...
    (OUT / "real_results.json").write_text(payload)
    # ... and the committed reference copy under results/ (derived from OUT so
    # the slow test stays isolated to its tmp dir).
    results_subdir = OUT / "results"
    results_subdir.mkdir(parents=True, exist_ok=True)
    (results_subdir / "real_results.json").write_text(payload)
    print("[OK] real_results.json (root mirror + results/)")

    # ---- figures: delegate to the importable plotting module (reads results/) -
    # run_eval stays the COMPUTE entry; the figure CODE lives in the module so it
    # can be re-run standalone. We pass start=OUT so it resolves the just-written
    # results/ (works under the monkeypatched tmp dir too).
    try:
        from scripts.generate_figures import generate_all
        generate_all(start=OUT, out_dirs=(OUT / "figures", OUT))
    except Exception as ex:                                   # pragma: no cover
        print("[WARN] figure step skipped:", repr(ex))


def _selective_fnr_by_coverage(y_true, p, coverage_grid, order="confidence"):
    """Selective FNR as a function of coverage for one ranking policy.

    Retain the top-`c` fraction of cases by `order`, predict (p>=0.5), and report
    the FNR among retained positives. `order`:
      * "confidence" -> rank by max(p, 1-p) desc  (softmax-response baseline)
      * "threshold"  -> accept cases with p>=tau, tau swept high->low (the
                        acceptance-threshold sweep behind the original AURC).
    Returns an FNR array aligned to coverage_grid (NaN where undefined).
    """
    y_true = np.asarray(y_true); p = np.asarray(p)
    n = len(p)
    out = np.full(len(coverage_grid), np.nan)
    if n == 0:
        return out
    if order == "confidence":
        rank = np.argsort(-np.maximum(p, 1 - p))      # most-confident first
        for i, c in enumerate(coverage_grid):
            k = int(round(c * n))
            if k <= 0:
                continue
            idx = rank[:k]
            out[i] = fnr_at(y_true[idx], p[idx], 0.5)
    else:  # acceptance-threshold sweep
        for i, c in enumerate(coverage_grid):
            k = int(round(c * n))
            if k <= 0:
                continue
            tau = np.sort(p)[::-1][min(k, n) - 1]      # threshold giving ~k accepted
            acc = p >= tau
            if acc.sum() == 0:
                continue
            out[i] = fnr_at(y_true[acc], p[acc], 0.5)
    return out


def build_figure_data(y_te, p_te, retained, cp_ret, ltt_coverage, ltt_op,
                      ablation, target_fnr, seed=SEED, n_boot=300):
    """Build & persist the arrays the two figures consume (deterministic).

    Risk-coverage panel arrays (all on one coverage grid):
      coverage_grid, ltt_curve (acceptance-threshold sweep == original AURC base),
      sr_curve (softmax-response baseline), boot_lo/boot_hi (95% bootstrap CI
      band around the SR/confidence-ranked curve), split_conformal operating
      point, ltt operating point, safe_fnr (the <=5% shaded ceiling).
    Ablation panel arrays: levels, base_fnr, retained_fnr, abstention.
    """
    y_te = np.asarray(y_te); p_te = np.asarray(p_te)
    cov_grid = np.round(np.linspace(0.05, 1.0, 20), 4)

    ltt_curve = _selective_fnr_by_coverage(y_te, p_te, cov_grid, order="threshold")
    sr_curve = _selective_fnr_by_coverage(y_te, p_te, cov_grid, order="confidence")

    # Deterministic bootstrap 95% CI around the confidence-ranked selective curve.
    rb = np.random.RandomState(seed)
    n = len(p_te)
    boot = np.full((n_boot, len(cov_grid)), np.nan)
    for b in range(n_boot):
        idx = rb.randint(0, n, n)
        boot[b] = _selective_fnr_by_coverage(y_te[idx], p_te[idx], cov_grid,
                                             order="confidence")
    boot_lo = np.nanpercentile(boot, 2.5, axis=0)
    boot_hi = np.nanpercentile(boot, 97.5, axis=0)

    # Split-conformal singleton-retention operating point.
    cp_cov = float(np.asarray(cp_ret).mean()) if len(cp_ret) else 0.0
    cp_fnr = (float(fnr_at(y_te[cp_ret], p_te[cp_ret], 0.5))
              if np.asarray(cp_ret).sum() else float("nan"))

    def _clean(a):
        return [None if (x is None or (isinstance(x, float) and np.isnan(x)))
                else float(x) for x in a]

    return {
        "moderate_mcar": {
            "coverage_grid": [float(c) for c in cov_grid],
            "ltt_curve": _clean(ltt_curve),
            "sr_curve": _clean(sr_curve),
            "sr_ci_lo": _clean(boot_lo),
            "sr_ci_hi": _clean(boot_hi),
            "ci_level": 0.95, "n_boot": int(n_boot),
            "ltt_op": [float(ltt_op[0]), float(ltt_op[1])],
            "split_conformal_op": [cp_cov, cp_fnr],
            "safe_fnr": float(target_fnr),
            "aurc": float(area_under_risk_coverage_curve(
                *coverage_risk_curve(y_te, p_te)[:2])),
        },
        "ablation": {
            "levels": [a["level"] for a in ablation],
            "base_fnr": [float(a["base_fnr"]) for a in ablation],
            "retained_fnr": [float(a["retained_fnr_test"]) for a in ablation],
            "abstention": [float(a["abstention"]) for a in ablation],
            "target_fnr": float(target_fnr),
        },
    }


if __name__ == "__main__":
    main()
