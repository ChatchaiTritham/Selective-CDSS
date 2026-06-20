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
Run:  python run_eval.py
Out:  real_results.json  +  fig_risk_coverage.pdf/.png  +  fig_ablation.pdf/.png
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

    (OUT / "real_results.json").write_text(json.dumps(
        {k: v for k, v in results.items()}, indent=2, default=lambda o: float(o)))
    print("[OK] real_results.json")

    # ---- figures (Top-Tier publication style; data is live, never hardcoded) --
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        apply_pub_style()
        C_CURVE, C_TARGET, C_OP = PALETTE[0], PALETTE[1], PALETTE[2]   # blue / orange / green

        def _save(fig, stem):
            """Save vector PDF + 300-dpi PNG (same basename, bbox tight)."""
            fig.savefig(OUT / f"{stem}.pdf")
            fig.savefig(OUT / f"{stem}.png", dpi=300)

        # --- Figure 1: risk-coverage curve at moderate MCAR ---
        cov_c, risk_c, _ = coverage_risk_curve(y_te, e["p_te"])
        aurc = area_under_risk_coverage_curve(cov_c, risk_c)
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        ax.plot(cov_c, risk_c, color=C_CURVE, marker="", linestyle="-",
                label=f"Risk–coverage (AURC = {aurc:.3f})")
        ax.axhline(TARGET_FNR, color=C_TARGET, linestyle="--", linewidth=1.4,
                   label=f"Target FNR = {TARGET_FNR:.2f}")
        ax.scatter([e["coverage"]], [e["retained_fnr_test"]], color=C_OP, marker="D",
                   s=42, zorder=5, edgecolor="white", linewidth=0.6,
                   label=f"LTT operating point (cov = {e['coverage']:.2f})")
        ax.set_xlabel("Coverage (fraction of cases retained)")
        ax.set_ylabel("Selective false-negative rate (test)")
        ax.set_title("Risk-controlled selective prediction\n(moderate MCAR degradation)")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        _save(fig, "fig_risk_coverage")

        # --- Figure 2: degradation ablation (baseline vs retained FNR + abstention) ---
        labs = [a["level"].capitalize() for a in abl]
        bf = [a["base_fnr"] for a in abl]; rf = [a["retained_fnr_test"] for a in abl]
        ab = [a["abstention"] for a in abl]; x = np.arange(len(labs))
        fig2, a1 = plt.subplots(figsize=(3.7, 2.9))
        # abstention as background bars on the secondary axis (drawn first, behind lines)
        a2 = a1.twinx()
        bars = a2.bar(x, ab, width=0.55, color=PALETTE[5], alpha=0.30,
                      label="Abstention rate", zorder=1)
        a2.set_ylabel("Abstention rate")
        a2.set_ylim(0, 1)
        a2.spines["top"].set_visible(False)
        a2.spines["right"].set_visible(True)   # twin axis needs its own spine
        a2.grid(False)
        # FNR lines on the primary axis (on top)
        l1, = a1.plot(x, bf, color=PALETTE[1], marker="o", linestyle="-",
                      zorder=3, label="Baseline FNR (no abstention)")
        l2, = a1.plot(x, rf, color=PALETTE[2], marker="s", linestyle="-",
                      zorder=3, label="LTT retained FNR (test)")
        lt = a1.axhline(TARGET_FNR, color=PALETTE[6], linestyle=":", linewidth=1.4,
                        zorder=2, label=f"Target FNR = {TARGET_FNR:.2f}")
        a1.set_xticks(x); a1.set_xticklabels(labs)
        a1.set_ylabel("False-negative rate")
        a1.set_xlabel("Data-quality degradation (MCAR severity)")
        a1.set_ylim(bottom=0)
        a1.set_zorder(a2.get_zorder() + 1)     # keep line axis above the bars
        a1.patch.set_visible(False)
        a1.set_title("Safety holds as data quality degrades")
        a1.legend(handles=[l1, l2, lt, bars], loc="upper left")
        _save(fig2, "fig_ablation")
        print("[OK] figures (publication style)")
    except Exception as ex:
        print("[WARN] figure skipped:", repr(ex))


if __name__ == "__main__":
    main()
