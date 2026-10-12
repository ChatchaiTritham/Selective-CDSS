"""ICT Express revision-2 analyses (simulated-review response). Reuses run_eval.py
of the Selective-CDSS repository unchanged (cohort, seed 42, splits, degradations).

E1  Selective-certified LTT: certifies the RETAINED FNR directly. For a fixed band
    m and each tau (ascending), conditional on the n_r(tau) retained calibration
    positives the retained misses k_r(tau) ~ Bin(n_r, rFNR(tau)) under
    exchangeability, so p = P(Bin(n_r, alpha) <= k_r) is a valid p-value for
    H0: rFNR(tau) > alpha. Fixed-sequence testing (pre-set order) controls FWER
    at delta whatever the monotonicity, so P(rFNR(tau_hat) <= alpha) >= 1-delta.
E2  Extra baselines at moderate MCAR: SGR-style confidence threshold
    (Geifman & El-Yaniv 2017) with binomial bound and Bonferroni over the grid;
    class-conditional (Mondrian) conformal threshold (no abstention).
    False-positive rate of retained decisions is reported for every method.
E3  Band-width sensitivity m in {0, .05, .10, .15, .20}.
E4  50 independent re-splits at moderate and severe MCAR.
E5  Real public dataset: Wisconsin Diagnostic Breast Cancer (sklearn built-in),
    same rules under MCAR/MNAR degradation.
E6  Runtime overhead of calibration and of the online gate.
Run:  python run_ictx_r2.py <repo> <out.json>
"""
import json, sys, time, platform
import numpy as np
from scipy.stats import binom
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

ALPHA, DELTA, M0 = R.TARGET_FNR, 0.10, R.MARGIN
GRID = np.linspace(0.01, 0.99, 99)


def cp(k, n):
    return list(R.clopper_pearson(k, n))


def tau_emp(pc, yc):
    ok = [t for t in GRID if R.fnr_at(yc, pc, t) <= ALPHA]
    return max(ok) if ok else None


def tau_cert(pc, yc):
    pos = pc[yc == 1]; n = len(pos); best = None
    for t in GRID:
        if binom.cdf(int((pos < t).sum()), n, ALPHA) <= DELTA:
            best = t
        else:
            break
    return best


def tau_selcert(pc, yc, m):
    best = None
    for t in GRID:
        ret = (np.abs(pc - t) >= m) & (yc == 1)
        n = int(ret.sum()); k = int((ret & (pc < t)).sum())
        if n > 0 and binom.cdf(k, n, ALPHA) <= DELTA:
            best = t
        else:
            break
    return best


def score(pt, yt, ret, pred):
    rp = int((yt[ret] == 1).sum()); fn = int(((yt == 1) & ret & ~pred).sum())
    rn = int((yt[ret] == 0).sum()); fp = int(((yt == 0) & ret & pred).sum())
    return dict(coverage=float(ret.mean()), retained_pos=rp, retained_fn=fn,
                retained_fnr=fn / rp if rp else 0.0, ci95=cp(fn, rp),
                retained_neg=rn, retained_fp=fp, retained_fpr=fp / rn if rn else 0.0,
                deferred=int((~ret).sum()), deferred_pos=int(((yt == 1) & ~ret).sum()))


def band(pt, yt, tau, m):
    if tau is None:
        return None
    d = score(pt, yt, np.abs(pt - tau) >= m, pt >= tau); d["tau"] = float(tau); return d


def rules(pc, yc, pt, yt, m=M0):
    return {"empirical": band(pt, yt, tau_emp(pc, yc), m),
            "certified": band(pt, yt, tau_cert(pc, yc), m),
            "selective_certified": band(pt, yt, tau_selcert(pc, yc, m), m)}


def sgr(pc, yc, pt, yt):
    """SGR-style: confidence kappa=max(p,1-p), label at 0.5; pick the lowest
    confidence cut whose retained calibration FNR passes a binomial test at
    delta/|grid| (Bonferroni)."""
    kc, kt = np.maximum(pc, 1 - pc), np.maximum(pt, 1 - pt)
    cuts = np.linspace(0.5, 0.99, 50); best = None
    for c in cuts:
        r = (kc >= c) & (yc == 1); n = int(r.sum()); k = int((r & (pc < .5)).sum())
        if n > 0 and binom.cdf(k, n, ALPHA) <= DELTA / len(cuts):
            best = c; break
    if best is None:
        return None
    d = score(pt, yt, kt >= best, pt >= .5); d["cut"] = float(best); return d


def mondrian(pc, yc, pt, yt):
    """Class-conditional conformal: flag positive iff p >= q, q = floor((n+1)alpha)-th
    smallest calibration-positive score (P(FNR) <= alpha marginally). No abstention."""
    s = np.sort(pc[yc == 1]); k = int(np.floor((len(s) + 1) * ALPHA))
    q = s[k - 1] if k >= 1 else 0.0
    d = score(pt, yt, np.ones(len(pt), bool), pt >= q); d["tau"] = float(q); return d


out = {"alpha": ALPHA, "delta": DELTA, "margin": M0}
X, y, cols = R.build_cohort()
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=R.SEED, stratify=y)
Xc0, Xt0, yc, yt = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=R.SEED, stratify=ytmp)
med = {c: float(np.median(Xtr[:, j])) for j, c in enumerate(cols)}
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=-1).fit(Xtr, ytr)
LV = [("clean", 0, 0), ("mild", .2, .3), ("moderate", .35, .6), ("severe", .5, .9)]
P = {}
for name, rate, sig in LV:
    Xc = Xc0 if rate == 0 else R.degrade(Xc0, cols, med, "mcar", rate, sig)
    Xt = Xt0 if rate == 0 else R.degrade(Xt0, cols, med, "mcar", rate, sig)
    P[name] = (clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xt)[:, 1])
for mn in ["mar", "mnar"]:
    P[mn] = (clf.predict_proba(R.degrade(Xc0, cols, med, mn, .35, .6))[:, 1],
             clf.predict_proba(R.degrade(Xt0, cols, med, mn, .35, .6))[:, 1])
out["E1_rules"] = {k: rules(pc, yc, pt, yt) for k, (pc, pt) in P.items()}
pc, pt = P["moderate"]
out["E2_baselines_moderate"] = {
    "full_automation": score(pt, yt, np.ones(len(pt), bool), pt >= .5),
    "sgr_confidence": sgr(pc, yc, pt, yt), "mondrian_conformal": mondrian(pc, yc, pt, yt),
    **rules(pc, yc, pt, yt)}
out["E2_baselines_severe"] = {"sgr_confidence": sgr(*P["severe"][:1], yc, P["severe"][1], yt),
                              "mondrian_conformal": mondrian(P["severe"][0], yc, P["severe"][1], yt),
                              **rules(P["severe"][0], yc, P["severe"][1], yt)}
out["E3_band"] = {str(m): rules(pc, yc, pt, yt, m) for m in [0.0, .05, .10, .15, .20]}

# E4: 50 re-splits
E4 = {}
for lvl, rate, sig in [("moderate", .35, .6), ("severe", .5, .9)]:
    runs = []
    for s in range(50):
        a, b_, ya, yb = train_test_split(X, y, test_size=0.5, random_state=s, stratify=y)
        ac, at, yac, yat = train_test_split(b_, yb, test_size=0.5, random_state=s, stratify=yb)
        ms = {c: float(np.median(a[:, j])) for j, c in enumerate(cols)}
        cs = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(a, ya)
        qc = cs.predict_proba(R.degrade(ac, cols, ms, "mcar", rate, sig, seed=s))[:, 1]
        qt = cs.predict_proba(R.degrade(at, cols, ms, "mcar", rate, sig, seed=s))[:, 1]
        r = rules(qc, yac, qt, yat); r["sgr_confidence"] = sgr(qc, yac, qt, yat); runs.append(r)
    summ = {}
    for key in ["empirical", "certified", "selective_certified", "sgr_confidence"]:
        v = [r[key] for r in runs if r[key] is not None]
        if not v:
            summ[key] = dict(n_certified=0); continue
        fn = sum(d["retained_fn"] for d in v); n = sum(d["retained_pos"] for d in v)
        f = np.array([d["retained_fnr"] for d in v])
        summ[key] = dict(n_certified=len(v), runs_above_target=int((f > ALPHA).sum()),
                         fnr_mean=float(f.mean()), fnr_max=float(f.max()), pooled_fn=fn, pooled_pos=n,
                         pooled_ci95=cp(fn, n),
                         abstention_mean=float(np.mean([1 - d["coverage"] for d in v])),
                         fpr_mean=float(np.mean([d["retained_fpr"] for d in v])))
    E4[lvl] = summ
    print(lvl, json.dumps(summ, indent=0))
out["E4_resplits50"] = E4

# E5: real public data (WDBC); positive class = malignant
d = load_breast_cancer(); Xb = d.data; yb = (d.target == 0).astype(int)
btr, btmp, ybtr, ybtmp = train_test_split(Xb, yb, test_size=0.5, random_state=R.SEED, stratify=yb)
bc0, bt0, ybc, ybt = train_test_split(btmp, ybtmp, test_size=0.5, random_state=R.SEED, stratify=ybtmp)
bmed, bsd = np.median(btr, 0), btr.std(0)


def bdeg(Z, mech, rate, sig, seed):
    r = np.random.RandomState(seed)
    Zn = Z + r.normal(0, 1, Z.shape) * sig * bsd             # noise in feature-SD units
    if mech == "mcar":
        p = np.full(Z.shape, rate)
    else:                                                    # MNAR: high values vanish more
        p = np.clip(rate * (1 + 0.8 * np.sign(Zn - bmed)), 0, 1)
    msk = r.rand(*Z.shape) < p
    return np.where(msk, bmed, Zn)


bclf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=-1).fit(btr, ybtr)
E5 = {"n": int(len(yb)), "n_pos": int(yb.sum()), "split": [len(ybtr), len(ybc), len(ybt)],
      "test_pos": int(ybt.sum())}
for nm, mech, rate, sig in [("clean", "mcar", 0, 0), ("mcar_moderate", "mcar", .35, .6),
                            ("mcar_severe", "mcar", .5, .9), ("mnar_moderate", "mnar", .35, .6)]:
    qc = bclf.predict_proba(bdeg(bc0, mech, rate, sig, 1))[:, 1]
    qt = bclf.predict_proba(bdeg(bt0, mech, rate, sig, 2))[:, 1]
    E5[nm] = {"base": score(qt, ybt, np.ones(len(qt), bool), qt >= .5), **rules(qc, ybc, qt, ybt)}
out["E5_wdbc"] = E5

# E6: runtime (this machine)
pc, pt = P["moderate"]; Xm = R.degrade(Xt0, cols, med, "mcar", .35, .6)
t0 = time.perf_counter(); [tau_selcert(pc, yc, M0) for _ in range(200)]; cal = (time.perf_counter() - t0) / 200
tau = tau_selcert(pc, yc, M0)
lat_rf, lat_gate = [], []
for i in range(200):
    x = Xm[i % len(Xm):i % len(Xm) + 1]
    t0 = time.perf_counter(); p = clf.predict_proba(x)[0, 1]; t1 = time.perf_counter()
    _ = "abstain" if abs(p - tau) < M0 else int(p >= tau); t2 = time.perf_counter()
    lat_rf.append(t1 - t0); lat_gate.append(t2 - t1)
out["E6_runtime"] = {"machine": platform.processor(), "calibration_s": cal,
                     "rf_predict_median_ms": float(np.median(lat_rf) * 1e3),
                     "gate_median_us": float(np.median(lat_gate) * 1e6),
                     "state_bytes": 16}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k in ["E1_rules", "E2_baselines_moderate", "E2_baselines_severe", "E3_band", "E5_wdbc", "E6_runtime"]:
    print(k, json.dumps(out[k], default=str)[:3000])
