"""ICT Express add-on analyses (reviewer-hardening). Reuses run_eval.py of the
Selective-CDSS repository unchanged (same cohort, seed 42, splits, degradations).

Adds, for every reported condition:
  * retained FN / retained positives and exact Clopper-Pearson 95% CI;
  * a CERTIFIED Learn-then-Test variant: binomial-tail p-value for
    H0: FNR(tau) > alpha on calibration positives, fixed-sequence testing from
    the strictest threshold upward (FWER <= delta), delta = 0.10. The same
    abstention band m is then applied and scored on the test fold.
Run:  PYTHONPATH=<repo>/src:<repo> python run_ictx_addons.py <repo> <out.json>
"""
import json, sys
import numpy as np
from scipy.stats import binom
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ALPHA, DELTA, M = R.TARGET_FNR, 0.10, R.MARGIN
GRID = np.linspace(0.01, 0.99, 99)


def ltt_certified(p_cal, y_cal):
    """Fixed-sequence LTT: test tau ascending (FNR non-decreasing in tau);
    stop at first non-rejection. Returns largest certified tau or None."""
    pos = p_cal[y_cal == 1]; n = len(pos); best = None
    for t in GRID:
        k = int((pos < t).sum())
        pval = binom.cdf(k, n, ALPHA)          # P(Bin(n,alpha) <= k) valid under H0: FNR>=alpha
        if pval <= DELTA:
            best = t
        else:
            break
    return best


def score(p_te, y_te, tau):
    ret = np.abs(p_te - tau) >= M
    pred = p_te >= tau
    rp = int((y_te[ret] == 1).sum()); fn = int(((y_te == 1) & ret & ~pred).sum())
    lo, hi = R.clopper_pearson(fn, rp)
    return dict(tau=float(tau), coverage=float(ret.mean()), retained_pos=rp, retained_fn=fn,
                retained_fnr=fn / rp if rp else 0.0, ci95=[lo, hi])


def cond(clf, Xc, yc, Xt, yt):
    pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xt)[:, 1]
    emp = [t for t in GRID if R.fnr_at(yc, pc, t) <= ALPHA]
    out = {"n_test_pos": int((yt == 1).sum()),
           "base_fn": int(((yt == 1) & (pt < 0.5)).sum()),
           "empirical": score(pt, yt, max(emp)) if emp else None}
    tc = ltt_certified(pc, yc)
    out["certified"] = score(pt, yt, tc) if tc is not None else None
    return out


X, y, cols = R.build_cohort()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=R.SEED, stratify=y)
X_c0, X_t0, y_c, y_t = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=R.SEED, stratify=y_tmp)
med = {c: float(np.median(X_tr[:, j])) for j, c in enumerate(cols)}
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=-1).fit(X_tr, y_tr)
res = {"alpha": ALPHA, "delta": DELTA, "margin": M, "mcar": {}, "mech": {}}
for name, rate, sig in [("clean", 0, 0), ("mild", .2, .3), ("moderate", .35, .6), ("severe", .5, .9)]:
    Xc = X_c0 if rate == 0 else R.degrade(X_c0, cols, med, "mcar", rate, sig)
    Xt = X_t0 if rate == 0 else R.degrade(X_t0, cols, med, "mcar", rate, sig)
    res["mcar"][name] = cond(clf, Xc, y_c, Xt, y_t)
for mn in ["mcar", "mar", "mnar"]:
    res["mech"][mn] = cond(clf, R.degrade(X_c0, cols, med, mn, .35, .6), y_c,
                           R.degrade(X_t0, cols, med, mn, .35, .6), y_t)
# certified variant over the 10 repeated splits (moderate MCAR)
reps = []
for s in range(10):
    a_tr, a_tmp, b_tr, b_tmp = train_test_split(X, y, test_size=0.5, random_state=s, stratify=y)
    a_c, a_t, b_c, b_t = train_test_split(a_tmp, b_tmp, test_size=0.5, random_state=s, stratify=b_tmp)
    ms = {c: float(np.median(a_tr[:, j])) for j, c in enumerate(cols)}
    cs = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(a_tr, b_tr)
    reps.append(cond(cs, R.degrade(a_c, cols, ms, "mcar", .35, .6, seed=s), b_c,
                     R.degrade(a_t, cols, ms, "mcar", .35, .6, seed=s), b_t))
res["repeated_splits_moderate"] = reps
json.dump(res, open(sys.argv[2], "w"), indent=1)
for k, v in list(res["mcar"].items()) + list(res["mech"].items()):
    e, c = v["empirical"], v["certified"]
    print(k, "EMP", e and (e["retained_fn"], e["retained_pos"], round(e["coverage"], 3), [round(x, 3) for x in e["ci95"]]),
          "CERT", c and (round(c["tau"], 2), c["retained_fn"], c["retained_pos"], round(c["coverage"], 3), [round(x, 3) for x in c["ci95"]]))
for r in reps:
    c = r["certified"]; e = r["empirical"]
    print("rep EMP", e["retained_fn"], e["retained_pos"], "CERT", c and (c["retained_fn"], c["retained_pos"], round(c["coverage"], 3)))
