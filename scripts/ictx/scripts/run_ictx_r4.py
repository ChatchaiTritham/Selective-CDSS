"""ICT Express revision-4 analyses (second simulated-review round). Reuses run_eval.py
of the Selective-CDSS repository unchanged (simulators, degradation operators) and the
rule implementations of run_ictx_r2.py.

F1  External-pool truth. A fresh pool of simulated patients (3,000 per condition,
    labelled with the severity cut of the development cohort) is degraded once per
    setting. For 50 re-splits of the development cohort (train/cal/test as in the
    paper) each rule's threshold is evaluated on the pool, which estimates the TRUE
    retained FNR of tau_hat, so P(rFNR(tau_hat) > alpha) is measured directly.
    Settings: MCAR moderate/severe, MAR, MNAR; also a half-size calibration fold.
F2  SGR as published (binary search over the confidence grid, delta/ceil(log2 K)) and a
    fixed-sequence SGR variant (most confident cut first), on the test fold and the pool.
F3  Softmax-response labelled at tau_hat (same coverage as each LTT rule) and FPR for
    softmax response and split conformal; delta sensitivity {0.05, 0.10, 0.20}.
F4  Calibration shift: tau calibrated on one degradation, deployed on another.
F5  Subgroups (sepsis / ARDS / ACS) of the pool, selective rule, pooled over re-splits.
F6  Runtime: warm-up, 1,000 repetitions, n_jobs=1, single query and batch.
Run:  python run_ictx_r4.py <repo> <out.json>
"""
import json, sys, time, platform, math
import numpy as np
import pandas as pd
from scipy.stats import binom
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ALPHA, DELTA, M0 = R.TARGET_FNR, 0.10, R.MARGIN
GRID = np.linspace(0.01, 0.99, 99)
N_POOL = 3000
SETTINGS = [("moderate", "mcar", .35, .6), ("severe", "mcar", .5, .9),
            ("mar", "mar", .35, .6), ("mnar", "mnar", .35, .6)]


def cp(k, n):
    return list(R.clopper_pearson(k, n))


def tau_emp(pc, yc):
    ok = [t for t in GRID if R.fnr_at(yc, pc, t) <= ALPHA]
    return max(ok) if ok else None


def tau_cert(pc, yc, delta=DELTA):
    pos = pc[yc == 1]; n = len(pos); best = None
    for t in GRID:
        if binom.cdf(int((pos < t).sum()), n, ALPHA) <= delta:
            best = t
        else:
            break
    return best


def tau_selcert(pc, yc, m=M0, delta=DELTA):
    best = None
    for t in GRID:
        ret = (np.abs(pc - t) >= m) & (yc == 1)
        n = int(ret.sum()); k = int((ret & (pc < t)).sum())
        if n > 0 and binom.cdf(k, n, ALPHA) <= delta:
            best = t
        else:
            break
    return best


def score(yt, ret, pred):
    rp = int((yt[ret] == 1).sum()); fn = int(((yt == 1) & ret & ~pred).sum())
    rn = int((yt[ret] == 0).sum()); fp = int(((yt == 0) & ret & pred).sum())
    return dict(coverage=float(ret.mean()), retained_pos=rp, retained_fn=fn,
                retained_fnr=fn / rp if rp else 0.0, ci95=cp(fn, rp),
                retained_fpr=fp / rn if rn else 0.0,
                deferred=int((~ret).sum()), deferred_pos=int(((yt == 1) & ~ret).sum()))


def band(pt, yt, tau, m=M0):
    if tau is None:
        return None
    d = score(yt, np.abs(pt - tau) >= m, pt >= tau); d["tau"] = float(tau); return d


def rules(pc, yc, pt, yt):
    return {"empirical": band(pt, yt, tau_emp(pc, yc)),
            "certified": band(pt, yt, tau_cert(pc, yc)),
            "selective_certified": band(pt, yt, tau_selcert(pc, yc))}


CUTS = np.linspace(0.5, 0.99, 50)


def _sgr_ok(pc, yc, c, lvl):
    r = (np.maximum(pc, 1 - pc) >= c) & (yc == 1)
    n = int(r.sum()); k = int((r & (pc < .5)).sum())
    return n > 0 and binom.cdf(k, n, ALPHA) <= lvl


def sgr_binary(pc, yc):
    """Geifman & El-Yaniv (2017): binary search over the sorted cut grid, each test at
    delta / ceil(log2 K); returns the lowest certified cut found."""
    lvl = DELTA / math.ceil(math.log2(len(CUTS)))
    lo, hi, best = 0, len(CUTS) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _sgr_ok(pc, yc, CUTS[mid], lvl):
            best = CUTS[mid]; hi = mid - 1
        else:
            lo = mid + 1
    return best


def sgr_fixed(pc, yc):
    """Fixed-sequence SGR: test cuts from most to least confident at level delta,
    stop at the first non-rejection; returns the lowest certified cut."""
    best = None
    for c in CUTS[::-1]:
        if _sgr_ok(pc, yc, c, DELTA):
            best = c
        else:
            break
    return best


def sgr_eval(pt, yt, cut):
    if cut is None:
        return None
    d = score(yt, np.maximum(pt, 1 - pt) >= cut, pt >= .5); d["cut"] = float(cut); return d


# ---- development cohort (identical to run_eval.build_cohort, but keeps condition and severity)
rows, sev, cond = [], [], []
for ci, (_, (cls, spec, lat)) in enumerate(R.BASELINES.items()):
    m = cls()
    for _ in range(R.N_PER):
        f, s = R.make_patient(m, spec, lat); rows.append(f); sev.append(s); cond.append(ci)
Xdf = pd.DataFrame(rows); sev = np.array(sev); cond = np.array(cond)
CUT = float(np.quantile(sev, 0.66))
y = (sev >= CUT).astype(int); cols = list(Xdf.columns)
X = Xdf[cols].apply(lambda c: c.fillna(c.median())).values
out = {"alpha": ALPHA, "delta": DELTA, "margin": M0, "n_dev": int(len(y)), "prev_dev": float(y.mean())}

# ---- external pool (continues the simulator's random stream; same severity cut)
prow, psev, pcond = [], [], []
for ci, (_, (cls, spec, lat)) in enumerate(R.BASELINES.items()):
    m = cls()
    for _ in range(N_POOL):
        f, s = R.make_patient(m, spec, lat); prow.append(f); psev.append(s); pcond.append(ci)
Pdf = pd.DataFrame(prow)[cols]; Xp = Pdf.apply(lambda c: c.fillna(c.median())).values
yp = (np.array(psev) >= CUT).astype(int); pcond = np.array(pcond)
out["pool"] = {"n": int(len(yp)), "positives": int(yp.sum()), "prevalence": float(yp.mean()),
               "positives_by_condition": [int(yp[pcond == c].sum()) for c in range(3)]}
print("pool", out["pool"], flush=True)

MED = {c: float(np.median(X[:, j])) for j, c in enumerate(cols)}   # full-cohort medians (pool + cal)
POOLDEG = {nm: R.degrade(Xp, cols, MED, mech, rate, sig, seed=777) for nm, mech, rate, sig in SETTINGS}
print("pool degraded", flush=True)

# ---- F1/F2/F5: 50 re-splits evaluated on the pool
F1, sub = {}, {nm: np.zeros((3, 2), int) for nm, *_ in SETTINGS}
for nm, mech, rate, sig in SETTINGS:
    runs = []
    for s in range(50):
        a, b_, ya, yb = train_test_split(X, y, test_size=0.5, random_state=s, stratify=y)
        ac, at, yac, yat = train_test_split(b_, yb, test_size=0.5, random_state=s, stratify=yb)
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(a, ya)
        qc = clf.predict_proba(R.degrade(ac, cols, MED, mech, rate, sig, seed=s))[:, 1]
        qp = clf.predict_proba(POOLDEG[nm])[:, 1]
        half = train_test_split(np.arange(len(yac)), test_size=0.5, random_state=s, stratify=yac)[0]
        r = {"full": rules(qc, yac, qp, yp), "half": rules(qc[half], yac[half], qp, yp),
             "sgr_binary": sgr_eval(qp, yp, sgr_binary(qc, yac)),
             "sgr_fixed": sgr_eval(qp, yp, sgr_fixed(qc, yac))}
        ts = tau_selcert(qc, yac)
        if ts is not None:
            ret = np.abs(qp - ts) >= M0
            for c in range(3):
                mk = ret & (yp == 1) & (pcond == c)
                sub[nm][c] += [int((mk & (qp < ts)).sum()), int(mk.sum())]
        runs.append(r)
    summ = {}
    for cal in ["full", "half"]:
        for key in ["empirical", "certified", "selective_certified"]:
            v = [r[cal][key] for r in runs if r[cal][key] is not None]
            f = np.array([d["retained_fnr"] for d in v]) if v else np.array([])
            summ[f"{cal}/{key}"] = dict(
                n_certified=len(v), true_exceed=int((f > ALPHA).sum()),
                true_rfnr_mean=float(f.mean()) if v else None, true_rfnr_max=float(f.max()) if v else None,
                abstention_mean=float(np.mean([1 - d["coverage"] for d in v])) if v else None,
                fpr_mean=float(np.mean([d["retained_fpr"] for d in v])) if v else None)
    for key in ["sgr_binary", "sgr_fixed"]:
        v = [r[key] for r in runs if r[key] is not None]
        f = np.array([d["retained_fnr"] for d in v]) if v else np.array([])
        summ[key] = dict(n_certified=len(v), true_exceed=int((f > ALPHA).sum()),
                         true_rfnr_mean=float(f.mean()) if v else None,
                         abstention_mean=float(np.mean([1 - d["coverage"] for d in v])) if v else None,
                         fpr_mean=float(np.mean([d["retained_fpr"] for d in v])) if v else None)
    F1[nm] = summ
    print(nm, json.dumps(summ), flush=True)
out["F1_pool_resplits50"] = F1
out["F5_subgroups_selective"] = {nm: [dict(condition=c, misses=int(v[c, 0]), retained_pos=int(v[c, 1]),
                                           rfnr=float(v[c, 0] / v[c, 1]) if v[c, 1] else None,
                                           ci95=cp(int(v[c, 0]), int(v[c, 1])))
                                      for c in range(3)] for nm, v in sub.items()}

# ---- seed-42 split (as in the paper) for F2-F4
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=R.SEED, stratify=y)
Xc0, Xt0, yc, yt = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=R.SEED, stratify=ytmp)
med = {c: float(np.median(Xtr[:, j])) for j, c in enumerate(cols)}
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=-1).fit(Xtr, ytr)
D = {nm: (clf.predict_proba(R.degrade(Xc0, cols, med, mech, rate, sig))[:, 1],
          clf.predict_proba(R.degrade(Xt0, cols, med, mech, rate, sig))[:, 1]) for nm, mech, rate, sig in SETTINGS}
D["mild"] = (clf.predict_proba(R.degrade(Xc0, cols, med, "mcar", .2, .3))[:, 1],
             clf.predict_proba(R.degrade(Xt0, cols, med, "mcar", .2, .3))[:, 1])
D["clean"] = (clf.predict_proba(Xc0)[:, 1], clf.predict_proba(Xt0)[:, 1])
pc, pt = D["moderate"]
F2 = {"sgr_binary": sgr_eval(pt, yt, sgr_binary(pc, yc)), "sgr_fixed": sgr_eval(pt, yt, sgr_fixed(pc, yc)),
      "sgr_binary_level": DELTA / math.ceil(math.log2(len(CUTS)))}
F3 = {}
for key, tf in [("empirical", tau_emp(pc, yc)), ("certified", tau_cert(pc, yc)), ("selective_certified", tau_selcert(pc, yc))]:
    cov = float((np.abs(pt - tf) >= M0).mean())
    conf = np.abs(pt - tf); k = int(round(cov * len(pt)))
    keep = np.zeros(len(pt), bool); keep[np.argsort(-conf)[:k]] = True
    F3[f"softmax_at_tau/{key}"] = score(yt, keep, pt >= tf)
    keep5 = np.zeros(len(pt), bool); keep5[np.argsort(-np.maximum(pt, 1 - pt))[:k]] = True
    F3[f"softmax_at_0.5/{key}"] = score(yt, keep5, pt >= .5)
# split conformal (LAC, alpha=0.05): score 1 - p_y on calibration; retain singletons
sc = np.where(yc == 1, 1 - pc, pc); n = len(sc)
q = np.sort(sc)[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)]
in1, in0 = (1 - pt) <= q, pt <= q
single = in1 ^ in0
F3["split_conformal"] = score(yt, single, in1 & ~in0)
F3["delta_sensitivity"] = {str(dl): {"certified": band(pt, yt, tau_cert(pc, yc, dl)),
                                     "selective_certified": band(pt, yt, tau_selcert(pc, yc, M0, dl))}
                           for dl in [0.05, 0.10, 0.20]}
F4 = {}
for cal, dep in [("clean", "mild"), ("clean", "moderate"), ("clean", "severe"),
                 ("moderate", "severe"), ("moderate", "mnar"), ("mnar", "moderate")]:
    tc = tau_emp(D[cal][0], yc); ts = tau_selcert(D[cal][0], yc)
    F4[f"{cal}->{dep}"] = {"empirical": band(D[dep][1], yt, tc), "selective_certified": band(D[dep][1], yt, ts)}
out.update({"F2_sgr_moderate": F2, "F3_baselines_moderate": F3, "F4_calibration_shift": F4})

# ---- F6 runtime
Xm = R.degrade(Xt0, cols, med, "mcar", .35, .6)
c1 = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=1).fit(Xtr, ytr)
tau = tau_selcert(pc, yc)
for i in range(20):
    c1.predict_proba(Xm[i:i + 1])
lat_rf, lat_gate = [], []
for i in range(1000):
    x = Xm[i % len(Xm):i % len(Xm) + 1]
    t0 = time.perf_counter(); p = c1.predict_proba(x)[0, 1]; t1 = time.perf_counter()
    _ = "abstain" if abs(p - tau) < M0 else int(p >= tau); t2 = time.perf_counter()
    lat_rf.append(t1 - t0); lat_gate.append(t2 - t1)
tb = []
for _ in range(20):
    t0 = time.perf_counter(); pb = c1.predict_proba(Xm)[:, 1]; tb.append(time.perf_counter() - t0)
tg = []
for _ in range(200):
    t0 = time.perf_counter(); _ = np.where(np.abs(pb - tau) < M0, -1, (pb >= tau).astype(int)); tg.append(time.perf_counter() - t0)
t0 = time.perf_counter(); [tau_selcert(pc, yc) for _ in range(200)]; cal_s = (time.perf_counter() - t0) / 200
out["F6_runtime"] = {"cpu": platform.processor(), "python": platform.python_version(), "sklearn": sklearn.__version__,
                     "n_jobs": 1, "reps": 1000, "warmup": 20,
                     "rf_single_ms_median": float(np.median(lat_rf) * 1e3),
                     "gate_single_us_median": float(np.median(lat_gate) * 1e6),
                     "rf_batch285_ms_median": float(np.median(tb) * 1e3),
                     "rf_per_query_in_batch_us": float(np.median(tb) / len(Xm) * 1e6),
                     "gate_batch285_us_median": float(np.median(tg) * 1e6),
                     "calibration_ms": cal_s * 1e3, "state_bytes": 16}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: out[k] for k in ["F2_sgr_moderate", "F6_runtime"]}, default=str))
