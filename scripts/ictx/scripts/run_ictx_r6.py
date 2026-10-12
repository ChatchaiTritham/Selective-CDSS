"""ICT Express revision-6 analyses (fourth simulated-review round). Re-creates the exact
independent-run design of run_ictx_r5.py (same generator stream, seeds and splits; the
selective/empirical exceedance counts are re-checked against results/r5.json) and adds:

H1  Per-run workload on the pool: deferrals, false alarms and human touches per 100
    queries; system FNR when a reviewer misses a fraction h of deferred positives.
H2  Calibration shift inside the 50-run loop: threshold calibrated on one degradation of
    the calibration fold, evaluated on the pool under another.
H3  Per-run true retained FNR (empirical and selective, full and half calibration fold)
    and per-run ACS retained FNR, saved for the figure.
Run:  python run_ictx_r6.py <repo> <out.json> [<r5.json>]
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import binom
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ALPHA, DELTA, M0 = R.TARGET_FNR, 0.10, R.MARGIN
GRID = np.linspace(0.01, 0.99, 99)
N_POOL, RUNS = 3000, 50
SET = {"clean": ("mcar", 0, 0), "moderate": ("mcar", .35, .6), "severe": ("mcar", .5, .9),
       "mar": ("mar", .35, .6), "mnar": ("mnar", .35, .6)}
MAIN = ["moderate", "severe", "mar", "mnar"]
SHIFT = [("clean", "severe"), ("moderate", "severe"), ("moderate", "mnar")]
H = [0.0, 0.1, 0.2]


def tau_emp(pc, yc):
    ok = [t for t in GRID if R.fnr_at(yc, pc, t) <= ALPHA]
    return max(ok) if ok else None


def tau_selcert(pc, yc, m=M0):
    best = None
    for t in GRID:
        ret = (np.abs(pc - t) >= m) & (yc == 1)
        n = int(ret.sum()); k = int((ret & (pc < t)).sum())
        if n > 0 and binom.cdf(k, n, ALPHA) <= DELTA:
            best = t
        else:
            break
    return best


def ev(pt, yt, tau, m=M0):
    if tau is None:
        return None
    ret = np.abs(pt - tau) >= m; pred = pt >= tau; n = len(yt); pos = int((yt == 1).sum())
    rp = int((yt[ret] == 1).sum()); fn = int(((yt == 1) & ret & ~pred).sum())
    rn = int((yt[ret] == 0).sum()); fp = int(((yt == 0) & ret & pred).sum())
    dpos = int(((yt == 1) & ~ret).sum())
    return dict(rfnr=fn / rp if rp else 0.0, fpr=fp / rn if rn else 0.0,
                defer_per100=100 * (~ret).sum() / n, alarms_per100=100 * fp / n,
                touches_per100=100 * ((~ret).sum() + fp) / n,
                sysfnr={str(h): (fn + h * dpos) / pos for h in H})


def gen(n_per):
    rows, sev, cond = [], [], []
    for ci, (_, (cls, spec, lat)) in enumerate(R.BASELINES.items()):
        m = cls()
        for _ in range(n_per):
            f, s = R.make_patient(m, spec, lat); rows.append(f); sev.append(s); cond.append(ci)
    return pd.DataFrame(rows), np.array(sev), np.array(cond)


dev, sev0, _ = gen(R.N_PER)
CUT = float(np.quantile(sev0, 0.66)); cols = list(dev.columns)
pdf, psev, pcond = gen(N_POOL)
Xp = pdf[cols].apply(lambda c: c.fillna(c.median())).values; yp = (psev >= CUT).astype(int)
cohorts = []
for s in range(RUNS):
    df, sv, cd = gen(R.N_PER)
    cohorts.append((df[cols].apply(lambda c: c.fillna(c.median())).values, (sv >= CUT).astype(int), cd))
print("generated", flush=True)

rec = {nm: {k: [] for k in ["full/empirical", "full/selective_certified", "half/empirical", "half/selective_certified"]} for nm in MAIN}
acs = {nm: [] for nm in MAIN}
shift = {f"{a}->{b}": {"empirical": [], "selective_certified": []} for a, b in SHIFT}
for s in range(RUNS):
    X, y, cd = cohorts[s]
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.5, random_state=s, stratify=y)
    c_, t_ = train_test_split(b, test_size=0.5, random_state=s, stratify=y[b])
    med = {c: float(np.median(X[a, j])) for j, c in enumerate(cols)}
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(X[a], y[a])
    yc = y[c_]
    half = train_test_split(np.arange(len(yc)), test_size=0.5, random_state=s, stratify=yc)[0]
    QC, QP = {}, {}
    for nm in ["clean"] + MAIN:
        mech, rate, sig = SET[nm]
        QC[nm] = clf.predict_proba(X[c_] if rate == 0 else R.degrade(X[c_], cols, med, mech, rate, sig, seed=s))[:, 1]
        QP[nm] = clf.predict_proba(Xp if rate == 0 else R.degrade(Xp, cols, med, mech, rate, sig, seed=1000 + s))[:, 1]
    for nm in MAIN:
        qc, qp = QC[nm], QP[nm]
        for cal, sel in [("full", np.arange(len(yc))), ("half", half)]:
            rec[nm][f"{cal}/empirical"].append(ev(qp, yp, tau_emp(qc[sel], yc[sel])))
            rec[nm][f"{cal}/selective_certified"].append(ev(qp, yp, tau_selcert(qc[sel], yc[sel])))
        ts = tau_selcert(qc, yc)
        if ts is not None:
            mk = pcond == 2
            acs[nm].append(ev(qp[mk], yp[mk], ts)["rfnr"])
    for ca, de in SHIFT:
        shift[f"{ca}->{de}"]["empirical"].append(ev(QP[de], yp, tau_emp(QC[ca], yc)))
        shift[f"{ca}->{de}"]["selective_certified"].append(ev(QP[de], yp, tau_selcert(QC[ca], yc)))
    if s % 10 == 9:
        print("run", s + 1, flush=True)


def summ(v):
    ok = [d for d in v if d is not None]
    f = np.array([d["rfnr"] for d in ok])
    return dict(n_certified=len(ok), exceed=int((f > ALPHA).sum()), rfnr_mean=float(f.mean()), rfnr_max=float(f.max()),
                fpr_mean=float(np.mean([d["fpr"] for d in ok])),
                defer_per100=float(np.mean([d["defer_per100"] for d in ok])),
                alarms_per100=float(np.mean([d["alarms_per100"] for d in ok])),
                touches_per100=float(np.mean([d["touches_per100"] for d in ok])),
                sysfnr_mean={h: float(np.mean([d["sysfnr"][h] for d in ok])) for h in map(str, H)})


out = {"runs": RUNS, "H1_workload": {nm: {k: summ(v) for k, v in r.items()} for nm, r in rec.items()},
       "H2_shift": {k: {r: summ(v) for r, v in d.items()} for k, d in shift.items()},
       "H3_per_run": {nm: {k: [None if d is None else d["rfnr"] for d in v] for k, v in r.items()} for nm, r in rec.items()},
       "H3_acs_per_run": acs}
allx = sum(out["H1_workload"][nm]["full/selective_certified"]["exceed"] for nm in MAIN)
out["pooled_selective_full"] = {"exceed": allx, "runs": 200, "ci95": list(R.clopper_pearson(allx, 200))}
if len(sys.argv) > 3:
    r5 = json.load(open(sys.argv[3]))["G1_independent_runs"]
    out["matches_r5"] = all(out["H1_workload"][nm][k]["exceed"] == r5[nm][k]["exceed"] for nm in MAIN for k in rec[nm])
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"matches_r5": out.get("matches_r5"), "pooled": out["pooled_selective_full"]}))
