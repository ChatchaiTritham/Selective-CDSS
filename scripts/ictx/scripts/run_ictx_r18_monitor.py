"""ICT Express r18: anytime-valid retained-FNR monitor on the simulated testbed (protocol addendum A2).

Same cohorts, seeds, splits, pool split (U/E) and same_fill preprocessing as run_ictx_r16_shiftgate.py.
The ungated selective-certified threshold is deployed; labelled retained positives of E arrive in
random order and feed the mixture e-process of scripts/realdata/monitor.py (alarm at 1/delta).
Run:  python run_ictx_r18_monitor.py <repo> <out.json>
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
sys.path.insert(0, sys.argv[1] + '/scripts/realdata')
from monitor import first_alarm, stream
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHA, DELTA, M0 = R.TARGET_FNR, 0.10, R.MARGIN
GRID = np.linspace(0.01, 0.99, 99)
N_POOL, RUNS, N_U = 3000, 50, 285
SET = {"clean": ("mcar", 0, 0), "moderate": ("mcar", .35, .6), "severe": ("mcar", .5, .9),
       "mar": ("mar", .35, .6), "mnar": ("mnar", .35, .6)}
PAIRS = [("clean", "severe"), ("moderate", "severe"), ("moderate", "mnar"),
         ("moderate", "moderate"), ("severe", "severe"), ("mnar", "mnar")]


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


def rfnr(pt, yt, tau, m=M0):
    ret = (np.abs(pt - tau) >= m) & (yt == 1)
    return float((ret & (pt < tau)).sum() / ret.sum()) if ret.any() else 0.0


def gate(Xc, Xu, seed):
    X = np.vstack([Xc, Xu]); d = np.r_[np.zeros(len(Xc)), np.ones(len(Xu))]; c = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, d):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        c[te] = clf.fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    p = float(mannwhitneyu(c[len(Xc):], c[:len(Xc)], alternative="greater").pvalue)
    return p <= DELTA, p


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
Xp0 = pdf[cols].apply(lambda c: c.fillna(c.median())).values; yp = (psev >= CUT).astype(int)
cohorts = []
for s in range(RUNS):
    df, sv, cd = gen(R.N_PER)
    cohorts.append((df[cols].apply(lambda c: c.fillna(c.median())).values, (sv >= CUT).astype(int), cd,
                    df[cols].median()))
SAME = True
print("generated", flush=True)

res = {f"{a}->{b}": [] for a, b in PAIRS}
for s in range(RUNS):
    X, y, cd, fill = cohorts[s]
    Xp = pdf[cols].fillna(fill).values if SAME else Xp0
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.5, random_state=s, stratify=y)
    c_, t_ = train_test_split(b, test_size=0.5, random_state=s, stratify=y[b])
    med = {c: float(np.median(X[a, j])) for j, c in enumerate(cols)}
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(X[a], y[a])
    yc = y[c_]
    XC, XP = {}, {}
    for nm, (mech, rate, sig) in SET.items():
        XC[nm] = X[c_] if rate == 0 else R.degrade(X[c_], cols, med, mech, rate, sig, seed=s)
        XP[nm] = Xp if rate == 0 else R.degrade(Xp, cols, med, mech, rate, sig, seed=1000 + s)
    rng = np.random.default_rng(5000 + s)
    perm = rng.permutation(len(yp)); U, E = perm[:N_U], perm[N_U:]
    for ca, de in PAIRS:
        tau = tau_selcert(clf.predict_proba(XC[ca])[:, 1], yc)
        shifted, p = gate(XC[ca], XP[de][U], s)
        pe = clf.predict_proba(XP[de][E])[:, 1]
        f = None if tau is None else rfnr(pe, yp[E], tau)
        al = None if tau is None else first_alarm(stream(pe, yp[E], tau, M0, np.random.default_rng(9000 + s)), ALPHA, DELTA)
        res[f"{ca}->{de}"].append(dict(certified=tau is not None, shift=shifted, p=p, rfnr=f, alarm_at=al))
    if s % 10 == 9:
        print("run", s + 1, flush=True)

out = {"runs": RUNS, "pairs": {}}
for k, v in res.items():
    cert = [d for d in v if d["certified"]]
    ex = [d for d in cert if d["rfnr"] > ALPHA]; ok = [d for d in cert if d["rfnr"] <= ALPHA]
    ga = [d for d in ex if not d["shift"]]
    out["pairs"][k] = dict(certified=len(cert), exceed=len(ex),
                           alarms_exceed=sum(d["alarm_at"] is not None for d in ex),
                           median_to_alarm_exceed=float(np.median([d["alarm_at"] for d in ex if d["alarm_at"] is not None])) if any(d["alarm_at"] is not None for d in ex) else None,
                           alarms_ok=sum(d["alarm_at"] is not None for d in ok), ok=len(ok),
                           exceed_missed_by_gate=len(ga),
                           of_which_alarmed=sum(d["alarm_at"] is not None for d in ga))
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["pairs"], indent=1))
