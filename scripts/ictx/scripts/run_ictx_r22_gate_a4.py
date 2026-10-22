"""ICT Express r22: protocol addendum A4 on the simulated testbed. Identical to run_ictx_r18_monitor.py
except that the gate is the sample-split permutation test of scripts/realdata/gate_split.py.
(Original r18 docstring follows.) Anytime-valid retained-FNR monitor (protocol addendum A2).

Same cohorts, seeds, splits, pool split (U/E) and same_fill preprocessing as run_ictx_r16_shiftgate.py.
The ungated selective-certified threshold is deployed; labelled retained positives of E arrive in
random order and feed the mixture e-process of scripts/realdata/monitor.py (alarm at 1/delta).
Run:  python run_ictx_r22_gate_a4.py <repo> <out.json>   (resumable: <out>_cache.pkl, <out>_parts/)
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
sys.path.insert(0, sys.argv[1] + '/scripts/realdata')
from monitor import first_alarm, stream
from gate_split import gate_split
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


import os, pickle
OUT = os.path.splitext(sys.argv[2])[0]
CACHE, PARTS = OUT + "_cache.pkl", OUT + "_parts"
os.makedirs(PARTS, exist_ok=True)
if os.path.exists(CACHE):
    cols, CUT, pdf, psev, pcond, Xp0, yp, cohorts = pickle.load(open(CACHE, "rb"))
else:
    dev, sev0, _ = gen(R.N_PER)
    CUT = float(np.quantile(sev0, 0.66)); cols = list(dev.columns)
    pdf, psev, pcond = gen(N_POOL)
    Xp0 = pdf[cols].apply(lambda c: c.fillna(c.median())).values; yp = (psev >= CUT).astype(int)
    cohorts = []
    for s in range(RUNS):
        df, sv, cd = gen(R.N_PER)
        cohorts.append((df[cols].apply(lambda c: c.fillna(c.median())).values, (sv >= CUT).astype(int), cd,
                        df[cols].median()))
    pickle.dump((cols, CUT, pdf, psev, pcond, Xp0, yp, cohorts), open(CACHE + ".tmp", "wb"))
    os.replace(CACHE + ".tmp", CACHE)
SAME = True
print("generated", flush=True)

res = {f"{a}->{b}": [] for a, b in PAIRS}
for s in range(RUNS):
    part = os.path.join(PARTS, f"run_{s:02d}.json")
    if os.path.exists(part):
        for k, v in json.load(open(part)).items():
            res[k].append(v)
        continue
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
        shifted, p, _ = gate_split(XC[ca], XP[de][U], s, DELTA)
        pe = clf.predict_proba(XP[de][E])[:, 1]
        f = None if tau is None else rfnr(pe, yp[E], tau)
        al = None if tau is None else first_alarm(stream(pe, yp[E], tau, M0, np.random.default_rng(9000 + s)), ALPHA, DELTA)
        res[f"{ca}->{de}"].append(dict(certified=tau is not None, shift=shifted, p=p, rfnr=f, alarm_at=al))
    json.dump({k: v[-1] for k, v in res.items()}, open(part + ".tmp", "w"), default=float)
    os.replace(part + ".tmp", part)
    print("done run", s, flush=True)
    if s % 10 == 9:
        print("run", s + 1, flush=True)

out = {"runs": RUNS, "gate": "A4 sample-split permutation (B=999)", "pairs": {}}
for k, v in res.items():
    cert = [d for d in v if d["certified"]]
    ex = [d for d in cert if d["rfnr"] > ALPHA]; ok = [d for d in cert if d["rfnr"] <= ALPHA]
    gm = [d for d in ex if not d["shift"]]
    out["pairs"][k] = dict(certified=len(cert), shift_detected=int(sum(d["shift"] for d in v)),
                           exceed_ungated=len(ex), exceed_gated=len(gm),
                           gate_missed_alarmed=sum(d["alarm_at"] is not None for d in gm),
                           undetected_by_gate_and_monitor=sum(d["alarm_at"] is None for d in gm),
                           monitor_false_alarms=sum(d["alarm_at"] is not None for d in ok), within=len(ok))
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["pairs"], indent=1))
