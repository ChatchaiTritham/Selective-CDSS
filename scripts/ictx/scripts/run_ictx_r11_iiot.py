"""ICT Express r11: selective-certified LTT rule on the industrial-IoT
bearing-fault domain of run_eval_secondary.py (which reports the empirical
rule only). Same generator, MCAR operator, band m=0.10, alpha=0.05,
delta=0.10, grid and fixed-sequence test as run_ictx_r2.py.

A) seed-42 split (300/150/150), four MCAR levels.
B) 50 runs: fresh 600-machine cohort per run (new split/forest/degradation
   seeds), thresholds evaluated on an independent pool of 9,000 machines
   (degraded anew per run) to estimate the true retained FNR.
Run (repo root):  python scripts/ictx/scripts/run_ictx_r11_iiot.py out.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binom
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import run_eval_secondary as S
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ALPHA, DELTA, M = 0.05, 0.10, 0.10
GRID = np.linspace(0.01, 0.99, 99)
LEVELS = [("clean", 0.0, 0.0), ("mild", 0.20, 0.3), ("moderate", 0.35, 0.6), ("severe", 0.50, 0.9)]


def fnr(y, p, t):
    pos = (y == 1).sum()
    return 0.0 if pos == 0 else float(((y == 1) & (p < t)).sum() / pos)


def tau_emp(pc, yc):
    ok = [t for t in GRID if fnr(yc, pc, t) <= ALPHA]
    return max(ok) if ok else None


def tau_cert(pc, yc):
    pos = pc[yc == 1]; best = None
    for t in GRID:
        if binom.cdf(int((pos < t).sum()), len(pos), ALPHA) <= DELTA: best = t
        else: break
    return best


def tau_sel(pc, yc):
    best = None
    for t in GRID:
        r = (np.abs(pc - t) >= M) & (yc == 1); n = int(r.sum()); k = int((r & (pc < t)).sum())
        if n > 0 and binom.cdf(k, n, ALPHA) <= DELTA: best = t
        else: break
    return best


def ev(pt, yt, t):
    if t is None: return None
    ret = np.abs(pt - t) >= M; rp = int((yt[ret] == 1).sum()); rn = int((yt[ret] == 0).sum())
    fn = int(((yt == 1) & ret & (pt < t)).sum()); fp = int(((yt == 0) & ret & (pt >= t)).sum())
    return dict(tau=float(t), abst=float(1 - ret.mean()), rpos=rp, rfn=fn,
                rfnr=fn / rp if rp else 0.0, rfpr=fp / rn if rn else 0.0)


RULES = {"empirical": tau_emp, "overall_cert": tau_cert, "selective_cert": tau_sel}


def cohort(n, rng):
    S.rng = rng; S.N_MACHINES = n
    return S.build_cohort()[:2]


def deg(X, rate, sig, seed):
    return X if rate == 0 else S.degrade_mcar(X, rate, sig, seed)


out = {"alpha": ALPHA, "delta": DELTA, "margin": M}
# A) seed-42 split, identical to run_eval_secondary.py
X, y = cohort(600, np.random.RandomState(42))
Xtr, Xtm, ytr, ytm = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
Xc0, Xt0, yc, yt = train_test_split(Xtm, ytm, test_size=0.5, random_state=42, stratify=ytm)
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1).fit(Xtr, ytr)
A = {}
for name, r, s in LEVELS:
    pc = clf.predict_proba(deg(Xc0, r, s, 42))[:, 1]; pt = clf.predict_proba(deg(Xt0, r, s, 42))[:, 1]
    A[name] = {"base_fnr": fnr(yt, pt, 0.5), "cal_pos": int(yc.sum()),
               **{k: ev(pt, yt, f(pc, yc)) for k, f in RULES.items()}}
out["seed42"] = A

# B) 50 runs, independent pool
Xp, yp = cohort(9000, np.random.RandomState(20261015))
B = {}
for name, r, s in LEVELS:
    rows = {k: [] for k in RULES}
    for run in range(50):
        sd = 1000 + run
        X, y = cohort(600, np.random.RandomState(sd))
        Xtr, Xtm, ytr, ytm = train_test_split(X, y, test_size=0.5, random_state=sd, stratify=y)
        Xc, _, yc, _ = train_test_split(Xtm, ytm, test_size=0.5, random_state=sd, stratify=ytm)
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=sd, n_jobs=-1).fit(Xtr, ytr)
        pc = clf.predict_proba(deg(Xc, r, s, sd))[:, 1]
        pp = clf.predict_proba(deg(Xp, r, s, sd + 5000))[:, 1]
        for k, f in RULES.items():
            rows[k].append(ev(pp, yp, f(pc, yc)))
    summ = {}
    for k, L in rows.items():
        c = [e for e in L if e]
        v = np.array([e["rfnr"] for e in c]) if c else np.array([])
        summ[k] = dict(certified=len(c), exceed=int((v > ALPHA).sum()),
                       mean_rfnr=float(v.mean()) if c else None, max_rfnr=float(v.max()) if c else None,
                       mean_abst=float(np.mean([e["abst"] for e in c])) if c else None,
                       mean_rfpr=float(np.mean([e["rfpr"] for e in c])) if c else None)
    B[name] = summ
    print(name, json.dumps(summ))
out["runs50_pool9000"] = B
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print(json.dumps(A, indent=1))
