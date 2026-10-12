"""ICT Express revision-5 analyses (third simulated-review round). Supersedes F1/F5 of
run_ictx_r4.py with a design whose runs are independent draws.

G1  Independent simulated pool, independent runs. The development cohort of the paper
    (seed-42 stream) fixes the severity cut. A pool of 9,000 simulated patients is drawn
    once from the same generator (same cut). Then, for each of 50 runs, a FRESH
    development cohort of 1,140 patients is simulated, split 570/285/285 (stratified),
    a forest is trained, masked entries are imputed with the TRAINING medians, and the
    calibration fold and the pool are degraded with run-specific seeds. Each rule's
    threshold is evaluated on the degraded pool, estimating its true retained FNR.
    Full and half calibration folds; MCAR moderate/severe, MAR, MNAR.
G2  Subgroups per run (sepsis/ARDS/ACS): distribution over runs of the selective rule's
    pool retained FNR, instead of pooling correlated runs.
G3  Condition-wise (Mondrian) selective calibration: one threshold per condition,
    calibrated on that condition's calibration patients only.
Run:  python run_ictx_r5.py <repo> <out.json>
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
SETTINGS = [("moderate", "mcar", .35, .6), ("severe", "mcar", .5, .9),
            ("mar", "mar", .35, .6), ("mnar", "mnar", .35, .6)]


def tau_emp(pc, yc):
    ok = [t for t in GRID if R.fnr_at(yc, pc, t) <= ALPHA]
    return max(ok) if ok else None


def tau_cert(pc, yc):
    pos = pc[yc == 1]; n = len(pos); best = None
    for t in GRID:
        if n > 0 and binom.cdf(int((pos < t).sum()), n, ALPHA) <= DELTA:
            best = t
        else:
            break
    return best


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


def evaluate(pt, yt, tau, m=M0):
    if tau is None:
        return None
    ret = np.abs(pt - tau) >= m; pred = pt >= tau
    rp = int((yt[ret] == 1).sum()); fn = int(((yt == 1) & ret & ~pred).sum())
    rn = int((yt[ret] == 0).sum()); fp = int(((yt == 0) & ret & pred).sum())
    return dict(rfnr=fn / rp if rp else 0.0, abst=float(1 - ret.mean()), fpr=fp / rn if rn else 0.0)


def gen(n_per):
    rows, sev, cond = [], [], []
    for ci, (_, (cls, spec, lat)) in enumerate(R.BASELINES.items()):
        m = cls()
        for _ in range(n_per):
            f, s = R.make_patient(m, spec, lat); rows.append(f); sev.append(s); cond.append(ci)
    df = pd.DataFrame(rows)
    return df, np.array(sev), np.array(cond)


# development cohort of the paper (fixes the severity cut and column order)
dev, sev0, _ = gen(R.N_PER)
CUT = float(np.quantile(sev0, 0.66)); cols = list(dev.columns)
# pool, drawn once from the continuing generator stream
pdf, psev, pcond = gen(N_POOL)
Xp = pdf[cols].apply(lambda c: c.fillna(c.median())).values; yp = (psev >= CUT).astype(int)
# fresh development cohorts, one per run
cohorts = []
for s in range(RUNS):
    df, sv, cd = gen(R.N_PER)
    cohorts.append((df[cols].apply(lambda c: c.fillna(c.median())).values, (sv >= CUT).astype(int), cd))
print("generated", flush=True)

out = {"alpha": ALPHA, "delta": DELTA, "margin": M0, "runs": RUNS, "cut": CUT,
       "pool": {"n": int(len(yp)), "positives": int(yp.sum()),
                "positives_by_condition": [int(yp[pcond == c].sum()) for c in range(3)]},
       "cohort_prevalence": [float(c[1].mean()) for c in cohorts]}
G1, G2, G3 = {}, {}, {}
for nm, mech, rate, sig in SETTINGS:
    rec = {f"{cal}/{k}": [] for cal in ["full", "half"] for k in ["empirical", "certified", "selective_certified"]}
    sub = [[] for _ in range(3)]
    mond = {"certified_all3": 0, "rfnr": [[] for _ in range(3)], "fpr": [], "abst": []}
    for s in range(RUNS):
        X, y, cd = cohorts[s]
        idx = np.arange(len(y))
        a, b = train_test_split(idx, test_size=0.5, random_state=s, stratify=y)
        c_, t_ = train_test_split(b, test_size=0.5, random_state=s, stratify=y[b])
        med = {c: float(np.median(X[a, j])) for j, c in enumerate(cols)}
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(X[a], y[a])
        qc = clf.predict_proba(R.degrade(X[c_], cols, med, mech, rate, sig, seed=s))[:, 1]
        qp = clf.predict_proba(R.degrade(Xp, cols, med, mech, rate, sig, seed=1000 + s))[:, 1]
        yc, cc = y[c_], cd[c_]
        half = train_test_split(np.arange(len(yc)), test_size=0.5, random_state=s, stratify=yc)[0]
        for cal, sel in [("full", np.arange(len(yc))), ("half", half)]:
            for k, fn in [("empirical", tau_emp), ("certified", tau_cert), ("selective_certified", tau_selcert)]:
                rec[f"{cal}/{k}"].append(evaluate(qp, yp, fn(qc[sel], yc[sel])))
        ts = tau_selcert(qc, yc)
        if ts is not None:
            for c in range(3):
                mk = pcond == c
                sub[c].append(evaluate(qp[mk], yp[mk], ts)["rfnr"])
        taus = [tau_selcert(qc[cc == c], yc[cc == c]) for c in range(3)]
        if all(t is not None for t in taus):
            mond["certified_all3"] += 1
            tp = np.array([taus[c] for c in pcond])
            ret = np.abs(qp - tp) >= M0; pred = qp >= tp
            for c in range(3):
                mk = (pcond == c) & ret & (yp == 1)
                mond["rfnr"][c].append(float((mk & ~pred).sum() / mk.sum()) if mk.sum() else 0.0)
            rn = ret & (yp == 0)
            mond["fpr"].append(float((rn & pred).sum() / rn.sum())); mond["abst"].append(float(1 - ret.mean()))
    summ = {}
    for key, v in rec.items():
        ok = [d for d in v if d is not None]; f = np.array([d["rfnr"] for d in ok])
        ex = int((f > ALPHA).sum())
        summ[key] = dict(n_certified=len(ok), exceed=ex,
                         p_binom_ge=float(binom.sf(ex - 1, len(ok), DELTA)) if ok else None,
                         rfnr_mean=float(f.mean()) if ok else None, rfnr_max=float(f.max()) if ok else None,
                         abst_mean=float(np.mean([d["abst"] for d in ok])) if ok else None,
                         fpr_mean=float(np.mean([d["fpr"] for d in ok])) if ok else None)
    G1[nm] = summ
    G2[nm] = [dict(condition=c, runs=len(sub[c]), mean=float(np.mean(sub[c])),
                   p2_5=float(np.percentile(sub[c], 2.5)), p97_5=float(np.percentile(sub[c], 97.5)),
                   runs_above=int((np.array(sub[c]) > ALPHA).sum())) for c in range(3)]
    G3[nm] = dict(certified_all3=mond["certified_all3"],
                  by_condition=[dict(condition=c, mean=float(np.mean(mond["rfnr"][c])) if mond["rfnr"][c] else None,
                                     runs_above=int((np.array(mond["rfnr"][c]) > ALPHA).sum()))
                                for c in range(3)],
                  fpr_mean=float(np.mean(mond["fpr"])) if mond["fpr"] else None,
                  abst_mean=float(np.mean(mond["abst"])) if mond["abst"] else None)
    print(nm, json.dumps({"G1": summ, "G2": G2[nm], "G3": G3[nm]}), flush=True)
out.update({"G1_independent_runs": G1, "G2_subgroups_per_run": G2, "G3_mondrian_selective": G3})
json.dump(out, open(sys.argv[2], "w"), indent=1)
