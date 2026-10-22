"""Addendum A4 (PROTOCOL.md): A1 design with the sample-split permutation gate, plus the A2 monitor
on the same evaluation half (gate + monitor combined).

Usage: python -I -W ignore run_shift_a4.py <features.csv.gz> <out_dir> [n_runs=50]
Writes runs.csv and summary.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import beta
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RL  # noqa: E402
from gate_split import gate_split  # noqa: E402
from monitor import first_alarm, stream  # noqa: E402
from run_realdata import NON_FEAT, model, split  # noqa: E402

ALPHA, DELTA, M, WMAX = 0.05, 0.10, 0.10, 20.0


def domain_scores(Xs, Xu, seed):
    """Cross-fitted P(target | x) for the stacked [Xs; Xu]."""
    X = np.vstack([Xs, Xu]); d = np.r_[np.zeros(len(Xs)), np.ones(len(Xu))]
    c = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, d):
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        c[te] = clf.fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    return c[:len(Xs)], c[len(Xs):]


def binom_cdf_real(k, n, a):
    """Pr[Bin(n, a) <= k] for real k, n (regularised incomplete beta)."""
    if k >= n:
        return 1.0
    return float(betainc(n - k, k + 1, 1 - a))


def tau_weighted(pc, yc, w, alpha, delta, m):
    best = None
    for t in RL.GRID:
        r = (np.abs(pc - t) >= m) & (yc == 1)
        if not r.any():
            break
        wr = w[r]; neff = wr.sum() ** 2 / (wr ** 2).sum()
        f = (wr * (pc[r] < t)).sum() / wr.sum()
        if binom_cdf_real(f * neff, neff, alpha) <= delta:
            best = t
        else:
            break
    return best


def main():
    df = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    df = df[df.included == 1]
    feats = [c for c in df.columns if c not in NON_FEAT]
    dropped = [c for c in feats if df.groupby("site")[c].apply(lambda v: v.notna().sum()).min() == 0]
    feats = [c for c in feats if c not in dropped]
    S = {s: (df[df.site == s][feats].to_numpy(float), df[df.site == s]["y"].to_numpy(int)) for s in "AB"}
    rows = []
    for run in range(n_runs):
        seed = 1000 + run
        sp = {s: split(S[s][1], seed) for s in "AB"}
        clf = {s: model("LR", seed).fit(S[s][0][sp[s][0]], S[s][1][sp[s][0]]) for s in "AB"}
        for s in "AB":
            for t in "AB":
                Xs, ys = S[s]; Xt, yt = S[t]
                ca, te = sp[s][1], sp[t][2]
                rng = np.random.default_rng(seed + (0 if s == t else 7))
                perm = rng.permutation(te); U, E = perm[: len(te) // 2], perm[len(te) // 2:]
                pc = clf[s].predict_proba(Xs[ca])[:, 1]; pe = clf[s].predict_proba(Xt[E])[:, 1]
                shift, pval, dauc = gate_split(Xs[ca], Xt[U], seed, DELTA)
                tau_sel = RL.tau_selcert(pc, ys[ca], ALPHA, DELTA, M)
                alarm = None if tau_sel is None else first_alarm(
                    stream(pe, yt[E], tau_sel, M, np.random.default_rng(9000 + run)), ALPHA, DELTA)
                taus = {"selective": tau_sel, "shift_gated": None if shift else tau_sel}
                for rule, tau in taus.items():
                    d = RL.band(pe, yt[E], tau, M)
                    r = dict(run=run, setting=f"{s}to{t}", rule=rule, refused=d is None,
                             domain_auc=dauc, shift_p=pval, shift=shift, alarm_at=alarm)
                    if d is not None:
                        r.update({k: v for k, v in d.items() if k != "ci95"})
                    rows.append(r)
        print("run", run, flush=True)
    R = pd.DataFrame(rows)
    R["viol"] = (~R.refused) & (R.retained_fnr > ALPHA)
    R.to_csv(out / "runs.csv", index=False)

    def ci(k, n):
        return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
                1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]
    summ = []
    for (st, ru), g in R.groupby(["setting", "rule"], sort=False):
        ok = g[~g.refused]; k = int(g.viol.sum())
        summ.append(dict(setting=st, rule=ru, runs=len(g), refusals=int(g.refused.sum()),
                         violations=k, viol_ci=ci(k, len(g)),
                         rfnr_mean=float(ok.retained_fnr.mean()) if len(ok) else None,
                         coverage_mean=float(ok.coverage.mean()) if len(ok) else None,
                         retained_fpr_mean=float(ok.retained_fpr.mean()) if len(ok) else None,
                         domain_auc_mean=float(g.domain_auc.mean()), shift_detected=int(g["shift"].sum()),
                         undetected_by_gate_and_monitor=int((g.viol & g.alarm_at.isna()).sum()),
                         monitor_false_alarms=int(((~g.refused) & ~g.viol & g.alarm_at.notna()).sum())))
    meta = dict(gate="A4 sample-split permutation (B=999)", alpha=ALPHA, delta=DELTA, margin=M,
                n_runs=n_runs, model="LR")
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, summary=summ), indent=1))
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
