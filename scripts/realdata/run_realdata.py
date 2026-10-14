"""Real-data study on PhysioNet/CinC 2019 (protocol: PROTOCOL.md).

Usage: python -I run_realdata.py <features.csv.gz> <out_dir> [n_runs=50]
Writes runs.csv (one row per run x setting x model x alpha x rule [x n_cal]),
summary.csv and summary.json.
"""
import json, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RL  # noqa: E402

DELTA, M = 0.10, 0.10
ALPHAS = (0.05, 0.10)
NCALS = (100, 250, 500, 1000, 2000, None)
NON_FEAT = {"pid", "site", "n_hours", "septic_in_window", "y", "included"}


def cp(k, n):
    return RL.cp(k, n)


def model(name, seed):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          class_weight="balanced", random_state=seed)


def split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=0.4, stratify=y, random_state=seed)
    ca, te = train_test_split(rest, train_size=1 / 3, stratify=y[rest], random_state=seed)
    return tr, ca, te


def main():
    df = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    df = df[df.included == 1]
    feats = [c for c in df.columns if c not in NON_FEAT]
    # drop value columns never observed at one of the sites (e.g. EtCO2 at site A);
    # their observed-indicator column is kept
    dropped = [c for c in feats if df.groupby("site")[c].apply(lambda v: v.notna().sum()).min() == 0]
    feats = [c for c in feats if c not in dropped]
    S = {s: (df[df.site == s][feats].to_numpy(float), df[df.site == s]["y"].to_numpy(int)) for s in "AB"}
    rows = []

    def rec(run, setting, mname, alpha, pc, yc, pt, yt, ncal="all"):
        res = RL.all_rules(pc, yc, pt, yt, alpha, DELTA, M)
        for rule, d in res.items():
            base = dict(run=run, setting=setting, model=mname, alpha=alpha, n_cal=ncal,
                        cal_n=len(yc), cal_pos=int(yc.sum()), rule=rule, refused=d is None)
            if d is not None:
                base.update({k: v for k, v in d.items() if k != "ci95"})
                base["ci_lo"], base["ci_hi"] = d["ci95"]
            rows.append(base)

    auc = []
    for run in range(n_runs):
        seed = 1000 + run
        sp = {s: split(S[s][1], seed) for s in "AB"}
        for mname in ("LR", "GB"):
            P = {}
            for s in "AB":
                X, y = S[s]; tr = sp[s][0]
                clf = model(mname, seed).fit(X[tr], y[tr])
                for t in "AB":
                    Xt = S[t][0]
                    P[(s, t)] = clf.predict_proba(Xt)[:, 1]
            for s in "AB":
                for t in "AB":
                    te = sp[t][2]
                    auc.append(dict(run=run, model=mname, train=s, test=t,
                                    auroc=roc_auc_score(S[t][1][te], P[(s, t)][te])))
            for alpha in ALPHAS:
                for s in "AB":
                    for t in "AB":
                        ys, yt_ = S[s][1], S[t][1]
                        ca_s, te_t, ca_t = sp[s][1], sp[t][2], sp[t][1]
                        p_st = P[(s, t)]
                        if s == t:
                            rec(run, f"within_{s}", mname, alpha, p_st[ca_s], ys[ca_s], p_st[te_t], yt_[te_t])
                        else:
                            rec(run, f"cross_{s}to{t}", mname, alpha, P[(s, s)][ca_s], ys[ca_s],
                                p_st[te_t], yt_[te_t])
                            rec(run, f"recal_{s}to{t}", mname, alpha, p_st[ca_t], yt_[ca_t],
                                p_st[te_t], yt_[te_t])
            # calibration-size sweep: site A, alpha = 0.05
            rng = np.random.default_rng(seed)
            ca, te = sp["A"][1], sp["A"][2]; yA = S["A"][1]; pA = P[("A", "A")]
            for nc in NCALS[:-1]:
                sub = rng.choice(ca, nc, replace=False)
                rec(run, "sweep_A", mname, 0.05, pA[sub], yA[sub], pA[te], yA[te], ncal=nc)
        print("run", run, flush=True)

    R = pd.DataFrame(rows)
    R["viol"] = (~R.refused) & (R.retained_fnr > R.alpha)
    R["viol_strict"] = (~R.refused) & (R.ci_lo > R.alpha)
    R.to_csv(out / "runs.csv", index=False)
    A = pd.DataFrame(auc); A.to_csv(out / "auroc_runs.csv", index=False)

    def ci(k, n):
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return [float(lo), float(hi)]

    summ = []
    for key, g in R.groupby(["setting", "model", "alpha", "n_cal", "rule"], sort=False):
        n = len(g); k = int(g.viol.sum()); nr = int(g.refused.sum()); ok = g[~g.refused]
        summ.append(dict(setting=key[0], model=key[1], alpha=key[2], n_cal=key[3], rule=key[4],
                         runs=n, refusals=nr, violations=k, viol_rate=k / n,
                         viol_ci_lo=ci(k, n)[0], viol_ci_hi=ci(k, n)[1],
                         viol_rate_nonrefused=(k / len(ok)) if len(ok) else None,
                         viol_strict=int(g.viol_strict.sum()),
                         cal_pos_mean=float(g.cal_pos.mean()),
                         rfnr_mean=float(ok.retained_fnr.mean()) if len(ok) else None,
                         rfnr_sd=float(ok.retained_fnr.std()) if len(ok) else None,
                         rfnr_q90=float(ok.retained_fnr.quantile(0.9)) if len(ok) else None,
                         coverage_mean=float(ok.coverage.mean()) if len(ok) else None,
                         coverage_sd=float(ok.coverage.std()) if len(ok) else None,
                         deferred_per_1000=float((1 - ok.coverage).mean() * 1000) if len(ok) else None,
                         retained_fpr_mean=float(ok.retained_fpr.mean()) if len(ok) else None,
                         retained_pos_test_mean=float(ok.retained_pos.mean()) if len(ok) else None))
    Sm = pd.DataFrame(summ); Sm.to_csv(out / "summary.csv", index=False)
    meta = dict(n_runs=n_runs, delta=DELTA, margin=M, alphas=ALPHAS,
                refusal_floor={str(a): RL.refusal_floor(a, DELTA) for a in ALPHAS},
                cohort={s: dict(n=int(len(S[s][1])), pos=int(S[s][1].sum()),
                                prev=float(S[s][1].mean())) for s in "AB"},
                n_features=len(feats), dropped_all_missing=dropped,
                auroc=A.groupby(["model", "train", "test"]).auroc.agg(["mean", "std"]).reset_index().to_dict("records"),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, sklearn=sklearn.__version__))
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, summary=summ), indent=1, default=str))


if __name__ == "__main__":
    main()
