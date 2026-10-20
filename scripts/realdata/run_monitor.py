"""Addendum A2 (PROTOCOL.md): anytime-valid monitor on real ICU data.

Same runs, splits and evaluation halves as run_shift.py. The ungated selective-certified threshold is
deployed; labelled retained positives of the evaluation half arrive in random order.
Usage: python -I -W ignore run_monitor.py <features.csv.gz> <out_dir> [n_runs=50]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RL  # noqa: E402
from monitor import first_alarm, stream  # noqa: E402
from run_realdata import NON_FEAT, model, split  # noqa: E402

ALPHA, DELTA, M = 0.05, 0.10, 0.10


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
                perm = rng.permutation(te); E = perm[len(te) // 2:]   # same E as run_shift.py
                pc = clf[s].predict_proba(Xs[ca])[:, 1]; pe = clf[s].predict_proba(Xt[E])[:, 1]
                tau = RL.tau_selcert(pc, ys[ca], ALPHA, DELTA, M)
                if tau is None:
                    continue
                d = RL.band(pe, yt[E], tau, M)
                x = stream(pe, yt[E], tau, M, np.random.default_rng(9000 + run))
                rows.append(dict(run=run, setting=f"{s}to{t}", rfnr=d["retained_fnr"], n_ret_pos=len(x),
                                 exceed=d["retained_fnr"] > ALPHA, alarm_at=first_alarm(x, ALPHA, DELTA)))
        print("run", run, flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "runs.csv", index=False)
    summ = []
    for (st, ex), g in R.groupby(["setting", "exceed"]):
        a = g.alarm_at.dropna()
        summ.append(dict(setting=st, exceed=bool(ex), runs=len(g), alarms=int(len(a)),
                         median_to_alarm=float(a.median()) if len(a) else None,
                         median_ret_pos=float(g.n_ret_pos.median())))
    (out / "summary.json").write_text(json.dumps(dict(alpha=ALPHA, delta=DELTA, summary=summ), indent=1))
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
