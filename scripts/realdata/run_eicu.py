"""Addendum A3 (PROTOCOL.md): refusal floor, label-free gate and anytime-valid monitor on the
eICU Collaborative Research Database Demo v2.0.1 (open access, doi:10.13026/4mxk-na84).

Usage: python -I -W ignore run_eicu.py <eicu_demo_dir> <out_dir> [n_runs=50]
<eicu_demo_dir> holds patient.csv.gz, hospital.csv.gz, apacheApsVar.csv.gz.
Writes cohort.json, runs.csv and summary.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RL  # noqa: E402
from monitor import first_alarm, stream  # noqa: E402
from run_realdata import model  # noqa: E402
from run_shift import domain_scores  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ALPHAS, DELTA, M = (0.05, 0.10), 0.10, 0.10
SOURCE, TARGET = {"Midwest", "West"}, {"South", "Northeast"}
APS = ["intubated", "vent", "dialysis", "eyes", "motor", "verbal", "meds", "urine", "wbc", "temperature",
       "respiratoryrate", "sodium", "heartrate", "meanbp", "ph", "hematocrit", "creatinine", "albumin",
       "pao2", "pco2", "bun", "glucose", "bilirubin", "fio2"]


def cohort(d):
    p = pd.read_csv(d / "patient.csv.gz"); h = pd.read_csv(d / "hospital.csv.gz")
    a = pd.read_csv(d / "apacheApsVar.csv.gz")
    first = p.groupby("patienthealthsystemstayid").unitvisitnumber.transform("min") == p.unitvisitnumber
    p = p[first & p.hospitaldischargestatus.isin(["Alive", "Expired"])].merge(h[["hospitalid", "region"]], on="hospitalid")
    p = p[p.region.isin(SOURCE | TARGET)].merge(a[["patientunitstayid"] + APS], on="patientunitstayid", how="inner")
    X = p[APS].astype(float).replace(-1.0, np.nan)
    for c in ["wbc", "temperature", "respiratoryrate", "sodium", "heartrate", "meanbp", "ph", "hematocrit",
              "creatinine", "albumin", "pao2", "pco2", "bun", "glucose", "bilirubin", "fio2", "urine"]:
        X[c + "_obs"] = X[c].notna().astype(float)
    X["age"] = pd.to_numeric(p.age.replace("> 89", "90"), errors="coerce").values
    X["male"] = (p.gender == "Male").astype(float).values
    y = (p.hospitaldischargestatus == "Expired").astype(int).to_numpy()
    site = np.where(p.region.isin(SOURCE), "S", "T")
    return X.to_numpy(float), y, site, p


def gate(Xc, Xu, seed):
    cs, cu = domain_scores(Xc, Xu, seed)
    auc = float(np.mean(cu[:, None] > cs[None, :]) + 0.5 * np.mean(cu[:, None] == cs[None, :]))
    return float(mannwhitneyu(cu, cs, alternative="greater").pvalue) <= DELTA, auc


def main():
    d = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    X, y, site, p = cohort(d)
    S, T = site == "S", site == "T"
    meta = dict(n=int(len(y)), deaths=int(y.sum()), hospitals=int(p.hospitalid.nunique()),
                source=dict(n=int(S.sum()), deaths=int(y[S].sum()), hospitals=int(p[S].hospitalid.nunique())),
                target=dict(n=int(T.sum()), deaths=int(y[T].sum()), hospitals=int(p[T].hospitalid.nunique())),
                floors={str(a): RL.refusal_floor(a, DELTA) for a in ALPHAS})
    (out / "cohort.json").write_text(json.dumps(meta, indent=1)); print(meta, flush=True)
    iS, iT = np.nonzero(S)[0], np.nonzero(T)[0]
    rows = []
    for run in range(n_runs):
        seed = 1000 + run
        tr, rest = train_test_split(iS, train_size=0.4, stratify=y[iS], random_state=seed)
        ca, te = train_test_split(rest, train_size=1 / 3, stratify=y[rest], random_state=seed)
        clf = model("LR", seed).fit(X[tr], y[tr])
        rng = np.random.default_rng(seed)
        evals = {}
        for name, idx in (("S->S", te), ("S->T", iT)):
            perm = rng.permutation(idx); evals[name] = (perm[: len(idx) // 2], perm[len(idx) // 2:])
        pc = clf.predict_proba(X[ca])[:, 1]
        for name, (U, E) in evals.items():
            shifted, dauc = gate(X[ca], X[U], seed)
            pe = clf.predict_proba(X[E])[:, 1]
            for alpha in ALPHAS:
                taus = {"selective": RL.tau_selcert(pc, y[ca], alpha, DELTA, M),
                        "uncorrected": RL.tau_emp(pc, y[ca], alpha)}
                for rule, tau in taus.items():
                    r = dict(run=run, setting=name, alpha=alpha, rule=rule, cal_pos=int(y[ca].sum()),
                             refused=tau is None, gate_shift=shifted, domain_auc=dauc)
                    if tau is not None:
                        dd = RL.band(pe, y[E], tau, M)
                        x = stream(pe, y[E], tau, M, np.random.default_rng(9000 + run))
                        r.update(rfnr=dd["retained_fnr"], ret_pos=dd["retained_pos"], coverage=dd["coverage"],
                                 retained_fpr=dd["retained_fpr"], exceed=dd["retained_fnr"] > alpha,
                                 alarm_at=first_alarm(x, alpha, DELTA))
                    rows.append(r)
        print("run", run, flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "runs.csv", index=False)
    summ = []
    for (st, a, ru), g in R.groupby(["setting", "alpha", "rule"]):
        ok = g[~g.refused]
        ex = ok[ok.exceed == True]; wi = ok[ok.exceed == False]  # noqa: E712
        summ.append(dict(setting=st, alpha=a, rule=ru, runs=len(g), refused=int(g.refused.sum()),
                         cal_pos_mean=float(g.cal_pos.mean()),
                         exceed=int(len(ex)), exceed_alarmed=int(ex.alarm_at.notna().sum()),
                         within=int(len(wi)), within_alarmed=int(wi.alarm_at.notna().sum()),
                         rfnr_mean=float(ok.rfnr.mean()) if len(ok) else None,
                         ret_pos_mean=float(ok.ret_pos.mean()) if len(ok) else None,
                         gate_refusals=int(g.gate_shift.sum()), domain_auc_mean=float(g.domain_auc.mean()),
                         exceed_and_gate_missed=int((ex.gate_shift == False).sum())))  # noqa: E712
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, summary=summ), indent=1))
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
