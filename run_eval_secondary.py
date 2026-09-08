"""
ICT Express letter -- SECONDARY-DOMAIN evaluation (industrial-IoT sensor fault
triage), added to demonstrate that the risk-controlled selective-inference
wrapper generalizes beyond the clinical instance in run_eval.py.

Same LTT wrapper (basics_cdss.clinical_metrics.conformal_prediction, which is
domain-agnostic -- it only needs a sklearn-style model + a risk function), same
seed, same abstention policy, same MCAR degradation protocol as run_eval.py --
applied to a SEPARATE synthetic generative process that has nothing to do with
the clinical digital twins: multivariate machine telemetry (vibration,
bearing temperature, motor current, rotational speed) drifting toward a
bearing-fault regime. No basics_cdss disease models are used here.

Run:  python run_eval_secondary.py
Out:  real_results_secondary.json
"""
import json
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from basics_cdss.clinical_metrics.conformal_prediction import risk_control_conformal

SEED = 42
MARGIN = 0.10
TARGET_FNR = 0.05
rng = np.random.RandomState(SEED)
OUT = Path(__file__).parent

N_MACHINES = 600          # 300 normal-only run, 300 develop a bearing fault
T_STEPS = 24
COLS = ["vibration_rms", "bearing_temp_c", "motor_current_a", "rpm"]


def simulate_machine(faulty, rng):
    """AR(1) telemetry; a faulty unit drifts toward a bearing-degradation
    regime over the observation window (higher vibration/temp, lower rpm)."""
    state = {
        "vibration_rms": rng.normal(2.0, 0.4),
        "bearing_temp_c": rng.normal(55.0, 4.0),
        "motor_current_a": rng.normal(10.0, 1.2),
        "rpm": rng.normal(1450.0, 30.0),
    }
    drift = {
        "vibration_rms": 0.11 if faulty else 0.0,
        "bearing_temp_c": 0.55 if faulty else 0.0,
        "motor_current_a": 0.05 if faulty else 0.0,
        "rpm": -1.4 if faulty else 0.0,
    }
    noise_sd = {"vibration_rms": 0.15, "bearing_temp_c": 0.8,
                "motor_current_a": 0.3, "rpm": 8.0}
    traj = {k: [] for k in COLS}
    for _ in range(T_STEPS):
        for k in COLS:
            state[k] = 0.85 * state[k] + 0.15 * (state[k] + drift[k]) + rng.normal(0, noise_sd[k])
            traj[k].append(state[k])
    feat = {}
    for k in COLS:
        c = np.array(traj[k])
        feat[f"{k}__last"], feat[f"{k}__mean"] = c[-1], c.mean()
        feat[f"{k}__max"], feat[f"{k}__slope"] = c.max(), (c[-1] - c[0]) / T_STEPS
    return feat


def build_cohort():
    rows, y = [], []
    for i in range(N_MACHINES):
        faulty = i >= N_MACHINES // 2
        rows.append(simulate_machine(faulty, rng))
        y.append(int(faulty))
    cols = list(rows[0].keys())
    X = np.array([[r[c] for c in cols] for r in rows])
    return X, np.array(y), cols


def fnr_at(y_true, y_proba, tau):
    yp = (y_proba >= tau).astype(int)
    pos = (y_true == 1).sum()
    return 0.0 if pos == 0 else ((y_true == 1) & (yp == 0)).sum() / pos


def degrade_mcar(X, rate, sigma, seed=SEED):
    """Same MCAR protocol as run_eval.py: Gaussian noise + independent masking
    to the per-column median."""
    r = np.random.RandomState(seed)
    out = X + r.normal(0, sigma, size=X.shape) * X.std(axis=0, keepdims=True)
    medians = np.median(X, axis=0)
    mask = r.rand(*X.shape) < rate
    out[mask] = np.broadcast_to(medians, X.shape)[mask]
    return out


def policy(p, tau, margin=MARGIN):
    abstain = np.abs(p - tau) < margin
    return ~abstain, (p >= tau).astype(int)


def evaluate(clf, X_cal, y_cal, X_te, y_te):
    rc = risk_control_conformal(clf, X_cal, y_cal, X_te, fnr_at, target_risk=TARGET_FNR)
    tau = rc.threshold
    p_te = clf.predict_proba(X_te)[:, 1]
    base_fnr = fnr_at(y_te, p_te, 0.5)
    retained, _ = policy(p_te, tau)
    cov = float(retained.mean())
    ret_fnr = fnr_at(y_te[retained], p_te[retained], tau) if retained.sum() else 0.0
    return dict(tau=float(tau), base_fnr=float(base_fnr), coverage=cov,
                abstention=float(1 - cov), retained_fnr_test=float(ret_fnr),
                risk_controlled=bool(rc.risk_controlled))


def main():
    print("[*] secondary-domain cohort (industrial-IoT bearing-fault triage)...")
    X, y, cols = build_cohort()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
    X_cal0, X_te0, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    print(f"    cohort={len(y)} feat={X.shape[1]} prev={y.mean():.3f} "
          f"train={len(y_tr)} cal={len(y_cal)} test={len(y_te)}")

    results = {"domain": "industrial-IoT bearing-fault triage (synthetic telemetry)",
               "cohort": {"n": int(len(y)), "n_features": int(X.shape[1]),
                          "prevalence": float(y.mean()), "seed": SEED,
                          "margin": MARGIN, "target_fnr": TARGET_FNR},
               "split": {"train": int(len(y_tr)), "cal": int(len(y_cal)), "test": int(len(y_te))}}

    clean = evaluate(clf, X_cal0, y_cal, X_te0, y_te)
    results["clean"] = clean
    print(f"    clean: baseFNR={clean['base_fnr']:.3f} retFNR={clean['retained_fnr_test']:.3f} "
          f"cov={clean['coverage']:.3f} ctrl={clean['risk_controlled']}")

    abl = []
    for name, rate, sig in [("clean", 0.0, 0.0), ("mild", 0.20, 0.3),
                             ("moderate", 0.35, 0.6), ("severe", 0.50, 0.9)]:
        if rate == 0.0:
            Xc, Xt = X_cal0, X_te0
        else:
            Xc = degrade_mcar(X_cal0, rate, sig)
            Xt = degrade_mcar(X_te0, rate, sig)
        e = evaluate(clf, Xc, y_cal, Xt, y_te)
        abl.append({"level": name, "rate": rate, **e})
        print(f"    {name}: baseFNR={e['base_fnr']:.3f} retFNR_test={e['retained_fnr_test']:.3f} "
              f"abstain={e['abstention']:.3f} ctrl={e['risk_controlled']}")
    results["ablation_mcar"] = abl

    (OUT / "real_results_secondary.json").write_text(json.dumps(results, indent=2, default=float))
    print("[OK] real_results_secondary.json")


if __name__ == "__main__":
    main()
