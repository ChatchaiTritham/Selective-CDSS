"""E6b: runtime with single-threaded RF inference (n_jobs=1), batch 1 and full test fold."""
import sys, time, json, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/src")
import run_eval as R
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X, y, cols = R.build_cohort()
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=R.SEED, stratify=y)
Xc0, Xt0, yc, yt = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=R.SEED, stratify=ytmp)
clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=R.SEED, n_jobs=1).fit(Xtr, ytr)
tau, m = 0.41, 0.10
one, gate = [], []
for i in range(300):
    x = Xt0[i % 285:i % 285 + 1]
    t0 = time.perf_counter(); p = clf.predict_proba(x)[0, 1]; t1 = time.perf_counter()
    d = "abstain" if abs(p - tau) < m else int(p >= tau); t2 = time.perf_counter()
    one.append(t1 - t0); gate.append(t2 - t1)
bt = []
for _ in range(20):
    t0 = time.perf_counter(); P = clf.predict_proba(Xt0)[:, 1]; t1 = time.perf_counter()
    g = np.abs(P - tau) < m; t2 = time.perf_counter(); bt.append((t1 - t0, t2 - t1))
bt = np.median(np.array(bt), 0)
out = {"rf_single_ms_median": float(np.median(one) * 1e3), "gate_single_us_median": float(np.median(gate) * 1e6),
       "rf_batch285_ms": float(bt[0] * 1e3), "gate_batch285_us": float(bt[1] * 1e6)}
print(out); json.dump(out, open(sys.argv[2], "w"), indent=1)
