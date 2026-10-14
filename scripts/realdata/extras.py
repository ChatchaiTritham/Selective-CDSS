"""Post-hoc descriptive extras (added after results; see PROTOCOL.md D2).
Degenerate certification = chosen tau <= margin, so no retained patient can be
predicted negative and the retained FNR is 0 by construction.
Usage: python -I extras.py <results_dir>
"""
import json, sys
from pathlib import Path
import pandas as pd
d = Path(sys.argv[1]); R = pd.read_csv(d / "runs.csv", dtype={"n_cal": str})
R = R[(R.rule == "ltt_selective") & (~R.refused)]
g = R.groupby(["setting", "model", "alpha", "n_cal"])
out = g[["tau", "coverage", "retained_pos"]].apply(lambda x: pd.Series(dict(n=len(x), degenerate=int((x.tau <= 0.10 + 1e-9).sum()),
                                      tau_median=float(x.tau.median()),
                                      deferred_per_1000=float((1 - x.coverage).mean() * 1000),
                                      retained_pos_test=float(x.retained_pos.mean())))).reset_index()
out.to_csv(d / "ltt_selective_extras.csv", index=False)
print(out.to_string(index=False))
