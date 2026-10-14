"""Patient-level features from the first 6 ICU hours (see PROTOCOL.md).

Usage: python -I build_features.py <data_dir> <out.csv.gz>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WIN = 6
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]


def one(path, site):
    d = pd.read_csv(path, sep="|")
    w = d[d["ICULOS"] <= WIN]
    after = d[d["ICULOS"] > WIN]
    r = {"pid": path.stem, "site": site, "n_hours": int(d["ICULOS"].max()),
         "septic_in_window": int(w["SepsisLabel"].max() == 1) if len(w) else 0,
         "y": int(after["SepsisLabel"].max() == 1) if len(after) else 0,
         "win_len": int(len(w))}
    for v in VITALS:
        s = w[v].dropna()
        r[v + "_last"] = s.iloc[-1] if len(s) else np.nan
        r[v + "_mean"] = s.mean() if len(s) else np.nan
        r[v + "_min"] = s.min() if len(s) else np.nan
        r[v + "_max"] = s.max() if len(s) else np.nan
        r[v + "_obs"] = int(len(s) > 0)
    for v in LABS:
        s = w[v].dropna()
        r[v + "_last"] = s.iloc[-1] if len(s) else np.nan
        r[v + "_obs"] = int(len(s) > 0)
    first = d.iloc[0]
    r["Age"] = first["Age"]; r["Gender"] = first["Gender"]; r["HospAdmTime"] = first["HospAdmTime"]
    u1, u2 = first["Unit1"], first["Unit2"]
    r["Unit1"] = 0 if pd.isna(u1) else u1; r["Unit2"] = 0 if pd.isna(u2) else u2
    r["Unit_missing"] = int(pd.isna(u1) and pd.isna(u2))
    return r


def main():
    root = Path(sys.argv[1]) / "training"
    rows = []
    for site in ("A", "B"):
        for f in sorted((root / f"training_set{site}").glob("*.psv")):
            rows.append(one(f, site))
    df = pd.DataFrame(rows)
    df["included"] = ((df["n_hours"] > WIN) & (df["septic_in_window"] == 0)).astype(int)
    df.to_csv(sys.argv[2], index=False)
    for s in ("A", "B"):
        a = df[df.site == s]; i = a[a.included == 1]
        print(s, "stays", len(a), "included", len(i), "positives", int(i.y.sum()),
              "prev %.4f" % i.y.mean())


if __name__ == "__main__":
    main()
