"""Sample-split classifier two-sample test with a permutation p-value (protocol addendum A4).

The domain classifier is fitted on one half of the pooled inputs and scored on the other half, so the
held-out scores are exchangeable under H0 (calibration and deployment inputs from one distribution);
permuting the domain labels among them gives a p-value that is valid in finite samples.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gate_split(Xc, Xu, seed, delta, B=999, impute=True):
    """Return (refuse, p_value, held-out AUROC)."""
    X = np.vstack([Xc, Xu]); d = np.r_[np.zeros(len(Xc)), np.ones(len(Xu))]
    a, b = train_test_split(np.arange(len(d)), test_size=0.5, stratify=d, random_state=seed)
    steps = ([SimpleImputer(strategy="median")] if impute else []) + [StandardScaler(), LogisticRegression(max_iter=2000)]
    c = make_pipeline(*steps).fit(X[a], d[a]).predict_proba(X[b])[:, 1]
    lab = d[b]; r = rankdata(c); n1 = int(lab.sum()); n0 = len(lab) - n1

    def auc(lb):
        return (r[lb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    obs = auc(lab)
    rng = np.random.default_rng(7000 + seed)
    perm = np.array([auc(rng.permutation(lab)) for _ in range(B)])
    p = (1 + int((perm >= obs).sum())) / (B + 1)
    return p <= delta, p, float(obs)
