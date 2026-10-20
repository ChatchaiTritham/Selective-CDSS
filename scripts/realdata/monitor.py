"""Anytime-valid monitor of the retained FNR (protocol addendum A2).

E_t = mean_lambda prod_{i<=t} (1 + lambda (X_i - alpha)), X_i = 1 if the i-th labelled retained
positive was missed. Under H0: rFNR <= alpha each product is a nonnegative supermartingale, so is the
mixture, and Ville's inequality gives Pr[sup_t E_t >= 1/delta] <= delta.
"""
import numpy as np

LAMBDAS = (1.0, 2.0, 5.0, 10.0, 15.0)


def first_alarm(x, alpha, delta, lambdas=LAMBDAS):
    """Index (1-based) of the first labelled retained positive at which E_t >= 1/delta, or None."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    logs = np.cumsum(np.log1p(np.outer(x - alpha, lambdas)), axis=0)
    e = np.exp(logs).mean(axis=1)
    hit = np.nonzero(e >= 1 / delta)[0]
    return int(hit[0]) + 1 if len(hit) else None


def stream(pt, yt, tau, m, rng):
    """Miss indicators of retained positives in random arrival order."""
    r = (np.abs(pt - tau) >= m) & (yt == 1)
    x = (pt[r] < tau).astype(float)
    return x[rng.permutation(len(x))]
