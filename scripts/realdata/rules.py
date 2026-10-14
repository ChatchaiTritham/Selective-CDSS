"""Selective / risk-control rules for the real-data study.

tau_emp, tau_cert, tau_selcert, score, band, sgr_binary are the run_ictx_r4.py
implementations with alpha/delta/margin turned into parameters (r4 reads them from
module globals); tests/test_realdata_rules.py checks equality against the r4 source.
scrc_i and crc are added baselines (see PROTOCOL.md).
"""
import math
import numpy as np
from scipy.stats import binom

GRID = np.linspace(0.01, 0.99, 99)
CUTS = np.linspace(0.5, 0.99, 50)


def cp(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def fnr_at(y, p, t):
    pos = y == 1
    return float((p[pos] < t).mean()) if pos.any() else 0.0


def tau_emp(pc, yc, alpha):
    ok = [t for t in GRID if fnr_at(yc, pc, t) <= alpha]
    return max(ok) if ok else None


def tau_cert(pc, yc, alpha, delta):
    pos = pc[yc == 1]; n = len(pos); best = None
    for t in GRID:
        if binom.cdf(int((pos < t).sum()), n, alpha) <= delta:
            best = t
        else:
            break
    return best


def tau_selcert(pc, yc, alpha, delta, m):
    best = None
    for t in GRID:
        ret = (np.abs(pc - t) >= m) & (yc == 1)
        n = int(ret.sum()); k = int((ret & (pc < t)).sum())
        if n > 0 and binom.cdf(k, n, alpha) <= delta:
            best = t
        else:
            break
    return best


def refusal_floor(alpha, delta):
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def score(yt, ret, pred):
    rp = int((yt[ret] == 1).sum()); fn = int(((yt == 1) & ret & ~pred).sum())
    rn = int((yt[ret] == 0).sum()); fp = int(((yt == 0) & ret & pred).sum())
    ci = cp(fn, rp)
    return dict(coverage=float(ret.mean()), retained_pos=rp, retained_fn=fn,
                retained_fnr=fn / rp if rp else 0.0, ci95=ci,
                retained_fpr=fp / rn if rn else 0.0,
                deferred=int((~ret).sum()), deferred_pos=int(((yt == 1) & ~ret).sum()),
                n_test=int(len(yt)))


def band(pt, yt, tau, m):
    if tau is None:
        return None
    d = score(yt, np.abs(pt - tau) >= m, pt >= tau); d["tau"] = float(tau); return d


def no_band(pt, yt, tau):
    if tau is None:
        return None
    d = score(yt, np.ones(len(pt), bool), pt >= tau); d["tau"] = float(tau); return d


def _sgr_ok(pc, yc, c, lvl, alpha):
    r = (np.maximum(pc, 1 - pc) >= c) & (yc == 1)
    n = int(r.sum()); k = int((r & (pc < .5)).sum())
    return n > 0 and binom.cdf(k, n, alpha) <= lvl


def sgr_binary(pc, yc, alpha, delta):
    lvl = delta / math.ceil(math.log2(len(CUTS)))
    lo, hi, best = 0, len(CUTS) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _sgr_ok(pc, yc, CUTS[mid], lvl, alpha):
            best = CUTS[mid]; hi = mid - 1
        else:
            lo = mid + 1
    return best


def sgr_eval(pt, yt, cut):
    if cut is None:
        return None
    d = score(yt, np.maximum(pt, 1 - pt) >= cut, pt >= .5); d["cut"] = float(cut); return d


def scrc_i(pc, yc, alpha, delta, xi=0.8):
    """SCRC-I (Xu, Guo, Wei 2025, Prop. 4) on calibration positives (X|Y=1).
    Selection g=max(p,1-p) >= c; lambda_1 grid = CUTS, target selection rate xi;
    loss = 1{p < tau} (missed positive); returns (cut, tau) or None (refusal)."""
    p = pc[yc == 1]; n = len(p)
    if n == 0:
        return None
    g = np.maximum(p, 1 - p)
    # smallest lambda_1 (= highest cut) whose empirical selection rate reaches xi
    rates = [(g >= c).mean() for c in CUTS]
    feas = [i for i, r in enumerate(rates) if r >= xi]
    if not feas:
        return None
    i = max(feas); c = CUTS[i]
    xi_lcb = max(rates[i] - math.sqrt(math.log(4 / delta) / (2 * n)), 0.0)
    eps = math.sqrt(math.log(4 * len(CUTS) * len(GRID) / delta) / (2 * n))
    sel = g >= c
    ok = [t for t in GRID if (sel & (p < t)).mean() + eps <= alpha * xi_lcb]
    if xi_lcb <= 0 or not ok:
        return None
    return float(c), float(max(ok))


def scrc_eval(pt, yt, sol):
    if sol is None:
        return None
    c, t = sol
    d = score(yt, np.maximum(pt, 1 - pt) >= c, pt >= t); d["cut"] = c; d["tau"] = t; return d


def crc(pc, yc, alpha):
    p = pc[yc == 1]; n = len(p)
    ok = [t for t in GRID if (n * (p < t).mean() + 1) / (n + 1) <= alpha]
    return max(ok) if ok else None


def all_rules(pc, yc, pt, yt, alpha, delta, m):
    return {"ltt_selective": band(pt, yt, tau_selcert(pc, yc, alpha, delta, m), m),
            "ltt_overall": band(pt, yt, tau_cert(pc, yc, alpha, delta), m),
            "empirical": band(pt, yt, tau_emp(pc, yc, alpha), m),
            "sgr": sgr_eval(pt, yt, sgr_binary(pc, yc, alpha, delta)),
            "scrc_i": scrc_eval(pt, yt, scrc_i(pc, yc, alpha, delta)),
            "crc": no_band(pt, yt, crc(pc, yc, alpha))}
