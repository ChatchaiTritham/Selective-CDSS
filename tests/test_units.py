"""Focused unit tests for deterministic pure functions in basics_cdss.

These target hand-computable metric/gating logic (calibration, coverage-risk,
harm-aware costs, confusion-matrix performance) on tiny inputs so the expected
values can be verified by hand. No training, network, or large data involved.

Repo uses a src/ layout; pytest.ini sets ``pythonpath = . src`` but we also
insert the path defensively so the file is runnable standalone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from basics_cdss.metrics import calibration as cal
from basics_cdss.metrics import coverage_risk as cr
from basics_cdss.metrics import harm as hm
from basics_cdss.metrics import performance as perf


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_brier_score_exact():
    # BS = mean((p - y)^2) = (0.01+0.04+0.09+0.09+0.04)/5 = 0.054
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    assert cal.brier_score(y_true, y_prob) == pytest.approx(0.054, abs=1e-12)


def test_brier_score_perfect_is_zero():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert cal.brier_score(y_true, y_prob) == pytest.approx(0.0, abs=1e-12)


def test_ece_bounds_and_empty():
    # ECE must lie in [0, 1]; empty input returns the EMPTY_METRIC_VALUE (0.0).
    rng = np.random.default_rng(42)
    y_prob = rng.random(200)
    y_true = (y_prob > 0.5).astype(int)
    ece = cal.expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert cal.expected_calibration_error(np.array([]), np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# coverage / risk selective-prediction logic
# --------------------------------------------------------------------------- #
def test_abstention_rate_threshold_gate():
    # 2 of 5 probs below 0.5 -> abstention rate 0.4.
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    assert cr.abstention_rate(y_prob, threshold=0.5) == pytest.approx(0.4)
    # Threshold 0 accepts everyone -> abstain 0; threshold >1 abstains all.
    assert cr.abstention_rate(y_prob, threshold=0.0) == pytest.approx(0.0)
    assert cr.abstention_rate(y_prob, threshold=1.01) == pytest.approx(1.0)


def test_coverage_risk_curve_monotone_coverage():
    # Coverage = fraction with prob >= tau; as tau rises over a sweep that
    # starts at 0 and ends at 1, coverage is non-increasing and bounded [0,1].
    y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4])
    cov, risk, thr = cr.coverage_risk_curve(y_true, y_prob, n_thresholds=50)
    assert cov.shape == risk.shape == thr.shape == (50,)
    assert np.all(cov >= -1e-12) and np.all(cov <= 1.0 + 1e-12)
    # non-increasing as threshold increases
    assert np.all(np.diff(cov) <= 1e-12)
    assert cov[0] == pytest.approx(1.0)  # tau=0 accepts all


def test_aurc_nonnegative_and_seed42_reproducible():
    rng = np.random.default_rng(42)
    y_prob = rng.random(300)
    y_true = rng.integers(0, 2, size=300)
    m1 = cr.selective_prediction_metrics(y_true, y_prob, n_thresholds=100)
    # recompute identically
    rng2 = np.random.default_rng(42)
    y_prob2 = rng2.random(300)
    y_true2 = rng2.integers(0, 2, size=300)
    m2 = cr.selective_prediction_metrics(y_true2, y_prob2, n_thresholds=100)
    assert m1.aurc >= 0.0
    assert m1.aurc == pytest.approx(m2.aurc, abs=1e-12)


# --------------------------------------------------------------------------- #
# harm-aware metrics
# --------------------------------------------------------------------------- #
def test_weighted_harm_loss_exact():
    # Errors at indices 1 (low,w=1) and 2 (high,w=10): (1*1 + 1*10)/4 = 2.75
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    tiers = np.array(["high", "low", "high", "low"])
    assert hm.weighted_harm_loss(y_true, y_pred, tiers) == pytest.approx(2.75)


def test_asymmetric_cost_matrix_exact():
    # y_true/y_pred -> tp=1, fp=1, fn=1, tn=1.
    # cost = fn*10 + fp*1 = 11 over 4 samples = 2.75
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    cost = hm.asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)
    assert cost == pytest.approx(2.75)


def test_escalation_failure_analysis_counts():
    # Missed 1 high-risk (idx0: true=1,pred=0), over-escalated 1 low-risk (idx2).
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])
    tiers = np.array(["high", "high", "low", "low"])
    out = hm.escalation_failure_analysis(y_true, y_pred, tiers)
    assert out["escalation_failures"] == 1
    assert out["false_escalations"] == 1
    assert out["high_risk_samples"] == 2
    assert out["low_risk_samples"] == 2


# --------------------------------------------------------------------------- #
# performance / confusion matrix
# --------------------------------------------------------------------------- #
def test_confusion_matrix_and_derived_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = perf.confusion_matrix(y_true, y_pred)
    # tn=1, fp=1, fn=1, tp=2
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
    assert cm.total == 5
    assert cm.prevalence == pytest.approx(3 / 5)

    m = perf.compute_performance_metrics(y_true, y_pred)
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.recall == pytest.approx(2 / 3)      # tp/(tp+fn)
    assert m.precision == pytest.approx(2 / 3)   # tp/(tp+fp)
    assert m.specificity == pytest.approx(1 / 2)  # tn/(tn+fp)
    assert 0.0 <= m.f1_score <= 1.0
