"""
Coverage-risk metrics for selective prediction evaluation.

This module implements selective prediction metrics from the manuscript:
- Coverage-Risk curves
- Area Under Risk-Coverage Curve (AURC)
- Abstention analysis

These metrics evaluate whether systems appropriately abstain when uncertain,
trading coverage for reduced conditional risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEFAULT_CONFIDENCE_ACCEPTANCE_THRESHOLD = 0.5
DEFAULT_SELECTIVE_THRESHOLD_COUNT = 100
MIN_THRESHOLD_VALUE = 0.0
MAX_THRESHOLD_VALUE = 1.0
EMPTY_METRIC_VALUE = 0.0


@dataclass
class SelectivePredictionMetrics:
    """Container for selective prediction evaluation results.

    Attributes:
        aurc: Area Under Risk-Coverage Curve (lower is better)
        coverage_at_risk_threshold: Coverage achieved at specified risk threshold
        risk_at_coverage_threshold: Risk at specified coverage threshold
        coverage_curve: Coverage values at different thresholds
        risk_curve: Conditional risk values at different thresholds
        thresholds: Confidence thresholds used
    """

    aurc: float
    coverage_at_risk_threshold: Optional[float] = None
    risk_at_coverage_threshold: Optional[float] = None
    coverage_curve: Optional[np.ndarray] = None
    risk_curve: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None


def coverage_risk_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_proxy: Optional[np.ndarray] = None,
    n_thresholds: int = DEFAULT_SELECTIVE_THRESHOLD_COUNT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coverage-risk curve for selective prediction.

    From manuscript:
    - Coverage(τ) = fraction of predictions retained at threshold τ
    - Risk(τ) = average risk among accepted predictions

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities (confidence scores)
        risk_proxy: Optional risk proxy for each sample. If None, uses 1 - y_true
        n_thresholds: Number of threshold points to evaluate

    Returns:
        Tuple of (coverages, risks, thresholds)

    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
        >>> coverages, risks, thresholds = coverage_risk_curve(y_true, y_prob)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return np.array([]), np.array([]), np.array([])

    # Default risk proxy: error rate (1 if wrong, 0 if correct)
    if risk_proxy is None:
        predicted_labels = (y_prob >= DEFAULT_CONFIDENCE_ACCEPTANCE_THRESHOLD).astype(
            float
        )
        risk_proxy_array = (predicted_labels != y_true).astype(float)
    else:
        risk_proxy_array = np.asarray(risk_proxy).astype(float)

    # Generate thresholds from min to max confidence
    thresholds = np.linspace(MIN_THRESHOLD_VALUE, MAX_THRESHOLD_VALUE, n_thresholds)

    coverage_values = []
    risk_values = []

    n_total = len(y_true)

    for tau in thresholds:
        # Accept predictions with confidence >= tau
        accepted_mask = y_prob >= tau
        n_accepted = accepted_mask.sum()

        if n_accepted == 0:
            # No predictions accepted: coverage = 0, risk undefined (use NaN)
            coverage_values.append(EMPTY_METRIC_VALUE)
            risk_values.append(np.nan)
        else:
            coverage_value = n_accepted / n_total
            conditional_risk = risk_proxy_array[accepted_mask].mean()

            coverage_values.append(coverage_value)
            risk_values.append(conditional_risk)

    return (np.array(coverage_values), np.array(risk_values), thresholds)


def area_under_risk_coverage_curve(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Compute Area Under Risk-Coverage Curve (AURC).

    From manuscript:
    AURC = ∫₀¹ Risk(c) dc

    Lower AURC indicates better selective prediction:
    ideal system maintains low risk even at high coverage.

    Args:
        coverages: Coverage values (x-axis)
        risks: Conditional risk values (y-axis)

    Returns:
        AURC value (lower is better)

    Example:
        >>> coverages = np.array([0.0, 0.5, 1.0])
        >>> risks = np.array([0.0, 0.1, 0.2])
        >>> aurc = area_under_risk_coverage_curve(coverages, risks)
    """
    coverages = np.asarray(coverages)
    risks = np.asarray(risks)

    # Remove NaN values (can occur at zero coverage)
    valid_mask = ~np.isnan(risks)
    coverages = coverages[valid_mask]
    risks = risks[valid_mask]

    if len(coverages) == 0:
        return EMPTY_METRIC_VALUE

    # Sort by coverage (should already be sorted, but ensure)
    sort_idx = np.argsort(coverages)
    coverages = coverages[sort_idx]
    risks = risks[sort_idx]

    # Trapezoidal integration
    try:
        # Use trapezoid (numpy >= 2.0)
        aurc = np.trapezoid(risks, coverages)
    except AttributeError:
        # Fallback to deprecated trapz for older numpy
        aurc = np.trapz(risks, coverages)

    return float(aurc)


def selective_prediction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_proxy: Optional[np.ndarray] = None,
    target_coverage: float = 0.8,
    target_risk: float = 0.1,
    n_thresholds: int = DEFAULT_SELECTIVE_THRESHOLD_COUNT,
) -> SelectivePredictionMetrics:
    """Compute comprehensive selective prediction metrics.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        risk_proxy: Optional risk proxy (default: error indicator)
        target_coverage: Target coverage for risk threshold analysis
        target_risk: Target risk for coverage threshold analysis
        n_thresholds: Number of threshold points

    Returns:
        SelectivePredictionMetrics object with AURC and curves

    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        >>> y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.85, 0.15, 0.4])
        >>> metrics = selective_prediction_metrics(y_true, y_prob)
        >>> print(f"AURC: {metrics.aurc:.4f}")
    """
    coverages, risks, thresholds = coverage_risk_curve(
        y_true, y_prob, risk_proxy, n_thresholds
    )

    aurc = area_under_risk_coverage_curve(coverages, risks)

    # Find risk at target coverage
    risk_at_coverage = None
    if len(coverages) > 0:
        # Find closest coverage to target
        valid_mask = ~np.isnan(risks)
        if valid_mask.any():
            valid_coverages = coverages[valid_mask]
            valid_risks = risks[valid_mask]

            idx = np.argmin(np.abs(valid_coverages - target_coverage))
            risk_at_coverage = valid_risks[idx]

    # Find coverage at target risk
    coverage_at_risk = None
    if len(risks) > 0:
        valid_mask = ~np.isnan(risks)
        if valid_mask.any():
            valid_coverages = coverages[valid_mask]
            valid_risks = risks[valid_mask]

            # Find threshold where risk is closest to target
            below_target = valid_risks <= target_risk
            if below_target.any():
                # Maximum coverage while maintaining risk <= target
                coverage_at_risk = valid_coverages[below_target].max()

    return SelectivePredictionMetrics(
        aurc=aurc,
        coverage_at_risk_threshold=coverage_at_risk,
        risk_at_coverage_threshold=risk_at_coverage,
        coverage_curve=coverages,
        risk_curve=risks,
        thresholds=thresholds,
    )


def abstention_rate(
    y_prob: np.ndarray,
    threshold: float = DEFAULT_CONFIDENCE_ACCEPTANCE_THRESHOLD,
) -> float:
    """Compute abstention rate at given confidence threshold.

    Args:
        y_prob: Predicted probabilities
        threshold: Minimum confidence threshold for prediction

    Returns:
        Fraction of samples abstained (confidence below threshold)

    Example:
        >>> y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
        >>> rate = abstention_rate(y_prob, threshold=0.5)
        >>> print(f"Abstention rate: {rate:.2f}")
    """
    y_prob = np.asarray(y_prob)

    if len(y_prob) == 0:
        return 0.0

    # Count predictions below threshold (ambiguous/uncertain)
    abstained = (y_prob < threshold).sum()
    rate = abstained / len(y_prob)

    return float(rate)


def stratified_selective_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_tiers: np.ndarray,
    risk_proxy: Optional[np.ndarray] = None,
    n_thresholds: int = 100,
) -> dict:
    """Compute selective prediction metrics stratified by risk tier.

    From manuscript: Selective prediction behavior should be evaluated
    separately for each risk tier to detect unsafe coverage expansion
    in high-risk scenarios.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        risk_tiers: Risk tier labels
        risk_proxy: Optional risk proxy
        n_thresholds: Number of threshold points

    Returns:
        Dictionary mapping risk_tier -> SelectivePredictionMetrics

    Example:
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
        >>> risk_tiers = np.array(["high", "low", "high", "low", "medium"])
        >>> metrics = stratified_selective_metrics(y_true, y_prob, risk_tiers)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    risk_tiers = np.asarray(risk_tiers)

    unique_tiers = np.unique(risk_tiers)
    stratified_metrics = {}

    for tier in unique_tiers:
        mask = risk_tiers == tier

        if mask.sum() == 0:
            continue

        tier_true = y_true[mask]
        tier_prob = y_prob[mask]
        tier_risk_proxy = risk_proxy[mask] if risk_proxy is not None else None

        metrics = selective_prediction_metrics(
            tier_true, tier_prob, risk_proxy=tier_risk_proxy, n_thresholds=n_thresholds
        )

        stratified_metrics[str(tier)] = metrics

    return stratified_metrics
