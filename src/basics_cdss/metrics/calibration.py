"""
Calibration metrics for clinical decision support evaluation.

This module implements calibration metrics from the manuscript:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability curves stratified by risk tier

These metrics assess whether model confidence aligns with empirical accuracy,
essential for safe clinical decision making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_CALIBRATION_BIN_COUNT = 10
MIN_CONFIDENCE_VALUE = 0.0
MAX_CONFIDENCE_VALUE = 1.0
EMPTY_METRIC_VALUE = 0.0


@dataclass
class CalibrationMetrics:
    """Container for calibration evaluation results.

    Attributes:
        ece: Expected Calibration Error (lower is better)
        brier_score: Brier score (lower is better)
        reliability_curve: Tuple of (bin_confidences, bin_accuracies, bin_counts)
    """

    ece: float
    brier_score: float
    reliability_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> float:
    """Expected Calibration Error (ECE).

    Implements Algorithm 3 from the manuscript:
    Measures weighted average of absolute differences between confidence and accuracy.

    ECE = Σ (|B_m|/N) |conf(B_m) - acc(B_m)|

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of calibration bins (default=10)

    Returns:
        ECE value (0 is perfect calibration)

    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
        >>> ece = expected_calibration_error(y_true, y_prob)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return EMPTY_METRIC_VALUE

    bins = np.linspace(MIN_CONFIDENCE_VALUE, MAX_CONFIDENCE_VALUE, n_bins + 1)
    expected_calibration_error_value = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include right boundary for last bin
        mask = (
            (y_prob >= lo) & (y_prob < hi)
            if i < n_bins - 1
            else (y_prob >= lo) & (y_prob <= hi)
        )

        if mask.sum() == 0:
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        bin_weight = mask.sum() / n_total

        expected_calibration_error_value += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return float(expected_calibration_error_value)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error between predictions and outcomes).

    From manuscript:
    BS = (1/N) Σ (ŷ_i - y_i)²

    Lower values indicate better calibration.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities

    Returns:
        Brier score (0 is perfect, 1 is worst)

    Example:
        >>> y_true = np.array([1, 0, 1])
        >>> y_prob = np.array([0.9, 0.1, 0.8])
        >>> bs = brier_score(y_true, y_prob)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return EMPTY_METRIC_VALUE

    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BIN_COUNT,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability curve (calibration curve).

    Returns bin-wise empirical accuracy vs. average confidence.
    Perfectly calibrated models produce a diagonal curve (y=x).

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy ("uniform" or "quantile")

    Returns:
        Tuple of (bin_confidences, bin_accuracies, bin_counts)

    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
        >>> confs, accs, counts = reliability_curve(y_true, y_prob, n_bins=5)
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return np.array([]), np.array([]), np.array([])

    if strategy == "uniform":
        bins = np.linspace(MIN_CONFIDENCE_VALUE, MAX_CONFIDENCE_VALUE, n_bins + 1)
    elif strategy == "quantile":
        bins = np.quantile(
            y_prob, np.linspace(MIN_CONFIDENCE_VALUE, MAX_CONFIDENCE_VALUE, n_bins + 1)
        )
        bins[0] = MIN_CONFIDENCE_VALUE
        bins[-1] = MAX_CONFIDENCE_VALUE
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (y_prob >= lo) & (y_prob < hi)
            if i < n_bins - 1
            else (y_prob >= lo) & (y_prob <= hi)
        )

        if mask.sum() == 0:
            continue

        bin_confidences.append(y_prob[mask].mean())
        bin_accuracies.append(y_true[mask].mean())
        bin_counts.append(mask.sum())

    return (np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts))


def stratified_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_tiers: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> Dict[str, CalibrationMetrics]:
    """Compute calibration metrics stratified by risk tier.

    From manuscript: Calibration should be assessed separately for each risk tier
    to detect tier-specific miscalibration patterns.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        risk_tiers: Risk tier labels (e.g., "low", "medium", "high")
        n_bins: Number of calibration bins

    Returns:
        Dictionary mapping risk_tier -> CalibrationMetrics

    Example:
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
        >>> risk_tiers = np.array(["high", "low", "high", "low", "medium"])
        >>> metrics = stratified_calibration_metrics(y_true, y_prob, risk_tiers)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).astype(float)
    risk_tiers = np.asarray(risk_tiers)

    unique_tiers = np.unique(risk_tiers)
    stratified_metrics = {}

    for tier in unique_tiers:
        mask = risk_tiers == tier

        if mask.sum() == 0:
            continue

        tier_true = y_true[mask].astype(float)
        tier_prob = y_prob[mask]

        ece = expected_calibration_error(tier_true, tier_prob, n_bins)
        bs = brier_score(tier_true, tier_prob)
        rel_curve = reliability_curve(tier_true, tier_prob, n_bins)

        stratified_metrics[str(tier)] = CalibrationMetrics(
            ece=ece, brier_score=bs, reliability_curve=rel_curve
        )

    return stratified_metrics


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_tiers: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict:
    """Compute comprehensive calibration summary.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        risk_tiers: Optional risk tier labels for stratification
        n_bins: Number of bins

    Returns:
        Dictionary with overall and (optionally) stratified calibration metrics
    """
    summary = {
        "overall": {
            "ece": expected_calibration_error(y_true, y_prob, n_bins),
            "brier_score": brier_score(y_true, y_prob),
            "reliability_curve": reliability_curve(y_true, y_prob, n_bins),
        }
    }

    if risk_tiers is not None:
        summary["by_risk_tier"] = stratified_calibration_metrics(
            y_true, y_prob, risk_tiers, n_bins
        )

    return summary
