"""
Harm-aware evaluation metrics for safety-critical decision support.

This module implements harm-aware metrics from the manuscript:
- Weighted harm loss by risk tier
- Escalation failure analysis
- Asymmetric cost evaluation

These metrics reflect that not all errors have equal clinical consequences:
failures in high-risk scenarios are weighted more heavily than low-risk errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class HarmMetrics:
    """Container for harm-aware evaluation results.

    Attributes:
        weighted_harm_loss: Overall weighted harm score
        harm_by_tier: Harm loss per risk tier
        escalation_failures: Count of failures to escalate high-risk cases
        false_escalations: Count of unnecessary escalations in low-risk cases
        harm_concentration: Fraction of total harm in high-risk tier
    """

    weighted_harm_loss: float
    harm_by_tier: Dict[str, float]
    escalation_failures: int
    false_escalations: int
    harm_concentration: float


# Default harm weights following clinical triage urgency
DEFAULT_HARM_WEIGHTS = {
    "high": 10.0,  # High-risk errors are 10x more harmful
    "medium": 3.0,  # Medium-risk errors are 3x more harmful
    "low": 1.0,  # Low-risk errors baseline
    "urgent": 10.0,  # Alias for high
    "non-urgent": 1.0,  # Alias for low
}


def weighted_harm_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_tiers: np.ndarray,
    harm_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted harm loss prioritizing high-risk scenarios.

    From manuscript:
    L_harm = (1/N) Σ w_r_i · 𝟙[ŷ_i ≠ y_i]

    where w_r_i is harm weight for risk tier r_i,
    with w_high >> w_medium > w_low.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        risk_tiers: Risk tier for each sample
        harm_weights: Dict mapping risk tier to weight (uses defaults if None)

    Returns:
        Weighted harm loss (lower is better)

    Example:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_pred = np.array([1, 1, 0, 0])  # 2 errors
        >>> risk_tiers = np.array(["high", "low", "high", "low"])
        >>> loss = weighted_harm_loss(y_true, y_pred, risk_tiers)
        # High-risk error weighted more than low-risk error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    risk_tiers = np.asarray(risk_tiers)

    if len(y_true) == 0:
        return 0.0

    if harm_weights is None:
        harm_weights = DEFAULT_HARM_WEIGHTS

    # Map risk tiers to weights
    weights = np.array(
        [harm_weights.get(str(tier).lower(), 1.0) for tier in risk_tiers]
    )

    # Indicator of error
    errors = (y_pred != y_true).astype(float)

    # Weighted loss
    weighted_loss = np.sum(weights * errors) / len(y_true)

    return float(weighted_loss)


def harm_by_risk_tier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_tiers: np.ndarray,
    harm_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute harm loss separately for each risk tier.

    Helps identify if errors disproportionately concentrate in high-risk scenarios.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        risk_tiers: Risk tier labels
        harm_weights: Harm weight mapping

    Returns:
        Dictionary mapping risk_tier -> tier-specific harm loss

    Example:
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([1, 1, 0, 0, 1])
        >>> risk_tiers = np.array(["high", "low", "high", "low", "medium"])
        >>> harm = harm_by_risk_tier(y_true, y_pred, risk_tiers)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    risk_tiers = np.asarray(risk_tiers)

    if harm_weights is None:
        harm_weights = DEFAULT_HARM_WEIGHTS

    unique_tiers = np.unique(risk_tiers)
    tier_harm = {}

    for tier in unique_tiers:
        mask = risk_tiers == tier
        tier_weight = harm_weights.get(str(tier).lower(), 1.0)

        if mask.sum() == 0:
            tier_harm[str(tier)] = 0.0
            continue

        tier_true = y_true[mask]
        tier_pred = y_pred[mask]

        # Error rate in this tier
        error_rate = (tier_pred != tier_true).mean()

        # Weighted by tier harm
        tier_harm[str(tier)] = float(tier_weight * error_rate)

    return tier_harm


def escalation_failure_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_tiers: np.ndarray,
    high_risk_labels: Optional[list] = None,
) -> Dict[str, int]:
    """Analyze escalation failures and false escalations.

    Escalation failure: Missing a high-risk case (false negative in high tier)
    False escalation: Over-escalating a low-risk case (false positive in low tier)

    Args:
        y_true: Ground truth (1 = needs escalation, 0 = no escalation)
        y_pred: Predictions (1 = escalate, 0 = defer)
        risk_tiers: Risk tier labels
        high_risk_labels: List of tier labels considered high-risk (default: ["high", "urgent"])

    Returns:
        Dictionary with escalation failure counts

    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([0, 1, 1, 0])  # Missed 1 high-risk, over-escalated 1 low-risk
        >>> risk_tiers = np.array(["high", "high", "low", "low"])
        >>> analysis = escalation_failure_analysis(y_true, y_pred, risk_tiers)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    risk_tiers = np.asarray(risk_tiers)

    if high_risk_labels is None:
        high_risk_labels = ["high", "urgent", "critical", "emergency"]

    # Identify high-risk and low-risk samples
    high_risk_mask = np.isin([str(t).lower() for t in risk_tiers], high_risk_labels)
    low_risk_mask = ~high_risk_mask

    # Escalation failures (false negatives in high-risk tier)
    high_risk_true = y_true[high_risk_mask]
    high_risk_pred = y_pred[high_risk_mask]
    escalation_failures = ((high_risk_true == 1) & (high_risk_pred == 0)).sum()

    # False escalations (false positives in low-risk tier)
    low_risk_true = y_true[low_risk_mask]
    low_risk_pred = y_pred[low_risk_mask]
    false_escalations = ((low_risk_true == 0) & (low_risk_pred == 1)).sum()

    return {
        "escalation_failures": int(escalation_failures),
        "false_escalations": int(false_escalations),
        "high_risk_samples": int(high_risk_mask.sum()),
        "low_risk_samples": int(low_risk_mask.sum()),
    }


def harm_concentration_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_tiers: np.ndarray,
    harm_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Measure fraction of total harm concentrated in high-risk tier.

    High concentration (close to 1.0) indicates errors predominantly
    occur in high-consequence scenarios—a critical safety concern.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        risk_tiers: Risk tier labels
        harm_weights: Harm weight mapping

    Returns:
        Concentration index [0, 1]: fraction of harm in high-risk tier

    Example:
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0, 0, 0, 0, 1])  # 2 errors in high-risk
        >>> risk_tiers = np.array(["high", "low", "high", "low", "low"])
        >>> concentration = harm_concentration_index(y_true, y_pred, risk_tiers)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    risk_tiers = np.asarray(risk_tiers)

    if harm_weights is None:
        harm_weights = DEFAULT_HARM_WEIGHTS

    tier_harm = harm_by_risk_tier(y_true, y_pred, risk_tiers, harm_weights)

    total_harm = sum(tier_harm.values())

    if total_harm == 0:
        return 0.0

    # Sum harm in high-risk tiers
    high_risk_tiers = ["high", "urgent", "critical", "emergency"]
    high_risk_harm = sum(
        harm for tier, harm in tier_harm.items() if tier.lower() in high_risk_tiers
    )

    concentration = high_risk_harm / total_harm

    return float(concentration)


def compute_harm_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_tiers: np.ndarray,
    harm_weights: Optional[Dict[str, float]] = None,
    high_risk_labels: Optional[list] = None,
) -> HarmMetrics:
    """Compute comprehensive harm-aware evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        risk_tiers: Risk tier labels
        harm_weights: Harm weight mapping
        high_risk_labels: Labels considered high-risk

    Returns:
        HarmMetrics object with all harm-aware metrics

    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0, 0])
        >>> y_pred = np.array([1, 0, 0, 1, 1, 0])
        >>> risk_tiers = np.array(["high", "high", "low", "medium", "low", "low"])
        >>> metrics = compute_harm_metrics(y_true, y_pred, risk_tiers)
        >>> print(f"Weighted harm: {metrics.weighted_harm_loss:.4f}")
    """
    weighted_loss = weighted_harm_loss(y_true, y_pred, risk_tiers, harm_weights)

    tier_harm = harm_by_risk_tier(y_true, y_pred, risk_tiers, harm_weights)

    escalation_analysis = escalation_failure_analysis(
        y_true, y_pred, risk_tiers, high_risk_labels
    )

    concentration = harm_concentration_index(y_true, y_pred, risk_tiers, harm_weights)

    return HarmMetrics(
        weighted_harm_loss=weighted_loss,
        harm_by_tier=tier_harm,
        escalation_failures=escalation_analysis["escalation_failures"],
        false_escalations=escalation_analysis["false_escalations"],
        harm_concentration=concentration,
    )


def asymmetric_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 1.0,
    cost_fp: float = 0.1,
    cost_tn: float = 0.0,
    cost_tp: float = 0.0,
) -> float:
    """Compute cost using asymmetric cost matrix.

    Useful when false negatives are more costly than false positives
    (e.g., missing a critical diagnosis vs. unnecessary test).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        cost_fn: Cost of false negative (default: 1.0)
        cost_fp: Cost of false positive (default: 0.1)
        cost_tn: Cost of true negative (default: 0.0)
        cost_tp: Cost of true positive (default: 0.0)

    Returns:
        Average cost per sample

    Example:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_pred = np.array([1, 1, 0, 0])
        >>> cost = asymmetric_cost_matrix(y_true, y_pred, cost_fn=10.0, cost_fp=1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    # Count confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Total cost
    total_cost = cost_tp * tp + cost_fp * fp + cost_tn * tn + cost_fn * fn

    # Average cost per sample
    avg_cost = total_cost / len(y_true)

    return float(avg_cost)
