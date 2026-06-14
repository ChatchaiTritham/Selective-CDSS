"""
Fairness Metrics Module

This module implements fairness and bias detection metrics critical for ethical
medical AI systems. Ensuring fairness across demographic groups is essential for:
- Regulatory compliance (FDA, EU AI Act)
- Ethical AI deployment in healthcare
- Health equity and reducing disparities
- Legal liability protection

Implements metrics from:
- Demographic Parity
- Equalized Odds
- Equal Opportunity
- Disparate Impact
- Calibration across groups

References:
    - Hardt M, Price E, Srebro N. (2016). Equality of opportunity in supervised
      learning. NIPS.
    - Chouldechova A. (2017). Fair prediction with disparate impact: A study of
      bias in recidivism prediction instruments. Big data, 5(2), 153-163.
    - Obermeyer Z, et al. (2019). Dissecting racial bias in an algorithm used to
      manage the health of populations. Science, 366(6464), 447-453.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class DemographicParityResult:
    """Container for Demographic Parity analysis results.

    Demographic Parity requires that the probability of positive prediction
    is the same across groups:
        P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1)

    where A is the protected attribute (e.g., race, sex, age group).

    Attributes:
        group_positive_rates: Positive prediction rate for each group
        parity_difference: Max difference in positive rates (0 = perfect parity)
        parity_ratio: Min/Max ratio of positive rates (1 = perfect parity)
        reference_group: Group used as reference
        is_fair: Whether the difference is within acceptable threshold
        threshold: Fairness threshold used
    """

    group_positive_rates: Dict[str, float]
    parity_difference: float
    parity_ratio: float
    reference_group: str
    is_fair: bool
    threshold: float


@dataclass
class EqualizedOddsResult:
    """Container for Equalized Odds analysis results.

    Equalized Odds requires equal True Positive Rate (TPR) and False Positive
    Rate (FPR) across groups:
        P(Y_hat = 1 | Y = y, A = 0) = P(Y_hat = 1 | Y = y, A = 1)  for y ∈ {0,1}

    This ensures the model performs equally well for all groups.

    Attributes:
        group_tpr: True Positive Rate for each group
        group_fpr: False Positive Rate for each group
        tpr_difference: Max difference in TPR across groups
        fpr_difference: Max difference in FPR across groups
        avg_odds_difference: Average of TPR and FPR differences
        is_fair: Whether both differences are within threshold
        threshold: Fairness threshold used
    """

    group_tpr: Dict[str, float]
    group_fpr: Dict[str, float]
    tpr_difference: float
    fpr_difference: float
    avg_odds_difference: float
    is_fair: bool
    threshold: float


@dataclass
class EqualOpportunityResult:
    """Container for Equal Opportunity analysis results.

    Equal Opportunity requires equal True Positive Rate (TPR) across groups:
        P(Y_hat = 1 | Y = 1, A = 0) = P(Y_hat = 1 | Y = 1, A = 1)

    This ensures equal sensitivity/recall for positive cases across groups.

    Attributes:
        group_tpr: True Positive Rate for each group
        tpr_difference: Max difference in TPR
        tpr_ratio: Min/Max ratio of TPR
        is_fair: Whether difference is within threshold
        threshold: Fairness threshold used
    """

    group_tpr: Dict[str, float]
    tpr_difference: float
    tpr_ratio: float
    is_fair: bool
    threshold: float


@dataclass
class DisparateImpactResult:
    """Container for Disparate Impact analysis results.

    Disparate Impact measures the ratio of positive prediction rates.
    The "80% rule" states that the ratio should be ≥ 0.8 to avoid discrimination.

        DI = P(Y_hat = 1 | A = unprivileged) / P(Y_hat = 1 | A = privileged)

    Attributes:
        disparate_impact_ratio: Ratio of positive rates (0.8-1.25 is fair)
        privileged_group: Reference/privileged group
        unprivileged_group: Group being compared
        privileged_positive_rate: Positive rate for privileged group
        unprivileged_positive_rate: Positive rate for unprivileged group
        is_fair: Whether ratio is within acceptable range
        four_fifths_rule: Whether passes 80% rule (DI ≥ 0.8)
    """

    disparate_impact_ratio: float
    privileged_group: str
    unprivileged_group: str
    privileged_positive_rate: float
    unprivileged_positive_rate: float
    is_fair: bool
    four_fifths_rule: bool


@dataclass
class CalibrationResult:
    """Container for calibration fairness analysis results.

    Calibration requires that predicted probabilities match observed frequencies
    across groups:
        P(Y = 1 | Score = s, A = a) = s  for all groups a

    Attributes:
        group_calibration: Dict mapping group -> (predicted, observed) arrays
        calibration_error: Mean calibration error for each group
        max_calibration_error: Maximum calibration error across groups
        is_calibrated: Whether all groups are well-calibrated
        threshold: Calibration error threshold
    """

    group_calibration: Dict[str, Tuple[np.ndarray, np.ndarray]]
    calibration_error: Dict[str, float]
    max_calibration_error: float
    is_calibrated: bool
    threshold: float


@dataclass
class FairnessReport:
    """Comprehensive fairness assessment report.

    Attributes:
        demographic_parity: Demographic parity results
        equalized_odds: Equalized odds results
        equal_opportunity: Equal opportunity results
        disparate_impact: Disparate impact results
        calibration: Calibration results
        overall_fair: Whether model passes all fairness criteria
        failed_criteria: List of failed fairness criteria
    """

    demographic_parity: DemographicParityResult
    equalized_odds: EqualizedOddsResult
    equal_opportunity: EqualOpportunityResult
    disparate_impact: Optional[DisparateImpactResult]
    calibration: CalibrationResult
    overall_fair: bool
    failed_criteria: List[str]


def demographic_parity(
    y_pred: np.ndarray, protected_attribute: np.ndarray, threshold: float = 0.1
) -> DemographicParityResult:
    """Assess demographic parity across protected groups.

    Demographic parity (also called statistical parity or independence)
    requires that predictions are independent of protected attributes.

    Args:
        y_pred: Binary predictions (0 or 1)
        protected_attribute: Group membership labels
        threshold: Maximum acceptable difference in positive rates (default: 0.1)

    Returns:
        DemographicParityResult with parity metrics

    Example:
        >>> result = demographic_parity(y_pred, race, threshold=0.1)
        >>> if not result.is_fair:
        ...     print(f"Demographic parity violated: {result.parity_difference:.3f}")
        >>> for group, rate in result.group_positive_rates.items():
        ...     print(f"{group}: {rate:.3f}")

    Clinical Interpretation:
        If violated, the model may systematically over- or under-predict
        for certain demographic groups, leading to unequal access to care.
    """
    y_pred = np.asarray(y_pred)
    protected_attribute = np.asarray(protected_attribute)

    unique_groups = np.unique(protected_attribute)
    group_rates = {}

    for group in unique_groups:
        mask = protected_attribute == group
        positive_rate = y_pred[mask].mean()
        group_rates[str(group)] = positive_rate

    # Calculate differences
    rates = list(group_rates.values())
    parity_difference = max(rates) - min(rates)

    # Calculate ratio
    if min(rates) > 0:
        parity_ratio = min(rates) / max(rates)
    else:
        parity_ratio = 0.0

    # Determine fairness
    is_fair = parity_difference <= threshold

    return DemographicParityResult(
        group_positive_rates=group_rates,
        parity_difference=parity_difference,
        parity_ratio=parity_ratio,
        reference_group=str(unique_groups[np.argmax(rates)]),
        is_fair=is_fair,
        threshold=threshold,
    )


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
    threshold: float = 0.1,
) -> EqualizedOddsResult:
    """Assess equalized odds across protected groups.

    Equalized odds requires equal TPR and FPR across groups, ensuring
    the model performs equally well regardless of protected attribute.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        protected_attribute: Group membership labels
        threshold: Maximum acceptable difference in TPR/FPR (default: 0.1)

    Returns:
        EqualizedOddsResult with TPR/FPR metrics

    Example:
        >>> result = equalized_odds(y_true, y_pred, sex, threshold=0.1)
        >>> if not result.is_fair:
        ...     print(f"TPR difference: {result.tpr_difference:.3f}")
        ...     print(f"FPR difference: {result.fpr_difference:.3f}")

    Clinical Interpretation:
        If violated, the model may have different sensitivity (ability to
        catch true cases) or specificity (ability to avoid false alarms)
        for different demographic groups.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attribute = np.asarray(protected_attribute)

    unique_groups = np.unique(protected_attribute)
    group_tpr = {}
    group_fpr = {}

    for group in unique_groups:
        mask = protected_attribute == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        if len(y_true_group) > 0:
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()

            # True Positive Rate (Sensitivity)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            group_tpr[str(group)] = tpr
            group_fpr[str(group)] = fpr

    # Calculate differences
    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())

    tpr_difference = max(tpr_values) - min(tpr_values)
    fpr_difference = max(fpr_values) - min(fpr_values)

    # Average odds difference
    avg_odds_diff = (tpr_difference + fpr_difference) / 2

    # Determine fairness (both must be within threshold)
    is_fair = (tpr_difference <= threshold) and (fpr_difference <= threshold)

    return EqualizedOddsResult(
        group_tpr=group_tpr,
        group_fpr=group_fpr,
        tpr_difference=tpr_difference,
        fpr_difference=fpr_difference,
        avg_odds_difference=avg_odds_diff,
        is_fair=is_fair,
        threshold=threshold,
    )


def equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
    threshold: float = 0.1,
) -> EqualOpportunityResult:
    """Assess equal opportunity across protected groups.

    Equal opportunity requires equal True Positive Rate (sensitivity)
    across groups, ensuring equal ability to identify positive cases.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        protected_attribute: Group membership labels
        threshold: Maximum acceptable TPR difference (default: 0.1)

    Returns:
        EqualOpportunityResult with TPR metrics

    Example:
        >>> result = equal_opportunity(y_true, y_pred, age_group, threshold=0.1)
        >>> for group, tpr in result.group_tpr.items():
        ...     print(f"{group}: TPR = {tpr:.3f}")

    Clinical Interpretation:
        If violated, the model may fail to detect disease at equal rates
        across demographic groups, disadvantaging certain populations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attribute = np.asarray(protected_attribute)

    unique_groups = np.unique(protected_attribute)
    group_tpr = {}

    for group in unique_groups:
        mask = protected_attribute == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        if len(y_true_group) > 0:
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()

            # True Positive Rate
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            group_tpr[str(group)] = tpr

    # Calculate differences
    tpr_values = list(group_tpr.values())
    tpr_difference = max(tpr_values) - min(tpr_values)

    # Calculate ratio
    if min(tpr_values) > 0:
        tpr_ratio = min(tpr_values) / max(tpr_values)
    else:
        tpr_ratio = 0.0

    # Determine fairness
    is_fair = tpr_difference <= threshold

    return EqualOpportunityResult(
        group_tpr=group_tpr,
        tpr_difference=tpr_difference,
        tpr_ratio=tpr_ratio,
        is_fair=is_fair,
        threshold=threshold,
    )


def disparate_impact(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
    privileged_group: str,
    unprivileged_group: Optional[str] = None,
) -> DisparateImpactResult:
    """Calculate disparate impact ratio.

    The "80% rule" (four-fifths rule) from US employment law states that
    the selection rate for any group should be at least 80% of the rate
    for the group with highest selection rate.

    Args:
        y_pred: Binary predictions
        protected_attribute: Group membership labels
        privileged_group: Reference/privileged group label
        unprivileged_group: Group to compare. If None, compares all others

    Returns:
        DisparateImpactResult with impact ratio

    Example:
        >>> result = disparate_impact(y_pred, race, privileged_group='White')
        >>> if not result.four_fifths_rule:
        ...     print(f"Fails 80% rule: DI = {result.disparate_impact_ratio:.3f}")

    Clinical Interpretation:
        DI < 0.8: Unprivileged group is under-selected (potential bias)
        DI > 1.25: Unprivileged group is over-selected
        0.8 ≤ DI ≤ 1.25: Acceptable range
    """
    y_pred = np.asarray(y_pred)
    protected_attribute = np.asarray(protected_attribute)

    # Calculate positive rates
    privileged_mask = protected_attribute == privileged_group
    privileged_rate = y_pred[privileged_mask].mean()

    if unprivileged_group is None:
        # Compare all other groups combined
        unprivileged_mask = protected_attribute != privileged_group
        unprivileged_label = f"Non-{privileged_group}"
    else:
        unprivileged_mask = protected_attribute == unprivileged_group
        unprivileged_label = str(unprivileged_group)

    unprivileged_rate = y_pred[unprivileged_mask].mean()

    # Disparate Impact ratio
    if privileged_rate > 0:
        di_ratio = unprivileged_rate / privileged_rate
    else:
        di_ratio = np.inf if unprivileged_rate > 0 else 1.0

    # Four-fifths rule (80% rule)
    four_fifths = di_ratio >= 0.8

    # Fair range: 0.8 to 1.25
    is_fair = 0.8 <= di_ratio <= 1.25

    return DisparateImpactResult(
        disparate_impact_ratio=di_ratio,
        privileged_group=str(privileged_group),
        unprivileged_group=unprivileged_label,
        privileged_positive_rate=privileged_rate,
        unprivileged_positive_rate=unprivileged_rate,
        is_fair=is_fair,
        four_fifths_rule=four_fifths,
    )


def calibration_by_group(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_attribute: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.1,
) -> CalibrationResult:
    """Assess calibration fairness across protected groups.

    Calibration requires that predicted probabilities match observed frequencies.
    Calibration fairness requires this to hold across all groups.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        protected_attribute: Group membership labels
        n_bins: Number of bins for calibration (default: 10)
        threshold: Maximum acceptable calibration error (default: 0.1)

    Returns:
        CalibrationResult with calibration metrics

    Example:
        >>> result = calibration_by_group(y_true, y_pred_proba, ethnicity)
        >>> for group, error in result.calibration_error.items():
        ...     print(f"{group}: Calibration Error = {error:.3f}")

    Clinical Interpretation:
        Poor calibration in a group means predicted probabilities are
        systematically too high or too low, leading to over- or under-treatment.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    protected_attribute = np.asarray(protected_attribute)

    unique_groups = np.unique(protected_attribute)
    group_calibration = {}
    calibration_errors = {}

    for group in unique_groups:
        mask = protected_attribute == group
        y_true_group = y_true[mask]
        y_pred_proba_group = y_pred_proba[mask]

        if len(y_true_group) > 0:
            # Bin predictions
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred_proba_group, bin_edges[1:-1])

            predicted_probs = []
            observed_freqs = []

            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    pred_prob = y_pred_proba_group[bin_mask].mean()
                    obs_freq = y_true_group[bin_mask].mean()
                    predicted_probs.append(pred_prob)
                    observed_freqs.append(obs_freq)

            predicted_probs = np.array(predicted_probs)
            observed_freqs = np.array(observed_freqs)

            group_calibration[str(group)] = (predicted_probs, observed_freqs)

            # Expected Calibration Error (ECE)
            if len(predicted_probs) > 0:
                ece = np.abs(predicted_probs - observed_freqs).mean()
            else:
                ece = 0.0

            calibration_errors[str(group)] = ece

    # Maximum calibration error across groups
    max_error = max(calibration_errors.values()) if calibration_errors else 0.0

    # Determine if well-calibrated
    is_calibrated = max_error <= threshold

    return CalibrationResult(
        group_calibration=group_calibration,
        calibration_error=calibration_errors,
        max_calibration_error=max_error,
        is_calibrated=is_calibrated,
        threshold=threshold,
    )


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_attribute: np.ndarray,
    privileged_group: Optional[str] = None,
    threshold: float = 0.1,
) -> FairnessReport:
    """Generate comprehensive fairness assessment report.

    Evaluates model across multiple fairness criteria:
    - Demographic Parity
    - Equalized Odds
    - Equal Opportunity
    - Disparate Impact
    - Calibration

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities
        protected_attribute: Group membership labels
        privileged_group: Reference group for disparate impact
        threshold: Fairness threshold for all metrics

    Returns:
        FairnessReport with all fairness metrics

    Example:
        >>> report = fairness_report(y_true, y_pred, y_pred_proba, race)
        >>> if not report.overall_fair:
        ...     print(f"Failed criteria: {report.failed_criteria}")
        >>> print(f"Demographic Parity: {report.demographic_parity.is_fair}")
        >>> print(f"Equalized Odds: {report.equalized_odds.is_fair}")
    """
    # Compute all fairness metrics
    dp_result = demographic_parity(y_pred, protected_attribute, threshold)
    eo_result = equalized_odds(y_true, y_pred, protected_attribute, threshold)
    eqopp_result = equal_opportunity(y_true, y_pred, protected_attribute, threshold)
    calib_result = calibration_by_group(
        y_true, y_pred_proba, protected_attribute, threshold=threshold
    )

    # Disparate impact (if privileged group specified)
    if privileged_group is not None:
        di_result = disparate_impact(y_pred, protected_attribute, privileged_group)
    else:
        di_result = None

    # Determine overall fairness
    failed = []
    if not dp_result.is_fair:
        failed.append("Demographic Parity")
    if not eo_result.is_fair:
        failed.append("Equalized Odds")
    if not eqopp_result.is_fair:
        failed.append("Equal Opportunity")
    if di_result and not di_result.is_fair:
        failed.append("Disparate Impact")
    if not calib_result.is_calibrated:
        failed.append("Calibration")

    overall_fair = len(failed) == 0

    return FairnessReport(
        demographic_parity=dp_result,
        equalized_odds=eo_result,
        equal_opportunity=eqopp_result,
        disparate_impact=di_result,
        calibration=calib_result,
        overall_fair=overall_fair,
        failed_criteria=failed,
    )
