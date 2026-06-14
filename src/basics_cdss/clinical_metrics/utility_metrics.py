"""
Clinical Utility Metrics Module

This module implements clinical utility metrics essential for medical AI validation,
including Decision Curve Analysis (DCA), Net Benefit, Number Needed to Treat (NNT),
and clinical impact assessment.

These metrics are critical for:
- FDA approval of medical AI systems
- Clinical decision-making evaluation
- Cost-effectiveness analysis
- Real-world deployment validation

References:
    - Vickers AJ, Elkin EB. (2006). Decision curve analysis: a novel method for
      evaluating prediction models. Medical Decision Making, 26(6), 565-574.
    - Vickers AJ, Van Calster B, Steyerberg EW. (2016). Net benefit approaches to
      the evaluation of prediction models, molecular markers, and diagnostic tests.
      BMJ, 352:i6.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class NetBenefitResult:
    """Container for Net Benefit calculation results.

    Net Benefit quantifies the clinical value of a prediction model by balancing
    the trade-off between true positives (benefits) and false positives (harms).

    Attributes:
        threshold: Probability threshold for classification
        net_benefit: Net benefit value at this threshold
        true_positive_rate: Proportion of true positives
        false_positive_rate: Proportion of false positives
        n_true_positives: Number of true positive cases
        n_false_positives: Number of false positive cases
        n_total: Total number of samples
        harm_to_benefit_ratio: Ratio of harm from FP to benefit from TP
    """

    threshold: float
    net_benefit: float
    true_positive_rate: float
    false_positive_rate: float
    n_true_positives: int
    n_false_positives: int
    n_total: int
    harm_to_benefit_ratio: float


@dataclass
class DecisionCurveResult:
    """Container for Decision Curve Analysis results.

    Decision Curve Analysis evaluates prediction models across a range of
    probability thresholds, comparing to "treat all" and "treat none" strategies.

    Attributes:
        thresholds: Array of probability thresholds
        net_benefit_model: Net benefit of the model at each threshold
        net_benefit_all: Net benefit of "treat all" strategy
        net_benefit_none: Net benefit of "treat none" strategy (always 0)
        standardized_net_benefit: Net benefit standardized by prevalence
        intervention_avoided: Proportion of interventions avoided vs treat-all
        threshold_range: Range of thresholds where model outperforms alternatives
    """

    thresholds: np.ndarray
    net_benefit_model: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray
    standardized_net_benefit: np.ndarray
    intervention_avoided: np.ndarray
    threshold_range: Tuple[float, float]


@dataclass
class NNTResult:
    """Container for Number Needed to Treat (NNT) results.

    NNT represents how many patients need to be treated to prevent one
    additional bad outcome, a key metric for clinical decision-making.

    Attributes:
        nnt: Number needed to treat (lower is better)
        arr: Absolute risk reduction (%)
        control_event_rate: Event rate in control/untreated group
        treatment_event_rate: Event rate in treatment/model-guided group
        nnth: Number needed to treat to harm (from false positives)
        confidence_interval: 95% CI for NNT
    """

    nnt: float
    arr: float  # Absolute Risk Reduction
    control_event_rate: float
    treatment_event_rate: float
    nnth: Optional[float] = None  # Number Needed to Treat to Harm
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class ClinicalImpactResult:
    """Container for clinical impact assessment results.

    Attributes:
        threshold: Probability threshold used
        n_high_risk: Number classified as high-risk
        n_low_risk: Number classified as low-risk
        percent_high_risk: Percentage classified as high-risk
        n_true_positives: True positives (correctly identified high-risk)
        n_false_positives: False positives (incorrectly identified high-risk)
        n_true_negatives: True negatives (correctly identified low-risk)
        n_false_negatives: False negatives (missed high-risk cases)
        ppv: Positive Predictive Value (precision)
        npv: Negative Predictive Value
        number_needed_to_screen: Number needed to screen to find one case
    """

    threshold: float
    n_high_risk: int
    n_low_risk: int
    percent_high_risk: float
    n_true_positives: int
    n_false_positives: int
    n_true_negatives: int
    n_false_negatives: int
    ppv: float
    npv: float
    number_needed_to_screen: float


def calculate_net_benefit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    harm_to_benefit_ratio: Optional[float] = None,
) -> NetBenefitResult:
    """Calculate net benefit for a given probability threshold.

    Net Benefit = (TP/N) - (FP/N) × (pt/(1-pt))

    where:
    - TP = True Positives
    - FP = False Positives
    - N = Total sample size
    - pt = probability threshold

    The term (pt/(1-pt)) represents the odds at threshold, quantifying the
    harm-to-benefit ratio that a decision-maker is willing to accept.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold for classification (default: 0.5)
        harm_to_benefit_ratio: Optional explicit harm-to-benefit ratio.
            If None, uses pt/(1-pt)

    Returns:
        NetBenefitResult object containing calculated metrics

    Example:
        >>> nb = calculate_net_benefit(y_true, y_pred_proba, threshold=0.3)
        >>> print(f"Net Benefit: {nb.net_benefit:.4f}")
        >>> print(f"True Positives: {nb.n_true_positives}/{nb.n_total}")
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Binary classification
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)

    # Calculate rates
    tpr = tp / n
    fpr = fp / n

    # Harm-to-benefit ratio
    if harm_to_benefit_ratio is None:
        if threshold >= 1.0:
            harm_to_benefit_ratio = np.inf
        else:
            harm_to_benefit_ratio = threshold / (1 - threshold)

    # Net Benefit formula
    net_benefit = tpr - fpr * harm_to_benefit_ratio

    return NetBenefitResult(
        threshold=threshold,
        net_benefit=net_benefit,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        n_true_positives=int(tp),
        n_false_positives=int(fp),
        n_total=int(n),
        harm_to_benefit_ratio=harm_to_benefit_ratio,
    )


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 100,
) -> DecisionCurveResult:
    """Perform Decision Curve Analysis across multiple thresholds.

    Decision Curve Analysis (DCA) evaluates the clinical usefulness of a
    prediction model by calculating net benefit across a range of probability
    thresholds, comparing to:
    - "Treat All" strategy: Everyone receives intervention
    - "Treat None" strategy: No one receives intervention

    The model is clinically useful when its net benefit exceeds both strategies.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: Array of thresholds to evaluate. If None, creates
            linearly spaced thresholds between 0.01 and 0.99
        n_thresholds: Number of thresholds if not specified (default: 100)

    Returns:
        DecisionCurveResult containing net benefit curves

    Example:
        >>> dca = decision_curve_analysis(y_true, y_pred_proba)
        >>> # Find threshold range where model is useful
        >>> print(f"Useful threshold range: {dca.threshold_range}")

    References:
        Vickers AJ, Elkin EB. (2006). Decision curve analysis: a novel method
        for evaluating prediction models. Medical Decision Making, 26(6), 565-574.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, n_thresholds)

    prevalence = y_true.mean()
    n = len(y_true)

    # Calculate net benefit for each threshold
    nb_model = []
    nb_all = []
    nb_none = []
    interventions_avoided = []

    for pt in thresholds:
        # Model net benefit
        nb_result = calculate_net_benefit(y_true, y_pred_proba, threshold=pt)
        nb_model.append(nb_result.net_benefit)

        # "Treat All" net benefit
        # All positives are caught (TPR=prevalence), but all negatives are FP
        nb_all_value = prevalence - (1 - prevalence) * (pt / (1 - pt))
        nb_all.append(nb_all_value)

        # "Treat None" net benefit is always 0
        nb_none.append(0.0)

        # Interventions avoided (relative to treat-all)
        if prevalence > 0:
            y_pred = (y_pred_proba >= pt).astype(int)
            n_interventions = y_pred.sum()
            n_all_interventions = n
            avoided = (n_all_interventions - n_interventions) / n_all_interventions
            interventions_avoided.append(avoided)
        else:
            interventions_avoided.append(0.0)

    nb_model = np.array(nb_model)
    nb_all = np.array(nb_all)
    nb_none = np.array(nb_none)
    interventions_avoided = np.array(interventions_avoided)

    # Standardized net benefit (per 100 patients)
    standardized_nb = nb_model * 100

    # Find threshold range where model outperforms both alternatives
    model_better = (nb_model > nb_all) & (nb_model > nb_none)
    if model_better.any():
        useful_indices = np.where(model_better)[0]
        threshold_range = (
            thresholds[useful_indices[0]],
            thresholds[useful_indices[-1]],
        )
    else:
        threshold_range = (np.nan, np.nan)

    return DecisionCurveResult(
        thresholds=thresholds,
        net_benefit_model=nb_model,
        net_benefit_all=nb_all,
        net_benefit_none=nb_none,
        standardized_net_benefit=standardized_nb,
        intervention_avoided=interventions_avoided,
        threshold_range=threshold_range,
    )


def calculate_nnt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    control_event_rate: Optional[float] = None,
    confidence_level: float = 0.95,
) -> NNTResult:
    """Calculate Number Needed to Treat (NNT).

    NNT = 1 / ARR
    ARR = |CER - EER|

    where:
    - ARR = Absolute Risk Reduction
    - CER = Control Event Rate (baseline/untreated)
    - EER = Experimental Event Rate (with model-guided treatment)

    A lower NNT indicates a more effective intervention. For example:
    - NNT = 5: Treat 5 patients to prevent 1 bad outcome
    - NNT = 50: Treat 50 patients to prevent 1 bad outcome

    Args:
        y_true: True binary labels (1 = adverse event)
        y_pred: Predicted binary labels (1 = high-risk, treat)
        control_event_rate: Baseline event rate without model.
            If None, uses event rate in predicted low-risk group
        confidence_level: Confidence level for CI (default: 0.95)

    Returns:
        NNTResult object with NNT and related metrics

    Example:
        >>> nnt_result = calculate_nnt(y_true, y_pred)
        >>> print(f"NNT: {nnt_result.nnt:.1f}")
        >>> print(f"Need to treat {nnt_result.nnt:.0f} patients to prevent 1 event")

    Clinical Interpretation:
        - NNT < 10: Very effective intervention
        - NNT 10-20: Moderately effective
        - NNT > 20: Less effective, consider cost-benefit
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If control rate not provided, use event rate in predicted low-risk group
    if control_event_rate is None:
        low_risk_mask = y_pred == 0
        if low_risk_mask.sum() > 0:
            control_event_rate = y_true[low_risk_mask].mean()
        else:
            # Fallback: use overall event rate
            control_event_rate = y_true.mean()

    # Event rate in high-risk (treated) group
    high_risk_mask = y_pred == 1
    if high_risk_mask.sum() > 0:
        treatment_event_rate = y_true[high_risk_mask].mean()
    else:
        treatment_event_rate = 0.0

    # Absolute Risk Reduction
    arr = abs(control_event_rate - treatment_event_rate)

    # NNT calculation
    if arr > 0:
        nnt = 1.0 / arr
    else:
        nnt = np.inf

    # Confidence interval (using Newcombe method)
    if confidence_level > 0 and arr > 0:
        from scipy import stats

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        n_control = (y_pred == 0).sum()
        n_treatment = (y_pred == 1).sum()

        if n_control > 0 and n_treatment > 0:
            se_control = np.sqrt(
                control_event_rate * (1 - control_event_rate) / n_control
            )
            se_treatment = np.sqrt(
                treatment_event_rate * (1 - treatment_event_rate) / n_treatment
            )
            se_arr = np.sqrt(se_control**2 + se_treatment**2)

            arr_lower = arr - z * se_arr
            arr_upper = arr + z * se_arr

            # NNT CI (note: reciprocal reverses order)
            if arr_lower > 0:
                nnt_upper = 1.0 / arr_lower
            else:
                nnt_upper = np.inf

            if arr_upper > 0:
                nnt_lower = 1.0 / arr_upper
            else:
                nnt_lower = -np.inf

            ci = (nnt_lower, nnt_upper)
        else:
            ci = None
    else:
        ci = None

    # Number Needed to Treat to Harm (NNTH)
    # Based on false positive rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
        nnth = 1.0 / fpr if fpr > 0 else np.inf
    else:
        nnth = None

    return NNTResult(
        nnt=nnt,
        arr=arr * 100,  # Convert to percentage
        control_event_rate=control_event_rate,
        treatment_event_rate=treatment_event_rate,
        nnth=nnth,
        confidence_interval=ci,
    )


def clinical_impact_analysis(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> ClinicalImpactResult:
    """Analyze the clinical impact of model predictions.

    This function assesses the practical implications of deploying a model
    in clinical practice, including:
    - How many patients are classified as high-risk
    - What proportion of high-risk classifications are correct (PPV)
    - How many patients need to be screened to find one true case

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold for high-risk classification

    Returns:
        ClinicalImpactResult with detailed impact metrics

    Example:
        >>> impact = clinical_impact_analysis(y_true, y_pred_proba, threshold=0.3)
        >>> print(f"Classify {impact.percent_high_risk:.1f}% as high-risk")
        >>> print(f"PPV: {impact.ppv:.3f} (precision of high-risk predictions)")
        >>> print(f"Screen {impact.number_needed_to_screen:.1f} to find 1 case")
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Classify
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_high_risk = int(tp + fp)
    n_low_risk = int(tn + fn)
    n_total = len(y_true)

    percent_high_risk = (n_high_risk / n_total) * 100

    # PPV and NPV
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Number needed to screen
    if tp > 0:
        nns = n_high_risk / tp
    else:
        nns = np.inf

    return ClinicalImpactResult(
        threshold=threshold,
        n_high_risk=n_high_risk,
        n_low_risk=n_low_risk,
        percent_high_risk=percent_high_risk,
        n_true_positives=int(tp),
        n_false_positives=int(fp),
        n_true_negatives=int(tn),
        n_false_negatives=int(fn),
        ppv=ppv,
        npv=npv,
        number_needed_to_screen=nns,
    )


def stratified_net_benefit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, NetBenefitResult]:
    """Calculate net benefit stratified by subgroups.

    Essential for assessing whether a model provides clinical utility
    across different patient populations (age groups, sex, ethnicity, etc.).

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        groups: Group labels for stratification
        threshold: Probability threshold

    Returns:
        Dictionary mapping group names to NetBenefitResult objects

    Example:
        >>> # Stratify by sex
        >>> results = stratified_net_benefit(y_true, y_pred_proba, sex, 0.3)
        >>> for group, nb in results.items():
        ...     print(f"{group}: Net Benefit = {nb.net_benefit:.4f}")
    """
    unique_groups = np.unique(groups)
    results = {}

    for group in unique_groups:
        mask = groups == group
        results[str(group)] = calculate_net_benefit(
            y_true[mask], y_pred_proba[mask], threshold=threshold
        )

    return results
