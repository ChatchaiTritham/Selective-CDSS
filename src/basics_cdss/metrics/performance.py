"""
Performance Metrics Module

Provides comprehensive classification performance metrics for CDSS evaluation:
- Confusion matrix analysis
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Multi-class and stratified metrics
- Statistical significance testing

Author: Chatchai Tritham
Affiliation: Naresuan University
Date: 2026-01-25
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score)
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import (f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for binary classification.

    Attributes:
        accuracy: Overall accuracy (TP+TN)/(TP+TN+FP+FN)
        precision: Precision (TP)/(TP+FP)
        recall: Recall/Sensitivity (TP)/(TP+FN)
        specificity: Specificity (TN)/(TN+FP)
        f1_score: F1-Score 2*(Precision*Recall)/(Precision+Recall)
        roc_auc: Area Under ROC Curve
        pr_auc: Area Under Precision-Recall Curve
        mcc: Matthews Correlation Coefficient
        kappa: Cohen's Kappa
        npv: Negative Predictive Value (TN)/(TN+FN)
        fpr: False Positive Rate (FP)/(FP+TN)
        fnr: False Negative Rate (FN)/(FN+TP)
    """

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    kappa: float
    npv: float
    fpr: float
    fnr: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'mcc': self.mcc,
            'kappa': self.kappa,
            'npv': self.npv,
            'fpr': self.fpr,
            'fnr': self.fnr,
        }


@dataclass
class ConfusionMatrixMetrics:
    """Confusion matrix with derived metrics.

    Attributes:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives
        total: Total samples
        prevalence: Disease prevalence (TP+FN)/Total
    """

    tn: int
    fp: int
    fn: int
    tp: int
    total: int
    prevalence: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return {
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'tp': self.tp,
            'total': self.total,
            'prevalence': self.prevalence,
        }

    def to_array(self) -> np.ndarray:
        """Return 2x2 confusion matrix as numpy array."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrixMetrics:
    """Compute confusion matrix with derived metrics.

    Parameters:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        ConfusionMatrixMetrics with counts and prevalence

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> print(f"Accuracy: {(cm.tp + cm.tn) / cm.total:.2f}")
    """
    cm = sklearn_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)
    prevalence = (tp + fn) / total if total > 0 else 0.0

    return ConfusionMatrixMetrics(
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        total=total,
        prevalence=prevalence,
    )


def compute_performance_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Parameters:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional, for ROC/PR-AUC)

    Returns:
        PerformanceMetrics with all metrics

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3])
        >>> metrics = compute_performance_metrics(y_true, y_pred, y_prob)
        >>> print(f"F1-Score: {metrics.f1_score:.3f}")
    """
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    # Specificity and NPV
    spec = cm.tn / (cm.tn + cm.fp) if (cm.tn + cm.fp) > 0 else 0.0
    npv = cm.tn / (cm.tn + cm.fn) if (cm.tn + cm.fn) > 0 else 0.0

    # FPR and FNR
    fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    fnr = cm.fn / (cm.fn + cm.tp) if (cm.fn + cm.tp) > 0 else 0.0

    # Advanced metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # ROC-AUC and PR-AUC (requires probabilities)
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            # Handle case where all labels are same class
            roc_auc = 0.0
            pr_auc = 0.0
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return PerformanceMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        f1_score=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=mcc,
        kappa=kappa,
        npv=npv,
        fpr=fpr,
        fnr=fnr,
    )


def stratified_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    strata_names: Optional[List[str]] = None,
) -> Dict[str, PerformanceMetrics]:
    """Compute performance metrics stratified by groups (e.g., risk tiers).

    Parameters:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional)
        strata: Group labels for stratification (optional)
        strata_names: Names for strata groups (optional)

    Returns:
        Dictionary mapping stratum name to PerformanceMetrics
        Includes 'overall' for all data combined

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 0])
        >>> strata = np.array(['high', 'high', 'high', 'low', 'low', 'low'])
        >>> metrics = stratified_performance_metrics(y_true, y_pred, strata=strata)
        >>> print(f"High-risk F1: {metrics['high'].f1_score:.3f}")
    """
    results = {}

    # Overall metrics
    results['overall'] = compute_performance_metrics(y_true, y_pred, y_prob)

    # Stratified metrics
    if strata is not None:
        unique_strata = np.unique(strata)

        for stratum in unique_strata:
            mask = strata == stratum

            # Skip if insufficient samples
            if np.sum(mask) < 2:
                continue

            y_true_s = y_true[mask]
            y_pred_s = y_pred[mask]
            y_prob_s = y_prob[mask] if y_prob is not None else None

            # Skip if only one class present
            if len(np.unique(y_true_s)) < 2:
                continue

            results[str(stratum)] = compute_performance_metrics(
                y_true_s, y_pred_s, y_prob_s
            )

    return results


def compute_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR, thresholds).

    Parameters:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class

    Returns:
        Tuple of (fpr, tpr, thresholds)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)
    """
    return roc_curve(y_true, y_prob)


def compute_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve.

    Parameters:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class

    Returns:
        Tuple of (precision, recall, thresholds)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> precision, recall, thresholds = compute_pr_curve(y_true, y_prob)
    """
    return precision_recall_curve(y_true, y_prob)


def sensitivity_specificity_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Analyze sensitivity and specificity across different thresholds.

    Parameters:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: Custom thresholds to evaluate (default: [0.3, 0.4, 0.5, 0.6, 0.7])

    Returns:
        DataFrame with threshold, sensitivity, specificity, F1, and Youden's J

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> df = sensitivity_specificity_analysis(y_true, y_prob)
    """
    if thresholds is None:
        thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_performance_metrics(y_true, y_pred, y_prob)

        # Youden's J statistic (sensitivity + specificity - 1)
        youdens_j = metrics.recall + metrics.specificity - 1.0

        results.append(
            {
                'threshold': threshold,
                'sensitivity': metrics.recall,
                'specificity': metrics.specificity,
                'precision': metrics.precision,
                'f1_score': metrics.f1_score,
                'youdens_j': youdens_j,
            }
        )

    return pd.DataFrame(results)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metric: str = 'f1_score',
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a performance metric.

    Parameters:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional, required for ROC-AUC/PR-AUC)
        metric: Metric name ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc')
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        >>> y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2, 0.85, 0.15])
        >>> mean, lower, upper = bootstrap_confidence_interval(
        ...     y_true, y_pred, y_prob, metric='f1_score'
        ... )
        >>> print(f"F1-Score: {mean:.3f} ({lower:.3f}, {upper:.3f})")
    """
    np.random.seed(seed)
    n = len(y_true)

    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices] if y_prob is not None else None

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        # Compute metric
        metrics = compute_performance_metrics(y_true_boot, y_pred_boot, y_prob_boot)
        metric_value = getattr(metrics, metric)
        bootstrap_metrics.append(metric_value)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Point estimate (original data)
    point_metrics = compute_performance_metrics(y_true, y_pred, y_prob)
    point_estimate = getattr(point_metrics, metric)

    # Confidence interval (percentile method)
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> Tuple[float, float]:
    """McNemar's test for comparing two classifiers.

    Tests whether two classifiers have significantly different error rates.

    Parameters:
        y_true: True binary labels
        y_pred_a: Predictions from classifier A
        y_pred_b: Predictions from classifier B

    Returns:
        Tuple of (test_statistic, p_value)
        p < 0.05 suggests classifiers have different performance

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> y_pred_a = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        >>> y_pred_b = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        >>> statistic, p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)
        >>> print(f"McNemar p-value: {p_value:.4f}")
    """
    # Create contingency table
    # n_01: A wrong, B correct
    # n_10: A correct, B wrong
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n_01 = np.sum(~correct_a & correct_b)
    n_10 = np.sum(correct_a & ~correct_b)

    # McNemar's test statistic (with continuity correction)
    if (n_01 + n_10) == 0:
        return 0.0, 1.0

    statistic = ((abs(n_01 - n_10) - 1) ** 2) / (n_01 + n_10)

    # Chi-squared test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


def multi_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute per-class metrics for multi-class classification.

    Parameters:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: Optional names for classes

    Returns:
        DataFrame with per-class precision, recall, F1-score, and support

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2])
        >>> df = multi_class_metrics(y_true, y_pred, class_names=['A', 'B', 'C'])
    """
    from sklearn.metrics import classification_report

    # Get classification report as dict
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Convert to DataFrame
    if class_names is not None:
        classes = class_names
    else:
        classes = [f"Class {i}" for i in np.unique(y_true)]

    rows = []
    for cls in classes:
        if cls in report:
            rows.append(
                {
                    'class': cls,
                    'precision': report[cls]['precision'],
                    'recall': report[cls]['recall'],
                    'f1_score': report[cls]['f1-score'],
                    'support': int(report[cls]['support']),
                }
            )

    # Add macro and weighted averages
    rows.append(
        {
            'class': 'macro avg',
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'support': int(report['macro avg']['support']),
        }
    )

    rows.append(
        {
            'class': 'weighted avg',
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'support': int(report['weighted avg']['support']),
        }
    )

    return pd.DataFrame(rows)


def performance_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
) -> Dict[str, Union[PerformanceMetrics, Dict]]:
    """Generate comprehensive performance summary.

    Parameters:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional)
        strata: Stratification groups (optional)

    Returns:
        Dictionary with overall metrics, confusion matrix, and stratified metrics

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 0])
        >>> y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2])
        >>> summary = performance_summary(y_true, y_pred, y_prob)
        >>> print(summary['metrics'].f1_score)
    """
    # Overall metrics
    metrics = compute_performance_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        'metrics': metrics,
        'confusion_matrix': cm,
    }

    # Stratified metrics
    if strata is not None:
        stratified = stratified_performance_metrics(y_true, y_pred, y_prob, strata)
        result['stratified'] = stratified

    return result
