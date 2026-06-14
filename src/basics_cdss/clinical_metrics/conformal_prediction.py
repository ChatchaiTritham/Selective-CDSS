"""
Conformal Prediction Module

This module implements conformal prediction methods for uncertainty quantification
with statistical guarantees. Conformal prediction provides:
- Prediction intervals with guaranteed coverage
- Distribution-free uncertainty estimates
- Calibrated confidence scores
- Adaptive prediction sets

Critical for medical AI because:
- Provides rigorous uncertainty quantification
- Enables safe deployment with "I don't know" option
- Increases trustworthiness and interpretability
- Helps identify out-of-distribution cases

References:
    - Vovk V, Gammerman A, Shafer G. (2005). Algorithmic Learning in a Random World.
    - Angelopoulos AN, Bates S. (2021). A gentle introduction to conformal prediction
      and distribution-free uncertainty quantification. arXiv:2107.07511.
    - Angelopoulos AN, et al. (2022). Learn then test: Calibrating predictive
      algorithms to achieve risk control. arXiv:2110.01052.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


@dataclass
class ConformalPredictionSet:
    """Container for conformal prediction results.

    Attributes:
        prediction_sets: List of prediction sets (one per sample)
        set_sizes: Size of each prediction set
        coverage: Empirical coverage rate
        target_coverage: Target coverage level (1 - alpha)
        calibration_scores: Nonconformity scores from calibration
        threshold: Quantile threshold used
        efficiency: Average prediction set size (smaller is better)
    """

    prediction_sets: List[Union[List[int], List[float]]]
    set_sizes: np.ndarray
    coverage: float
    target_coverage: float
    calibration_scores: np.ndarray
    threshold: float
    efficiency: float


@dataclass
class ConformalInterval:
    """Container for conformal prediction intervals.

    For regression problems, provides intervals [lower, upper] with
    guaranteed coverage.

    Attributes:
        lower_bounds: Lower bound of prediction interval for each sample
        upper_bounds: Upper bound of prediction interval for each sample
        point_predictions: Point predictions
        coverage: Empirical coverage rate
        target_coverage: Target coverage level
        interval_widths: Width of each interval
        average_width: Average interval width
        calibration_residuals: Residuals from calibration set
        threshold: Quantile threshold used
    """

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    point_predictions: np.ndarray
    coverage: float
    target_coverage: float
    interval_widths: np.ndarray
    average_width: float
    calibration_residuals: np.ndarray
    threshold: float


@dataclass
class AdaptiveConformalResult:
    """Container for adaptive conformal prediction results.

    Adaptive conformal prediction adjusts the prediction set size based
    on uncertainty, providing tighter sets when confident and wider sets
    when uncertain.

    Attributes:
        prediction_sets: Adaptive prediction sets
        set_sizes: Size of each prediction set
        difficulty_scores: Difficulty/uncertainty score for each sample
        coverage: Empirical coverage
        target_coverage: Target coverage level
        conditional_coverage: Coverage stratified by difficulty
        efficiency_gain: Improvement over standard conformal prediction
    """

    prediction_sets: List[List[int]]
    set_sizes: np.ndarray
    difficulty_scores: np.ndarray
    coverage: float
    target_coverage: float
    conditional_coverage: Dict[str, float]
    efficiency_gain: float


@dataclass
class RiskControlResult:
    """Container for risk control results.

    Risk control calibration ensures that a risk metric (e.g., false negative
    rate) is controlled below a target level.

    Attributes:
        threshold: Calibrated threshold
        empirical_risk: Observed risk on test set
        target_risk: Target risk level
        risk_controlled: Whether risk is controlled
        n_rejected: Number of samples where model abstains (high uncertainty)
        rejection_rate: Proportion of samples rejected
    """

    threshold: float
    empirical_risk: float
    target_risk: float
    risk_controlled: bool
    n_rejected: int
    rejection_rate: float


def split_conformal_classification(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    score_function: Optional[Callable] = None,
) -> ConformalPredictionSet:
    """Split conformal prediction for classification.

    Provides prediction sets with guaranteed coverage: P(Y ∈ C(X)) ≥ 1 - α

    The algorithm:
    1. Train model on training set
    2. Compute nonconformity scores on calibration set
    3. Find (1-α) quantile of scores
    4. For test samples, include all labels with score ≤ threshold

    Args:
        model: Sklearn-compatible classifier (must have predict_proba)
        X_train: Training features
        y_train: Training labels
        X_cal: Calibration features
        y_cal: Calibration labels
        X_test: Test features
        alpha: Miscoverage rate (default: 0.1 for 90% coverage)
        score_function: Custom nonconformity score. If None, uses 1 - P(Y=y)

    Returns:
        ConformalPredictionSet with guaranteed coverage

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> result = split_conformal_classification(
        ...     model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
        ... )
        >>> print(f"Coverage: {result.coverage:.3f} (target: {result.target_coverage:.3f})")
        >>> print(f"Average set size: {result.efficiency:.2f}")

    Clinical Use Case:
        When the prediction set contains multiple diagnoses, the clinician
        should order additional tests to disambiguate. A singleton set indicates
        high confidence.
    """
    # Train model
    model.fit(X_train, y_train)

    # Get predicted probabilities for calibration set
    probs_cal = model.predict_proba(X_cal)
    n_cal = len(y_cal)
    n_classes = probs_cal.shape[1]
    classes = model.classes_

    # Compute nonconformity scores on calibration set
    if score_function is None:
        # Default: 1 - P(Y = y_true)
        cal_scores = np.array(
            [
                1 - probs_cal[i, np.where(classes == y_cal[i])[0][0]]
                for i in range(n_cal)
            ]
        )
    else:
        cal_scores = np.array(
            [score_function(probs_cal[i], y_cal[i]) for i in range(n_cal)]
        )

    # Compute quantile threshold (adjusted for finite sample)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)  # Ensure valid quantile
    threshold = np.quantile(cal_scores, q_level)

    # Generate prediction sets for test samples
    probs_test = model.predict_proba(X_test)
    n_test = len(X_test)

    prediction_sets = []
    set_sizes = []

    for i in range(n_test):
        # Include all labels with score ≤ threshold
        if score_function is None:
            test_scores = 1 - probs_test[i]
        else:
            test_scores = np.array(
                [score_function(probs_test[i], cls) for cls in classes]
            )

        pred_set = [classes[j] for j in range(n_classes) if test_scores[j] <= threshold]
        prediction_sets.append(pred_set)
        set_sizes.append(len(pred_set))

    set_sizes = np.array(set_sizes)
    efficiency = set_sizes.mean()

    return ConformalPredictionSet(
        prediction_sets=prediction_sets,
        set_sizes=set_sizes,
        coverage=np.nan,  # Coverage computed on labeled test set
        target_coverage=1 - alpha,
        calibration_scores=cal_scores,
        threshold=threshold,
        efficiency=efficiency,
    )


def split_conformal_regression(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> ConformalInterval:
    """Split conformal prediction for regression.

    Provides prediction intervals with guaranteed coverage:
        P(Y ∈ [Y_hat - q, Y_hat + q]) ≥ 1 - α

    where q is the (1-α) quantile of absolute residuals on calibration set.

    Args:
        model: Sklearn-compatible regressor
        X_train: Training features
        y_train: Training target values
        X_cal: Calibration features
        y_cal: Calibration target values
        X_test: Test features
        alpha: Miscoverage rate (default: 0.1 for 90% coverage)

    Returns:
        ConformalInterval with prediction intervals

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> result = split_conformal_regression(
        ...     model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
        ... )
        >>> print(f"Average interval width: {result.average_width:.2f}")

    Clinical Use Case:
        For predicting continuous outcomes (e.g., survival time, biomarker levels),
        provides uncertainty intervals. Wide intervals indicate high uncertainty.
    """
    # Train model
    model.fit(X_train, y_train)

    # Compute residuals on calibration set
    y_cal_pred = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - y_cal_pred)

    # Compute quantile threshold
    n_cal = len(y_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = np.quantile(cal_residuals, q_level)

    # Generate prediction intervals for test set
    y_test_pred = model.predict(X_test)
    lower_bounds = y_test_pred - threshold
    upper_bounds = y_test_pred + threshold

    interval_widths = upper_bounds - lower_bounds
    average_width = interval_widths.mean()

    return ConformalInterval(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        point_predictions=y_test_pred,
        coverage=np.nan,  # Computed on labeled test set
        target_coverage=1 - alpha,
        interval_widths=interval_widths,
        average_width=average_width,
        calibration_residuals=cal_residuals,
        threshold=threshold,
    )


def adaptive_conformal_classification(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    difficulty_estimator: Optional[Callable] = None,
) -> AdaptiveConformalResult:
    """Adaptive conformal prediction with difficulty-based adjustment.

    Adjusts prediction set sizes based on sample difficulty, providing:
    - Smaller sets for easy/typical samples
    - Larger sets for difficult/atypical samples
    - Improved efficiency while maintaining coverage

    Args:
        model: Classifier with predict_proba
        X_train: Training features
        y_train: Training labels
        X_cal: Calibration features
        y_cal: Calibration labels
        X_test: Test features
        alpha: Target miscoverage rate
        difficulty_estimator: Function to estimate sample difficulty.
            If None, uses entropy of predicted probabilities

    Returns:
        AdaptiveConformalResult with adaptive prediction sets

    Example:
        >>> result = adaptive_conformal_classification(
        ...     model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
        ... )
        >>> print(f"Efficiency gain: {result.efficiency_gain:.1%}")

    Clinical Use Case:
        For complex/borderline cases (high difficulty), provide larger
        prediction sets to reflect uncertainty. For clear-cut cases,
        provide singleton sets.
    """
    from scipy.stats import entropy

    # Train model
    model.fit(X_train, y_train)

    # Default difficulty: entropy of predicted distribution
    if difficulty_estimator is None:

        def difficulty_estimator(probs):
            return entropy(probs + 1e-10)  # Add epsilon for numerical stability

    # Compute difficulty scores for calibration set
    probs_cal = model.predict_proba(X_cal)
    difficulty_cal = np.array([difficulty_estimator(p) for p in probs_cal])

    # Stratify calibration set by difficulty (tertiles)
    difficulty_bins = np.percentile(difficulty_cal, [33.33, 66.67])
    bin_labels = np.digitize(difficulty_cal, difficulty_bins)

    # Compute thresholds for each difficulty bin
    n_cal = len(y_cal)
    classes = model.classes_
    n_classes = len(classes)

    # Nonconformity scores
    cal_scores = np.array(
        [1 - probs_cal[i, np.where(classes == y_cal[i])[0][0]] for i in range(n_cal)]
    )

    # Adaptive thresholds
    thresholds = {}
    for bin_idx in [0, 1, 2]:
        mask = bin_labels == bin_idx
        if mask.sum() > 0:
            n_bin = mask.sum()
            q_level = np.ceil((n_bin + 1) * (1 - alpha)) / n_bin
            q_level = min(q_level, 1.0)
            thresholds[bin_idx] = np.quantile(cal_scores[mask], q_level)
        else:
            thresholds[bin_idx] = np.quantile(cal_scores, 1 - alpha)

    # Generate adaptive prediction sets for test samples
    probs_test = model.predict_proba(X_test)
    difficulty_test = np.array([difficulty_estimator(p) for p in probs_test])
    bin_labels_test = np.digitize(difficulty_test, difficulty_bins)

    prediction_sets = []
    set_sizes = []

    for i in range(len(X_test)):
        bin_idx = bin_labels_test[i]
        threshold = thresholds.get(bin_idx, thresholds[1])  # Default to middle bin

        test_scores = 1 - probs_test[i]
        pred_set = [classes[j] for j in range(n_classes) if test_scores[j] <= threshold]

        prediction_sets.append(pred_set)
        set_sizes.append(len(pred_set))

    set_sizes = np.array(set_sizes)

    # Conditional coverage by difficulty
    conditional_cov = {'easy': np.nan, 'medium': np.nan, 'hard': np.nan}

    # Efficiency gain (compare to standard conformal)
    standard_result = split_conformal_classification(
        model, X_train, y_train, X_cal, y_cal, X_test, alpha
    )
    efficiency_gain = (
        standard_result.efficiency - set_sizes.mean()
    ) / standard_result.efficiency

    return AdaptiveConformalResult(
        prediction_sets=prediction_sets,
        set_sizes=set_sizes,
        difficulty_scores=difficulty_test,
        coverage=np.nan,
        target_coverage=1 - alpha,
        conditional_coverage=conditional_cov,
        efficiency_gain=efficiency_gain,
    )


def risk_control_conformal(
    model: BaseEstimator,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    risk_function: Callable,
    target_risk: float = 0.05,
) -> RiskControlResult:
    """Learn Then Test (LTT) risk control using conformal prediction.

    Calibrates a threshold to ensure a risk metric (e.g., false negative rate)
    stays below a target level with high probability.

    Args:
        model: Pre-trained model with predict_proba
        X_cal: Calibration features
        y_cal: Calibration labels
        X_test: Test features
        risk_function: Function computing risk for a given threshold.
            Should take (y_true, y_pred_proba, threshold) and return risk value
        target_risk: Maximum acceptable risk level (e.g., 0.05 for 5% FNR)

    Returns:
        RiskControlResult with calibrated threshold

    Example:
        >>> def fnr_risk(y_true, y_proba, threshold):
        ...     y_pred = (y_proba >= threshold).astype(int)
        ...     fn = ((y_true == 1) & (y_pred == 0)).sum()
        ...     return fn / (y_true == 1).sum()
        >>> result = risk_control_conformal(
        ...     model, X_cal, y_cal, X_test, fnr_risk, target_risk=0.05
        ... )
        >>> print(f"Calibrated threshold: {result.threshold:.3f}")

    Clinical Use Case:
        Control false negative rate to ensure no more than 5% of disease
        cases are missed. Model abstains (rejects) when uncertain.
    """
    # Get predicted probabilities on calibration set
    y_cal_proba = model.predict_proba(X_cal)[:, 1]  # Assume binary classification

    # Grid search over thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    risks = []

    for threshold in thresholds:
        risk = risk_function(y_cal, y_cal_proba, threshold)
        risks.append(risk)

    risks = np.array(risks)

    # Find highest threshold that satisfies risk constraint
    valid_thresholds = thresholds[risks <= target_risk]

    if len(valid_thresholds) > 0:
        calibrated_threshold = valid_thresholds.max()
        risk_controlled = True
    else:
        # No threshold satisfies constraint; use most conservative
        calibrated_threshold = thresholds[np.argmin(risks)]
        risk_controlled = False

    # Apply to test set
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Rejection: samples with probability close to threshold
    rejection_margin = 0.1
    uncertain_mask = np.abs(y_test_proba - calibrated_threshold) < rejection_margin
    n_rejected = uncertain_mask.sum()
    rejection_rate = n_rejected / len(X_test)

    # Empirical risk on calibration set at calibrated threshold
    empirical_risk = risk_function(y_cal, y_cal_proba, calibrated_threshold)

    return RiskControlResult(
        threshold=calibrated_threshold,
        empirical_risk=empirical_risk,
        target_risk=target_risk,
        risk_controlled=risk_controlled,
        n_rejected=int(n_rejected),
        rejection_rate=rejection_rate,
    )


def conformal_pvalue(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test_candidate: int,
) -> float:
    """Compute conformal p-value for a candidate label.

    The conformal p-value quantifies how "strange" it would be for a sample
    to have a given label, based on the calibration set.

    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        X_cal: Calibration features
        y_cal: Calibration labels
        X_test: Single test sample (shape: (1, n_features))
        y_test_candidate: Candidate label to test

    Returns:
        p-value in [0, 1]. Low values indicate the candidate label is unlikely.

    Example:
        >>> X_test_sample = X_test[[0]]  # Single sample
        >>> for label in [0, 1]:
        ...     pval = conformal_pvalue(model, X_train, y_train, X_cal, y_cal,
        ...                             X_test_sample, label)
        ...     print(f"P(Y={label}): {pval:.3f}")

    Clinical Use Case:
        Use p-values to rank candidate diagnoses by plausibility.
    """
    model.fit(X_train, y_train)

    # Get probabilities
    probs_cal = model.predict_proba(X_cal)
    probs_test = model.predict_proba(X_test)[0]

    classes = model.classes_
    y_test_idx = np.where(classes == y_test_candidate)[0][0]

    # Nonconformity score for candidate
    test_score = 1 - probs_test[y_test_idx]

    # Nonconformity scores for calibration set
    n_cal = len(y_cal)
    cal_scores = np.array(
        [1 - probs_cal[i, np.where(classes == y_cal[i])[0][0]] for i in range(n_cal)]
    )

    # p-value: proportion of calibration scores ≥ test score
    p_value = (np.sum(cal_scores >= test_score) + 1) / (n_cal + 1)

    return p_value
