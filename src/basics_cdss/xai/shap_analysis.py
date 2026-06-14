"""
SHAP (SHapley Additive exPlanations) Analysis Module

Implements Shapley value-based feature importance analysis rooted in cooperative
game theory. In the context of clinical decision support, features (symptoms,
vital signs, lab results) are treated as "players" in a cooperative game where
the "payoff" is the prediction accuracy.

Theoretical Foundation:
    Shapley values provide a fair attribution of the prediction to each feature
    based on their marginal contribution across all possible coalitions. This
    aligns with the clinical intuition that:

    - Critical symptoms (e.g., chest pain, severe dyspnea) = Major players
      with high Shapley values, contributing significantly to triage decisions

    - Uncertain/ambiguous symptoms = Minor players with low Shapley values,
      contributing minimally to final predictions

    - Interactions between symptoms are captured through Shapley interaction values

Mathematical Foundation:
    For feature i, the Shapley value φᵢ is:

    φᵢ = Σ_{S⊆N\\{i}} [|S|!(|N|-|S|-1)!]/|N|! × [v(S∪{i}) - v(S)]

    where:
    - N is the set of all features
    - S is a coalition (subset) of features
    - v(S) is the prediction with features in S
    - The sum is over all possible coalitions not containing i

Author: Chatchai Tritham
Affiliation: Department of Computer Science and Information Technology,
             Faculty of Science, Naresuan University
Date: 2026-01-25
Version: 2.0.0 (XAI Enhancement)

References:
    - Shapley, L. S. (1953). A value for n-person games.
    - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
      model predictions. NeurIPS.
    - Molnar, C. (2022). Interpretable Machine Learning.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not installed. Install with: pip install shap")


@dataclass
class SHAPValues:
    """SHAP values and associated metadata.

    Attributes:
        values: SHAP values array (n_samples, n_features)
        base_value: Expected value of model output
        data: Original feature values
        feature_names: Names of features
        expected_value: Expected model output (alias for base_value)
    """

    values: np.ndarray
    base_value: Union[float, np.ndarray]
    data: np.ndarray
    feature_names: List[str]
    expected_value: Union[float, np.ndarray] = None

    def __post_init__(self):
        if self.expected_value is None:
            self.expected_value = self.base_value


@dataclass
class SHAPInteractionValues:
    """SHAP interaction values for feature interactions.

    Attributes:
        values: Interaction values array (n_samples, n_features, n_features)
        base_value: Expected value
        data: Original feature values
        feature_names: Names of features
    """

    values: np.ndarray
    base_value: Union[float, np.ndarray]
    data: np.ndarray
    feature_names: List[str]


@dataclass
class FeatureImportance:
    """Feature importance ranking based on SHAP values.

    Attributes:
        feature_names: Names of features
        importance_scores: Mean absolute SHAP values
        importance_rank: Rank (1 = most important)
        critical_features: Features above importance threshold
        non_critical_features: Features below importance threshold
        threshold: Threshold for critical/non-critical classification
    """

    feature_names: List[str]
    importance_scores: np.ndarray
    importance_rank: np.ndarray
    critical_features: List[str]
    non_critical_features: List[str]
    threshold: float


@dataclass
class GameTheoreticExplanation:
    """Game-theoretic interpretation of SHAP values.

    Attributes:
        major_players: Features with high contribution (critical symptoms)
        minor_players: Features with low contribution (uncertain symptoms)
        coalition_values: Contribution by feature coalitions
        marginal_contributions: Marginal contribution of each feature
        feature_interactions: Pairwise feature interactions
    """

    major_players: Dict[str, float]
    minor_players: Dict[str, float]
    coalition_values: Dict[str, float]
    marginal_contributions: Dict[str, float]
    feature_interactions: pd.DataFrame


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_type: str = "auto",
    algorithm: str = "auto",
    background_data: Optional[np.ndarray] = None,
    n_background_samples: int = 100,
    check_additivity: bool = True,
    random_state: int = 42,
) -> SHAPValues:
    """Compute SHAP values for model predictions.

    Automatically selects appropriate SHAP explainer based on model type:
    - TreeExplainer for tree-based models (XGBoost, Random Forest, etc.)
    - LinearExplainer for linear models
    - KernelExplainer for any model (model-agnostic)
    - DeepExplainer for neural networks

    Parameters:
        model: Trained model (sklearn, xgboost, etc.)
        X: Feature matrix (n_samples, n_features)
        feature_names: Names of features (optional)
        model_type: Type of model ('tree', 'linear', 'kernel', 'deep', 'auto')
        algorithm: SHAP algorithm ('auto', 'permutation', 'partition', 'tree')
        background_data: Background dataset for KernelExplainer
        n_background_samples: Number of background samples to use
        check_additivity: Whether to check SHAP values sum to prediction difference
        random_state: Random seed for reproducibility

    Returns:
        SHAPValues object containing values, base values, and metadata

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> shap_vals = compute_shap_values(model, X_test, feature_names=features)
        >>> print(f"Critical features: {shap_vals.feature_names[:5]}")

    Note:
        Game Theory Interpretation:
        - High SHAP values = Major players (critical symptoms like chest pain)
        - Low SHAP values = Minor players (uncertain symptoms)
        - Sign indicates direction of contribution (positive/negative)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not installed. Install with: pip install shap")

    np.random.seed(random_state)

    # Generate feature names if not provided
    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Select explainer based on model type
    if model_type == "auto":
        model_type = _detect_model_type(model)

    # Create explainer
    if model_type == "tree":
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation=(
                algorithm if algorithm != "auto" else "tree_path_dependent"
            ),
            model_output="raw",
        )
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)

        # Handle multi-class output (use positive class for binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

    elif model_type == "deep":
        if background_data is None:
            # Sample background data
            indices = np.random.choice(
                X.shape[0], min(n_background_samples, X.shape[0]), replace=False
            )
            background_data = X[indices]

        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

    else:  # kernel (model-agnostic)
        if background_data is None:
            # Sample background data for kernel SHAP
            indices = np.random.choice(
                X.shape[0], min(n_background_samples, X.shape[0]), replace=False
            )
            background_data = X[indices]

        # Create predict function
        if hasattr(model, 'predict_proba'):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = model.predict

        explainer = shap.KernelExplainer(predict_fn, background_data)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

    return SHAPValues(
        values=shap_values,
        base_value=base_value,
        data=X,
        feature_names=feature_names,
        expected_value=base_value,
    )


def compute_shap_interaction_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> SHAPInteractionValues:
    """Compute SHAP interaction values for feature interactions.

    SHAP interaction values capture how features interact to influence predictions.
    For features i and j, the interaction value quantifies how the effect of
    feature i depends on the value of feature j.

    Parameters:
        model: Trained tree-based model (XGBoost, Random Forest, etc.)
        X: Feature matrix (n_samples, n_features)
        feature_names: Names of features
        random_state: Random seed

    Returns:
        SHAPInteractionValues object

    Example:
        >>> interactions = compute_shap_interaction_values(model, X_test, features)
        >>> # Interaction between "chest_pain" and "troponin_level"
        >>> idx1 = features.index("chest_pain")
        >>> idx2 = features.index("troponin_level")
        >>> interaction_strength = np.abs(interactions.values[:, idx1, idx2]).mean()

    Note:
        Only available for tree-based models. Interaction values form a matrix
        where diagonal elements are main effects and off-diagonal elements are
        pairwise interactions.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not installed")

    np.random.seed(random_state)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Tree SHAP supports interaction values
    try:
        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(X)

        # Handle multi-class
        if isinstance(shap_interaction_values, list):
            shap_interaction_values = shap_interaction_values[1]

        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

    except Exception as e:
        raise ValueError(f"Model does not support interaction values: {e}")

    return SHAPInteractionValues(
        values=shap_interaction_values,
        base_value=base_value,
        data=X,
        feature_names=feature_names,
    )


def feature_importance_ranking(
    shap_values: SHAPValues,
    method: str = "mean_abs",
    threshold_percentile: float = 75.0,
) -> FeatureImportance:
    """Rank features by importance based on SHAP values.

    Classifies features into:
    - Critical features (major players): High SHAP values, strong influence
    - Non-critical features (minor players): Low SHAP values, weak influence

    Parameters:
        shap_values: SHAP values from compute_shap_values()
        method: Importance metric ('mean_abs', 'mean', 'max_abs', 'std')
        threshold_percentile: Percentile for critical/non-critical split (0-100)

    Returns:
        FeatureImportance object with rankings and classifications

    Example:
        >>> importance = feature_importance_ranking(shap_vals, threshold_percentile=80)
        >>> print("Major players (critical symptoms):")
        >>> for feat in importance.critical_features:
        ...     print(f"  {feat}: {importance.importance_scores[...]:.4f}")
        >>> print("Minor players (uncertain symptoms):")
        >>> for feat in importance.non_critical_features:
        ...     print(f"  {feat}: {importance.importance_scores[...]:.4f}")
    """
    # Compute importance scores
    if method == "mean_abs":
        scores = np.abs(shap_values.values).mean(axis=0)
    elif method == "mean":
        scores = shap_values.values.mean(axis=0)
    elif method == "max_abs":
        scores = np.abs(shap_values.values).max(axis=0)
    elif method == "std":
        scores = shap_values.values.std(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle multi-dimensional scores (e.g., multi-class/multi-output)
    # If scores is 2D, take mean across second dimension
    if len(scores.shape) > 1:
        scores = scores.mean(axis=-1)

    # Ensure 1D
    scores = np.asarray(scores).flatten()

    # Rank features
    ranks = stats.rankdata(
        -scores, method='ordinal'
    )  # Higher score = lower rank number

    # Determine threshold
    threshold = np.percentile(scores, threshold_percentile)

    # Classify features
    critical_mask = scores >= threshold
    critical_features = [
        shap_values.feature_names[i] for i in np.where(critical_mask)[0]
    ]
    non_critical_features = [
        shap_values.feature_names[i] for i in np.where(~critical_mask)[0]
    ]

    return FeatureImportance(
        feature_names=shap_values.feature_names,
        importance_scores=scores,
        importance_rank=ranks,
        critical_features=critical_features,
        non_critical_features=non_critical_features,
        threshold=threshold,
    )


def shapley_coalition_values(
    shap_values: SHAPValues, sample_idx: int = 0, top_k: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compute coalition values from game-theoretic perspective.

    Interprets SHAP values as contributions from feature coalitions in a
    cooperative game. This provides intuition about which combinations of
    symptoms contribute most to the triage decision.

    Parameters:
        shap_values: SHAP values object
        sample_idx: Index of sample to explain
        top_k: Number of top coalitions to return

    Returns:
        Dictionary with coalition information:
        - 'individual_contributions': Each feature's standalone contribution
        - 'top_coalitions': Most impactful feature combinations
        - 'cumulative_contribution': Running sum of contributions

    Example:
        >>> coalitions = shapley_coalition_values(shap_vals, sample_idx=0, top_k=5)
        >>> for feat, value in coalitions['individual_contributions'].items():
        ...     print(f"{feat}: {value:.4f}")
    """
    values = shap_values.values[sample_idx]
    features = shap_values.feature_names

    # Handle multi-output (binary classification)
    if len(values.shape) > 1:
        values = values[:, 1] if values.shape[1] > 1 else values.flatten()

    # Individual contributions
    individual = {features[i]: float(values[i]) for i in range(len(features))}

    # Sort by absolute value
    sorted_indices = np.argsort(-np.abs(values))

    # Top coalitions (simplified: single features and pairs)
    top_single = {features[i]: float(values[i]) for i in sorted_indices[:top_k]}

    # Cumulative contribution
    cumulative = {}
    base_val = shap_values.base_value
    # Handle multi-output base value
    if isinstance(base_val, np.ndarray) and len(base_val.shape) > 0:
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
    running_sum = float(base_val)
    for i in sorted_indices:
        running_sum += float(values[i])
        cumulative[features[i]] = running_sum

    return {
        'individual_contributions': individual,
        'top_coalitions': top_single,
        'cumulative_contribution': cumulative,
        'base_value': float(base_val),
    }


def game_theoretic_explanation(
    shap_values: SHAPValues,
    sample_idx: int = 0,
    major_player_percentile: float = 75.0,
    interaction_values: Optional[SHAPInteractionValues] = None,
) -> GameTheoreticExplanation:
    """Generate game-theoretic explanation of prediction.

    Frames the explanation in terms of:
    - Major players: Features with high Shapley values (critical symptoms)
    - Minor players: Features with low Shapley values (uncertain symptoms)
    - Coalitions: Groups of features working together
    - Interactions: How features influence each other

    This aligns with clinical reasoning where certain symptoms are clearly
    diagnostic (major players) while others are ambiguous (minor players).

    Parameters:
        shap_values: SHAP values object
        sample_idx: Sample to explain
        major_player_percentile: Threshold for major/minor classification
        interaction_values: Optional interaction values

    Returns:
        GameTheoreticExplanation object

    Example:
        >>> explanation = game_theoretic_explanation(shap_vals, sample_idx=0)
        >>> print("Major players (critical for this decision):")
        >>> for player, value in explanation.major_players.items():
        ...     print(f"  {player}: Shapley value = {value:.4f}")
        >>> print("Minor players (uncertain/ambiguous):")
        >>> for player, value in explanation.minor_players.items():
        ...     print(f"  {player}: Shapley value = {value:.4f}")
    """
    values = shap_values.values[sample_idx]
    features = shap_values.feature_names

    # Handle multi-output (binary classification)
    if len(values.shape) > 1:
        values = values[:, 1] if values.shape[1] > 1 else values.flatten()

    # Determine threshold
    abs_values = np.abs(values)
    threshold = np.percentile(abs_values, major_player_percentile)

    # Classify players
    major_mask = abs_values >= threshold
    major_players = {features[i]: float(values[i]) for i in np.where(major_mask)[0]}
    minor_players = {features[i]: float(values[i]) for i in np.where(~major_mask)[0]}

    # Coalition values
    coalitions = shapley_coalition_values(shap_values, sample_idx)

    # Marginal contributions (sorted by absolute value)
    marginal = {features[i]: float(values[i]) for i in np.argsort(-abs_values)}

    # Feature interactions
    if interaction_values is not None:
        interactions_matrix = interaction_values.values[sample_idx]
        # Handle multi-output (binary classification)
        if len(interactions_matrix.shape) == 3:
            interactions_matrix = interactions_matrix[:, :, 1]  # Positive class
        # Create DataFrame of interactions
        interaction_df = pd.DataFrame(
            interactions_matrix, index=features, columns=features
        )
    else:
        interaction_df = pd.DataFrame()

    return GameTheoreticExplanation(
        major_players=major_players,
        minor_players=minor_players,
        coalition_values=coalitions,
        marginal_contributions=marginal,
        feature_interactions=interaction_df,
    )


def stratified_shap_analysis(
    shap_values: SHAPValues,
    strata: np.ndarray,
    strata_names: Optional[List[str]] = None,
) -> Dict[str, FeatureImportance]:
    """Compute SHAP-based feature importance stratified by subgroups.

    Analyzes whether feature importance differs across patient subgroups
    (e.g., age groups, risk tiers, diagnoses). This can reveal whether
    certain symptoms are critical for some patient populations but not others.

    Parameters:
        shap_values: SHAP values object
        strata: Group labels for each sample
        strata_names: Names for strata (optional)

    Returns:
        Dictionary mapping stratum name to FeatureImportance

    Example:
        >>> # Analyze by risk tier
        >>> stratified = stratified_shap_analysis(shap_vals, risk_tiers)
        >>> for tier, importance in stratified.items():
        ...     print(f"\n{tier} risk patients:")
        ...     print(f"  Critical features: {importance.critical_features}")
    """
    unique_strata = np.unique(strata)

    if strata_names is None:
        strata_names = [str(s) for s in unique_strata]

    results = {}

    for stratum, name in zip(unique_strata, strata_names):
        mask = strata == stratum

        # Create SHAP values for this stratum
        stratum_shap = SHAPValues(
            values=shap_values.values[mask],
            base_value=shap_values.base_value,
            data=shap_values.data[mask],
            feature_names=shap_values.feature_names,
        )

        # Compute importance
        importance = feature_importance_ranking(stratum_shap)
        results[name] = importance

    return results


def temporal_shap_analysis(
    shap_values_by_time: List[SHAPValues], time_points: List[str]
) -> pd.DataFrame:
    """Analyze how feature importance changes over time.

    Useful for understanding concept drift or seasonal variations in which
    symptoms are most predictive.

    Parameters:
        shap_values_by_time: List of SHAP values for different time periods
        time_points: Labels for time periods (e.g., ['2020', '2021', '2022'])

    Returns:
        DataFrame with feature importance over time

    Example:
        >>> temporal_df = temporal_shap_analysis(shap_by_year, years)
        >>> # Plot importance trends
        >>> temporal_df.T.plot(figsize=(12, 6))
    """
    importance_by_time = []

    for shap_vals, time_label in zip(shap_values_by_time, time_points):
        importance = feature_importance_ranking(shap_vals)
        importance_dict = {
            'time': time_label,
            **{
                feat: score
                for feat, score in zip(
                    importance.feature_names, importance.importance_scores
                )
            },
        }
        importance_by_time.append(importance_dict)

    return pd.DataFrame(importance_by_time).set_index('time')


def shap_based_feature_selection(
    shap_values: SHAPValues,
    n_features: Optional[int] = None,
    importance_threshold: Optional[float] = None,
    method: str = "mean_abs",
) -> List[str]:
    """Select features based on SHAP importance.

    Provides data-driven feature selection by identifying features with
    consistently high Shapley values.

    Parameters:
        shap_values: SHAP values object
        n_features: Number of top features to select (optional)
        importance_threshold: Minimum importance threshold (optional)
        method: Importance metric

    Returns:
        List of selected feature names

    Example:
        >>> selected = shap_based_feature_selection(shap_vals, n_features=10)
        >>> print(f"Selected features: {selected}")
    """
    importance = feature_importance_ranking(shap_values, method=method)

    if n_features is not None:
        # Select top N features
        top_indices = np.argsort(-importance.importance_scores)[:n_features]
        selected = [importance.feature_names[i] for i in top_indices]
    elif importance_threshold is not None:
        # Select features above threshold
        mask = importance.importance_scores >= importance_threshold
        selected = [feat for feat, m in zip(importance.feature_names, mask) if m]
    else:
        # Default: use critical features
        selected = importance.critical_features

    return selected


def _detect_model_type(model: Any) -> str:
    """Detect model type for SHAP explainer selection."""
    model_class = model.__class__.__name__.lower()

    if any(
        tree in model_class for tree in ['tree', 'forest', 'xgb', 'lgb', 'catboost']
    ):
        return 'tree'
    elif any(
        linear in model_class for linear in ['linear', 'logistic', 'ridge', 'lasso']
    ):
        return 'linear'
    elif any(
        deep in model_class for deep in ['neural', 'keras', 'torch', 'sequential']
    ):
        return 'deep'
    else:
        return 'kernel'  # Default to model-agnostic


# Convenience function for quick analysis
def explain_prediction(
    model: Any,
    X: np.ndarray,
    sample_idx: int,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    compute_interactions: bool = False,
) -> GameTheoreticExplanation:
    """Quick explanation of a single prediction using SHAP.

    Convenience function that computes SHAP values and generates a
    game-theoretic explanation in one call.

    Parameters:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        feature_names: Feature names
        background_data: Background data for kernel SHAP
        compute_interactions: Whether to compute interaction values

    Returns:
        GameTheoreticExplanation for the specified sample

    Example:
        >>> explanation = explain_prediction(model, X_test, 0, feature_names)
        >>> print("Why was this patient triaged as high-risk?")
        >>> for symptom, value in explanation.major_players.items():
        ...     direction = "increases" if value > 0 else "decreases"
        ...     print(f"  - {symptom} {direction} risk (Shapley: {value:.4f})")
    """
    # Compute SHAP values
    shap_vals = compute_shap_values(
        model, X, feature_names, background_data=background_data
    )

    # Compute interactions if requested
    interactions = None
    if compute_interactions:
        try:
            interactions = compute_shap_interaction_values(model, X, feature_names)
        except:
            warnings.warn("Could not compute interaction values for this model")

    # Generate explanation
    explanation = game_theoretic_explanation(
        shap_vals, sample_idx=sample_idx, interaction_values=interactions
    )

    return explanation
