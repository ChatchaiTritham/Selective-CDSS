"""
Counterfactual Explanations Module

Generates counterfactual explanations for clinical decision support systems.
Counterfactuals answer "what-if" questions: "What would need to change for
this patient to receive a different triage decision?"

This is particularly valuable in clinical settings because counterfactuals:
1. Provide actionable insights for clinicians
2. Suggest potential interventions
3. Identify modifiable risk factors
4. Support shared decision-making with patients

Theoretical Foundation:
    A counterfactual explanation is a minimal change to input features that
    would result in a different prediction. Formally, for instance x with
    prediction f(x) = y, we seek x' such that:

    1. f(x') = y' (desired outcome)
    2. distance(x, x') is minimized
    3. x' satisfies domain constraints (feasibility)
    4. Changes are actionable (modifiable features only)

Clinical Example:
    Original: Patient triaged as HIGH risk
    Counterfactual: "If systolic BP was 20 mmHg lower AND troponin was
    in normal range, patient would be triaged as LOW risk"

    This suggests:
    - Blood pressure control is critical
    - Cardiac biomarker elevation drives high-risk classification
    - Potential interventions: antihypertensives, cardiac workup

Author: Chatchai Tritham
Affiliation: Department of Computer Science and Information Technology,
             Faculty of Science, Naresuan University
Date: 2026-01-25
Version: 2.0.0 (XAI Enhancement)

References:
    - Wachter et al. (2018). Counterfactual explanations without opening
      the black box. Harvard Journal of Law & Technology.
    - Mothilal et al. (2020). Explaining machine learning classifiers through
      diverse counterfactual explanations. FAT*.
    - Verma et al. (2020). Counterfactual explanations for machine learning.
      ICML Tutorial.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cityblock, cosine, euclidean


@dataclass
class CounterfactualExample:
    """A single counterfactual explanation.

    Attributes:
        original: Original feature values
        counterfactual: Counterfactual feature values
        original_prediction: Original prediction
        counterfactual_prediction: Counterfactual prediction
        feature_changes: Dictionary of {feature: (old_value, new_value)}
        distance: Distance between original and counterfactual
        feasible: Whether counterfactual satisfies constraints
        actionable: Whether all changes are actionable
        feature_names: Names of features
    """

    original: np.ndarray
    counterfactual: np.ndarray
    original_prediction: Union[int, float]
    counterfactual_prediction: Union[int, float]
    feature_changes: Dict[str, Tuple[float, float]]
    distance: float
    feasible: bool
    actionable: bool
    feature_names: List[str]


@dataclass
class CounterfactualSet:
    """A diverse set of counterfactual explanations.

    Attributes:
        counterfactuals: List of counterfactual examples
        diversity_score: Diversity among counterfactuals
        coverage: Proportion of feature space covered
        num_counterfactuals: Number of counterfactuals
    """

    counterfactuals: List[CounterfactualExample]
    diversity_score: float
    coverage: float
    num_counterfactuals: int


@dataclass
class InterventionSuggestion:
    """Clinical intervention suggestions based on counterfactuals.

    Attributes:
        feature_name: Name of feature to modify
        current_value: Current value
        target_value: Target value (counterfactual)
        change_magnitude: Absolute change needed
        change_percentage: Percentage change needed
        priority: Priority ranking (1 = highest)
        actionable: Whether this intervention is clinically actionable
        intervention_type: Type of intervention (medication, lifestyle, etc.)
    """

    feature_name: str
    current_value: float
    target_value: float
    change_magnitude: float
    change_percentage: float
    priority: int
    actionable: bool
    intervention_type: str


def generate_counterfactual(
    model: Any,
    x: np.ndarray,
    feature_names: List[str],
    desired_class: Optional[int] = None,
    method: str = "gradient",
    distance_metric: str = "euclidean",
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    immutable_features: Optional[List[str]] = None,
    actionable_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-3,
    random_state: int = 42,
) -> CounterfactualExample:
    """Generate a counterfactual explanation for a single instance.

    Finds minimal changes to input features that would result in a different
    prediction while respecting domain constraints.

    Parameters:
        model: Trained classifier with predict or predict_proba method
        x: Original instance (1D array of features)
        feature_names: Names of features
        desired_class: Target class (if None, flip to opposite class)
        method: Optimization method ('gradient', 'random', 'genetic')
        distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        feature_ranges: Valid ranges for each feature {name: (min, max)}
        immutable_features: Features that cannot be changed (e.g., age, gender)
        actionable_features: Features that can be changed (if None, all except immutable)
        categorical_features: Categorical feature names
        max_iterations: Maximum optimization iterations
        learning_rate: Learning rate for gradient-based methods
        tolerance: Convergence tolerance
        random_state: Random seed

    Returns:
        CounterfactualExample object

    Example:
        >>> # Patient currently HIGH risk - find counterfactual for LOW risk
        >>> cf = generate_counterfactual(
        ...     model, patient_features, feature_names,
        ...     desired_class=0,  # LOW risk
        ...     immutable_features=['age', 'gender'],
        ...     feature_ranges={'sbp': (90, 200), 'hr': (40, 180)}
        ... )
        >>> print("Required changes for LOW risk triage:")
        >>> for feat, (old, new) in cf.feature_changes.items():
        ...     print(f"  {feat}: {old:.1f} → {new:.1f}")
    """
    np.random.seed(random_state)

    # Get original prediction
    if hasattr(model, 'predict_proba'):
        orig_proba = model.predict_proba(x.reshape(1, -1))[0]
        orig_pred = np.argmax(orig_proba)
    else:
        orig_pred = model.predict(x.reshape(1, -1))[0]

    # Determine desired class
    if desired_class is None:
        # Binary flip
        desired_class = 1 - orig_pred if orig_pred in [0, 1] else 0

    # Initialize immutable and actionable features
    if immutable_features is None:
        immutable_features = []
    if actionable_features is None:
        actionable_features = [f for f in feature_names if f not in immutable_features]
    if categorical_features is None:
        categorical_features = []

    # Create masks
    immutable_mask = np.array([f in immutable_features for f in feature_names])
    actionable_mask = np.array([f in actionable_features for f in feature_names])

    # Generate counterfactual based on method
    if method == "gradient":
        x_cf = _gradient_based_counterfactual(
            model,
            x,
            desired_class,
            actionable_mask,
            max_iterations,
            learning_rate,
            tolerance,
        )
    elif method == "random":
        x_cf = _random_search_counterfactual(
            model,
            x,
            desired_class,
            actionable_mask,
            feature_ranges,
            feature_names,
            max_iterations,
            random_state,
        )
    elif method == "genetic":
        x_cf = _genetic_counterfactual(
            model,
            x,
            desired_class,
            actionable_mask,
            feature_ranges,
            feature_names,
            max_iterations,
            random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Keep immutable features unchanged
    x_cf[immutable_mask] = x[immutable_mask]

    # Apply feature range constraints
    if feature_ranges is not None:
        for i, feat in enumerate(feature_names):
            if feat in feature_ranges:
                min_val, max_val = feature_ranges[feat]
                x_cf[i] = np.clip(x_cf[i], min_val, max_val)

    # Get counterfactual prediction
    if hasattr(model, 'predict_proba'):
        cf_proba = model.predict_proba(x_cf.reshape(1, -1))[0]
        cf_pred = np.argmax(cf_proba)
    else:
        cf_pred = model.predict(x_cf.reshape(1, -1))[0]

    # Calculate distance
    if distance_metric == "euclidean":
        dist = euclidean(x, x_cf)
    elif distance_metric == "manhattan":
        dist = cityblock(x, x_cf)
    elif distance_metric == "cosine":
        dist = cosine(x, x_cf)
    else:
        dist = np.linalg.norm(x - x_cf)

    # Identify changes
    changes = {}
    for i, feat in enumerate(feature_names):
        if not np.isclose(x[i], x_cf[i], atol=tolerance):
            changes[feat] = (float(x[i]), float(x_cf[i]))

    # Check feasibility and actionability
    feasible = True
    if feature_ranges is not None:
        for feat, (old, new) in changes.items():
            if feat in feature_ranges:
                min_val, max_val = feature_ranges[feat]
                if not (min_val <= new <= max_val):
                    feasible = False
                    break

    actionable = all(feat in actionable_features for feat in changes.keys())

    return CounterfactualExample(
        original=x,
        counterfactual=x_cf,
        original_prediction=int(orig_pred),
        counterfactual_prediction=int(cf_pred),
        feature_changes=changes,
        distance=float(dist),
        feasible=feasible,
        actionable=actionable,
        feature_names=feature_names,
    )


def generate_diverse_counterfactuals(
    model: Any,
    x: np.ndarray,
    feature_names: List[str],
    num_counterfactuals: int = 5,
    diversity_weight: float = 1.0,
    **kwargs,
) -> CounterfactualSet:
    """Generate a diverse set of counterfactual explanations.

    Creates multiple counterfactuals that differ from each other, providing
    diverse paths to achieve the desired outcome. This is valuable clinically
    because it offers multiple intervention strategies.

    Parameters:
        model: Trained classifier
        x: Original instance
        feature_names: Feature names
        num_counterfactuals: Number of counterfactuals to generate
        diversity_weight: Weight for diversity term in optimization
        **kwargs: Additional arguments for generate_counterfactual()

    Returns:
        CounterfactualSet with diverse explanations

    Example:
        >>> cf_set = generate_diverse_counterfactuals(
        ...     model, patient, features, num_counterfactuals=3
        ... )
        >>> print(f"Found {cf_set.num_counterfactuals} diverse explanations:")
        >>> for i, cf in enumerate(cf_set.counterfactuals):
        ...     print(f"\nOption {i+1}:")
        ...     for feat, (old, new) in cf.feature_changes.items():
        ...         print(f"  {feat}: {old:.1f} → {new:.1f}")
    """
    counterfactuals = []

    for i in range(num_counterfactuals):
        # Add randomization to promote diversity
        seed = kwargs.get('random_state', 42) + i * 100

        # Generate counterfactual
        cf = generate_counterfactual(
            model,
            x,
            feature_names,
            random_state=seed,
            **{k: v for k, v in kwargs.items() if k != 'random_state'},
        )

        counterfactuals.append(cf)

    # Calculate diversity score (mean pairwise distance)
    if len(counterfactuals) > 1:
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                dist = euclidean(
                    counterfactuals[i].counterfactual, counterfactuals[j].counterfactual
                )
                distances.append(dist)
        diversity_score = np.mean(distances)
    else:
        diversity_score = 0.0

    # Calculate coverage (proportion of features changed across all CFs)
    all_changed_features = set()
    for cf in counterfactuals:
        all_changed_features.update(cf.feature_changes.keys())
    coverage = len(all_changed_features) / len(feature_names)

    return CounterfactualSet(
        counterfactuals=counterfactuals,
        diversity_score=float(diversity_score),
        coverage=float(coverage),
        num_counterfactuals=len(counterfactuals),
    )


def minimal_counterfactual(
    model: Any,
    x: np.ndarray,
    feature_names: List[str],
    max_features_changed: int = 3,
    **kwargs,
) -> CounterfactualExample:
    """Generate counterfactual with minimal number of feature changes.

    Finds the smallest set of features that, when changed, results in the
    desired prediction. This is clinically valuable because it identifies
    the most critical modifiable risk factors.

    Parameters:
        model: Trained classifier
        x: Original instance
        feature_names: Feature names
        max_features_changed: Maximum number of features to change
        **kwargs: Additional arguments for generate_counterfactual()

    Returns:
        CounterfactualExample with minimal changes

    Example:
        >>> minimal_cf = minimal_counterfactual(model, patient, features, max_features_changed=2)
        >>> print(f"Minimum changes needed: {len(minimal_cf.feature_changes)}")
        >>> for feat, (old, new) in minimal_cf.feature_changes.items():
        ...     print(f"  {feat}: {old:.1f} → {new:.1f}")
    """
    best_cf = None
    min_changes = float('inf')

    # Try different random seeds to find minimal solution
    for seed in range(10):
        cf = generate_counterfactual(
            model,
            x,
            feature_names,
            random_state=kwargs.get('random_state', 42) + seed,
            **{k: v for k, v in kwargs.items() if k != 'random_state'},
        )

        num_changes = len(cf.feature_changes)

        if num_changes < min_changes and num_changes <= max_features_changed:
            min_changes = num_changes
            best_cf = cf

    if best_cf is None:
        # Fallback to regular counterfactual
        best_cf = generate_counterfactual(model, x, feature_names, **kwargs)

    return best_cf


def actionable_interventions(
    counterfactual: CounterfactualExample,
    intervention_types: Optional[Dict[str, str]] = None,
    clinical_priority: Optional[Dict[str, int]] = None,
) -> List[InterventionSuggestion]:
    """Generate actionable clinical intervention suggestions from counterfactual.

    Translates counterfactual changes into concrete clinical interventions
    ranked by priority and feasibility.

    Parameters:
        counterfactual: Counterfactual explanation
        intervention_types: Mapping {feature: intervention_type}
                          e.g., {'sbp': 'medication', 'bmi': 'lifestyle'}
        clinical_priority: Priority ranking {feature: priority_score}

    Returns:
        List of InterventionSuggestion objects sorted by priority

    Example:
        >>> interventions = actionable_interventions(cf)
        >>> print("Recommended interventions (in priority order):")
        >>> for interv in interventions:
        ...     print(f"{interv.priority}. {interv.feature_name}:")
        ...     print(f"   Current: {interv.current_value:.1f}")
        ...     print(f"   Target: {interv.target_value:.1f}")
        ...     print(f"   Change: {interv.change_magnitude:.1f} ({interv.change_percentage:.1f}%)")
        ...     print(f"   Type: {interv.intervention_type}")
    """
    if intervention_types is None:
        intervention_types = {}

    if clinical_priority is None:
        clinical_priority = {}

    suggestions = []

    for feat, (old_val, new_val) in counterfactual.feature_changes.items():
        change_mag = abs(new_val - old_val)
        change_pct = (
            (abs(new_val - old_val) / abs(old_val) * 100) if old_val != 0 else 0
        )

        priority = clinical_priority.get(feat, 999)  # Default low priority
        interv_type = intervention_types.get(feat, "unspecified")

        suggestion = InterventionSuggestion(
            feature_name=feat,
            current_value=float(old_val),
            target_value=float(new_val),
            change_magnitude=float(change_mag),
            change_percentage=float(change_pct),
            priority=int(priority),
            actionable=counterfactual.actionable,
            intervention_type=interv_type,
        )

        suggestions.append(suggestion)

    # Sort by priority (lower number = higher priority)
    suggestions.sort(key=lambda x: x.priority)

    return suggestions


def whatif_analysis(
    model: Any,
    x: np.ndarray,
    feature_names: List[str],
    feature_to_vary: str,
    value_range: Tuple[float, float],
    num_points: int = 20,
) -> pd.DataFrame:
    """Perform what-if analysis by varying a single feature.

    Analyzes how predictions change as a single feature varies across a range
    of values, holding all other features constant.

    Parameters:
        model: Trained classifier
        x: Original instance
        feature_names: Feature names
        feature_to_vary: Name of feature to vary
        value_range: (min, max) range for the feature
        num_points: Number of points to sample

    Returns:
        DataFrame with feature values and corresponding predictions

    Example:
        >>> # How does systolic BP affect risk?
        >>> whatif_df = whatif_analysis(
        ...     model, patient, features,
        ...     feature_to_vary='systolic_bp',
        ...     value_range=(90, 200),
        ...     num_points=30
        ... )
        >>> # Plot results
        >>> whatif_df.plot(x='systolic_bp', y='prediction_proba')
    """
    feature_idx = feature_names.index(feature_to_vary)

    # Generate range of values
    values = np.linspace(value_range[0], value_range[1], num_points)

    results = []

    for val in values:
        # Create modified instance
        x_modified = x.copy()
        x_modified[feature_idx] = val

        # Get prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(x_modified.reshape(1, -1))[0]
            pred = np.argmax(proba)
            results.append(
                {
                    feature_to_vary: val,
                    'prediction': pred,
                    'prediction_proba': proba[1] if len(proba) == 2 else proba,
                }
            )
        else:
            pred = model.predict(x_modified.reshape(1, -1))[0]
            results.append({feature_to_vary: val, 'prediction': pred})

    return pd.DataFrame(results)


def counterfactual_stability(
    model: Any,
    counterfactual: CounterfactualExample,
    noise_level: float = 0.01,
    num_trials: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """Assess stability of counterfactual explanation.

    Tests whether small perturbations to the counterfactual still result in
    the desired prediction. High stability indicates robust counterfactuals.

    Parameters:
        model: Trained classifier
        counterfactual: Counterfactual to test
        noise_level: Standard deviation of Gaussian noise
        num_trials: Number of perturbation trials
        random_state: Random seed

    Returns:
        Dictionary with stability metrics

    Example:
        >>> stability = counterfactual_stability(model, cf, noise_level=0.02)
        >>> print(f"Stability: {stability['stability_score']:.2%}")
        >>> print(f"Prediction maintained in {stability['success_rate']:.2%} of trials")
    """
    np.random.seed(random_state)

    x_cf = counterfactual.counterfactual
    desired_pred = counterfactual.counterfactual_prediction

    successes = 0

    for _ in range(num_trials):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=x_cf.shape)
        x_perturbed = x_cf + noise

        # Get prediction
        if hasattr(model, 'predict_proba'):
            pred = np.argmax(model.predict_proba(x_perturbed.reshape(1, -1))[0])
        else:
            pred = model.predict(x_perturbed.reshape(1, -1))[0]

        if pred == desired_pred:
            successes += 1

    stability_score = successes / num_trials

    return {
        'stability_score': float(stability_score),
        'success_rate': float(stability_score),
        'num_trials': num_trials,
        'noise_level': noise_level,
    }


def _gradient_based_counterfactual(
    model: Any,
    x: np.ndarray,
    desired_class: int,
    actionable_mask: np.ndarray,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
) -> np.ndarray:
    """Generate counterfactual using gradient-based optimization."""
    x_cf = x.copy()

    for iteration in range(max_iterations):
        # Compute gradient (finite differences)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(x_cf.reshape(1, -1))[0]
            current_class = np.argmax(proba)

            # If already at desired class, stop
            if current_class == desired_class:
                break

            # Compute gradient via finite differences
            gradients = np.zeros_like(x_cf)
            epsilon = 1e-5

            for i in range(len(x_cf)):
                if actionable_mask[i]:
                    x_plus = x_cf.copy()
                    x_plus[i] += epsilon
                    proba_plus = model.predict_proba(x_plus.reshape(1, -1))[0]

                    gradients[i] = (
                        proba_plus[desired_class] - proba[desired_class]
                    ) / epsilon

            # Update (gradient ascent to maximize desired class probability)
            x_cf[actionable_mask] += learning_rate * gradients[actionable_mask]

        else:
            # For non-probabilistic models, use random search
            x_cf = _random_search_counterfactual(
                model, x, desired_class, actionable_mask, None, None, max_iterations, 42
            )
            break

    return x_cf


def _random_search_counterfactual(
    model: Any,
    x: np.ndarray,
    desired_class: int,
    actionable_mask: np.ndarray,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]],
    feature_names: Optional[List[str]],
    max_iterations: int,
    random_state: int,
) -> np.ndarray:
    """Generate counterfactual using random search."""
    np.random.seed(random_state)

    best_x_cf = x.copy()
    best_distance = float('inf')

    for _ in range(max_iterations):
        # Random perturbation
        x_cf = x.copy()
        perturbation = np.random.normal(0, 0.5, size=x.shape)
        x_cf[actionable_mask] += perturbation[actionable_mask]

        # Apply range constraints
        if feature_ranges is not None and feature_names is not None:
            for i, feat in enumerate(feature_names):
                if feat in feature_ranges:
                    min_val, max_val = feature_ranges[feat]
                    x_cf[i] = np.clip(x_cf[i], min_val, max_val)

        # Check if desired class
        if hasattr(model, 'predict_proba'):
            pred = np.argmax(model.predict_proba(x_cf.reshape(1, -1))[0])
        else:
            pred = model.predict(x_cf.reshape(1, -1))[0]

        if pred == desired_class:
            dist = euclidean(x, x_cf)
            if dist < best_distance:
                best_distance = dist
                best_x_cf = x_cf

    return best_x_cf


def _genetic_counterfactual(
    model: Any,
    x: np.ndarray,
    desired_class: int,
    actionable_mask: np.ndarray,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]],
    feature_names: Optional[List[str]],
    max_iterations: int,
    random_state: int,
    population_size: int = 50,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """Generate counterfactual using genetic algorithm."""
    np.random.seed(random_state)

    # Initialize population
    population = [x.copy() for _ in range(population_size)]

    for generation in range(max_iterations // population_size):
        # Evaluate fitness
        fitness_scores = []

        for individual in population:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(individual.reshape(1, -1))[0]
                pred = np.argmax(proba)

                # Fitness: probability of desired class - distance from original
                if pred == desired_class:
                    fitness = proba[desired_class] - 0.1 * euclidean(x, individual)
                else:
                    fitness = -euclidean(x, individual)
            else:
                pred = model.predict(individual.reshape(1, -1))[0]
                fitness = 1.0 if pred == desired_class else -1.0

            fitness_scores.append(fitness)

        # Select parents (tournament selection)
        fitness_scores = np.array(fitness_scores)
        parents_idx = np.argsort(-fitness_scores)[: population_size // 2]
        parents = [population[i] for i in parents_idx]

        # Create offspring
        offspring = []
        for _ in range(population_size - len(parents)):
            # Crossover
            idx = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[idx[0]], parents[idx[1]]
            child = parent1.copy()
            crossover_mask = np.random.rand(len(x)) > 0.5
            child[crossover_mask & actionable_mask] = parent2[
                crossover_mask & actionable_mask
            ]

            # Mutation
            mutation_mask = (np.random.rand(len(x)) < mutation_rate) & actionable_mask
            child[mutation_mask] += np.random.normal(0, 0.3, size=np.sum(mutation_mask))

            offspring.append(child)

        # New population
        population = parents + offspring

    # Return best individual
    best_idx = np.argmax(fitness_scores)
    return population[best_idx]
