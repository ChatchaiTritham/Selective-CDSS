"""Causal-specific evaluation metrics.

This module provides metrics for evaluating causal models, interventions,
and confounding adjustment quality.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error


def causal_consistency_score(
    data: pd.DataFrame,
    graph: 'CausalGraph',
    test: str = 'regression',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate if data is consistent with causal graph structure.

    Tests conditional independence relationships implied by graph
    using d-separation criterion.

    Args:
        data: Observational dataset
        graph: Causal graph
        test: Test type ('regression', 'correlation')
        alpha: Significance level for independence tests

    Returns:
        Dictionary with consistency scores:
            - consistency_score: Proportion of satisfied independence tests
            - n_tests: Number of tests performed
            - violations: List of violated independence relationships
            - passed_tests: Number of passed tests

    Example:
        >>> from basics_cdss.causal import CausalGraph, causal_consistency_score
        >>>
        >>> score = causal_consistency_score(
        ...     data=generated_data,
        ...     graph=causal_graph
        ... )
        >>> print(f"Consistency: {score['consistency_score']:.2%}")
    """
    violations = []
    passed_tests = 0
    total_tests = 0

    # Get all variables
    variables = list(graph.graph.nodes())

    # Test pairwise conditional independencies
    for i, var1 in enumerate(variables):
        for var2 in variables[i + 1 :]:
            # Find minimal conditioning set
            # For simplicity, use Markov blanket
            markov_blanket_1 = graph.get_markov_blanket(var1)
            markov_blanket_2 = graph.get_markov_blanket(var2)

            # Conditioning set: intersection of Markov blankets
            conditioning_vars = list(markov_blanket_1 & markov_blanket_2)
            conditioning_vars = [v for v in conditioning_vars if v in data.columns]

            if len(conditioning_vars) == 0:
                continue

            # Test if var1 ⊥⊥ var2 | conditioning_vars using graph
            if graph.d_separated({var1}, {var2}, set(conditioning_vars)):
                # Should be independent - test in data
                total_tests += 1

                p_value = _test_conditional_independence(
                    data, var1, var2, conditioning_vars, test
                )

                if p_value > alpha:
                    # Independence holds
                    passed_tests += 1
                else:
                    # Independence violated
                    violations.append(
                        {
                            'var1': var1,
                            'var2': var2,
                            'conditioning': conditioning_vars,
                            'p_value': p_value,
                        }
                    )

    consistency_score = passed_tests / total_tests if total_tests > 0 else 1.0

    return {
        'consistency_score': consistency_score,
        'n_tests': total_tests,
        'passed_tests': passed_tests,
        'violations': violations,
        'n_violations': len(violations),
    }


def _test_conditional_independence(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    conditioning_vars: List[str],
    test: str = 'regression',
) -> float:
    """Test conditional independence: X ⊥⊥ Y | Z.

    Args:
        data: Dataset
        var1: First variable
        var2: Second variable
        conditioning_vars: Conditioning variables
        test: Test type

    Returns:
        p-value for independence test
    """
    if test == 'regression':
        # Partial correlation test via regression
        from sklearn.linear_model import LinearRegression

        # Regress var1 on conditioning vars
        X_cond = data[conditioning_vars].values
        y1 = data[var1].values

        model1 = LinearRegression()
        model1.fit(X_cond, y1)
        residuals1 = y1 - model1.predict(X_cond)

        # Regress var2 on conditioning vars
        y2 = data[var2].values
        model2 = LinearRegression()
        model2.fit(X_cond, y2)
        residuals2 = y2 - model2.predict(X_cond)

        # Test correlation of residuals
        corr, p_value = stats.pearsonr(residuals1, residuals2)
        return p_value

    elif test == 'correlation':
        # Simple correlation (ignoring conditioning for now)
        corr, p_value = stats.pearsonr(data[var1], data[var2])
        return p_value

    else:
        raise ValueError(f"Unknown test: {test}")


def intervention_effect_size(
    scm: 'StructuralCausalModel',
    intervention: Dict[str, Any],
    outcome: str,
    baseline_data: Optional[pd.DataFrame] = None,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """Measure effect size of intervention on outcome.

    Computes standardized effect sizes (Cohen's d, Hedges' g) for
    intervention effects.

    Args:
        scm: Structural Causal Model
        intervention: Intervention specification
        outcome: Outcome variable
        baseline_data: Baseline (observational) data for comparison
        n_samples: Number of samples for intervention

    Returns:
        Dictionary with effect size measures:
            - cohens_d: Cohen's d
            - hedges_g: Hedges' g (bias-corrected)
            - mean_diff: Mean difference
            - relative_change: Relative change (%)

    Example:
        >>> effect = intervention_effect_size(
        ...     scm=scm,
        ...     intervention={'antibiotic': True},
        ...     outcome='mortality'
        ... )
        >>> print(f"Cohen's d: {effect['cohens_d']:.2f}")
    """
    # Sample baseline (no intervention)
    if baseline_data is None:
        baseline_data = scm.sample(n=n_samples, return_dataframe=True)

    # Sample with intervention
    intervention_data = scm.do_intervention(
        interventions=intervention, n=n_samples, return_dataframe=True
    )

    # Compute effect sizes
    baseline_mean = baseline_data[outcome].mean()
    intervention_mean = intervention_data[outcome].mean()

    baseline_std = baseline_data[outcome].std()
    intervention_std = intervention_data[outcome].std()

    # Pooled standard deviation
    n1, n2 = len(baseline_data), len(intervention_data)
    pooled_std = np.sqrt(
        ((n1 - 1) * baseline_std**2 + (n2 - 1) * intervention_std**2) / (n1 + n2 - 2)
    )

    # Cohen's d
    cohens_d = (intervention_mean - baseline_mean) / pooled_std

    # Hedges' g (bias correction for small samples)
    correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohens_d * correction_factor

    # Mean difference
    mean_diff = intervention_mean - baseline_mean

    # Relative change
    relative_change = (
        (mean_diff / baseline_mean * 100) if baseline_mean != 0 else np.nan
    )

    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'mean_diff': mean_diff,
        'relative_change': relative_change,
        'baseline_mean': baseline_mean,
        'intervention_mean': intervention_mean,
        'pooled_std': pooled_std,
    }


def confounding_bias_estimate(
    data_observational: pd.DataFrame,
    data_interventional: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> Dict[str, float]:
    """Estimate confounding bias by comparing observational and interventional effects.

    Confounding bias = Observational association - Causal effect

    Args:
        data_observational: Observational data
        data_interventional: Interventional data (from RCT or SCM)
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        Dictionary with bias estimates:
            - bias: Absolute bias
            - relative_bias: Relative bias (%)
            - observational_effect: Effect from observational data
            - causal_effect: True causal effect

    Example:
        >>> bias = confounding_bias_estimate(
        ...     data_observational=obs_data,
        ...     data_interventional=rct_data,
        ...     treatment='antibiotic',
        ...     outcome='mortality'
        ... )
        >>> print(f"Confounding bias: {bias['bias']:.3f}")
    """
    # Observational effect (naive estimate)
    obs_treated = data_observational[data_observational[treatment] == 1]
    obs_control = data_observational[data_observational[treatment] == 0]

    observational_effect = obs_treated[outcome].mean() - obs_control[outcome].mean()

    # Causal effect (from interventional data)
    int_treated = data_interventional[data_interventional[treatment] == 1]
    int_control = data_interventional[data_interventional[treatment] == 0]

    causal_effect = int_treated[outcome].mean() - int_control[outcome].mean()

    # Bias
    bias = observational_effect - causal_effect

    # Relative bias
    relative_bias = (bias / causal_effect * 100) if causal_effect != 0 else np.nan

    return {
        'bias': bias,
        'relative_bias': relative_bias,
        'observational_effect': observational_effect,
        'causal_effect': causal_effect,
    }


def causal_discovery_score(
    true_graph: 'CausalGraph', learned_graph: 'CausalGraph'
) -> Dict[str, float]:
    """Evaluate quality of learned causal graph against ground truth.

    Computes structural metrics for graph recovery:
    - Structural Hamming Distance (SHD)
    - Precision, Recall, F1 for edges

    Args:
        true_graph: Ground truth causal graph
        learned_graph: Learned/estimated causal graph

    Returns:
        Dictionary with graph recovery metrics

    Example:
        >>> score = causal_discovery_score(
        ...     true_graph=ground_truth,
        ...     learned_graph=discovered_graph
        ... )
        >>> print(f"F1 score: {score['f1']:.2f}")
    """
    import networkx as nx

    true_edges = set(true_graph.graph.edges())
    learned_edges = set(learned_graph.graph.edges())

    # True positives, false positives, false negatives
    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Structural Hamming Distance
    shd = fp + fn

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_true_edges': len(true_edges),
        'n_learned_edges': len(learned_edges),
    }


def calibration_error_interventional(
    predicted_outcomes: np.ndarray, observed_outcomes: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Compute calibration error for interventional predictions.

    Measures how well predicted intervention effects match observed effects.

    Args:
        predicted_outcomes: Predicted outcomes under intervention
        observed_outcomes: Observed outcomes under intervention
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - rmse: Root Mean Squared Error

    Example:
        >>> error = calibration_error_interventional(
        ...     predicted_outcomes=model.predict(X_intervention),
        ...     observed_outcomes=y_intervention
        ... )
    """
    # Bin predictions
    bin_edges = np.linspace(
        predicted_outcomes.min(), predicted_outcomes.max(), n_bins + 1
    )

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        # Get samples in this bin
        mask = (predicted_outcomes >= bin_edges[i]) & (
            predicted_outcomes < bin_edges[i + 1]
        )

        if mask.sum() == 0:
            continue

        # Mean predicted and observed in bin
        mean_predicted = predicted_outcomes[mask].mean()
        mean_observed = observed_outcomes[mask].mean()

        # Calibration error in bin
        bin_error = abs(mean_predicted - mean_observed)
        bin_weight = mask.sum() / len(predicted_outcomes)

        ece += bin_weight * bin_error
        mce = max(mce, bin_error)

    # RMSE
    rmse = np.sqrt(mean_squared_error(observed_outcomes, predicted_outcomes))

    return {'ece': ece, 'mce': mce, 'rmse': rmse}


def counterfactual_consistency(
    scm: 'StructuralCausalModel',
    observations: List[Dict[str, Any]],
    interventions: List[Dict[str, Any]],
    query_variables: List[str],
    n_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate consistency of counterfactual predictions.

    Tests whether counterfactual predictions are consistent across
    different observations with same structural properties.

    Args:
        scm: Structural Causal Model
        observations: List of observed states
        interventions: List of counterfactual interventions
        query_variables: Variables to query
        n_samples: Samples per counterfactual

    Returns:
        Dictionary with consistency metrics

    Example:
        >>> consistency = counterfactual_consistency(
        ...     scm=scm,
        ...     observations=[obs1, obs2, obs3],
        ...     interventions=[{'antibiotic': True}],
        ...     query_variables=['mortality']
        ... )
    """
    results = []

    for obs in observations:
        for intervention in interventions:
            cf_result = scm.counterfactual(
                observation=obs,
                intervention=intervention,
                query_variables=query_variables,
                n_samples=n_samples,
            )
            results.append(cf_result)

    # Compute variance in counterfactual predictions
    consistency_scores = {}

    for var in query_variables:
        values = [r[var] for r in results]
        consistency_scores[var] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan,
        }

    return consistency_scores


def markov_compatibility_test(
    data: pd.DataFrame, graph: 'CausalGraph', alpha: float = 0.05
) -> Dict[str, Any]:
    """Test if data satisfies Markov condition for causal graph.

    Markov condition: Each variable is independent of its non-descendants
    given its parents.

    Args:
        data: Dataset
        graph: Causal graph
        alpha: Significance level

    Returns:
        Dictionary with test results

    Example:
        >>> result = markov_compatibility_test(
        ...     data=generated_data,
        ...     graph=causal_graph
        ... )
        >>> print(f"Markov compatible: {result['is_compatible']}")
    """
    violations = []
    n_tests = 0

    for variable in graph.graph.nodes():
        parents = graph.get_parents(variable)
        descendants = graph.get_descendants(variable)

        # Non-descendants (excluding variable itself)
        all_vars = set(graph.graph.nodes())
        non_descendants = all_vars - descendants - {variable}

        # Test: variable ⊥⊥ non_descendant | parents
        for non_desc in non_descendants:
            if non_desc in parents:
                continue

            n_tests += 1

            # Test conditional independence
            p_value = _test_conditional_independence(
                data, variable, non_desc, parents, test='regression'
            )

            if p_value < alpha:
                violations.append(
                    {
                        'variable': variable,
                        'non_descendant': non_desc,
                        'parents': parents,
                        'p_value': p_value,
                    }
                )

    is_compatible = len(violations) == 0

    return {
        'is_compatible': is_compatible,
        'n_tests': n_tests,
        'n_violations': len(violations),
        'violations': violations,
        'compatibility_score': 1 - (len(violations) / n_tests) if n_tests > 0 else 1.0,
    }
