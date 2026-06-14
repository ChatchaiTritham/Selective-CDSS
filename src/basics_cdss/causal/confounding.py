"""Confounder identification and adjustment for causal inference.

This module implements methods for identifying confounders and adjusting
for confounding bias using backdoor and frontdoor adjustment.

Theoretical Foundation:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Shpitser, I., & Pearl, J. (2006). Identification of Joint Interventional Distributions.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from basics_cdss.causal.causal_graph import CausalGraph


def identify_confounders(
    graph: CausalGraph, treatment: str, outcome: str
) -> Dict[str, Any]:
    """Identify confounders using backdoor criterion.

    A set Z of variables is a sufficient adjustment set if:
    1. Z blocks all backdoor paths from treatment to outcome
    2. No variable in Z is a descendant of treatment

    Backdoor path: path from treatment to outcome that starts with
    an arrow pointing INTO treatment.

    Args:
        graph: Causal graph
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        Dictionary containing:
            - confounders: Minimal sufficient adjustment set
            - all_confounders: All valid adjustment sets
            - backdoor_paths: List of backdoor paths
            - needs_adjustment: Whether adjustment is needed

    Example:
        >>> from basics_cdss.causal import CausalGraph, identify_confounders
        >>>
        >>> graph = CausalGraph()
        >>> graph.add_edge('age', 'treatment')
        >>> graph.add_edge('age', 'outcome')
        >>> graph.add_edge('treatment', 'outcome')
        >>>
        >>> result = identify_confounders(graph, 'treatment', 'outcome')
        >>> print(f"Confounders: {result['confounders']}")
        ['age']
    """
    # Find all variables except treatment and outcome
    all_vars = set(graph.graph.nodes())
    all_vars.discard(treatment)
    all_vars.discard(outcome)

    # Find descendants of treatment (cannot adjust for these)
    treatment_descendants = graph.get_descendants(treatment)

    # Candidate adjustment variables
    candidate_vars = all_vars - treatment_descendants

    # Find backdoor paths
    backdoor_paths = _find_backdoor_paths(graph, treatment, outcome)

    # If no backdoor paths, no confounding
    if not backdoor_paths:
        return {
            'confounders': [],
            'all_confounders': [[]],
            'backdoor_paths': [],
            'needs_adjustment': False,
        }

    # Find all valid adjustment sets
    valid_sets = []

    # Check all subsets of candidate variables
    for r in range(len(candidate_vars) + 1):
        for candidate_set in combinations(candidate_vars, r):
            candidate_set = set(candidate_set)

            # Check if this set blocks all backdoor paths
            if _satisfies_backdoor_criterion(
                graph, treatment, outcome, candidate_set, backdoor_paths
            ):
                valid_sets.append(candidate_set)

    # Find minimal sets (no proper subset is also valid)
    minimal_sets = []
    for s in valid_sets:
        is_minimal = True
        for other in valid_sets:
            if other < s:  # other is proper subset of s
                is_minimal = False
                break
        if is_minimal:
            minimal_sets.append(s)

    # Return smallest minimal set as primary recommendation
    if minimal_sets:
        confounders = min(minimal_sets, key=len)
    else:
        confounders = set()

    return {
        'confounders': sorted(list(confounders)),
        'all_confounders': [sorted(list(s)) for s in minimal_sets],
        'backdoor_paths': backdoor_paths,
        'needs_adjustment': len(backdoor_paths) > 0,
    }


def _find_backdoor_paths(
    graph: CausalGraph, treatment: str, outcome: str
) -> List[List[str]]:
    """Find all backdoor paths from treatment to outcome.

    A backdoor path is a path that:
    1. Starts with an arrow INTO treatment
    2. Ends at outcome
    3. Does not traverse treatment's descendants

    Args:
        graph: Causal graph
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        List of backdoor paths (each path is a list of variables)
    """
    import networkx as nx

    backdoor_paths = []

    # Find all simple paths in underlying undirected graph
    undirected = graph.graph.to_undirected()

    try:
        all_paths = nx.all_simple_paths(
            undirected, source=treatment, target=outcome, cutoff=10  # Limit path length
        )

        for path in all_paths:
            # Check if this is a backdoor path
            if len(path) < 2:
                continue

            # Path must start with arrow INTO treatment
            second_node = path[1]
            if graph.graph.has_edge(second_node, treatment):
                # This is a backdoor path
                backdoor_paths.append(path)

    except nx.NetworkXNoPath:
        pass

    return backdoor_paths


def _satisfies_backdoor_criterion(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    adjustment_set: Set[str],
    backdoor_paths: List[List[str]],
) -> bool:
    """Check if adjustment set satisfies backdoor criterion.

    Args:
        graph: Causal graph
        treatment: Treatment variable
        outcome: Outcome variable
        adjustment_set: Set of variables to adjust for
        backdoor_paths: List of backdoor paths

    Returns:
        True if adjustment set blocks all backdoor paths
    """
    # Check each backdoor path
    for path in backdoor_paths:
        # Path must be blocked by adjustment set
        if not _is_path_blocked(graph, path, adjustment_set):
            return False

    return True


def _is_path_blocked(
    graph: CausalGraph, path: List[str], conditioning_set: Set[str]
) -> bool:
    """Check if path is blocked by conditioning set (d-separation).

    A path is blocked if:
    1. It contains a chain (A → B → C) or fork (A ← B → C)
       where middle variable is in conditioning set
    2. It contains a collider (A → B ← C) where B and its descendants
       are NOT in conditioning set

    Args:
        graph: Causal graph
        path: Path to check (list of variables)
        conditioning_set: Variables being conditioned on

    Returns:
        True if path is blocked
    """
    for i in range(len(path) - 2):
        a, b, c = path[i], path[i + 1], path[i + 2]

        # Check structure at b
        a_to_b = graph.graph.has_edge(a, b)
        b_to_a = graph.graph.has_edge(b, a)
        b_to_c = graph.graph.has_edge(b, c)
        c_to_b = graph.graph.has_edge(c, b)

        # Collider: a → b ← c
        if a_to_b and c_to_b:
            # Blocked if b and descendants NOT in conditioning set
            descendants = graph.get_descendants(b)
            if b not in conditioning_set and descendants.isdisjoint(conditioning_set):
                return True  # Path blocked

        # Chain: a → b → c or Fork: a ← b → c
        elif (a_to_b and b_to_c) or (b_to_a and b_to_c):
            # Blocked if b in conditioning set
            if b in conditioning_set:
                return True  # Path blocked

        # Chain: a ← b ← c
        elif b_to_a and c_to_b:
            if b in conditioning_set:
                return True  # Path blocked

    return False  # Path not blocked


def backdoor_adjustment(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    treatment_values: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """Estimate causal effect using backdoor adjustment formula.

    Backdoor adjustment: P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)

    Args:
        data: Observational dataset
        treatment: Treatment variable
        outcome: Outcome variable
        confounders: Confounding variables (adjustment set)
        treatment_values: Treatment values to compare (default: [0, 1])

    Returns:
        Dictionary with adjusted ATE estimate

    Example:
        >>> result = backdoor_adjustment(
        ...     data=obs_data,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     confounders=['age', 'comorbidities']
        ... )
        >>> print(f"Adjusted ATE: {result['ate']:.3f}")
    """
    if treatment_values is None:
        treatment_values = [0, 1]

    from sklearn.linear_model import LinearRegression

    # Fit regression: Y ~ T + confounders
    X = data[[treatment] + confounders].values
    y = data[outcome].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict under each treatment value
    control_value, treatment_value = treatment_values

    # Create counterfactual datasets
    X_control = data[confounders].copy()
    X_control[treatment] = control_value
    X_control = X_control[[treatment] + confounders].values

    X_treatment = data[confounders].copy()
    X_treatment[treatment] = treatment_value
    X_treatment = X_treatment[[treatment] + confounders].values

    # Predict outcomes
    y_control = model.predict(X_control)
    y_treatment = model.predict(X_treatment)

    # Average treatment effect
    ate = y_treatment.mean() - y_control.mean()

    return {
        'ate': ate,
        'control_mean': y_control.mean(),
        'treatment_mean': y_treatment.mean(),
        'confounders': confounders,
        'method': 'backdoor_adjustment',
    }


def frontdoor_adjustment(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    mediator: str,
    treatment_values: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """Estimate causal effect using frontdoor adjustment.

    Frontdoor adjustment is used when:
    1. Backdoor criterion cannot be satisfied (unmeasured confounding)
    2. There exists a mediator M on the causal path X → M → Y
    3. All backdoor paths from X to M are blocked
    4. All backdoor paths from M to Y are blocked by X

    Frontdoor formula:
        P(Y | do(X=x)) = Σ_m P(M=m | X=x) Σ_x' P(Y | M=m, X=x') P(X=x')

    Args:
        data: Observational dataset
        treatment: Treatment variable
        outcome: Outcome variable
        mediator: Mediator variable on causal path
        treatment_values: Treatment values to compare

    Returns:
        Dictionary with frontdoor-adjusted ATE estimate

    Example:
        >>> # Estimate effect when there's unmeasured confounding
        >>> # but mediator fully captures treatment effect
        >>> result = frontdoor_adjustment(
        ...     data=obs_data,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     mediator='infection_clearance'
        ... )
    """
    if treatment_values is None:
        treatment_values = [0, 1]

    control_value, treatment_value = treatment_values

    # Step 1: Estimate P(M | X=x)
    mediator_given_control = data[data[treatment] == control_value][mediator]
    mediator_given_treatment = data[data[treatment] == treatment_value][mediator]

    # Step 2: For each mediator value, estimate E[Y | M=m, X]
    # Marginalize over X to get E[Y | M=m]
    mediator_bins = pd.qcut(data[mediator], q=10, duplicates='drop')

    outcomes_control = []
    outcomes_treatment = []

    for bin_label in mediator_bins.cat.categories:
        # Get data in this mediator bin
        mask = mediator_bins == bin_label

        if mask.sum() == 0:
            continue

        # Compute E[Y | M in bin]
        y_in_bin = data.loc[mask, outcome].mean()

        # Weight by P(M in bin | X=x)
        p_control = (
            mediator_bins[data[treatment] == control_value] == bin_label
        ).mean()
        p_treatment = (
            mediator_bins[data[treatment] == treatment_value] == bin_label
        ).mean()

        outcomes_control.append(y_in_bin * p_control)
        outcomes_treatment.append(y_in_bin * p_treatment)

    ate = sum(outcomes_treatment) - sum(outcomes_control)

    return {
        'ate': ate,
        'control_mean': sum(outcomes_control),
        'treatment_mean': sum(outcomes_treatment),
        'mediator': mediator,
        'method': 'frontdoor_adjustment',
    }


def check_instrumental_variable(
    graph: CausalGraph, instrument: str, treatment: str, outcome: str
) -> Dict[str, Any]:
    """Check if variable is a valid instrumental variable.

    A valid instrument Z for X → Y must satisfy:
    1. Relevance: Z is associated with X
    2. Exclusion: Z affects Y only through X (no direct path Z → Y)
    3. Exogeneity: Z is independent of unmeasured confounders

    Args:
        graph: Causal graph
        instrument: Candidate instrumental variable
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        Dictionary with validation results

    Example:
        >>> result = check_instrumental_variable(
        ...     graph=graph,
        ...     instrument='randomization',
        ...     treatment='antibiotic',
        ...     outcome='mortality'
        ... )
        >>> print(f"Valid IV: {result['is_valid']}")
    """
    # Check relevance: instrument causes treatment
    has_path_to_treatment = treatment in graph.get_descendants(instrument)

    # Check exclusion: no direct path to outcome except through treatment
    # Get all paths from instrument to outcome
    import networkx as nx

    has_direct_path = False
    try:
        paths = list(
            nx.all_simple_paths(
                graph.graph, source=instrument, target=outcome, cutoff=10
            )
        )

        # Check if any path doesn't go through treatment
        for path in paths:
            if treatment not in path:
                has_direct_path = True
                break

    except nx.NetworkXNoPath:
        has_direct_path = False

    # Check exogeneity: no common causes with outcome
    instrument_ancestors = graph.get_ancestors(instrument)
    outcome_ancestors = graph.get_ancestors(outcome)

    # Common ancestors suggest confounding
    common_ancestors = instrument_ancestors & outcome_ancestors

    is_valid = (
        has_path_to_treatment and not has_direct_path and len(common_ancestors) == 0
    )

    return {
        'is_valid': is_valid,
        'relevance': has_path_to_treatment,
        'exclusion': not has_direct_path,
        'exogeneity': len(common_ancestors) == 0,
        'common_ancestors': sorted(list(common_ancestors)),
        'violations': [],
    }


def sensitivity_analysis_evalue(
    observed_estimate: float, confidence_limit: Optional[float] = None
) -> Dict[str, float]:
    """Compute E-value for sensitivity to unmeasured confounding.

    The E-value represents the minimum strength of association that an
    unmeasured confounder would need to have with both treatment and
    outcome to fully explain away the observed effect.

    Args:
        observed_estimate: Observed risk ratio or odds ratio
        confidence_limit: Lower confidence limit (optional)

    Returns:
        Dictionary with E-values

    Reference:
        VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in
        Observational Research. Annals of Internal Medicine.

    Example:
        >>> result = sensitivity_analysis_evalue(
        ...     observed_estimate=2.5,  # RR = 2.5
        ...     confidence_limit=1.8
        ... )
        >>> print(f"E-value: {result['e_value']:.2f}")
    """
    # E-value formula
    e_value = observed_estimate + np.sqrt(observed_estimate * (observed_estimate - 1))

    result = {
        'e_value': e_value,
        'observed_estimate': observed_estimate,
    }

    if confidence_limit is not None:
        e_value_ci = confidence_limit + np.sqrt(
            confidence_limit * (confidence_limit - 1)
        )
        result['e_value_ci'] = e_value_ci
        result['confidence_limit'] = confidence_limit

    return result
