"""BASICS-CDSS Causal Module: Structural Causal Model Simulation.

This module enables causal simulation using Structural Causal Models (SCMs)
to ensure generated scenarios satisfy causal relationships and support
do-calculus interventions.

Key capabilities:
1. Define causal graphs (DAGs) for clinical domains
2. Sample from SCMs with causal consistency
3. Perform interventions using do-calculus
4. Counterfactual reasoning with causal constraints
5. Identify and control for confounding

Theoretical Foundation:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Hernán, M. A., & Robins, J. M. (2020). Causal Inference: What If.

Example:
    >>> from basics_cdss.causal import CausalGraph, StructuralCausalModel
    >>>
    >>> # Define causal graph for sepsis
    >>> graph = CausalGraph()
    >>> graph.add_edge('infection', 'temperature')
    >>> graph.add_edge('infection', 'white_blood_cell_count')
    >>> graph.add_edge('temperature', 'heart_rate')
    >>>
    >>> # Create SCM
    >>> scm = StructuralCausalModel(graph, seed=42)
    >>>
    >>> # Sample observational data
    >>> data = scm.sample(n=100)
    >>>
    >>> # Perform intervention: do(antibiotic=True)
    >>> interventional_data = scm.do_intervention({'antibiotic': True}, n=100)
"""

from basics_cdss.causal.causal_graph import (CausalGraph,
                                             create_cardiac_causal_graph,
                                             create_respiratory_causal_graph,
                                             create_sepsis_causal_graph)
from basics_cdss.causal.causal_metrics import (causal_consistency_score,
                                               confounding_bias_estimate,
                                               intervention_effect_size)
from basics_cdss.causal.confounding import (backdoor_adjustment,
                                            frontdoor_adjustment,
                                            identify_confounders)
from basics_cdss.causal.interventions import (DoIntervention, compute_ate,
                                              compute_cate,
                                              perform_do_intervention)
from basics_cdss.causal.scm import CausalMechanism, StructuralCausalModel

__all__ = [
    # SCM
    "StructuralCausalModel",
    "CausalMechanism",
    # Graphs
    "CausalGraph",
    "create_sepsis_causal_graph",
    "create_cardiac_causal_graph",
    "create_respiratory_causal_graph",
    # Interventions
    "DoIntervention",
    "perform_do_intervention",
    "compute_ate",
    "compute_cate",
    # Confounding
    "identify_confounders",
    "backdoor_adjustment",
    "frontdoor_adjustment",
    # Metrics
    "causal_consistency_score",
    "intervention_effect_size",
    "confounding_bias_estimate",
]
