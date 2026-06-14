"""
XAI (Explainable AI) Module

Provides SHAP-based feature importance analysis and counterfactual explanations
for clinical decision support systems.

This module implements two complementary XAI approaches:

1. SHAP Values (Shapley Additive exPlanations):
   - Rooted in cooperative game theory
   - Features as "players" contributing to predictions
   - Critical symptoms = major players (high Shapley values)
   - Uncertain symptoms = minor players (low Shapley values)

2. Counterfactual Explanations:
   - "What-if" analysis for clinical decisions
   - Actionable intervention suggestions
   - Minimal changes for desired outcomes

Author: Chatchai Tritham
Affiliation: Department of Computer Science and Information Technology,
             Faculty of Science, Naresuan University
Date: 2026-01-25
Version: 2.0.0 (XAI Enhancement)
"""

# Counterfactual Explanations
from .counterfactual import (  # Core counterfactual functions; Data classes
    CounterfactualExample, CounterfactualSet, InterventionSuggestion,
    actionable_interventions, counterfactual_stability,
    generate_counterfactual, generate_diverse_counterfactuals,
    minimal_counterfactual, whatif_analysis)
# SHAP Analysis
from .shap_analysis import (  # Core SHAP functions; Data classes
    FeatureImportance, GameTheoreticExplanation, SHAPInteractionValues,
    SHAPValues, compute_shap_interaction_values, compute_shap_values,
    explain_prediction, feature_importance_ranking, game_theoretic_explanation,
    shap_based_feature_selection, shapley_coalition_values,
    stratified_shap_analysis, temporal_shap_analysis)

__all__ = [
    # SHAP Analysis
    'compute_shap_values',
    'compute_shap_interaction_values',
    'feature_importance_ranking',
    'shapley_coalition_values',
    'game_theoretic_explanation',
    'stratified_shap_analysis',
    'temporal_shap_analysis',
    'shap_based_feature_selection',
    'explain_prediction',
    'SHAPValues',
    'SHAPInteractionValues',
    'FeatureImportance',
    'GameTheoreticExplanation',
    # Counterfactual Explanations
    'generate_counterfactual',
    'generate_diverse_counterfactuals',
    'minimal_counterfactual',
    'actionable_interventions',
    'whatif_analysis',
    'counterfactual_stability',
    'CounterfactualExample',
    'CounterfactualSet',
    'InterventionSuggestion',
]
