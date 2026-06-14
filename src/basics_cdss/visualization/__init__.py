"""
Visualization module for BASICS-CDSS evaluation results.

This module provides manuscript-preparation plotting functions for:
- Calibration reliability diagrams
- Coverage-risk curves
- Harm concentration visualizations
- Uncertainty profile distributions
- Comparative evaluation plots
- Temporal trajectories and disease progression (Tier 1)
- Causal DAGs and intervention effects (Tier 2)
- Multi-agent interactions and workflow analysis (Tier 3)
- XAI: SHAP values and counterfactual explanations (v2.0.0)
- Clinical Metrics: Decision curves, fairness, conformal prediction (v2.1.0)
"""

# Advanced Charts
from .advanced_charts import (plot_3d_performance_surface,
                              plot_3d_scatter_performance,
                              plot_contour_performance,
                              plot_multi_radar_comparison,
                              plot_parallel_coordinates, plot_radar_chart,
                              plot_stratified_heatmap)
from .calibration_plots import (plot_calibration_comparison,
                                plot_reliability_diagram,
                                plot_stratified_calibration)
# Tier 2: Causal Simulation
from .causal_plots import (plot_backdoor_adjustment, plot_cate_heterogeneity,
                           plot_causal_dag, plot_confounding_analysis,
                           plot_intervention_effects)
# Clinical Metrics Plots (Phase 1: Medical AI Validation)
from .clinical_plots import (  # Clinical Utility Metrics; Fairness Metrics; Conformal Prediction
    plot_adaptive_efficiency_3d, plot_calibration_by_group,
    plot_clinical_impact, plot_clinical_impact_3d, plot_conformal_intervals,
    plot_coverage_vs_alpha, plot_decision_curve, plot_demographic_parity,
    plot_disparate_impact, plot_equalized_odds, plot_fairness_radar,
    plot_nnt_comparison, plot_prediction_set_sizes,
    plot_standardized_net_benefit)
from .comparison_plots import (create_evaluation_dashboard,
                               plot_metric_comparison,
                               plot_model_comparison_radar)
from .coverage_risk_plots import (plot_abstention_analysis,
                                  plot_coverage_risk_curve,
                                  plot_selective_prediction_comparison)
from .harm_plots import (plot_escalation_analysis, plot_harm_by_tier,
                         plot_harm_concentration)
# Tier 3: Multi-Agent Simulation
from .multiagent_plots import (plot_agent_interaction_network,
                               plot_alert_fatigue_dynamics,
                               plot_override_rates_comparison,
                               plot_system_resilience, plot_workflow_timeline)
# Performance Plots
from .performance_plots import (plot_confusion_matrix,
                                plot_metrics_comparison_bar,
                                plot_multi_class_confusion_matrix,
                                plot_multi_model_roc, plot_pr_curve,
                                plot_roc_curve,
                                plot_sensitivity_specificity_curve,
                                plot_threshold_analysis)
from .scenario_plots import (plot_perturbation_effects, plot_scenario_summary,
                             plot_uncertainty_distribution)
# Tier 1: Digital Twin / Temporal Analysis
from .temporal_plots import (plot_counterfactual_analysis,
                             plot_disease_progression,
                             plot_intervention_timing_analysis,
                             plot_temporal_trajectory)
# XAI (Explainable AI) Plots
from .xai_plots import (  # SHAP Visualizations; Counterfactual Visualizations
    plot_counterfactual_comparison, plot_counterfactual_diversity,
    plot_feature_changes, plot_intervention_priority, plot_shap_bar,
    plot_shap_dependence, plot_shap_heatmap, plot_shap_interaction_heatmap,
    plot_shap_summary, plot_shap_waterfall, plot_whatif_curve)

__all__ = [
    # Calibration
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_stratified_calibration",
    # Coverage-Risk
    "plot_coverage_risk_curve",
    "plot_selective_prediction_comparison",
    "plot_abstention_analysis",
    # Harm-Aware
    "plot_harm_by_tier",
    "plot_escalation_analysis",
    "plot_harm_concentration",
    # Scenarios
    "plot_uncertainty_distribution",
    "plot_perturbation_effects",
    "plot_scenario_summary",
    # Comparison
    "plot_metric_comparison",
    "plot_model_comparison_radar",
    "create_evaluation_dashboard",
    # Tier 1: Temporal/Digital Twin
    "plot_temporal_trajectory",
    "plot_disease_progression",
    "plot_counterfactual_analysis",
    "plot_intervention_timing_analysis",
    # Tier 2: Causal
    "plot_causal_dag",
    "plot_intervention_effects",
    "plot_cate_heterogeneity",
    "plot_confounding_analysis",
    "plot_backdoor_adjustment",
    # Tier 3: Multi-Agent
    "plot_agent_interaction_network",
    "plot_workflow_timeline",
    "plot_alert_fatigue_dynamics",
    "plot_override_rates_comparison",
    "plot_system_resilience",
    # Performance Plots
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_sensitivity_specificity_curve",
    "plot_threshold_analysis",
    "plot_multi_model_roc",
    "plot_metrics_comparison_bar",
    "plot_multi_class_confusion_matrix",
    # Advanced Charts
    "plot_3d_performance_surface",
    "plot_contour_performance",
    "plot_stratified_heatmap",
    "plot_radar_chart",
    "plot_multi_radar_comparison",
    "plot_parallel_coordinates",
    "plot_3d_scatter_performance",
    # XAI Plots - SHAP
    "plot_shap_waterfall",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_dependence",
    "plot_shap_heatmap",
    "plot_shap_interaction_heatmap",
    # XAI Plots - Counterfactual
    "plot_counterfactual_comparison",
    "plot_feature_changes",
    "plot_intervention_priority",
    "plot_whatif_curve",
    "plot_counterfactual_diversity",
    # Clinical Metrics - Utility
    "plot_decision_curve",
    "plot_standardized_net_benefit",
    "plot_nnt_comparison",
    "plot_clinical_impact",
    "plot_clinical_impact_3d",
    # Clinical Metrics - Fairness
    "plot_demographic_parity",
    "plot_equalized_odds",
    "plot_disparate_impact",
    "plot_calibration_by_group",
    "plot_fairness_radar",
    # Clinical Metrics - Conformal Prediction
    "plot_prediction_set_sizes",
    "plot_conformal_intervals",
    "plot_coverage_vs_alpha",
    "plot_adaptive_efficiency_3d",
]
