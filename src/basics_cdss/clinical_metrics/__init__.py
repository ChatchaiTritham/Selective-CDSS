"""
Clinical Metrics Module for Medical AI Validation

This module provides comprehensive metrics for evaluating medical AI systems,
focusing on clinical utility, fairness, and uncertainty quantification.

Phase 1: Critical for Medical AI
=================================

1. Clinical Utility Metrics
---------------------------
Essential for FDA approval and clinical decision-making:
- Net Benefit (Decision Curve Analysis)
- Number Needed to Treat (NNT)
- Clinical Impact Assessment

2. Fairness Metrics
-------------------
Ensure ethical AI deployment and health equity:
- Demographic Parity
- Equalized Odds
- Equal Opportunity
- Disparate Impact
- Calibration across groups

3. Conformal Prediction
-----------------------
Rigorous uncertainty quantification with guarantees:
- Prediction sets with guaranteed coverage
- Prediction intervals for regression
- Adaptive conformal prediction
- Risk control calibration

Example Usage:
--------------
    from basics_cdss.clinical_metrics import (
        calculate_net_benefit,
        decision_curve_analysis,
        calculate_nnt,
        fairness_report,
        split_conformal_classification
    )

    # Clinical Utility
    nb = calculate_net_benefit(y_true, y_pred_proba, threshold=0.3)
    print(f"Net Benefit: {nb.net_benefit:.4f}")

    dca = decision_curve_analysis(y_true, y_pred_proba)
    print(f"Useful threshold range: {dca.threshold_range}")

    nnt = calculate_nnt(y_true, y_pred)
    print(f"NNT: {nnt.nnt:.1f} (treat {nnt.nnt:.0f} to prevent 1 event)")

    # Fairness
    report = fairness_report(y_true, y_pred, y_pred_proba, race)
    if not report.overall_fair:
        print(f"Failed: {report.failed_criteria}")

    # Conformal Prediction
    conf_result = split_conformal_classification(
        model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1
    )
    print(f"Coverage: {conf_result.target_coverage:.1%}")
    print(f"Avg set size: {conf_result.efficiency:.2f}")
"""

from .conformal_prediction import (  # Dataclasses; Functions
    AdaptiveConformalResult, ConformalInterval, ConformalPredictionSet,
    RiskControlResult, adaptive_conformal_classification, conformal_pvalue,
    risk_control_conformal, split_conformal_classification,
    split_conformal_regression)
from .fairness_metrics import (CalibrationResult,  # Dataclasses; Functions
                               DemographicParityResult, DisparateImpactResult,
                               EqualizedOddsResult, EqualOpportunityResult,
                               FairnessReport, calibration_by_group,
                               demographic_parity, disparate_impact,
                               equal_opportunity, equalized_odds,
                               fairness_report)
from .utility_metrics import (ClinicalImpactResult,  # Dataclasses; Functions
                              DecisionCurveResult, NetBenefitResult, NNTResult,
                              calculate_net_benefit, calculate_nnt,
                              clinical_impact_analysis,
                              decision_curve_analysis, stratified_net_benefit)

__all__ = [
    # Clinical Utility Metrics
    'NetBenefitResult',
    'DecisionCurveResult',
    'NNTResult',
    'ClinicalImpactResult',
    'calculate_net_benefit',
    'decision_curve_analysis',
    'calculate_nnt',
    'clinical_impact_analysis',
    'stratified_net_benefit',
    # Fairness Metrics
    'DemographicParityResult',
    'EqualizedOddsResult',
    'EqualOpportunityResult',
    'DisparateImpactResult',
    'CalibrationResult',
    'FairnessReport',
    'demographic_parity',
    'equalized_odds',
    'equal_opportunity',
    'disparate_impact',
    'calibration_by_group',
    'fairness_report',
    # Conformal Prediction
    'ConformalPredictionSet',
    'ConformalInterval',
    'AdaptiveConformalResult',
    'RiskControlResult',
    'split_conformal_classification',
    'split_conformal_regression',
    'adaptive_conformal_classification',
    'risk_control_conformal',
    'conformal_pvalue',
]
