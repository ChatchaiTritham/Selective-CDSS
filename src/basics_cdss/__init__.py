"""BASICS-CDSS package.

Beyond Accuracy: Simulation-based Integrated Critical-Safety evaluation
for Clinical Decision Support Systems.

Version 2.0.0 introduces XAI (Explainable AI) methods:
- SHAP (Shapley Additive exPlanations) analysis
- Counterfactual explanations for clinical decisions

Version 2.1.0 introduces Phase 1 Clinical Metrics for Medical AI:
- Clinical Utility Metrics (Net Benefit, NNT, Decision Curves)
- Fairness Metrics (Demographic Parity, Equalized Odds, Calibration)
- Conformal Prediction (Uncertainty Quantification with Guarantees)
"""

from basics_cdss.constants import FRAMEWORK_NAME, PACKAGE_VERSION

__version__ = PACKAGE_VERSION

__all__ = [
    "scenario",
    "metrics",
    "governance",
    "visualization",
    "xai",
    "clinical_metrics",
    "FRAMEWORK_NAME",
]
