"""DRAS-5: Decision Risk-Action States (5-Level Urgency Classification).

This module implements the DRAS-5 component of the TRI-X framework, which
classifies patients into 5 urgency levels that drive routing and resource
allocation decisions.

DRAS-5 Levels:
- Level 5 (Immediate Emergency): Stroke within 2-hour thrombolysis window
- Level 4 (Urgent Specialist): Requires same-day specialist evaluation
- Level 3 (Scheduled Specialist OPD): Outpatient specialist visit (reduces ED crowding)
- Level 2 (Lower Urgency): Routine GP appointment acceptable
- Level 1 (Safe/No Danger): Self-care or GP follow-up only

The DRAS-5 classification integrates:
- ESI triage assessment
- SRGL red flag screening
- TiTrATE risk trajectory
- Digital twin disease prediction (from Tier 1)
- Clinical findings (HINTS, imaging, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DRAS5Level(Enum):
    """DRAS-5 urgency levels."""

    LEVEL_1 = 1  # Safe, no danger
    LEVEL_2 = 2  # Lower urgency
    LEVEL_3 = 3  # Scheduled specialist OPD
    LEVEL_4 = 4  # Urgent specialist (same-day)
    LEVEL_5 = 5  # Immediate emergency


@dataclass
class DRAS5Features:
    """Features used for DRAS-5 classification."""

    # Patient demographics
    age: int
    gender: str

    # Risk factors
    atrial_fibrillation: bool = False
    hypertension: bool = False
    diabetes: bool = False
    prior_stroke: bool = False

    # Clinical presentation
    symptom_duration_hours: float = 0.0
    symptom_severity: int = 5  # 0-10 scale

    # HINTS exam
    hints_performed: bool = False
    hints_central: bool = False  # True = central cause concern

    # Neurological exam
    focal_neurological_deficit: bool = False
    ataxia: bool = False
    dysarthria: bool = False
    diplopia: bool = False

    # Vital signs
    systolic_bp: float = 120
    heart_rate: float = 80

    # Imaging
    imaging_performed: bool = False
    imaging_positive_stroke: bool = False

    # Disease prediction (from digital twin, if available)
    predicted_disease: Optional[str] = None  # "bppv", "stroke", "vn", etc.
    stroke_probability: float = 0.0  # 0-1

    # Symptom pattern
    positional_trigger: bool = False
    continuous_vertigo: bool = False


@dataclass
class DRAS5Classification:
    """Complete DRAS-5 classification result."""

    level: DRAS5Level
    confidence: float  # 0-1 (classification confidence)
    rationale: str  # Explanation of classification
    feature_importance: Dict[str, float] = field(
        default_factory=dict
    )  # XAI: which features drove decision
    recommended_pathway: str = ""  # Care pathway recommendation
    time_sensitivity: str = ""  # Time-critical considerations
    uncertainty_factors: List[str] = field(
        default_factory=list
    )  # Sources of uncertainty

    def get_target_care_setting(self) -> str:
        """Get recommended care setting based on DRAS level."""
        if self.level == DRAS5Level.LEVEL_5:
            return "Emergency Department (Immediate)"
        elif self.level == DRAS5Level.LEVEL_4:
            return "Emergency Department (Urgent) or Same-Day Neurology/ENT"
        elif self.level == DRAS5Level.LEVEL_3:
            return "Outpatient Neurology/ENT Clinic (1-7 days)"
        elif self.level == DRAS5Level.LEVEL_2:
            return "Primary Care (GP, 1-2 weeks)"
        else:  # LEVEL_1
            return "Self-Care or GP Follow-up (as needed)"

    def get_alert_threshold(self) -> bool:
        """Check if CDSS alert should be generated.

        Typically alert for DRAS 4-5 (high urgency).
        """
        return self.level.value >= 4


def classify_urgency_level(
    features: DRAS5Features,
    esi_level: Optional[int] = None,
    red_flags_critical: bool = False,
    titrate_risk_score: Optional[float] = None,
) -> DRAS5Classification:
    """Classify patient into DRAS-5 urgency level.

    This integrates multiple sources of information:
    1. ESI triage level (initial assessment)
    2. SRGL red flags (safety screening)
    3. TiTrATE risk score (time-bounded assessment)
    4. Clinical features (HINTS, imaging, etc.)
    5. Digital twin prediction (if available)

    Args:
        features: DRAS5Features with all relevant patient data
        esi_level: ESI triage level (1-5), if available
        red_flags_critical: True if critical red flags from SRGL
        titrate_risk_score: Risk score from TiTrATE (0-10), if available

    Returns:
        DRAS5Classification with level, confidence, and rationale

    Example:
        >>> features = DRAS5Features(
        ...     age=72,
        ...     gender="male",
        ...     atrial_fibrillation=True,
        ...     symptom_duration_hours=1.5,
        ...     hints_performed=True,
        ...     hints_central=True,
        ...     stroke_probability=0.85
        ... )
        >>> classification = classify_urgency_level(
        ...     features,
        ...     esi_level=2,
        ...     red_flags_critical=True,
        ...     titrate_risk_score=8.5
        ... )
        >>> print(classification.level)
        DRAS5Level.LEVEL_5  # Immediate emergency
    """
    # Initialize score (0-10 scale, will map to DRAS 1-5)
    score = 5.0  # Start at neutral
    confidence = 0.8  # Default confidence
    rationale_components = []
    feature_importance = {}
    uncertainty_factors = []

    # === CRITICAL OVERRIDES ===
    # Red flags override all other factors
    if red_flags_critical:
        score += 5.0
        rationale_components.append("Critical red flags present")
        feature_importance["red_flags_critical"] = 5.0

    # Confirmed stroke on imaging
    if features.imaging_positive_stroke:
        score += 5.0
        rationale_components.append("Confirmed stroke on imaging")
        feature_importance["imaging_positive_stroke"] = 5.0

    # === TIME-CRITICAL STROKE WINDOW ===
    # Within 2-hour optimal thrombolysis window + high stroke suspicion
    if features.symptom_duration_hours < 2.0:
        if features.stroke_probability > 0.7 or features.hints_central:
            score += 3.0
            rationale_components.append(
                f"Within thrombolysis window ({features.symptom_duration_hours:.1f} hours)"
            )
            feature_importance["symptom_duration_hours"] = 3.0

    # Within extended thrombolysis window (2-4.5 hours)
    elif features.symptom_duration_hours < 4.5:
        if features.stroke_probability > 0.7 or features.hints_central:
            score += 2.0
            rationale_components.append(
                f"Within extended thrombolysis window ({features.symptom_duration_hours:.1f} hours)"
            )
            feature_importance["symptom_duration_hours"] = 2.0

    # === CLINICAL FINDINGS ===
    # HINTS exam (strongest clinical predictor)
    if features.hints_performed:
        if features.hints_central:
            score += 3.0
            rationale_components.append("HINTS exam suggests central cause")
            feature_importance["hints_central"] = 3.0
        else:
            score -= 1.5
            rationale_components.append("HINTS exam suggests peripheral cause (benign)")
            feature_importance["hints_central"] = -1.5
            confidence = min(confidence, 0.9)  # Peripheral HINTS is reassuring
    else:
        uncertainty_factors.append("HINTS exam not performed")
        confidence *= 0.9

    # Focal neurological deficits
    neuro_deficit_count = sum(
        [
            features.focal_neurological_deficit,
            features.ataxia,
            features.dysarthria,
            features.diplopia,
        ]
    )
    if neuro_deficit_count >= 2:
        score += 3.0
        rationale_components.append(
            f"Multiple neurological deficits ({neuro_deficit_count})"
        )
        feature_importance["neurological_deficits"] = 3.0
    elif neuro_deficit_count == 1:
        score += 1.5
        rationale_components.append("Single neurological deficit")
        feature_importance["neurological_deficits"] = 1.5

    # === STROKE RISK FACTORS ===
    risk_factor_score = 0.0
    risk_factor_count = 0

    if features.atrial_fibrillation:
        risk_factor_score += 2.0
        risk_factor_count += 1
        rationale_components.append("Atrial fibrillation (high stroke risk)")

    if features.prior_stroke:
        risk_factor_score += 2.0
        risk_factor_count += 1
        rationale_components.append("Prior stroke/TIA")

    if features.age >= 75:
        risk_factor_score += 1.5
        risk_factor_count += 1
        rationale_components.append(f"Age {features.age} (increased stroke risk)")
    elif features.age >= 60:
        risk_factor_score += 0.5
        risk_factor_count += 1

    if features.hypertension:
        risk_factor_score += 0.5
        risk_factor_count += 1

    if features.diabetes:
        risk_factor_score += 0.5
        risk_factor_count += 1

    score += risk_factor_score
    if risk_factor_count > 0:
        feature_importance["risk_factors"] = risk_factor_score

    # === DIGITAL TWIN PREDICTION ===
    if features.predicted_disease is not None:
        if features.predicted_disease == "stroke":
            score += 3.0
            rationale_components.append(
                f"Digital twin predicts stroke (probability: {features.stroke_probability:.2f})"
            )
            feature_importance["digital_twin_stroke"] = 3.0
        elif features.predicted_disease == "bppv":
            score -= 2.0
            rationale_components.append("Digital twin predicts BPPV (benign)")
            feature_importance["digital_twin_bppv"] = -2.0
            confidence = max(confidence, 0.85)
        elif features.predicted_disease == "vestibular_neuritis":
            score += 0.0  # Neutral, needs specialist but not emergency
            rationale_components.append("Digital twin predicts vestibular neuritis")
    else:
        uncertainty_factors.append("No digital twin prediction available")
        confidence *= 0.95

    # === SYMPTOM PATTERN (BPPV indicators reduce urgency) ===
    if features.positional_trigger and not features.continuous_vertigo:
        if not features.hints_central and neuro_deficit_count == 0:
            score -= 2.0
            rationale_components.append("Positional episodic vertigo (suggests BPPV)")
            feature_importance["positional_trigger"] = -2.0

    # === IMAGING ===
    if features.imaging_performed and not features.imaging_positive_stroke:
        score -= 1.0  # Negative imaging is reassuring
        rationale_components.append("Imaging negative for acute stroke")
        feature_importance["imaging_negative"] = -1.0

    # === ESI TRIAGE INTEGRATION ===
    if esi_level is not None:
        if esi_level == 1:
            score += 2.0
            rationale_components.append("ESI Level 1 (immediate life threat)")
        elif esi_level == 2:
            score += 1.0
            rationale_components.append("ESI Level 2 (high risk)")
        feature_importance["esi_level"] = (6 - esi_level) * 0.5

    # === TITRATE RISK SCORE INTEGRATION ===
    if titrate_risk_score is not None:
        # TiTrATE score is 0-10, maps directly to scoring
        titrate_contribution = (titrate_risk_score - 5.0) * 0.5  # Scale to contribution
        score += titrate_contribution
        rationale_components.append(f"TiTrATE risk score: {titrate_risk_score:.1f}")
        feature_importance["titrate_risk_score"] = titrate_contribution

    # === MAP SCORE TO DRAS LEVEL ===
    dras_level, time_sensitivity, pathway = _map_score_to_dras(score, features)

    # === CONFIDENCE ADJUSTMENT ===
    # Lower confidence if multiple uncertainty factors
    if len(uncertainty_factors) >= 2:
        confidence *= 0.85

    # === BUILD RATIONALE ===
    rationale = " | ".join(rationale_components)
    if not rationale:
        rationale = f"Score: {score:.1f} (neutral assessment)"

    return DRAS5Classification(
        level=dras_level,
        confidence=confidence,
        rationale=rationale,
        feature_importance=feature_importance,
        recommended_pathway=pathway,
        time_sensitivity=time_sensitivity,
        uncertainty_factors=uncertainty_factors,
    )


def _map_score_to_dras(
    score: float, features: DRAS5Features
) -> Tuple[DRAS5Level, str, str]:
    """Map numerical score to DRAS level.

    Thresholds (with safety margin for stroke):
    - Score >= 8.0 → DRAS-5 (Immediate Emergency)
    - Score >= 6.0 → DRAS-4 (Urgent Specialist)
    - Score >= 4.0 → DRAS-3 (Scheduled Specialist)
    - Score >= 2.0 → DRAS-2 (Lower Urgency)
    - Score < 2.0 → DRAS-1 (Safe)

    Returns:
        Tuple of (DRAS5Level, time_sensitivity, care_pathway)
    """
    if score >= 8.0:
        return (
            DRAS5Level.LEVEL_5,
            "IMMEDIATE (within 30 minutes)",
            "ED → Stroke Protocol → CT/MRI → Neurology Stat → Thrombolysis Evaluation",
        )

    elif score >= 6.0:
        return (
            DRAS5Level.LEVEL_4,
            "URGENT (same-day, within 4 hours)",
            "ED → Urgent Neurology Consult → Imaging → Admission vs Observation",
        )

    elif score >= 4.0:
        # DRAS-3: Key for ED crowding reduction
        # Route benign causes (BPPV, VN) to outpatient specialist
        if features.hints_performed and not features.hints_central:
            pathway = "ED Evaluation → ENT/Neurology Outpatient (1-7 days) → Discharge with Precautions"
        else:
            pathway = "ED Evaluation → Specialist Outpatient (1-7 days) → Discharge with Precautions"

        return (DRAS5Level.LEVEL_3, "SCHEDULED (1-7 days acceptable)", pathway)

    elif score >= 2.0:
        return (
            DRAS5Level.LEVEL_2,
            "ROUTINE (1-2 weeks acceptable)",
            "GP Evaluation → Specialist Referral if Needed → Conservative Management",
        )

    else:
        return (
            DRAS5Level.LEVEL_1,
            "NON-URGENT (as needed)",
            "Self-Care → GP Follow-up if Symptoms Persist",
        )


def calibrate_thresholds(
    validation_data: List[Dict],
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.80,
) -> Dict[str, float]:
    """Calibrate DRAS-5 thresholds to achieve target sensitivity/specificity.

    This is typically done on a validation dataset with known outcomes.

    For stroke detection:
    - High sensitivity (>90%) is critical (minimize false negatives)
    - Moderate specificity (70-80%) acceptable (some false alarms OK)

    Args:
        validation_data: List of cases with features and true outcomes
        target_sensitivity: Target sensitivity for stroke detection (default 0.95)
        target_specificity: Target specificity (default 0.80)

    Returns:
        Dictionary with calibrated thresholds

    Example:
        >>> validation_data = [
        ...     {"features": {...}, "true_stroke": True, "true_dras": 5},
        ...     {"features": {...}, "true_stroke": False, "true_dras": 3},
        ...     # ... more cases
        ... ]
        >>> thresholds = calibrate_thresholds(validation_data)
        >>> print(thresholds)
        {'dras_5_threshold': 7.5, 'dras_4_threshold': 5.8, ...}
    """
    # Extract scores and true outcomes
    scores = []
    true_strokes = []
    true_dras_levels = []

    for case in validation_data:
        features = case["features"]
        # Calculate score for this case
        classification = classify_urgency_level(features)
        score = sum(classification.feature_importance.values())  # Reconstruct score
        scores.append(score)
        true_strokes.append(case.get("true_stroke", False))
        true_dras_levels.append(case.get("true_dras", 3))

    scores = np.array(scores)
    true_strokes = np.array(true_strokes)

    # Find DRAS-5/4 threshold (for stroke detection)
    # Sweep through thresholds to find best sensitivity/specificity trade-off
    thresholds_to_test = np.arange(4.0, 10.0, 0.1)
    best_threshold = 8.0
    best_metric = 0.0

    for threshold in thresholds_to_test:
        predicted_high_risk = scores >= threshold

        # Calculate sensitivity and specificity for stroke detection
        tp = np.sum(predicted_high_risk & true_strokes)
        fn = np.sum(~predicted_high_risk & true_strokes)
        fp = np.sum(predicted_high_risk & ~true_strokes)
        tn = np.sum(~predicted_high_risk & ~true_strokes)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Prioritize sensitivity (stroke detection critical)
        if sensitivity >= target_sensitivity:
            metric = sensitivity + 0.5 * specificity  # Weight sensitivity 2x
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold

    return {
        "dras_5_threshold": best_threshold,
        "dras_4_threshold": best_threshold - 2.0,
        "dras_3_threshold": best_threshold - 4.0,
        "dras_2_threshold": best_threshold - 6.0,
        "target_sensitivity": target_sensitivity,
        "target_specificity": target_specificity,
        "achieved_sensitivity": best_metric / 1.5,  # Approximate
    }


def explain_classification(classification: DRAS5Classification) -> str:
    """Generate human-readable explanation of DRAS-5 classification.

    Uses SHAP-style feature importance to explain decision.

    Args:
        classification: DRAS5Classification to explain

    Returns:
        Human-readable explanation string

    Example:
        >>> features = DRAS5Features(age=75, atrial_fibrillation=True, hints_central=True)
        >>> classification = classify_urgency_level(features)
        >>> explanation = explain_classification(classification)
        >>> print(explanation)
        DRAS-5 Level: LEVEL_5 (Immediate Emergency)
        Confidence: 0.85
        Key factors (in order of importance):
          1. HINTS central (+3.0): Suggests central (stroke) cause
          2. Atrial fibrillation (+2.0): High stroke risk
          3. Age 75 (+1.5): Increased stroke risk
        Recommended pathway: ED → Stroke Protocol → ...
    """
    lines = []
    lines.append(
        f"DRAS-5 Level: {classification.level.name} ({classification.level.value})"
    )
    lines.append(f"Confidence: {classification.confidence:.2f}")

    if classification.feature_importance:
        lines.append("\nKey factors (in order of importance):")
        # Sort by absolute importance
        sorted_features = sorted(
            classification.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            direction = "increases" if importance > 0 else "decreases"
            lines.append(f"  {i}. {feature} ({importance:+.1f}): {direction} urgency")

    lines.append(f"\nTime sensitivity: {classification.time_sensitivity}")
    lines.append(f"Recommended pathway: {classification.recommended_pathway}")

    if classification.uncertainty_factors:
        lines.append(f"\nUncertainty factors:")
        for factor in classification.uncertainty_factors:
            lines.append(f"  - {factor}")

    lines.append(f"\nRationale: {classification.rationale}")

    return "\n".join(lines)
