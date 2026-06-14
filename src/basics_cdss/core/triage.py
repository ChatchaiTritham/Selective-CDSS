"""ESI Triage Assessment for Dizziness and Vertigo.

This module implements the Emergency Severity Index (ESI) triage system
specifically adapted for dizziness and vertigo presentations. It provides
initial urgency classification that feeds into the TRI-X framework.

The ESI is a 5-level triage system:
- Level 1: Immediate life threat (requires immediate physician evaluation)
- Level 2: High risk/severe pain or distress (should be seen quickly)
- Level 3: Multiple resources needed (stable but complex)
- Level 4: One resource needed (simple problem)
- Level 5: No resources needed (minor issue)

For dizziness, we focus on identifying life-threatening central causes
(particularly stroke) while efficiently routing benign peripheral causes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class ESILevel(Enum):
    """Emergency Severity Index levels."""

    LEVEL_1 = 1  # Immediate life threat
    LEVEL_2 = 2  # High risk, emergent
    LEVEL_3 = 3  # Multiple resources, urgent
    LEVEL_4 = 4  # One resource, less urgent
    LEVEL_5 = 5  # No resources, non-urgent


@dataclass
class VitalSigns:
    """Patient vital signs for triage assessment."""

    heart_rate: float  # beats per minute
    systolic_bp: float  # mmHg
    diastolic_bp: float  # mmHg
    respiratory_rate: float  # breaths per minute
    temperature: float  # Celsius
    oxygen_saturation: float  # percentage

    def is_abnormal(self) -> bool:
        """Check if any vital signs are abnormal."""
        abnormal = False

        # Heart rate: Normal 60-100 bpm
        if self.heart_rate < 50 or self.heart_rate > 110:
            abnormal = True

        # Blood pressure: Systolic <90 or >200, Diastolic >120
        if self.systolic_bp < 90 or self.systolic_bp > 200:
            abnormal = True
        if self.diastolic_bp > 120:
            abnormal = True

        # Respiratory rate: Normal 12-20
        if self.respiratory_rate < 10 or self.respiratory_rate > 24:
            abnormal = True

        # Temperature: Fever >38.0°C
        if self.temperature > 38.0:
            abnormal = True

        # Oxygen saturation: <92%
        if self.oxygen_saturation < 92:
            abnormal = True

        return abnormal


@dataclass
class DizzinessSymptoms:
    """Dizziness-specific symptoms for triage."""

    # Symptom description (patient reported)
    symptom_description: (
        str  # "dizzy", "vertigo", "spinning", "lightheaded", "imbalance"
    )
    duration_hours: float  # Time since onset
    episode_duration_seconds: Optional[
        float
    ]  # Duration of individual episodes (for episodic vertigo)

    # Associated symptoms (stroke warning signs)
    headache: bool = False
    headache_severity: int = 0  # 0-10 scale
    nausea_vomiting: bool = False
    diplopia: bool = False  # Double vision
    dysarthria: bool = False  # Slurred speech
    dysphagia: bool = False  # Difficulty swallowing
    ataxia: bool = False  # Inability to walk
    numbness_weakness: bool = False  # Focal neurological deficit
    altered_mental_status: bool = False
    hearing_loss: bool = False
    tinnitus: bool = False

    # Triggers
    positional_trigger: bool = False  # Triggers with head movement (suggests BPPV)
    continuous_vertigo: bool = False  # Continuous vs episodic (suggests VN vs BPPV)

    def has_central_warning_signs(self) -> bool:
        """Check for warning signs of central (stroke) cause."""
        # BEFAST signs: Balance, Eyes, Face, Arms, Speech, Time
        # In dizzy patient: diplopia, dysarthria, dysphagia, ataxia, numbness/weakness
        central_signs = [
            self.diplopia,
            self.dysarthria,
            self.dysphagia,
            self.ataxia and not self.positional_trigger,  # Ataxia with BPPV is benign
            self.numbness_weakness,
            self.altered_mental_status,
            self.headache and self.headache_severity >= 7,  # Severe sudden headache
        ]
        return any(central_signs)

    def suggests_bppv(self) -> bool:
        """Check if symptoms suggest benign BPPV."""
        # Classic BPPV: Positional, brief episodes (<60 sec), no neurological signs
        if self.positional_trigger and not self.continuous_vertigo:
            if (
                self.episode_duration_seconds is not None
                and self.episode_duration_seconds < 60
            ):
                if not self.has_central_warning_signs():
                    return True
        return False


@dataclass
class PatientRiskFactors:
    """Patient risk factors for stroke."""

    age: int
    atrial_fibrillation: bool = False
    hypertension: bool = False
    diabetes: bool = False
    prior_stroke_tia: bool = False
    coronary_artery_disease: bool = False
    smoking: bool = False
    anticoagulation: bool = False  # On anticoagulants

    def calculate_stroke_risk_score(self) -> float:
        """Calculate stroke risk score based on risk factors."""
        score = 0.0

        # Age (strongest risk factor)
        if self.age >= 60:
            score += 2.0
        elif self.age >= 50:
            score += 1.0

        # Atrial fibrillation (very high risk for cardioembolic stroke)
        if self.atrial_fibrillation:
            score += 3.0

        # Prior stroke/TIA (highest risk)
        if self.prior_stroke_tia:
            score += 3.0

        # Hypertension (modifiable risk factor)
        if self.hypertension:
            score += 1.0

        # Diabetes
        if self.diabetes:
            score += 1.0

        # CAD
        if self.coronary_artery_disease:
            score += 1.0

        # Smoking
        if self.smoking:
            score += 0.5

        return score


@dataclass
class ESITriageAssessment:
    """Complete ESI triage assessment for dizziness patient."""

    vital_signs: VitalSigns
    symptoms: DizzinessSymptoms
    risk_factors: PatientRiskFactors
    esi_level: Optional[ESILevel] = None
    initial_dras_level: Optional[int] = None  # Mapped to DRAS-5 (1-5)
    rationale: str = ""

    def perform_assessment(self) -> ESILevel:
        """Perform ESI triage assessment.

        Decision tree:
        1. Does patient require immediate life-saving intervention? → ESI 1
        2. Is patient high risk or in severe distress? → ESI 2
        3. How many resources needed? ≥2 → ESI 3, 1 → ESI 4, 0 → ESI 5

        For dizziness:
        - ESI 1: Altered mental status, unstable vitals, clear stroke
        - ESI 2: High stroke risk, severe symptoms, abnormal vitals
        - ESI 3: Moderate risk, needs imaging/specialist
        - ESI 4: Low risk, simple benign cause (BPPV)
        - ESI 5: Minimal symptoms, no resources needed
        """

        # Step 1: Immediate life-saving intervention needed?
        if self._requires_immediate_intervention():
            self.esi_level = ESILevel.LEVEL_1
            self.rationale = "Requires immediate intervention (altered mental status, unstable vitals, or clear stroke)"
            return ESILevel.LEVEL_1

        # Step 2: High risk situation?
        if self._is_high_risk():
            self.esi_level = ESILevel.LEVEL_2
            self.rationale = "High risk for stroke (central warning signs, high-risk factors, or abnormal vitals)"
            return ESILevel.LEVEL_2

        # Step 3: Resource prediction
        num_resources = self._predict_resources_needed()

        if num_resources >= 2:
            self.esi_level = ESILevel.LEVEL_3
            self.rationale = f"Multiple resources needed ({num_resources}): imaging, specialist consult, labs"
            return ESILevel.LEVEL_3
        elif num_resources == 1:
            self.esi_level = ESILevel.LEVEL_4
            self.rationale = (
                "Single resource needed: clinical exam or simple intervention"
            )
            return ESILevel.LEVEL_4
        else:
            self.esi_level = ESILevel.LEVEL_5
            self.rationale = "No resources needed: reassurance or self-care advice"
            return ESILevel.LEVEL_5

    def _requires_immediate_intervention(self) -> bool:
        """Check if patient requires immediate life-saving intervention."""
        # Altered mental status
        if self.symptoms.altered_mental_status:
            return True

        # Unstable vitals
        if self.vital_signs.systolic_bp < 80:  # Severe hypotension
            return True
        if (
            self.vital_signs.heart_rate < 40 or self.vital_signs.heart_rate > 150
        ):  # Severe brady/tachycardia
            return True
        if (
            self.vital_signs.respiratory_rate < 8
            or self.vital_signs.respiratory_rate > 30
        ):  # Severe respiratory distress
            return True
        if self.vital_signs.oxygen_saturation < 85:  # Severe hypoxemia
            return True

        # Clear stroke presentation (multiple central signs)
        central_sign_count = sum(
            [
                self.symptoms.diplopia,
                self.symptoms.dysarthria,
                self.symptoms.dysphagia,
                self.symptoms.ataxia and not self.symptoms.positional_trigger,
                self.symptoms.numbness_weakness,
            ]
        )
        if central_sign_count >= 2:  # Multiple focal deficits = likely stroke
            return True

        return False

    def _is_high_risk(self) -> bool:
        """Check if patient is high risk (ESI 2)."""
        # Central warning signs present
        if self.symptoms.has_central_warning_signs():
            return True

        # High stroke risk score
        stroke_risk = self.risk_factors.calculate_stroke_risk_score()
        if stroke_risk >= 4.0:  # High risk profile
            return True

        # Abnormal vital signs
        if self.vital_signs.is_abnormal():
            return True

        # Acute onset (<6 hours) in high-risk patient
        if self.symptoms.duration_hours < 6 and stroke_risk >= 2.0:
            return True

        return False

    def _predict_resources_needed(self) -> int:
        """Predict number of ED resources needed.

        Resources for dizziness:
        - Neuroimaging (CT/MRI)
        - Specialist consult (neurology, ENT)
        - Laboratory tests (CBC, metabolic panel)
        - EKG (if AFib suspected)
        - Vestibular testing (if available)
        - Clinical exam only
        """
        resources = 0
        stroke_risk = self.risk_factors.calculate_stroke_risk_score()

        # Imaging needed?
        # - Moderate stroke risk (score 2-4) or
        # - Any central warning sign or
        # - Persistent symptoms >24 hours without clear benign cause
        if (
            stroke_risk >= 2.0
            or self.symptoms.has_central_warning_signs()
            or (self.symptoms.duration_hours > 24 and not self.symptoms.suggests_bppv())
        ):
            resources += 1  # CT or MRI

        # Specialist consult needed?
        # - High stroke risk or
        # - Unclear diagnosis or
        # - Persistent symptoms requiring subspecialty
        if stroke_risk >= 2.0 or not self.symptoms.suggests_bppv():
            resources += 1  # Neurology or ENT consult

        # Labs/EKG needed?
        # - New AFib suspected (irregular pulse + risk factors) or
        # - Abnormal vitals suggesting metabolic cause
        if self.risk_factors.atrial_fibrillation or self.vital_signs.is_abnormal():
            resources += 1  # Labs, EKG

        # If clear BPPV (positional, brief episodes, no red flags), only needs exam
        if self.symptoms.suggests_bppv() and stroke_risk < 2.0:
            resources = 1  # Clinical exam + possible Dix-Hallpike

        return resources


def perform_dizziness_triage(
    vital_signs: VitalSigns,
    symptoms: DizzinessSymptoms,
    risk_factors: PatientRiskFactors,
) -> ESITriageAssessment:
    """Perform ESI triage assessment for dizziness patient.

    Args:
        vital_signs: Patient vital signs
        symptoms: Dizziness symptoms and associated features
        risk_factors: Patient risk factors for stroke

    Returns:
        ESITriageAssessment with ESI level and rationale

    Example:
        >>> vitals = VitalSigns(
        ...     heart_rate=85, systolic_bp=140, diastolic_bp=85,
        ...     respiratory_rate=16, temperature=37.0, oxygen_saturation=98
        ... )
        >>> symptoms = DizzinessSymptoms(
        ...     symptom_description="spinning",
        ...     duration_hours=2,
        ...     episode_duration_seconds=45,
        ...     positional_trigger=True,
        ...     continuous_vertigo=False
        ... )
        >>> risk_factors = PatientRiskFactors(age=65, hypertension=True)
        >>> assessment = perform_dizziness_triage(vitals, symptoms, risk_factors)
        >>> print(assessment.esi_level)
        ESILevel.LEVEL_3
    """
    assessment = ESITriageAssessment(
        vital_signs=vital_signs, symptoms=symptoms, risk_factors=risk_factors
    )
    assessment.perform_assessment()
    return assessment


def map_esi_to_dras(esi_level: ESILevel, stroke_risk_score: float) -> int:
    """Map ESI level to initial DRAS-5 classification.

    This provides initial urgency classification that will be refined by
    the TiTrATE assessment and digital twin prediction.

    Mapping strategy:
    - ESI 1 (immediate) → DRAS 5 (immediate emergency)
    - ESI 2 (high risk) → DRAS 4-5 depending on stroke risk
    - ESI 3 (multiple resources) → DRAS 3-4
    - ESI 4 (one resource) → DRAS 2-3
    - ESI 5 (no resources) → DRAS 1

    Args:
        esi_level: ESI triage level
        stroke_risk_score: Calculated stroke risk score

    Returns:
        Initial DRAS level (1-5)

    Example:
        >>> map_esi_to_dras(ESILevel.LEVEL_2, stroke_risk_score=5.0)
        5  # High stroke risk → DRAS-5
        >>> map_esi_to_dras(ESILevel.LEVEL_4, stroke_risk_score=0.5)
        2  # Low risk, simple problem → DRAS-2
    """
    if esi_level == ESILevel.LEVEL_1:
        return 5  # Immediate emergency

    elif esi_level == ESILevel.LEVEL_2:
        # High risk: DRAS 4-5 depending on stroke risk
        if stroke_risk_score >= 4.0:
            return 5  # Very high stroke risk
        else:
            return 4  # Moderate-high stroke risk

    elif esi_level == ESILevel.LEVEL_3:
        # Multiple resources: DRAS 3-4
        if stroke_risk_score >= 3.0:
            return 4  # Urgent specialist
        else:
            return 3  # Scheduled specialist OPD

    elif esi_level == ESILevel.LEVEL_4:
        # One resource: DRAS 2-3
        if stroke_risk_score >= 2.0:
            return 3  # Some concern, outpatient specialist
        else:
            return 2  # Low urgency, GP follow-up

    else:  # ESI Level 5
        return 1  # Safe, no danger


def calculate_uncertainty_in_triage(
    symptoms: DizzinessSymptoms, health_literacy: float = 0.5
) -> float:
    """Calculate uncertainty in triage assessment due to symptom reporting.

    Key TRI-X theme: Patient reporting uncertainty affects triage accuracy.

    Uncertainty factors:
    - Vague symptom description ("dizzy" vs "vertigo")
    - Imprecise timing (critical for stroke window)
    - Forgotten triggers or associated symptoms
    - Health literacy affects reporting accuracy

    Args:
        symptoms: Dizziness symptoms as reported by patient
        health_literacy: Patient health literacy (0-1 scale, default 0.5)

    Returns:
        Uncertainty score (0-1, higher = more uncertain)

    Example:
        >>> symptoms = DizzinessSymptoms(
        ...     symptom_description="dizzy",  # Vague
        ...     duration_hours=3.0,  # Imprecise
        ...     positional_trigger=False
        ... )
        >>> uncertainty = calculate_uncertainty_in_triage(symptoms, health_literacy=0.3)
        >>> print(f"Triage uncertainty: {uncertainty:.2f}")
        Triage uncertainty: 0.65  # High uncertainty
    """
    uncertainty = 0.0

    # Symptom description vagueness
    vague_terms = ["dizzy", "lightheaded", "off-balance", "weird"]
    if any(term in symptoms.symptom_description.lower() for term in vague_terms):
        uncertainty += 0.3

    # Duration imprecision (rounded hours suggest estimate)
    if symptoms.duration_hours == int(symptoms.duration_hours):
        uncertainty += 0.2

    # Missing episode duration (important for BPPV diagnosis)
    if symptoms.positional_trigger and symptoms.episode_duration_seconds is None:
        uncertainty += 0.2

    # Health literacy factor (low literacy → higher uncertainty)
    uncertainty += (1 - health_literacy) * 0.3

    # Cap at 1.0
    return min(uncertainty, 1.0)
