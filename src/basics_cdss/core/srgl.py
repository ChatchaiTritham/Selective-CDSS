"""SRGL: Screening-First Risk Governance Logic.

This module implements the SRGL component of the TRI-X framework, which
provides a safety governance layer that screens for red flags BEFORE
risk stratification and ensures safety protocols are in place throughout
the clinical decision process.

SRGL Philosophy:
- Screen FIRST: Check for immediate danger signs before any risk calculation
- Risk governance: Ensure safety protocols at every decision point
- Logic-based: Convert clinical guidelines to explicit logic rules

For dizziness:
- Red flags: Stroke warning signs (BEFAST), severe vital sign abnormalities
- Safety protocols: Ensure no contraindications to treatments
- Governance: Override mechanisms for clinical judgment
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RedFlagCategory(Enum):
    """Categories of red flags requiring immediate attention."""

    STROKE_SIGNS = "stroke_signs"  # BEFAST signs
    VITAL_INSTABILITY = "vital_instability"  # Hemodynamic instability
    ALTERED_CONSCIOUSNESS = "altered_consciousness"  # GCS < 15
    SEVERE_HEADACHE = "severe_headache"  # Thunderclap or worst headache
    HEMORRHAGE_RISK = "hemorrhage_risk"  # On anticoagulation with head trauma


class SafetyProtocol(Enum):
    """Safety protocols that must be checked."""

    THROMBOLYSIS_ELIGIBILITY = "thrombolysis_eligibility"  # No contraindications to tPA
    IMAGING_SAFETY = "imaging_safety"  # MRI contraindications (pacemaker, etc.)
    MEDICATION_SAFETY = "medication_safety"  # Drug interactions, allergies
    DISCHARGE_SAFETY = (
        "discharge_safety"  # Safe to discharge (home support, return precautions)
    )
    PROCEDURE_CONSENT = "procedure_consent"  # Informed consent for procedures


@dataclass
class RedFlag:
    """A single red flag finding."""

    category: RedFlagCategory
    description: str
    severity: int  # 1-5 (5 = most severe)
    action_required: str  # What must be done immediately
    timestamp: str


@dataclass
class SafetyCheck:
    """A safety protocol check result."""

    protocol: SafetyProtocol
    passed: bool
    reason: str = ""
    contraindications: List[str] = field(default_factory=list)


@dataclass
class SRGLScreening:
    """Complete SRGL screening result."""

    red_flags: List[RedFlag] = field(default_factory=list)
    safety_checks: List[SafetyCheck] = field(default_factory=list)
    screening_passed: bool = True  # False if any critical red flags
    governance_override_required: bool = False  # True if clinical judgment needed
    recommendations: List[str] = field(default_factory=list)

    def has_critical_red_flags(self) -> bool:
        """Check if any critical (severity >= 4) red flags present."""
        return any(flag.severity >= 4 for flag in self.red_flags)

    def get_failed_safety_checks(self) -> List[SafetyCheck]:
        """Get list of failed safety checks."""
        return [check for check in self.safety_checks if not check.passed]

    def is_safe_to_proceed(self) -> bool:
        """Check if safe to proceed with proposed actions."""
        # Cannot proceed if critical red flags or failed safety checks
        if self.has_critical_red_flags():
            return False
        if self.get_failed_safety_checks():
            return False
        return True


def perform_red_flag_screening(
    symptoms: Dict[str, any],
    vital_signs: Dict[str, float],
    neurological_exam: Dict[str, bool],
    medical_history: Dict[str, any],
) -> SRGLScreening:
    """Perform red flag screening for dizziness patient.

    This is the FIRST step in TRI-X, before any risk calculation or
    DRAS-5 classification. Red flags mandate immediate action.

    Args:
        symptoms: Patient symptoms (headache, nausea, visual changes, etc.)
        vital_signs: Vital signs (HR, BP, RR, O2 sat, temp)
        neurological_exam: Neurological findings (diplopia, dysarthria, ataxia, etc.)
        medical_history: Relevant history (anticoagulation, prior stroke, etc.)

    Returns:
        SRGLScreening with red flags and recommendations

    Example:
        >>> symptoms = {
        ...     "headache": True,
        ...     "headache_severity": 9,
        ...     "diplopia": True,
        ...     "ataxia": True
        ... }
        >>> vital_signs = {"heart_rate": 85, "systolic_bp": 140}
        >>> neuro_exam = {"diplopia": True, "ataxia": True, "dysarthria": False}
        >>> history = {"anticoagulation": False}
        >>> screening = perform_red_flag_screening(symptoms, vital_signs, neuro_exam, history)
        >>> print(screening.has_critical_red_flags())
        True  # Diplopia + ataxia = stroke concern
    """
    screening = SRGLScreening()

    # Check for BEFAST stroke signs
    _check_stroke_signs(symptoms, neurological_exam, screening)

    # Check vital sign stability
    _check_vital_stability(vital_signs, screening)

    # Check consciousness level
    _check_consciousness(symptoms, neurological_exam, screening)

    # Check for severe headache
    _check_severe_headache(symptoms, screening)

    # Check for hemorrhage risk
    _check_hemorrhage_risk(symptoms, medical_history, screening)

    # Set overall screening status
    if screening.has_critical_red_flags():
        screening.screening_passed = False
        screening.recommendations.append(
            "CRITICAL RED FLAGS IDENTIFIED - IMMEDIATE PHYSICIAN EVALUATION REQUIRED"
        )

    return screening


def _check_stroke_signs(
    symptoms: Dict[str, any],
    neurological_exam: Dict[str, bool],
    screening: SRGLScreening,
):
    """Check for BEFAST stroke warning signs.

    BEFAST:
    - Balance: Sudden trouble with balance or coordination (ataxia)
    - Eyes: Sudden vision changes (diplopia, vision loss)
    - Face: Facial droop
    - Arms: Arm weakness or numbness
    - Speech: Slurred speech or difficulty speaking
    - Time: Time to call 911 (act fast)
    """
    stroke_signs_present = []

    # Balance
    if neurological_exam.get("ataxia", False):
        stroke_signs_present.append("ataxia")

    # Eyes
    if neurological_exam.get("diplopia", False):
        stroke_signs_present.append("diplopia")
    if symptoms.get("vision_loss", False):
        stroke_signs_present.append("vision_loss")

    # Face
    if neurological_exam.get("facial_droop", False):
        stroke_signs_present.append("facial_droop")

    # Arms
    if neurological_exam.get("arm_weakness", False):
        stroke_signs_present.append("arm_weakness")
    if symptoms.get("numbness", False):
        stroke_signs_present.append("numbness")

    # Speech
    if neurological_exam.get("dysarthria", False):
        stroke_signs_present.append("dysarthria")
    if neurological_exam.get("aphasia", False):
        stroke_signs_present.append("aphasia")

    # If multiple BEFAST signs, critical red flag
    if len(stroke_signs_present) >= 2:
        screening.red_flags.append(
            RedFlag(
                category=RedFlagCategory.STROKE_SIGNS,
                description=f"Multiple stroke warning signs: {', '.join(stroke_signs_present)}",
                severity=5,  # Critical
                action_required="Immediate stroke protocol activation, imaging within 30 minutes",
                timestamp="",
            )
        )
    elif len(stroke_signs_present) == 1:
        screening.red_flags.append(
            RedFlag(
                category=RedFlagCategory.STROKE_SIGNS,
                description=f"Stroke warning sign present: {stroke_signs_present[0]}",
                severity=4,  # High
                action_required="Urgent neurology consult, imaging within 1 hour",
                timestamp="",
            )
        )


def _check_vital_stability(vital_signs: Dict[str, float], screening: SRGLScreening):
    """Check for hemodynamic instability."""
    instability_findings = []

    # Severe hypotension (SBP < 90)
    if vital_signs.get("systolic_bp", 120) < 90:
        instability_findings.append("severe hypotension (SBP < 90)")

    # Severe hypertension (SBP > 220 or DBP > 120)
    if vital_signs.get("systolic_bp", 120) > 220:
        instability_findings.append("severe hypertension (SBP > 220)")
    if vital_signs.get("diastolic_bp", 80) > 120:
        instability_findings.append("hypertensive emergency (DBP > 120)")

    # Severe tachycardia (HR > 150)
    if vital_signs.get("heart_rate", 80) > 150:
        instability_findings.append("severe tachycardia (HR > 150)")

    # Severe bradycardia (HR < 40)
    if vital_signs.get("heart_rate", 80) < 40:
        instability_findings.append("severe bradycardia (HR < 40)")

    # Hypoxemia (O2 sat < 90%)
    if vital_signs.get("oxygen_saturation", 98) < 90:
        instability_findings.append("hypoxemia (O2 sat < 90%)")

    if instability_findings:
        severity = 5 if len(instability_findings) >= 2 else 4
        screening.red_flags.append(
            RedFlag(
                category=RedFlagCategory.VITAL_INSTABILITY,
                description=f"Vital sign instability: {', '.join(instability_findings)}",
                severity=severity,
                action_required="Immediate resuscitation, consider ICU",
                timestamp="",
            )
        )


def _check_consciousness(
    symptoms: Dict[str, any],
    neurological_exam: Dict[str, bool],
    screening: SRGLScreening,
):
    """Check for altered level of consciousness."""
    if symptoms.get("altered_mental_status", False):
        gcs = symptoms.get("gcs", 15)  # Glasgow Coma Scale
        if gcs < 13:
            screening.red_flags.append(
                RedFlag(
                    category=RedFlagCategory.ALTERED_CONSCIOUSNESS,
                    description=f"Altered consciousness (GCS {gcs})",
                    severity=5,  # Critical
                    action_required="Immediate airway assessment, imaging, neurology consult",
                    timestamp="",
                )
            )
        elif gcs < 15:
            screening.red_flags.append(
                RedFlag(
                    category=RedFlagCategory.ALTERED_CONSCIOUSNESS,
                    description=f"Mildly altered consciousness (GCS {gcs})",
                    severity=4,  # High
                    action_required="Close monitoring, consider imaging",
                    timestamp="",
                )
            )


def _check_severe_headache(symptoms: Dict[str, any], screening: SRGLScreening):
    """Check for severe or thunderclap headache.

    Thunderclap headache (sudden, severe, "worst headache of life") suggests:
    - Subarachnoid hemorrhage
    - Vertebral artery dissection
    - Intracranial hemorrhage
    """
    if symptoms.get("headache", False):
        severity = symptoms.get("headache_severity", 0)
        sudden_onset = symptoms.get("headache_sudden_onset", False)

        # Thunderclap headache (severe + sudden)
        if severity >= 8 and sudden_onset:
            screening.red_flags.append(
                RedFlag(
                    category=RedFlagCategory.SEVERE_HEADACHE,
                    description="Thunderclap headache (sudden, severe)",
                    severity=5,  # Critical
                    action_required="Immediate CT head, consider subarachnoid hemorrhage or dissection",
                    timestamp="",
                )
            )
        # Severe headache alone
        elif severity >= 8:
            screening.red_flags.append(
                RedFlag(
                    category=RedFlagCategory.SEVERE_HEADACHE,
                    description=f"Severe headache (severity {severity}/10)",
                    severity=3,  # Moderate
                    action_required="Consider imaging if persistent or associated neurological signs",
                    timestamp="",
                )
            )


def _check_hemorrhage_risk(
    symptoms: Dict[str, any], medical_history: Dict[str, any], screening: SRGLScreening
):
    """Check for hemorrhage risk (anticoagulation + head trauma)."""
    anticoagulation = medical_history.get("anticoagulation", False)
    head_trauma = symptoms.get("head_trauma_recent", False)

    if anticoagulation and head_trauma:
        screening.red_flags.append(
            RedFlag(
                category=RedFlagCategory.HEMORRHAGE_RISK,
                description="Anticoagulation + recent head trauma",
                severity=5,  # Critical
                action_required="Immediate CT head to rule out intracranial hemorrhage",
                timestamp="",
            )
        )
    elif anticoagulation:
        # Anticoagulation alone is not a red flag but requires caution
        screening.recommendations.append(
            "Patient on anticoagulation - consider lower threshold for imaging if any neurological signs"
        )


def check_safety_protocols(
    proposed_action: str,
    patient_data: Dict[str, any],
    clinical_findings: Dict[str, any],
) -> SafetyCheck:
    """Check if proposed action is safe given patient data.

    Args:
        proposed_action: Action being considered ("thrombolysis", "mri", "discharge", etc.)
        patient_data: Patient demographics, medical history, medications
        clinical_findings: Current clinical findings

    Returns:
        SafetyCheck indicating if action is safe

    Example:
        >>> patient = {
        ...     "age": 75,
        ...     "recent_surgery": True,
        ...     "on_anticoagulation": True
        ... }
        >>> findings = {"imaging_result": "acute_infarct", "time_from_onset_hours": 2.0}
        >>> check = check_safety_protocols("thrombolysis", patient, findings)
        >>> print(check.passed)
        False  # Recent surgery = contraindication
    """
    if proposed_action == "thrombolysis":
        return _check_thrombolysis_safety(patient_data, clinical_findings)
    elif proposed_action == "mri":
        return _check_mri_safety(patient_data)
    elif proposed_action == "discharge":
        return _check_discharge_safety(patient_data, clinical_findings)
    elif proposed_action == "vestibular_suppressant":
        return _check_medication_safety(patient_data, "vestibular_suppressant")
    else:
        # Default: assume safe
        return SafetyCheck(
            protocol=SafetyProtocol.PROCEDURE_CONSENT,
            passed=True,
            reason=f"No specific safety contraindications identified for {proposed_action}",
        )


def _check_thrombolysis_safety(
    patient_data: Dict[str, any], clinical_findings: Dict[str, any]
) -> SafetyCheck:
    """Check for thrombolysis (tPA) contraindications.

    Absolute contraindications:
    - Recent surgery or trauma (<3 months)
    - Active bleeding
    - Hemorrhagic stroke on imaging
    - BP >185/110 uncontrolled
    - Recent stroke <3 months
    - Intracranial hemorrhage history
    """
    contraindications = []

    # Time window (4.5 hours for IV tPA)
    time_from_onset = clinical_findings.get("time_from_onset_hours", 999)
    if time_from_onset > 4.5:
        contraindications.append(
            f"Outside time window ({time_from_onset:.1f} hours > 4.5 hours)"
        )

    # Hemorrhage on imaging
    if clinical_findings.get("imaging_result") == "hemorrhage":
        contraindications.append("Hemorrhagic stroke on imaging")

    # Recent surgery or trauma
    if patient_data.get("recent_surgery", False):
        contraindications.append("Recent surgery (<3 months)")
    if patient_data.get("recent_trauma", False):
        contraindications.append("Recent major trauma (<3 months)")

    # Active bleeding
    if patient_data.get("active_bleeding", False):
        contraindications.append("Active bleeding")

    # Uncontrolled hypertension
    systolic_bp = clinical_findings.get("systolic_bp", 120)
    diastolic_bp = clinical_findings.get("diastolic_bp", 80)
    if systolic_bp > 185 or diastolic_bp > 110:
        contraindications.append(
            f"Uncontrolled hypertension (BP {systolic_bp}/{diastolic_bp})"
        )

    # Recent stroke
    if patient_data.get("stroke_within_3_months", False):
        contraindications.append("Recent stroke (<3 months)")

    # Prior ICH
    if patient_data.get("prior_intracranial_hemorrhage", False):
        contraindications.append("History of intracranial hemorrhage")

    # Determine pass/fail
    passed = len(contraindications) == 0

    return SafetyCheck(
        protocol=SafetyProtocol.THROMBOLYSIS_ELIGIBILITY,
        passed=passed,
        reason="Thrombolysis eligible" if passed else "Thrombolysis contraindicated",
        contraindications=contraindications,
    )


def _check_mri_safety(patient_data: Dict[str, any]) -> SafetyCheck:
    """Check for MRI contraindications."""
    contraindications = []

    # Pacemaker (most MRIs not compatible)
    if patient_data.get("pacemaker", False):
        if not patient_data.get("mri_compatible_pacemaker", False):
            contraindications.append("Cardiac pacemaker (non-MRI compatible)")

    # Metallic implants
    if patient_data.get("metallic_implant", False):
        implant_type = patient_data.get("implant_type", "unknown")
        contraindications.append(
            f"Metallic implant ({implant_type}) - needs verification"
        )

    # Severe claustrophobia (relative contraindication)
    if patient_data.get("severe_claustrophobia", False):
        contraindications.append(
            "Severe claustrophobia (may require sedation or open MRI)"
        )

    passed = len(contraindications) == 0

    return SafetyCheck(
        protocol=SafetyProtocol.IMAGING_SAFETY,
        passed=passed,
        reason="MRI safe" if passed else "MRI contraindications present - consider CT",
        contraindications=contraindications,
    )


def _check_discharge_safety(
    patient_data: Dict[str, any], clinical_findings: Dict[str, any]
) -> SafetyCheck:
    """Check if safe to discharge patient."""
    contraindications = []

    # Unstable vitals
    if clinical_findings.get("vitals_abnormal", False):
        contraindications.append("Unstable vital signs")

    # Ongoing symptoms severity
    symptom_severity = clinical_findings.get("symptom_severity", 0)
    if symptom_severity >= 7:
        contraindications.append(
            f"Severe ongoing symptoms (severity {symptom_severity}/10)"
        )

    # No home support
    if not patient_data.get("home_support", True):
        if patient_data.get("age", 0) > 75 or clinical_findings.get("ataxia", False):
            contraindications.append("No home support + high fall risk")

    # Unable to understand return precautions
    if not patient_data.get("understands_precautions", True):
        contraindications.append("Unable to understand return precautions")

    # Suspected stroke but imaging not done
    if clinical_findings.get("stroke_concern", False) and not clinical_findings.get(
        "imaging_performed", False
    ):
        contraindications.append("Stroke concern but imaging not performed")

    passed = len(contraindications) == 0

    return SafetyCheck(
        protocol=SafetyProtocol.DISCHARGE_SAFETY,
        passed=passed,
        reason="Safe for discharge" if passed else "Discharge safety concerns",
        contraindications=contraindications,
    )


def _check_medication_safety(
    patient_data: Dict[str, any], medication: str
) -> SafetyCheck:
    """Check medication safety (drug interactions, allergies)."""
    contraindications = []

    # Allergy check
    allergies = patient_data.get("allergies", [])
    if medication in allergies or "meclizine" in allergies:
        contraindications.append(f"Allergy to {medication}")

    # Vestibular suppressant specific checks
    if medication == "vestibular_suppressant":
        # Elderly: Increased fall risk with sedation
        if patient_data.get("age", 0) > 75:
            contraindications.append(
                "Age >75: High fall risk with vestibular suppressant sedation"
            )

        # Should not use >3 days (delays central compensation)
        contraindications.append(
            "CAUTION: Use maximum 3 days (delays vestibular compensation if prolonged)"
        )

    passed = len(contraindications) == 0

    return SafetyCheck(
        protocol=SafetyProtocol.MEDICATION_SAFETY,
        passed=passed,
        reason="Medication safe" if passed else "Medication safety concerns",
        contraindications=contraindications,
    )
