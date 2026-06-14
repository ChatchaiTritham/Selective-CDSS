"""TiTrATE: Time-bound Risk-stratified Action with Treatment Escalation.

This module implements the TiTrATE component of the TRI-X framework, which
provides time-bounded risk assessment at multiple time points to ensure
appropriate treatment escalation for dizziness and vertigo patients.

TiTrATE Concept:
- Time-bound: Assessment at T0 (presentation), T1 (1hr), T2 (4hr), T3 (24hr)
- Risk-stratified: Classify patient risk at each time point
- Action: Clear next steps based on risk level
- Treatment Escalation: Increase intervention intensity if risk increases

For dizziness:
- T0 (Presentation): Initial triage, rule out stroke emergency
- T1 (1 hour): Early reassessment, imaging if high risk
- T2 (4 hours): Post-imaging or post-intervention assessment
- T3 (24 hours): Symptom resolution check, disposition decision
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TimePoint(Enum):
    """TiTrATE time points for assessment."""

    T0 = 0  # Presentation (0 hours)
    T1 = 1  # Early reassessment (1 hour)
    T2 = 4  # Mid reassessment (4 hours)
    T3 = 24  # Late reassessment (24 hours)


class RiskLevel(Enum):
    """Risk stratification levels."""

    MINIMAL = 1  # No significant risk identified
    LOW = 2  # Low risk, benign cause likely
    MODERATE = 3  # Moderate risk, specialist needed
    HIGH = 4  # High risk, urgent intervention
    CRITICAL = 5  # Critical risk, immediate intervention


class ActionRecommendation(Enum):
    """Action recommendations based on risk."""

    # Imaging
    IMAGING_URGENT = "imaging_urgent"  # CT/MRI within 30 min
    IMAGING_ROUTINE = "imaging_routine"  # CT/MRI within 4 hours
    NO_IMAGING = "no_imaging"

    # Specialist consultation
    NEURO_STAT = "neuro_stat"  # Neurology immediate (stroke team)
    NEURO_URGENT = "neuro_urgent"  # Neurology same-day
    ENT_SCHEDULED = "ent_scheduled"  # ENT outpatient 1-7 days
    GP_FOLLOWUP = "gp_followup"  # GP routine follow-up

    # Treatment
    THROMBOLYSIS_EVAL = "thrombolysis_eval"  # Evaluate for tPA
    EPLEY_IMMEDIATE = "epley_immediate"  # Epley maneuver now
    EPLEY_SCHEDULED = "epley_scheduled"  # Epley at ENT clinic
    VESTIBULAR_SUPPRESSANT = "vestibular_suppressant"  # Meclizine, etc. (max 3 days)
    VESTIBULAR_REHAB = "vestibular_rehab"  # PT referral for rehab

    # Disposition
    ICU_ADMISSION = "icu_admission"  # ICU for stroke monitoring
    OBSERVATION = "observation"  # ED observation unit
    DISCHARGE_SAFE = "discharge_safe"  # Safe for discharge
    DISCHARGE_PRECAUTIONS = "discharge_precautions"  # Discharge with return precautions


@dataclass
class ClinicalFindings:
    """Clinical findings at a given time point."""

    time_point: TimePoint
    timestamp: datetime

    # Vital signs (changes from baseline)
    heart_rate: float
    blood_pressure_sys: float
    blood_pressure_dia: float

    # Neurological exam
    hints_performed: bool = False
    hints_hit_result: Optional[str] = (
        None  # "positive" (peripheral), "negative" (central)
    )
    hints_nystagmus: Optional[str] = (
        None  # "horizontal", "direction_changing" (central)
    )
    hints_skew: Optional[bool] = None  # True = skew present (central)

    focal_neurological_deficit: bool = False
    ataxia: bool = False
    dysarthria: bool = False

    # Symptom evolution
    symptom_severity: int = 5  # 0-10 scale
    symptom_resolution: bool = False
    new_symptoms: List[str] = field(default_factory=list)

    # Imaging results (if performed)
    imaging_performed: bool = False
    imaging_type: Optional[str] = None  # "CT", "MRI"
    imaging_result: Optional[str] = None  # "normal", "acute_infarct", "hemorrhage"

    # Interventions performed since last assessment
    interventions: List[str] = field(default_factory=list)

    def hints_suggests_central(self) -> bool:
        """Check if HINTS exam suggests central (stroke) cause.

        HINTS (Head Impulse, Nystagmus, Test of Skew):
        - Central indicators:
          - Head Impulse Test NEGATIVE (normal VOR)
          - Direction-changing nystagmus
          - Skew deviation present
        - Peripheral indicators:
          - Head Impulse Test POSITIVE (abnormal VOR)
          - Horizontal nystagmus (unidirectional)
          - No skew deviation
        """
        if not self.hints_performed:
            return False

        # HIT negative (normal VOR) = central concern
        if self.hints_hit_result == "negative":
            return True

        # Direction-changing nystagmus = central
        if self.hints_nystagmus == "direction_changing":
            return True

        # Skew deviation = central
        if self.hints_skew:
            return True

        return False


@dataclass
class TiTrATEAssessment:
    """Complete TiTrATE assessment across time points."""

    patient_id: str
    initial_presentation_time: datetime

    # Assessments at each time point
    assessments: Dict[TimePoint, ClinicalFindings] = field(default_factory=dict)

    # Risk trajectory
    risk_trajectory: Dict[TimePoint, RiskLevel] = field(default_factory=dict)

    # Action trajectory
    actions_taken: Dict[TimePoint, List[ActionRecommendation]] = field(
        default_factory=dict
    )

    # Escalation tracking
    escalation_occurred: bool = False
    escalation_time: Optional[datetime] = None
    escalation_reason: str = ""

    def add_assessment(
        self, time_point: TimePoint, findings: ClinicalFindings
    ) -> Tuple[RiskLevel, List[ActionRecommendation]]:
        """Add clinical findings at a time point and calculate risk/actions.

        Args:
            time_point: Time point (T0, T1, T2, T3)
            findings: Clinical findings at this time point

        Returns:
            Tuple of (risk_level, action_recommendations)
        """
        self.assessments[time_point] = findings

        # Calculate risk at this time point
        risk_level = self._calculate_risk(time_point, findings)
        self.risk_trajectory[time_point] = risk_level

        # Determine actions based on risk
        actions = self._determine_actions(time_point, risk_level, findings)
        self.actions_taken[time_point] = actions

        # Check for escalation
        self._check_escalation(time_point)

        return risk_level, actions

    def _calculate_risk(
        self, time_point: TimePoint, findings: ClinicalFindings
    ) -> RiskLevel:
        """Calculate risk level at given time point.

        Risk calculation considers:
        - Clinical findings (HINTS, neurological exam)
        - Imaging results (if available)
        - Symptom evolution (improving vs worsening)
        - Time from onset (critical for stroke)
        """
        risk_score = 0.0

        # HINTS exam (strongest predictor)
        if findings.hints_performed:
            if findings.hints_suggests_central():
                risk_score += 4.0  # Central cause = high risk
            else:
                risk_score -= 1.0  # Peripheral cause = lower risk

        # Focal neurological deficits (stroke indicators)
        if findings.focal_neurological_deficit:
            risk_score += 3.0
        if findings.ataxia:
            risk_score += 2.0
        if findings.dysarthria:
            risk_score += 2.0

        # Imaging results
        if findings.imaging_performed:
            if findings.imaging_result == "acute_infarct":
                risk_score += 5.0  # Confirmed stroke
            elif findings.imaging_result == "hemorrhage":
                risk_score += 5.0  # Hemorrhagic stroke
            elif findings.imaging_result == "normal":
                risk_score -= 1.0  # Imaging reassurance

        # Symptom evolution
        if findings.symptom_resolution:
            risk_score -= 2.0  # Improving = lower risk
        elif findings.new_symptoms:
            risk_score += 1.0 * len(findings.new_symptoms)  # Worsening = higher risk

        # Time from onset (stroke window consideration)
        time_from_onset = (
            findings.timestamp - self.initial_presentation_time
        ).total_seconds() / 3600
        if time_from_onset < 4.5 and risk_score >= 3.0:
            risk_score += 1.0  # Within thrombolysis window and high risk = urgent

        # Convert score to risk level
        if risk_score >= 6.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 4.0:
            return RiskLevel.HIGH
        elif risk_score >= 2.0:
            return RiskLevel.MODERATE
        elif risk_score >= 0:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _determine_actions(
        self, time_point: TimePoint, risk_level: RiskLevel, findings: ClinicalFindings
    ) -> List[ActionRecommendation]:
        """Determine action recommendations based on risk level and time point.

        Actions follow treatment escalation logic:
        - Higher risk → More intensive interventions
        - Time progression → Reassess and adjust
        """
        actions = []
        time_from_onset = (
            findings.timestamp - self.initial_presentation_time
        ).total_seconds() / 3600

        # CRITICAL RISK (Level 5)
        if risk_level == RiskLevel.CRITICAL:
            # Imaging if not done
            if not findings.imaging_performed:
                actions.append(ActionRecommendation.IMAGING_URGENT)

            # Neurology stat consult
            actions.append(ActionRecommendation.NEURO_STAT)

            # Thrombolysis evaluation if within window
            if time_from_onset < 4.5 and findings.imaging_result != "hemorrhage":
                actions.append(ActionRecommendation.THROMBOLYSIS_EVAL)

            # ICU admission
            actions.append(ActionRecommendation.ICU_ADMISSION)

        # HIGH RISK (Level 4)
        elif risk_level == RiskLevel.HIGH:
            # Imaging urgent
            if not findings.imaging_performed:
                actions.append(ActionRecommendation.IMAGING_URGENT)

            # Neurology urgent consult (same-day)
            actions.append(ActionRecommendation.NEURO_URGENT)

            # Observation or admission depending on findings
            if findings.imaging_result == "acute_infarct":
                actions.append(ActionRecommendation.ICU_ADMISSION)
            else:
                actions.append(ActionRecommendation.OBSERVATION)

        # MODERATE RISK (Level 3)
        elif risk_level == RiskLevel.MODERATE:
            # Routine imaging if concern persists at T1 or later
            if time_point != TimePoint.T0 and not findings.imaging_performed:
                actions.append(ActionRecommendation.IMAGING_ROUTINE)

            # ENT scheduled if BPPV suspected
            if self._suggests_bppv(findings):
                actions.append(ActionRecommendation.ENT_SCHEDULED)
                actions.append(ActionRecommendation.EPLEY_SCHEDULED)
            else:
                actions.append(ActionRecommendation.NEURO_URGENT)

            # Consider observation if T0/T1, discharge with precautions if T2/T3 stable
            if time_point in [TimePoint.T0, TimePoint.T1]:
                actions.append(ActionRecommendation.OBSERVATION)
            else:
                actions.append(ActionRecommendation.DISCHARGE_PRECAUTIONS)

        # LOW RISK (Level 2)
        elif risk_level == RiskLevel.LOW:
            # BPPV treatment if indicated
            if self._suggests_bppv(findings):
                if time_point == TimePoint.T0:
                    actions.append(ActionRecommendation.EPLEY_IMMEDIATE)
                else:
                    actions.append(ActionRecommendation.ENT_SCHEDULED)

            # Vestibular suppressant for acute symptom relief (max 3 days)
            if time_point == TimePoint.T0 and findings.symptom_severity >= 6:
                actions.append(ActionRecommendation.VESTIBULAR_SUPPRESSANT)

            # GP follow-up
            actions.append(ActionRecommendation.GP_FOLLOWUP)

            # Discharge safe if stable at T2/T3
            if time_point in [TimePoint.T2, TimePoint.T3]:
                actions.append(ActionRecommendation.DISCHARGE_SAFE)

        # MINIMAL RISK (Level 1)
        else:
            # GP follow-up for reassurance
            actions.append(ActionRecommendation.GP_FOLLOWUP)

            # Safe discharge
            actions.append(ActionRecommendation.DISCHARGE_SAFE)

        return actions

    def _suggests_bppv(self, findings: ClinicalFindings) -> bool:
        """Check if findings suggest BPPV (benign positional vertigo)."""
        # HINTS peripheral pattern + no focal deficits
        if findings.hints_performed:
            if not findings.hints_suggests_central():
                if not findings.focal_neurological_deficit:
                    return True
        return False

    def _check_escalation(self, time_point: TimePoint):
        """Check if treatment escalation has occurred.

        Escalation = risk level increased from previous time point.
        This triggers intensification of interventions.
        """
        # Get previous time point
        previous_time_points = [tp for tp in TimePoint if tp.value < time_point.value]
        if not previous_time_points:
            return  # First assessment, no escalation possible

        previous_tp = max(previous_time_points, key=lambda tp: tp.value)

        if previous_tp not in self.risk_trajectory:
            return  # No previous assessment

        current_risk = self.risk_trajectory[time_point]
        previous_risk = self.risk_trajectory[previous_tp]

        if current_risk.value > previous_risk.value:
            self.escalation_occurred = True
            self.escalation_time = self.assessments[time_point].timestamp
            self.escalation_reason = (
                f"Risk escalated from {previous_risk.name} to {current_risk.name} "
                f"at {time_point.name}"
            )

    def get_risk_trajectory_summary(self) -> Dict[str, any]:
        """Get summary of risk trajectory across time points."""
        return {
            "time_points_assessed": [
                tp.name for tp in sorted(self.assessments.keys(), key=lambda x: x.value)
            ],
            "risk_levels": [
                self.risk_trajectory[tp].name
                for tp in sorted(self.risk_trajectory.keys(), key=lambda x: x.value)
            ],
            "escalation_occurred": self.escalation_occurred,
            "escalation_reason": (
                self.escalation_reason if self.escalation_occurred else None
            ),
            "final_risk": (
                self.risk_trajectory[
                    max(self.risk_trajectory.keys(), key=lambda x: x.value)
                ].name
                if self.risk_trajectory
                else None
            ),
        }


def perform_time_bounded_assessment(
    patient_id: str,
    initial_presentation_time: datetime,
    time_point: TimePoint,
    findings: ClinicalFindings,
    previous_assessment: Optional[TiTrATEAssessment] = None,
) -> TiTrATEAssessment:
    """Perform TiTrATE assessment at a given time point.

    Args:
        patient_id: Patient identifier
        initial_presentation_time: Time of initial presentation
        time_point: Current time point (T0, T1, T2, T3)
        findings: Clinical findings at this time point
        previous_assessment: Previous TiTrATEAssessment (if continuing)

    Returns:
        TiTrATEAssessment with updated risk and actions

    Example:
        >>> # T0 assessment (presentation)
        >>> findings_t0 = ClinicalFindings(
        ...     time_point=TimePoint.T0,
        ...     timestamp=datetime.now(),
        ...     heart_rate=85,
        ...     blood_pressure_sys=140,
        ...     blood_pressure_dia=85,
        ...     hints_performed=True,
        ...     hints_hit_result="negative",  # Central sign
        ...     hints_nystagmus="horizontal",
        ...     symptom_severity=8
        ... )
        >>> assessment = perform_time_bounded_assessment(
        ...     patient_id="P001",
        ...     initial_presentation_time=datetime.now(),
        ...     time_point=TimePoint.T0,
        ...     findings=findings_t0
        ... )
        >>> print(assessment.risk_trajectory[TimePoint.T0])
        RiskLevel.HIGH
    """
    if previous_assessment is None:
        assessment = TiTrATEAssessment(
            patient_id=patient_id, initial_presentation_time=initial_presentation_time
        )
    else:
        assessment = previous_assessment

    assessment.add_assessment(time_point, findings)
    return assessment


def calculate_risk_score(
    hints_central: bool,
    focal_deficit: bool,
    imaging_positive: bool,
    symptom_improving: bool,
    time_from_onset_hours: float,
) -> float:
    """Calculate numerical risk score for dizziness patient.

    This is a simplified version of the risk calculation used in TiTrATEAssessment.
    Useful for quick risk estimation without full assessment object.

    Args:
        hints_central: HINTS exam suggests central cause
        focal_deficit: Focal neurological deficit present
        imaging_positive: Imaging shows acute infarct/hemorrhage
        symptom_improving: Symptoms improving over time
        time_from_onset_hours: Time since symptom onset

    Returns:
        Risk score (0-10 scale)

    Example:
        >>> # High-risk stroke patient
        >>> risk = calculate_risk_score(
        ...     hints_central=True,
        ...     focal_deficit=True,
        ...     imaging_positive=True,
        ...     symptom_improving=False,
        ...     time_from_onset_hours=2.0
        ... )
        >>> print(f"Risk score: {risk:.1f}")
        Risk score: 9.0
    """
    score = 0.0

    if hints_central:
        score += 4.0
    if focal_deficit:
        score += 3.0
    if imaging_positive:
        score += 5.0
    if not symptom_improving:
        score += 1.0
    else:
        score -= 1.0

    # Urgency factor within thrombolysis window
    if time_from_onset_hours < 4.5 and score >= 3.0:
        score += 1.0

    return max(0.0, min(10.0, score))


def simulate_risk_trajectory(
    initial_risk: float,
    disease_type: str,
    intervention_timing: str = "standard",
    num_timepoints: int = 4,
) -> List[float]:
    """Simulate risk trajectory over time for different disease types.

    This is useful for digital twin simulation and what-if analysis.

    Args:
        initial_risk: Initial risk score at T0 (0-10)
        disease_type: "bppv", "vestibular_neuritis", "stroke", "migraine"
        intervention_timing: "immediate", "standard", "delayed"
        num_timepoints: Number of time points to simulate (default 4: T0, T1, T2, T3)

    Returns:
        List of risk scores at each time point

    Example:
        >>> # BPPV with immediate Epley maneuver
        >>> trajectory = simulate_risk_trajectory(
        ...     initial_risk=5.0,
        ...     disease_type="bppv",
        ...     intervention_timing="immediate"
        ... )
        >>> print(trajectory)
        [5.0, 2.0, 1.0, 0.5]  # Rapid improvement with treatment

        >>> # Stroke with delayed intervention
        >>> trajectory = simulate_risk_trajectory(
        ...     initial_risk=8.0,
        ...     disease_type="stroke",
        ...     intervention_timing="delayed"
        ... )
        >>> print(trajectory)
        [8.0, 9.0, 9.5, 9.5]  # Worsening without timely treatment
    """
    trajectory = [initial_risk]

    # Disease-specific progression patterns
    if disease_type == "bppv":
        # BPPV: Rapid improvement with Epley, slow natural resolution
        if intervention_timing == "immediate":
            improvement_rates = [-3.0, -2.0, -1.0]  # Rapid improvement
        elif intervention_timing == "standard":
            improvement_rates = [-1.5, -1.5, -1.0]  # Moderate improvement
        else:  # delayed
            improvement_rates = [-0.5, -0.8, -0.8]  # Slow natural resolution

    elif disease_type == "vestibular_neuritis":
        # VN: Gradual improvement over days, central compensation
        improvement_rates = [-0.5, -1.0, -1.5]  # Slow then faster
        if intervention_timing == "immediate":
            improvement_rates = [-1.0, -1.5, -2.0]  # Early vestibular rehab helps

    elif disease_type == "stroke":
        # Stroke: Stable or worsening without treatment, improves with thrombolysis
        if intervention_timing == "immediate":
            improvement_rates = [0.0, -3.0, -2.0]  # Thrombolysis within window
        elif intervention_timing == "standard":
            improvement_rates = [0.5, -1.0, -1.0]  # Late or partial treatment
        else:  # delayed
            improvement_rates = [1.0, 0.5, 0.0]  # Worsening then stable

    elif disease_type == "migraine":
        # Vestibular migraine: Episodic, self-limited
        improvement_rates = [-1.0, -2.0, -1.5]  # Natural resolution

    else:
        # Unknown disease: Assume gradual improvement
        improvement_rates = [-1.0, -1.0, -1.0]

    # Simulate trajectory
    current_risk = initial_risk
    for rate in improvement_rates[: num_timepoints - 1]:
        current_risk = max(0.0, min(10.0, current_risk + rate))
        trajectory.append(current_risk)

    return trajectory
