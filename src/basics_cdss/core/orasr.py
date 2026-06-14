"""ORASR: Operational Reasoning-Action Safety Routing.

This module implements the ORASR component of the TRI-X framework, which
translates DRAS-5 urgency classifications into operational routing decisions
with safety checks at each step.

ORASR Functions:
- Operational: Practical routing to available resources
- Reasoning: Logic-based decision making
- Action: Specific next steps
- Safety: Continuous safety verification
- Routing: Direct to appropriate care setting

ORASR ensures that DRAS-5 classifications result in appropriate,
safe, and feasible care pathways given real-world constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CarePathway(Enum):
    """Care pathway options."""

    ED_IMMEDIATE = "ed_immediate"  # Emergency department immediate
    ED_URGENT = "ed_urgent"  # Emergency department urgent (fast track)
    NEURO_CLINIC = "neurology_clinic"  # Outpatient neurology
    ENT_CLINIC = "ent_clinic"  # Outpatient ENT
    GP_CLINIC = "gp_clinic"  # Primary care
    TELEHEALTH = "telehealth"  # Virtual visit
    SELF_CARE = "self_care"  # Home care with instructions


class ResourceType(Enum):
    """Types of resources needed."""

    IMAGING_CT = "imaging_ct"
    IMAGING_MRI = "imaging_mri"
    NEUROLOGY_CONSULT = "neurology_consult"
    ENT_CONSULT = "ent_consult"
    STROKE_TEAM = "stroke_team"
    ICU_BED = "icu_bed"
    OBSERVATION_BED = "observation_bed"


@dataclass
class ResourceAvailability:
    """Current resource availability."""

    imaging_ct_available: bool = True
    imaging_ct_wait_minutes: int = 15
    imaging_mri_available: bool = True
    imaging_mri_wait_minutes: int = 45

    neurology_available: bool = True
    neurology_response_minutes: int = 30
    neurology_on_site: bool = False  # vs teleconsult/on-call

    ent_available: bool = False  # Typically outpatient only
    ent_next_appointment_hours: int = 48

    stroke_team_available: bool = True  # Comprehensive stroke center

    icu_beds_available: int = 2
    observation_beds_available: int = 4

    current_time: str = "day"  # "day", "evening", "night"

    def get_bottleneck_resources(self) -> List[str]:
        """Identify resource constraints."""
        bottlenecks = []

        if not self.imaging_ct_available or self.imaging_ct_wait_minutes > 60:
            bottlenecks.append("CT imaging (delayed or unavailable)")

        if not self.imaging_mri_available or self.imaging_mri_wait_minutes > 120:
            bottlenecks.append("MRI imaging (delayed or unavailable)")

        if not self.neurology_available or self.neurology_response_minutes > 60:
            bottlenecks.append("Neurology consult (delayed)")

        if self.icu_beds_available == 0:
            bottlenecks.append("ICU beds (full)")

        if self.observation_beds_available == 0:
            bottlenecks.append("Observation beds (full)")

        return bottlenecks


@dataclass
class PatientContext:
    """Patient-specific context affecting routing."""

    can_travel_independently: bool = True
    has_transportation: bool = True
    home_support: bool = True
    lives_alone: bool = False
    distance_to_hospital_km: float = 5.0
    health_literacy: float = 0.7  # 0-1 scale
    insurance_status: str = "insured"  # "insured", "uninsured", "medicare"
    language_barrier: bool = False


@dataclass
class ORASRRouting:
    """Complete ORASR routing decision."""

    primary_pathway: CarePathway
    alternative_pathways: List[CarePathway] = field(default_factory=list)
    required_resources: List[ResourceType] = field(default_factory=list)
    timeline: str = ""  # When patient should reach care
    specific_actions: List[str] = field(default_factory=list)
    safety_checks_passed: bool = True
    routing_rationale: str = ""
    barriers_identified: List[str] = field(default_factory=list)
    contingency_plan: str = ""

    def get_routing_summary(self) -> Dict[str, any]:
        """Get summary of routing decision."""
        return {
            "pathway": self.primary_pathway.value,
            "timeline": self.timeline,
            "actions": self.specific_actions,
            "resources_needed": [r.value for r in self.required_resources],
            "barriers": self.barriers_identified,
            "safe_to_proceed": self.safety_checks_passed,
        }


def route_to_care_pathway(
    dras_level: int,
    patient_context: PatientContext,
    resource_availability: ResourceAvailability,
    clinical_features: Dict[str, any],
    red_flags_present: bool = False,
) -> ORASRRouting:
    """Route patient to appropriate care pathway based on DRAS level and constraints.

    Args:
        dras_level: DRAS-5 classification (1-5)
        patient_context: Patient-specific factors (transportation, home support, etc.)
        resource_availability: Current resource constraints
        clinical_features: Clinical findings (stroke probability, BPPV likelihood, etc.)
        red_flags_present: Critical red flags from SRGL

    Returns:
        ORASRRouting with pathway, timeline, actions, and safety checks

    Example:
        >>> patient = PatientContext(
        ...     can_travel_independently=True,
        ...     has_transportation=True,
        ...     home_support=True
        ... )
        >>> resources = ResourceAvailability(
        ...     imaging_ct_available=True,
        ...     neurology_available=True
        ... )
        >>> clinical = {"stroke_probability": 0.85, "hints_central": True}
        >>> routing = route_to_care_pathway(
        ...     dras_level=5,
        ...     patient_context=patient,
        ...     resource_availability=resources,
        ...     clinical_features=clinical,
        ...     red_flags_present=True
        ... )
        >>> print(routing.primary_pathway)
        CarePathway.ED_IMMEDIATE
    """
    routing = ORASRRouting(primary_pathway=CarePathway.SELF_CARE)  # Default

    # === DRAS-5: IMMEDIATE EMERGENCY ===
    if dras_level == 5 or red_flags_present:
        routing.primary_pathway = CarePathway.ED_IMMEDIATE
        routing.timeline = (
            "IMMEDIATE (within 15 minutes if not in ED, within 30 min if in ED)"
        )
        routing.routing_rationale = (
            "DRAS-5 immediate emergency: Suspected stroke or critical condition"
        )

        # Required resources
        routing.required_resources = [
            ResourceType.IMAGING_CT,  # Immediate CT to rule out hemorrhage
            ResourceType.NEUROLOGY_CONSULT,  # Neurology stat consult
        ]

        # If stroke confirmed, need MRI and possibly ICU
        if clinical_features.get("stroke_probability", 0) > 0.7:
            routing.required_resources.append(ResourceType.IMAGING_MRI)
            routing.required_resources.append(ResourceType.STROKE_TEAM)
            if clinical_features.get("severe", False):
                routing.required_resources.append(ResourceType.ICU_BED)

        # Specific actions
        routing.specific_actions = [
            "Activate stroke protocol (CODE STROKE)",
            "Non-contrast CT head within 15 minutes",
            "Neurology stat consult (in-person or telestroke)",
            "Check thrombolysis eligibility (time window, contraindications)",
            "Prepare for possible tPA administration",
            "Monitor vitals continuously",
            "NPO (nothing by mouth) until swallow evaluated",
        ]

        # Safety checks
        routing.safety_checks_passed = _check_imaging_availability(
            routing.required_resources, resource_availability
        )

        # Barriers and contingency
        bottlenecks = resource_availability.get_bottleneck_resources()
        if bottlenecks:
            routing.barriers_identified = bottlenecks
            if "CT imaging" in str(bottlenecks):
                routing.contingency_plan = (
                    "If CT unavailable, transfer to nearest stroke center with imaging"
                )
            if "ICU beds" in str(bottlenecks):
                routing.contingency_plan = "If ICU full, ED observation with continuous monitoring, consider transfer"

    # === DRAS-4: URGENT SPECIALIST ===
    elif dras_level == 4:
        routing.primary_pathway = CarePathway.ED_URGENT
        routing.timeline = "URGENT (same-day, within 4 hours)"
        routing.routing_rationale = (
            "DRAS-4 urgent: Moderate-high stroke risk, requires same-day evaluation"
        )

        # Required resources
        routing.required_resources = [
            ResourceType.IMAGING_CT,  # Urgent imaging (within 1 hour)
            ResourceType.NEUROLOGY_CONSULT,  # Same-day neurology
        ]

        # If outpatient capable and neurology clinic available
        if (
            resource_availability.neurology_available
            and resource_availability.current_time == "day"
        ):
            routing.alternative_pathways = [CarePathway.NEURO_CLINIC]
            routing.routing_rationale += (
                " | Alternative: Direct to neurology clinic if imaging can be arranged"
            )

        # Specific actions
        routing.specific_actions = [
            "ED triage and assessment",
            "CT head within 1 hour",
            "Neurology consult (same-day, in-person or teleconsult)",
            "Consider observation if symptoms persist",
            "Discharge with 24-hour neurology follow-up if stable",
        ]

        # Safety checks
        routing.safety_checks_passed = True  # DRAS-4 has more flexibility

        # Barriers
        if not resource_availability.neurology_available:
            routing.barriers_identified.append(
                "Neurology consult delayed (after-hours)"
            )
            routing.contingency_plan = (
                "ED observation overnight, neurology consult in AM"
            )

    # === DRAS-3: SCHEDULED SPECIALIST (KEY FOR ED CROWDING REDUCTION) ===
    elif dras_level == 3:
        # Determine subspecialty needed
        if clinical_features.get("bppv_likely", False):
            routing.primary_pathway = CarePathway.ENT_CLINIC
            routing.timeline = "SCHEDULED (1-7 days acceptable)"
            routing.routing_rationale = (
                "DRAS-3: Likely BPPV, ENT outpatient Epley maneuver"
            )

            routing.required_resources = [ResourceType.ENT_CONSULT]

            routing.specific_actions = [
                "Schedule ENT clinic appointment (1-7 days)",
                "Provide BPPV education and Brandt-Daroff exercises",
                "Prescribe vestibular suppressant (meclizine 25mg PRN, max 3 days)",
                "Discharge with return precautions (new neurological symptoms)",
                "Follow-up if symptoms not improved by ENT visit",
            ]

            # Check ENT availability
            if resource_availability.ent_next_appointment_hours > 168:  # >7 days
                routing.barriers_identified.append(
                    f"ENT appointment delayed ({resource_availability.ent_next_appointment_hours/24:.0f} days)"
                )
                routing.contingency_plan = "Teach Epley maneuver in ED if trained physician available, otherwise GP referral"

        else:
            # General dizziness, needs specialist eval
            routing.primary_pathway = CarePathway.NEURO_CLINIC
            routing.timeline = "SCHEDULED (1-7 days acceptable)"
            routing.routing_rationale = (
                "DRAS-3: Non-emergent dizziness, outpatient neurology evaluation"
            )

            routing.required_resources = [ResourceType.NEUROLOGY_CONSULT]

            routing.specific_actions = [
                "Schedule neurology clinic appointment (1-7 days)",
                "Provide dizziness diary for patient to track symptoms",
                "Discharge with return precautions (worsening symptoms, new deficits)",
                "Follow-up if symptoms not improved by neurology visit",
            ]

        # Safety checks for discharge
        discharge_safe = _check_discharge_safety(patient_context, clinical_features)
        routing.safety_checks_passed = discharge_safe

        if not discharge_safe:
            routing.barriers_identified.append(
                "Discharge safety concerns (see specific actions)"
            )
            routing.specific_actions.append(
                "Consider short ED observation or admission if discharge unsafe"
            )

    # === DRAS-2: LOWER URGENCY (GP) ===
    elif dras_level == 2:
        routing.primary_pathway = CarePathway.GP_CLINIC
        routing.timeline = "ROUTINE (1-2 weeks acceptable)"
        routing.routing_rationale = (
            "DRAS-2: Low-risk dizziness, primary care evaluation adequate"
        )

        routing.specific_actions = [
            "Schedule GP appointment (1-2 weeks)",
            "Provide general dizziness education",
            "Advise on fall prevention if elderly",
            "Return precautions: severe headache, neurological symptoms, worsening",
        ]

        # Alternative: Telehealth if appropriate
        if (
            patient_context.has_transportation == False
            or patient_context.distance_to_hospital_km > 50
        ):
            routing.alternative_pathways = [CarePathway.TELEHEALTH]
            routing.specific_actions.append(
                "Consider telehealth visit if transportation barrier"
            )

        routing.safety_checks_passed = True

    # === DRAS-1: SAFE (SELF-CARE) ===
    else:  # dras_level == 1
        routing.primary_pathway = CarePathway.SELF_CARE
        routing.timeline = "AS NEEDED"
        routing.routing_rationale = "DRAS-1: Minimal risk, self-care appropriate"

        routing.specific_actions = [
            "Provide dizziness self-care education",
            "Hydration and rest recommendations",
            "GP follow-up if symptoms persist >2 weeks",
            "Return precautions: severe symptoms, neurological changes",
        ]

        routing.safety_checks_passed = True

    # === PATIENT-SPECIFIC MODIFICATIONS ===
    routing = _apply_patient_context_modifications(routing, patient_context)

    return routing


def _check_imaging_availability(
    required_resources: List[ResourceType], availability: ResourceAvailability
) -> bool:
    """Check if required imaging resources are available in acceptable timeframe."""
    for resource in required_resources:
        if resource == ResourceType.IMAGING_CT:
            if not availability.imaging_ct_available:
                return False
            if (
                availability.imaging_ct_wait_minutes > 60
            ):  # >1 hour unacceptable for DRAS-5
                return False

        elif resource == ResourceType.IMAGING_MRI:
            if not availability.imaging_mri_available:
                return False  # MRI is critical for stroke diagnosis

    return True


def _check_discharge_safety(
    patient_context: PatientContext, clinical_features: Dict[str, any]
) -> bool:
    """Check if patient can be safely discharged (DRAS-3 decision point)."""
    # Cannot discharge if:
    # - Lives alone and cannot ambulate safely (fall risk)
    # - No transportation and symptoms prevent driving
    # - High symptom severity still
    # - Stroke probability still moderate

    if patient_context.lives_alone and not patient_context.can_travel_independently:
        return False  # Fall risk at home

    if (
        not patient_context.has_transportation
        and not patient_context.can_travel_independently
    ):
        return False  # Cannot get to follow-up appointment

    if clinical_features.get("symptom_severity", 0) >= 7:
        return False  # Still too symptomatic

    if clinical_features.get("stroke_probability", 0) > 0.3:
        return False  # Still moderate stroke concern

    return True


def _apply_patient_context_modifications(
    routing: ORASRRouting, patient_context: PatientContext
) -> ORASRRouting:
    """Apply patient-specific modifications to routing plan."""
    # Language barrier → Need interpreter
    if patient_context.language_barrier:
        routing.specific_actions.append("Arrange interpreter for all encounters")

    # Low health literacy → Enhanced education
    if patient_context.health_literacy < 0.5:
        routing.specific_actions.append(
            "Provide simplified instructions with visual aids"
        )
        routing.specific_actions.append("Teach-back method to confirm understanding")

    # No transportation → Consider telehealth or transportation assistance
    if not patient_context.has_transportation:
        if routing.primary_pathway in [
            CarePathway.NEURO_CLINIC,
            CarePathway.ENT_CLINIC,
            CarePathway.GP_CLINIC,
        ]:
            routing.barriers_identified.append("No transportation for follow-up")
            routing.alternative_pathways.append(CarePathway.TELEHEALTH)
            routing.specific_actions.append(
                "Arrange medical transportation or offer telehealth alternative"
            )

    # Lives alone + high risk → Extra safety
    if (
        patient_context.lives_alone
        and routing.primary_pathway != CarePathway.ED_IMMEDIATE
    ):
        routing.specific_actions.append(
            "Ensure patient has emergency contact system (phone, medical alert)"
        )
        routing.specific_actions.append("Consider home health check-in within 24 hours")

    return routing


def perform_safety_checks(
    routing: ORASRRouting,
    patient_data: Dict[str, any],
    resource_availability: ResourceAvailability,
) -> Tuple[bool, List[str]]:
    """Perform comprehensive safety checks before finalizing routing.

    This is the final governance step before routing is implemented.

    Args:
        routing: Proposed routing plan
        patient_data: Complete patient data
        resource_availability: Current resource status

    Returns:
        Tuple of (all_checks_passed, list_of_issues)

    Example:
        >>> routing = ORASRRouting(primary_pathway=CarePathway.ED_IMMEDIATE, ...)
        >>> patient = {"anticoagulation": True, "recent_fall": True}
        >>> resources = ResourceAvailability(imaging_ct_available=False)
        >>> passed, issues = perform_safety_checks(routing, patient, resources)
        >>> print(passed)
        False
        >>> print(issues)
        ['CT imaging unavailable for DRAS-5 patient']
    """
    issues = []

    # Check 1: Critical resources available for high-urgency patients
    if routing.primary_pathway == CarePathway.ED_IMMEDIATE:
        if ResourceType.IMAGING_CT in routing.required_resources:
            if not resource_availability.imaging_ct_available:
                issues.append(
                    "CT imaging unavailable for DRAS-5 patient - consider immediate transfer"
                )

        if ResourceType.ICU_BED in routing.required_resources:
            if resource_availability.icu_beds_available == 0:
                issues.append(
                    "ICU bed unavailable - patient may need transfer after stabilization"
                )

        if ResourceType.STROKE_TEAM in routing.required_resources:
            if not resource_availability.stroke_team_available:
                issues.append(
                    "Stroke team unavailable - activate telestroke or transfer protocol"
                )

    # Check 2: Discharge safety
    if routing.primary_pathway == CarePathway.SELF_CARE:
        if patient_data.get("high_fall_risk", False):
            issues.append("High fall risk patient - reconsider self-care pathway")

        if patient_data.get("lives_alone", False) and patient_data.get("age", 0) > 75:
            issues.append(
                "Elderly patient living alone - ensure emergency contact system before discharge"
            )

    # Check 3: Follow-up feasibility
    if routing.primary_pathway in [CarePathway.NEURO_CLINIC, CarePathway.ENT_CLINIC]:
        if not patient_data.get("has_transportation", True):
            issues.append(
                "No transportation - telehealth alternative or transportation assistance needed"
            )

    # Check 4: Medication safety
    for action in routing.specific_actions:
        if "meclizine" in action.lower() or "vestibular suppressant" in action.lower():
            if patient_data.get("age", 0) > 75:
                issues.append(
                    "Elderly patient - vestibular suppressant increases fall risk, use cautiously"
                )

    # Check 5: Time-critical pathways
    if routing.timeline.startswith("IMMEDIATE"):
        time_from_onset = patient_data.get("symptom_duration_hours", 999)
        if time_from_onset > 4.5:
            issues.append(
                f"Symptom onset {time_from_onset:.1f} hours ago - outside thrombolysis window but may still benefit from intervention"
            )

    all_passed = len(issues) == 0

    return all_passed, issues


def generate_routing_documentation(routing: ORASRRouting) -> str:
    """Generate documentation for routing decision (for medical record).

    Args:
        routing: ORASRRouting decision to document

    Returns:
        Formatted documentation string

    Example:
        >>> routing = ORASRRouting(
        ...     primary_pathway=CarePathway.ED_IMMEDIATE,
        ...     timeline="IMMEDIATE",
        ...     routing_rationale="DRAS-5 stroke concern"
        ... )
        >>> doc = generate_routing_documentation(routing)
        >>> print(doc)
        TRI-X ORASR ROUTING DECISION
        ============================
        Primary Pathway: ED_IMMEDIATE
        ...
    """
    lines = []
    lines.append("TRI-X ORASR ROUTING DECISION")
    lines.append("=" * 50)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nPrimary Pathway: {routing.primary_pathway.value.upper()}")
    lines.append(f"Timeline: {routing.timeline}")
    lines.append(f"\nRationale: {routing.routing_rationale}")

    if routing.alternative_pathways:
        lines.append(f"\nAlternative Pathways:")
        for pathway in routing.alternative_pathways:
            lines.append(f"  - {pathway.value}")

    if routing.required_resources:
        lines.append(f"\nRequired Resources:")
        for resource in routing.required_resources:
            lines.append(f"  - {resource.value}")

    lines.append(f"\nSpecific Actions:")
    for i, action in enumerate(routing.specific_actions, 1):
        lines.append(f"  {i}. {action}")

    if routing.barriers_identified:
        lines.append(f"\nBarriers Identified:")
        for barrier in routing.barriers_identified:
            lines.append(f"  - {barrier}")

        if routing.contingency_plan:
            lines.append(f"\nContingency Plan: {routing.contingency_plan}")

    lines.append(
        f"\nSafety Checks: {'PASSED' if routing.safety_checks_passed else 'FAILED - SEE ISSUES'}"
    )
    lines.append("=" * 50)

    return "\n".join(lines)
