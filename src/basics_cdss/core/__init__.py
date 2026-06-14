"""TRI-X Core Module: Triage-TiTrATE-XAI Framework Components.

This module contains the core components of the TRI-X framework for
clinical decision support in dizziness and vertigo diagnosis:

- Triage: ESI-based urgency assessment
- TiTrATE: Time-bound Risk-stratified Action with Treatment Escalation
- SRGL: Screening-First Risk Governance Logic
- DRAS-5: Decision Risk-Action States (5-level urgency classification)
- ORASR: Operational Reasoning-Action Safety Routing

These components work together to provide safe, efficient, and transparent
clinical decision support for acute dizziness presentations.
"""

from basics_cdss.core.dras5 import (DRAS5Classification, calibrate_thresholds,
                                    classify_urgency_level)
from basics_cdss.core.orasr import (ORASRRouting, perform_safety_checks,
                                    route_to_care_pathway)
from basics_cdss.core.srgl import (SRGLScreening, check_safety_protocols,
                                   perform_red_flag_screening)
from basics_cdss.core.titrate import (TiTrATEAssessment, calculate_risk_score,
                                      perform_time_bounded_assessment)
from basics_cdss.core.triage import (ESITriageAssessment, map_esi_to_dras,
                                     perform_dizziness_triage)

__all__ = [
    # Triage
    "ESITriageAssessment",
    "perform_dizziness_triage",
    "map_esi_to_dras",
    # TiTrATE
    "TiTrATEAssessment",
    "perform_time_bounded_assessment",
    "calculate_risk_score",
    # SRGL
    "SRGLScreening",
    "perform_red_flag_screening",
    "check_safety_protocols",
    # DRAS-5
    "DRAS5Classification",
    "classify_urgency_level",
    "calibrate_thresholds",
    # ORASR
    "ORASRRouting",
    "route_to_care_pathway",
    "perform_safety_checks",
]
