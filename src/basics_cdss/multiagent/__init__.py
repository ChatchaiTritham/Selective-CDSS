"""BASICS-CDSS Multi-Agent Module: System-Level Simulation.

This module enables multi-agent simulation of clinical decision support
systems embedded in complex healthcare environments, capturing:

1. Agent interactions (Patient, Clinician, CDSS, Nurse, Admin)
2. Clinical workflows and protocols
3. System-level effects (alert fatigue, workflow disruption)
4. Emergent phenomena from CDSS deployment

Key Components:
    - Agent classes: Patient, Clinician, CDSS, Nurse
    - Environment: Hospital simulation with wards, resources
    - Workflow: Clinical protocols and task sequences
    - Interaction: Communication and decision protocols
    - SystemicMetrics: Alert fatigue, override rates, workflow impact

Theoretical Foundation:
    - Multi-agent systems theory
    - Sociotechnical systems
    - Human factors in health IT
    - Workflow analysis

Example:
    >>> from basics_cdss.multiagent import (
    ...     HospitalEnvironment, ClinicianAgent, CDSSAgent, PatientAgent
    ... )
    >>>
    >>> # Create hospital environment
    >>> hospital = HospitalEnvironment(n_beds=20, icu_beds=8)
    >>>
    >>> # Create agents
    >>> cdss = CDSSAgent(model=sepsis_model, alert_threshold=0.8)
    >>> clinician = ClinicianAgent(
    ...     experience_level='senior',
    ...     workload_capacity=5
    ... )
    >>> patient = PatientAgent(archetype_id='A001', digital_twin=twin)
    >>>
    >>> # Run simulation
    >>> hospital.add_agent(cdss)
    >>> hospital.add_agent(clinician)
    >>> hospital.add_agent(patient)
    >>>
    >>> results = hospital.simulate(duration_hours=24)
    >>>
    >>> # Analyze systemic effects
    >>> from basics_cdss.multiagent import compute_alert_fatigue
    >>> fatigue = compute_alert_fatigue(results)
"""

from basics_cdss.multiagent.agents import (Agent, CDSSAgent, ClinicianAgent,
                                           NurseAgent, PatientAgent)
from basics_cdss.multiagent.environment import (HospitalEnvironment, Resource,
                                                Ward)
from basics_cdss.multiagent.interaction import (AlertMessage, DecisionRequest,
                                                InteractionProtocol, Message,
                                                perform_interaction)
from basics_cdss.multiagent.systemic_metrics import (
    compute_alert_fatigue, compute_coordination_efficiency,
    compute_override_rate, compute_time_to_action, compute_workflow_disruption)
from basics_cdss.multiagent.workflow import (ClinicalWorkflow, Task,
                                             WorkflowState,
                                             create_acs_workflow,
                                             create_sepsis_workflow)

__all__ = [
    # Agents
    "Agent",
    "PatientAgent",
    "ClinicianAgent",
    "CDSSAgent",
    "NurseAgent",
    # Environment
    "HospitalEnvironment",
    "Ward",
    "Resource",
    # Workflow
    "ClinicalWorkflow",
    "Task",
    "WorkflowState",
    "create_sepsis_workflow",
    "create_acs_workflow",
    # Interaction
    "InteractionProtocol",
    "Message",
    "AlertMessage",
    "DecisionRequest",
    "perform_interaction",
    # Metrics
    "compute_alert_fatigue",
    "compute_override_rate",
    "compute_workflow_disruption",
    "compute_time_to_action",
    "compute_coordination_efficiency",
]
