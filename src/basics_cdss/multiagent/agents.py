"""Agent classes for multi-agent clinical simulation.

This module defines autonomous agents representing different actors
in the healthcare system.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class AgentType(Enum):
    """Types of agents in the simulation."""

    PATIENT = "patient"
    CLINICIAN = "clinician"
    NURSE = "nurse"
    CDSS = "cdss"
    ADMINISTRATOR = "administrator"


class AgentState(Enum):
    """States an agent can be in."""

    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    RESPONDING = "responding"


@dataclass
class AgentAction:
    """Represents an action taken by an agent.

    Attributes:
        agent_id: Agent performing action
        action_type: Type of action
        target: Target of action (e.g., patient_id, task_id)
        parameters: Action parameters
        timestamp: When action was taken
    """

    agent_id: str
    action_type: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class Agent(ABC):
    """Base class for all agents in the simulation.

    An agent is an autonomous entity that:
    1. Perceives its environment
    2. Makes decisions based on goals and constraints
    3. Takes actions that affect the environment
    4. Interacts with other agents

    Example:
        >>> class CustomAgent(Agent):
        ...     def perceive(self, environment):
        ...         return environment.get_state()
        ...
        ...     def decide(self, perception):
        ...         return {'action': 'custom_action'}
        ...
        ...     def act(self, decision, environment):
        ...         return AgentAction(self.agent_id, 'custom_action')
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: AgentType = AgentType.CLINICIAN,
        name: Optional[str] = None,
    ):
        """Initialize agent.

        Args:
            agent_id: Unique identifier (auto-generated if None)
            agent_type: Type of agent
            name: Human-readable name
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_{self.agent_id[:8]}"

        self.state = AgentState.IDLE
        self.history: List[AgentAction] = []
        self.beliefs: Dict[str, Any] = {}

    @abstractmethod
    def perceive(self, environment: 'HospitalEnvironment') -> Dict[str, Any]:
        """Perceive the environment and form beliefs.

        Args:
            environment: Hospital environment

        Returns:
            Dictionary of perceptions
        """
        pass

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on perceptions.

        Args:
            perception: Current perceptions

        Returns:
            Decision dictionary
        """
        pass

    @abstractmethod
    def act(
        self, decision: Dict[str, Any], environment: 'HospitalEnvironment'
    ) -> AgentAction:
        """Execute action based on decision.

        Args:
            decision: Decision to execute
            environment: Hospital environment

        Returns:
            Action taken
        """
        pass

    def step(
        self, environment: 'HospitalEnvironment', dt: float = 1.0
    ) -> Optional[AgentAction]:
        """Execute one simulation step.

        Args:
            environment: Hospital environment
            dt: Time step in hours

        Returns:
            Action taken (if any)
        """
        # Perceive
        perception = self.perceive(environment)

        # Decide
        decision = self.decide(perception)

        # Act
        if decision:
            action = self.act(decision, environment)
            self.history.append(action)
            return action

        return None


class PatientAgent(Agent):
    """Patient agent with evolving clinical state.

    Represents a patient whose condition evolves over time according
    to a digital twin model.

    Attributes:
        archetype_id: Source archetype identifier
        digital_twin: Temporal digital twin model
        current_state: Current clinical state
        risk_scores: Historical risk scores from CDSS

    Example:
        >>> from basics_cdss.temporal import PatientDigitalTwin, SepsisModel
        >>>
        >>> twin = PatientDigitalTwin(
        ...     archetype_id='A001',
        ...     initial_state={'temperature': 38.5, 'hr': 110},
        ...     disease_model=SepsisModel()
        ... )
        >>>
        >>> patient = PatientAgent(
        ...     archetype_id='A001',
        ...     digital_twin=twin
        ... )
    """

    def __init__(
        self,
        archetype_id: str,
        digital_twin: 'PatientDigitalTwin',
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(agent_id, AgentType.PATIENT, name)
        self.archetype_id = archetype_id
        self.digital_twin = digital_twin
        self.current_state = digital_twin.current_state
        self.risk_scores: List[Dict[str, Any]] = []
        self.interventions_received: List[Dict[str, Any]] = []

    def perceive(self, environment: 'HospitalEnvironment') -> Dict[str, Any]:
        """Perceive environment (patients don't actively perceive much)."""
        return {
            'current_state': self.current_state.features,
            'timestamp': self.current_state.timestamp,
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Patients don't make active decisions in basic model."""
        return {}

    def act(
        self, decision: Dict[str, Any], environment: 'HospitalEnvironment'
    ) -> AgentAction:
        """Patients don't take active actions."""
        return AgentAction(
            agent_id=self.agent_id,
            action_type='evolve',
            timestamp=self.current_state.timestamp,
        )

    def evolve(self, dt: float = 1.0, interventions: Optional[Dict[str, Any]] = None):
        """Evolve patient state according to digital twin.

        Args:
            dt: Time step in hours
            interventions: Medical interventions applied
        """
        self.current_state = self.digital_twin.step(dt, interventions)

        if interventions:
            self.interventions_received.append(
                {
                    'timestamp': self.current_state.timestamp,
                    'interventions': interventions,
                }
            )


class ClinicianAgent(Agent):
    """Clinician agent that makes medical decisions.

    Attributes:
        experience_level: Experience level (junior, senior, expert)
        specialty: Medical specialty
        workload: Current number of active patients
        workload_capacity: Maximum patients
        response_time_fn: Function computing response time
        cdss_trust: Trust in CDSS recommendations (0-1)

    Example:
        >>> clinician = ClinicianAgent(
        ...     experience_level='senior',
        ...     specialty='emergency_medicine',
        ...     workload_capacity=5,
        ...     cdss_trust=0.8
        ... )
    """

    def __init__(
        self,
        experience_level: str = 'senior',
        specialty: str = 'general',
        workload_capacity: int = 5,
        cdss_trust: float = 0.7,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(agent_id, AgentType.CLINICIAN, name)
        self.experience_level = experience_level
        self.specialty = specialty
        self.workload = 0
        self.workload_capacity = workload_capacity
        self.cdss_trust = cdss_trust

        # Performance characteristics
        self.base_accuracy = self._get_base_accuracy()
        self.response_time_fn = self._get_response_time_fn()

        # Alerts and decisions
        self.alerts_received: List[Dict[str, Any]] = []
        self.decisions_made: List[Dict[str, Any]] = []

    def _get_base_accuracy(self) -> float:
        """Get base diagnostic accuracy by experience level."""
        accuracy_map = {'junior': 0.75, 'mid': 0.85, 'senior': 0.90, 'expert': 0.95}
        return accuracy_map.get(self.experience_level, 0.85)

    def _get_response_time_fn(self) -> Callable:
        """Get response time function based on experience."""

        def response_time(workload_ratio):
            # Response time increases with workload
            base_time = {
                'junior': 30,  # minutes
                'mid': 20,
                'senior': 15,
                'expert': 10,
            }.get(self.experience_level, 20)

            # Exponential increase with workload
            return base_time * np.exp(workload_ratio)

        return response_time

    def perceive(self, environment: 'HospitalEnvironment') -> Dict[str, Any]:
        """Perceive patients and alerts."""
        perception = {
            'patients': environment.get_patients_for_clinician(self.agent_id),
            'pending_alerts': environment.get_pending_alerts(self.agent_id),
            'workload': self.workload,
            'timestamp': environment.current_time,
        }
        return perception

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make clinical decision.

        Decision process:
        1. Triage patients by urgency
        2. Review CDSS alerts
        3. Decide on interventions
        """
        decision = {}

        # Process alerts
        for alert in perception['pending_alerts']:
            # Decide whether to follow CDSS recommendation
            follow_alert = self._evaluate_alert(alert)

            self.alerts_received.append(
                {
                    'timestamp': perception['timestamp'],
                    'alert': alert,
                    'followed': follow_alert,
                }
            )

            if follow_alert:
                decision['intervention'] = alert['recommendation']
                decision['patient_id'] = alert['patient_id']

        return decision

    def _evaluate_alert(self, alert: Dict[str, Any]) -> bool:
        """Evaluate whether to follow CDSS alert.

        Factors:
        - CDSS trust level
        - Alert confidence
        - Current workload
        - Alert fatigue
        """
        # Base probability of following
        base_follow_prob = self.cdss_trust

        # Adjust for alert confidence
        alert_confidence = alert.get('confidence', 0.5)
        follow_prob = base_follow_prob * alert_confidence

        # Reduce if high workload (alert fatigue)
        workload_ratio = self.workload / self.workload_capacity
        fatigue_factor = np.exp(-workload_ratio)  # Decreases with workload
        follow_prob *= fatigue_factor

        # Stochastic decision
        return np.random.random() < follow_prob

    def act(
        self, decision: Dict[str, Any], environment: 'HospitalEnvironment'
    ) -> AgentAction:
        """Execute clinical action."""
        if 'intervention' in decision:
            action = AgentAction(
                agent_id=self.agent_id,
                action_type='intervention',
                target=decision['patient_id'],
                parameters={'intervention': decision['intervention']},
                timestamp=environment.current_time,
            )

            self.decisions_made.append(
                {'timestamp': environment.current_time, 'decision': decision}
            )

            return action

        return AgentAction(
            agent_id=self.agent_id,
            action_type='monitor',
            timestamp=environment.current_time,
        )


class CDSSAgent(Agent):
    """CDSS agent that generates alerts and recommendations.

    Attributes:
        model: Predictive model for risk assessment
        alert_threshold: Risk threshold for alerts
        alert_cooldown: Minimum time between alerts (hours)
        alerts_generated: History of generated alerts

    Example:
        >>> cdss = CDSSAgent(
        ...     model=sepsis_model,
        ...     alert_threshold=0.8,
        ...     alert_cooldown=2.0
        ... )
    """

    def __init__(
        self,
        model: Any,
        alert_threshold: float = 0.8,
        alert_cooldown: float = 2.0,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(agent_id, AgentType.CDSS, name)
        self.model = model
        self.alert_threshold = alert_threshold
        self.alert_cooldown = alert_cooldown
        self.alerts_generated: List[Dict[str, Any]] = []
        self.last_alert_time: Dict[str, float] = {}  # patient_id -> time

    def perceive(self, environment: 'HospitalEnvironment') -> Dict[str, Any]:
        """Perceive patient states."""
        return {
            'patients': environment.get_all_patients(),
            'timestamp': environment.current_time,
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk and decide whether to generate alert."""
        decisions = []

        for patient in perception['patients']:
            # Get patient state
            patient_state = patient.current_state.features

            # Assess risk
            risk_score = self._assess_risk(patient_state)

            # Check if alert should be generated
            if self._should_alert(
                patient.agent_id, risk_score, perception['timestamp']
            ):
                decisions.append(
                    {
                        'patient_id': patient.agent_id,
                        'risk_score': risk_score,
                        'recommendation': self._generate_recommendation(
                            patient_state, risk_score
                        ),
                    }
                )

        return {'alerts': decisions}

    def _assess_risk(self, patient_state: Dict[str, Any]) -> float:
        """Assess patient risk using model."""
        # Convert state to model input format
        # This is simplified - actual implementation depends on model
        try:
            risk = self.model.predict_proba([patient_state])[0][1]
        except:
            # Fallback if model format doesn't match
            risk = 0.5

        return risk

    def _should_alert(
        self, patient_id: str, risk_score: float, current_time: float
    ) -> bool:
        """Determine if alert should be generated."""
        # Check threshold
        if risk_score < self.alert_threshold:
            return False

        # Check cooldown
        if patient_id in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[patient_id]
            if time_since_last < self.alert_cooldown:
                return False

        return True

    def _generate_recommendation(
        self, patient_state: Dict[str, Any], risk_score: float
    ) -> Dict[str, Any]:
        """Generate treatment recommendation."""
        # Simplified recommendation logic
        if risk_score > 0.9:
            return {
                'urgency': 'critical',
                'actions': ['immediate_intervention', 'icu_transfer'],
            }
        elif risk_score > 0.8:
            return {
                'urgency': 'high',
                'actions': ['close_monitoring', 'consider_intervention'],
            }
        else:
            return {'urgency': 'moderate', 'actions': ['monitor']}

    def act(
        self, decision: Dict[str, Any], environment: 'HospitalEnvironment'
    ) -> AgentAction:
        """Generate alert actions."""
        if 'alerts' in decision and decision['alerts']:
            for alert_info in decision['alerts']:
                alert = AgentAction(
                    agent_id=self.agent_id,
                    action_type='generate_alert',
                    target=alert_info['patient_id'],
                    parameters=alert_info,
                    timestamp=environment.current_time,
                )

                self.alerts_generated.append(alert_info)
                self.last_alert_time[alert_info['patient_id']] = (
                    environment.current_time
                )

                return alert

        return AgentAction(
            agent_id=self.agent_id,
            action_type='monitor',
            timestamp=environment.current_time,
        )


class NurseAgent(Agent):
    """Nurse agent for patient monitoring and care.

    Attributes:
        assigned_patients: List of patient IDs
        monitoring_frequency: How often to check patients (hours)

    Example:
        >>> nurse = NurseAgent(
        ...     assigned_patients=['patient1', 'patient2'],
        ...     monitoring_frequency=1.0
        ... )
    """

    def __init__(
        self,
        assigned_patients: Optional[List[str]] = None,
        monitoring_frequency: float = 1.0,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(agent_id, AgentType.NURSE, name)
        self.assigned_patients = assigned_patients or []
        self.monitoring_frequency = monitoring_frequency
        self.observations: List[Dict[str, Any]] = []

    def perceive(self, environment: 'HospitalEnvironment') -> Dict[str, Any]:
        """Perceive assigned patients."""
        patients = [
            p
            for p in environment.get_all_patients()
            if p.agent_id in self.assigned_patients
        ]

        return {'patients': patients, 'timestamp': environment.current_time}

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide monitoring actions."""
        return {'action': 'monitor_patients', 'patients': perception['patients']}

    def act(
        self, decision: Dict[str, Any], environment: 'HospitalEnvironment'
    ) -> AgentAction:
        """Execute monitoring."""
        # Record observations
        for patient in decision.get('patients', []):
            self.observations.append(
                {
                    'timestamp': environment.current_time,
                    'patient_id': patient.agent_id,
                    'state': patient.current_state.features,
                }
            )

        return AgentAction(
            agent_id=self.agent_id,
            action_type='monitor',
            timestamp=environment.current_time,
        )
