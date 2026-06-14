"""Hospital environment for multi-agent simulation.

This module defines the simulated hospital environment where agents
interact and clinical workflows unfold.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np


class WardType(Enum):
    """Types of hospital wards."""

    EMERGENCY = "emergency"
    ICU = "icu"
    GENERAL = "general"
    STEPDOWN = "stepdown"


@dataclass
class Resource:
    """Represents a hospital resource.

    Attributes:
        resource_id: Unique identifier
        resource_type: Type of resource (bed, ventilator, etc.)
        available: Whether resource is available
        location: Ward where resource is located
    """

    resource_id: str
    resource_type: str
    available: bool = True
    location: Optional[str] = None


@dataclass
class Ward:
    """Represents a hospital ward or unit.

    Attributes:
        ward_id: Unique identifier
        ward_type: Type of ward
        capacity: Total bed capacity
        occupied_beds: Current number of occupied beds
        patients: List of patient IDs in ward
        resources: Available resources in ward
    """

    ward_id: str
    ward_type: WardType
    capacity: int
    occupied_beds: int = 0
    patients: List[str] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)

    @property
    def available_beds(self) -> int:
        """Number of available beds."""
        return self.capacity - self.occupied_beds

    @property
    def occupancy_rate(self) -> float:
        """Ward occupancy rate (0-1)."""
        return self.occupied_beds / self.capacity if self.capacity > 0 else 0.0

    def admit_patient(self, patient_id: str) -> bool:
        """Admit patient to ward.

        Args:
            patient_id: Patient identifier

        Returns:
            True if admission successful
        """
        if self.available_beds > 0:
            self.patients.append(patient_id)
            self.occupied_beds += 1
            return True
        return False

    def discharge_patient(self, patient_id: str):
        """Discharge patient from ward.

        Args:
            patient_id: Patient identifier
        """
        if patient_id in self.patients:
            self.patients.remove(patient_id)
            self.occupied_beds -= 1


class HospitalEnvironment:
    """Simulated hospital environment for multi-agent system.

    The environment manages:
    1. Physical resources (beds, equipment)
    2. Agent registry and states
    3. Message passing between agents
    4. Event queue and timing
    5. State transitions and workflows

    Example:
        >>> from basics_cdss.multiagent import (
        ...     HospitalEnvironment, PatientAgent, ClinicianAgent, CDSSAgent
        ... )
        >>>
        >>> # Create environment
        >>> hospital = HospitalEnvironment(n_beds=20, icu_beds=8)
        >>>
        >>> # Add agents
        >>> patient = PatientAgent(archetype_id='A001', digital_twin=twin)
        >>> clinician = ClinicianAgent(experience_level='senior')
        >>> cdss = CDSSAgent(model=sepsis_model)
        >>>
        >>> hospital.add_agent(patient)
        >>> hospital.add_agent(clinician)
        >>> hospital.add_agent(cdss)
        >>>
        >>> # Run simulation
        >>> results = hospital.simulate(duration_hours=24, dt=1.0)
    """

    def __init__(
        self,
        n_beds: int = 20,
        icu_beds: int = 8,
        ed_beds: int = 15,
        seed: Optional[int] = None,
    ):
        """Initialize hospital environment.

        Args:
            n_beds: Number of general ward beds
            icu_beds: Number of ICU beds
            ed_beds: Number of emergency department beds
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Time
        self.current_time: float = 0.0

        # Wards
        self.wards: Dict[str, Ward] = {
            'emergency': Ward(
                ward_id='emergency', ward_type=WardType.EMERGENCY, capacity=ed_beds
            ),
            'icu': Ward(ward_id='icu', ward_type=WardType.ICU, capacity=icu_beds),
            'general': Ward(
                ward_id='general', ward_type=WardType.GENERAL, capacity=n_beds
            ),
        }

        # Agents
        self.agents: Dict[str, 'Agent'] = {}
        self.patients: Dict[str, 'PatientAgent'] = {}
        self.clinicians: Dict[str, 'ClinicianAgent'] = {}
        self.cdss_agents: Dict[str, 'CDSSAgent'] = {}
        self.nurses: Dict[str, 'NurseAgent'] = {}

        # Communication
        self.message_queue: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

        # History
        self.event_log: List[Dict[str, Any]] = []

    def add_agent(self, agent: 'Agent'):
        """Add agent to environment.

        Args:
            agent: Agent to add
        """
        self.agents[agent.agent_id] = agent

        # Add to type-specific registry
        from basics_cdss.multiagent.agents import AgentType

        if agent.agent_type == AgentType.PATIENT:
            self.patients[agent.agent_id] = agent
        elif agent.agent_type == AgentType.CLINICIAN:
            self.clinicians[agent.agent_id] = agent
        elif agent.agent_type == AgentType.CDSS:
            self.cdss_agents[agent.agent_id] = agent
        elif agent.agent_type == AgentType.NURSE:
            self.nurses[agent.agent_id] = agent

        # Log event
        self._log_event(
            {
                'type': 'agent_added',
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type.value,
                'timestamp': self.current_time,
            }
        )

    def remove_agent(self, agent_id: str):
        """Remove agent from environment.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]

            # Remove from type-specific registry
            from basics_cdss.multiagent.agents import AgentType

            if agent.agent_type == AgentType.PATIENT:
                del self.patients[agent_id]
            elif agent.agent_type == AgentType.CLINICIAN:
                del self.clinicians[agent_id]
            elif agent.agent_type == AgentType.CDSS:
                del self.cdss_agents[agent_id]
            elif agent.agent_type == AgentType.NURSE:
                del self.nurses[agent_id]

            self._log_event(
                {
                    'type': 'agent_removed',
                    'agent_id': agent_id,
                    'timestamp': self.current_time,
                }
            )

    def get_all_patients(self) -> List['PatientAgent']:
        """Get all patient agents."""
        return list(self.patients.values())

    def get_patients_for_clinician(self, clinician_id: str) -> List['PatientAgent']:
        """Get patients assigned to clinician.

        Args:
            clinician_id: Clinician identifier

        Returns:
            List of patient agents
        """
        # Simplified: return all patients
        # In real implementation, track assignments
        return list(self.patients.values())

    def get_pending_alerts(self, clinician_id: str) -> List[Dict[str, Any]]:
        """Get pending alerts for clinician.

        Args:
            clinician_id: Clinician identifier

        Returns:
            List of alert dictionaries
        """
        # Return alerts not yet acknowledged
        pending = [
            alert for alert in self.alerts if not alert.get('acknowledged', False)
        ]
        return pending

    def send_alert(self, from_agent: str, to_agent: str, alert_data: Dict[str, Any]):
        """Send alert from one agent to another.

        Args:
            from_agent: Sender agent ID
            to_agent: Receiver agent ID
            alert_data: Alert content
        """
        alert = {
            'from': from_agent,
            'to': to_agent,
            'timestamp': self.current_time,
            'acknowledged': False,
            **alert_data,
        }

        self.alerts.append(alert)

        self._log_event(
            {
                'type': 'alert_sent',
                'from': from_agent,
                'to': to_agent,
                'timestamp': self.current_time,
            }
        )

    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert.

        Args:
            alert_index: Index of alert in alerts list
        """
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]['acknowledged'] = True

    def step(self, dt: float = 1.0):
        """Advance simulation by one time step.

        Args:
            dt: Time step in hours
        """
        # Update time
        self.current_time += dt

        # Step all agents
        for agent in self.agents.values():
            action = agent.step(self, dt)

            if action:
                self._log_event(
                    {
                        'type': 'agent_action',
                        'agent_id': action.agent_id,
                        'action_type': action.action_type,
                        'target': action.target,
                        'timestamp': self.current_time,
                    }
                )

        # Evolve patients
        for patient in self.patients.values():
            patient.evolve(dt)

        # Process messages
        self._process_messages()

    def _process_messages(self):
        """Process messages in queue."""
        while self.message_queue:
            message = self.message_queue.pop(0)
            # Process message
            # Implementation depends on message type

    def _log_event(self, event: Dict[str, Any]):
        """Log event to history.

        Args:
            event: Event dictionary
        """
        self.event_log.append(event)

    def simulate(self, duration_hours: float, dt: float = 1.0) -> Dict[str, Any]:
        """Run simulation for specified duration.

        Args:
            duration_hours: Simulation duration in hours
            dt: Time step in hours

        Returns:
            Dictionary with simulation results:
                - event_log: All events
                - patient_trajectories: Patient state histories
                - alerts: All alerts generated
                - metrics: Aggregate metrics

        Example:
            >>> results = hospital.simulate(duration_hours=24, dt=1.0)
            >>> print(f"Total events: {len(results['event_log'])}")
            >>> print(f"Total alerts: {len(results['alerts'])}")
        """
        n_steps = int(duration_hours / dt)

        for _ in range(n_steps):
            self.step(dt)

        # Compile results
        results = {
            'event_log': self.event_log,
            'alerts': self.alerts,
            'duration_hours': duration_hours,
            'final_time': self.current_time,
            'n_agents': len(self.agents),
            'n_patients': len(self.patients),
            'n_clinicians': len(self.clinicians),
            'patient_trajectories': self._get_patient_trajectories(),
            'metrics': self._compute_metrics(),
        }

        return results

    def _get_patient_trajectories(self) -> Dict[str, List]:
        """Get patient state trajectories."""
        trajectories = {}

        for patient_id, patient in self.patients.items():
            trajectories[patient_id] = [
                {'timestamp': state.timestamp, 'features': state.features}
                for state in patient.digital_twin.history
            ]

        return trajectories

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        return {
            'total_events': len(self.event_log),
            'total_alerts': len(self.alerts),
            'alert_rate': (
                len(self.alerts) / self.current_time if self.current_time > 0 else 0
            ),
            'occupancy': {
                ward_id: ward.occupancy_rate for ward_id, ward in self.wards.items()
            },
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state.

        Returns:
            Dictionary with environment state
        """
        return {
            'current_time': self.current_time,
            'n_agents': len(self.agents),
            'n_patients': len(self.patients),
            'wards': {
                ward_id: {'occupancy': ward.occupancy_rate, 'patients': ward.patients}
                for ward_id, ward in self.wards.items()
            },
            'pending_alerts': len(
                [a for a in self.alerts if not a.get('acknowledged')]
            ),
        }

    def reset(self):
        """Reset environment to initial state."""
        self.current_time = 0.0
        self.message_queue.clear()
        self.alerts.clear()
        self.event_log.clear()

        # Reset wards
        for ward in self.wards.values():
            ward.patients.clear()
            ward.occupied_beds = 0

        # Reset patient digital twins
        for patient in self.patients.values():
            patient.digital_twin.reset()
            patient.current_state = patient.digital_twin.current_state
