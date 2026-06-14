"""Agent interaction protocols and message passing.

This module defines communication protocols between agents in the
multi-agent clinical system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class MessageType(Enum):
    """Types of messages between agents."""

    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HANDOFF = "handoff"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Base class for inter-agent messages.

    Attributes:
        message_id: Unique identifier
        message_type: Type of message
        sender: Sender agent ID
        recipient: Recipient agent ID
        content: Message content
        priority: Message priority
        timestamp: When message was sent
        acknowledged: Whether message was acknowledged

    Example:
        >>> msg = Message(
        ...     message_id='msg_001',
        ...     message_type=MessageType.NOTIFICATION,
        ...     sender='nurse_1',
        ...     recipient='clinician_1',
        ...     content={'patient_id': 'P001', 'event': 'vital_signs_abnormal'},
        ...     priority=MessagePriority.HIGH
        ... )
    """

    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = 0.0
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def acknowledge(self):
        """Mark message as acknowledged."""
        self.acknowledged = True


@dataclass
class AlertMessage(Message):
    """CDSS alert message to clinician.

    Attributes:
        patient_id: Patient the alert concerns
        alert_type: Type of clinical alert
        risk_score: Predicted risk score
        recommendation: Recommended action
        confidence: Model confidence (0-1)

    Example:
        >>> alert = AlertMessage(
        ...     message_id='alert_001',
        ...     sender='cdss_1',
        ...     recipient='clinician_1',
        ...     patient_id='P001',
        ...     alert_type='sepsis_risk',
        ...     risk_score=0.85,
        ...     recommendation={'action': 'obtain_lactate', 'urgency': 'high'},
        ...     confidence=0.90
        ... )
    """

    patient_id: str = ""
    alert_type: str = ""
    risk_score: float = 0.0
    recommendation: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self):
        """Set message type to ALERT."""
        self.message_type = MessageType.ALERT


@dataclass
class DecisionRequest(Message):
    """Request for clinical decision.

    Attributes:
        patient_id: Patient requiring decision
        decision_type: Type of decision needed
        context: Clinical context for decision
        deadline: Time by which decision is needed

    Example:
        >>> request = DecisionRequest(
        ...     message_id='req_001',
        ...     sender='nurse_1',
        ...     recipient='clinician_1',
        ...     patient_id='P001',
        ...     decision_type='intervention',
        ...     context={'vitals': {...}, 'labs': {...}},
        ...     deadline=2.0  # 2 hours
        ... )
    """

    patient_id: str = ""
    decision_type: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None

    def __post_init__(self):
        """Set message type to REQUEST."""
        self.message_type = MessageType.REQUEST


@dataclass
class HandoffMessage(Message):
    """Patient handoff between clinicians.

    Attributes:
        patient_id: Patient being handed off
        handoff_type: Type of handoff (shift, transfer, consult)
        summary: Clinical summary
        pending_tasks: Tasks that need completion
        concerns: Clinical concerns to follow up

    Example:
        >>> handoff = HandoffMessage(
        ...     message_id='handoff_001',
        ...     sender='clinician_day',
        ...     recipient='clinician_night',
        ...     patient_id='P001',
        ...     handoff_type='shift_change',
        ...     summary='62F with sepsis, on antibiotics x 6h',
        ...     pending_tasks=['repeat_lactate', 'assess_response'],
        ...     concerns=['Persistent hypotension despite fluids']
        ... )
    """

    patient_id: str = ""
    handoff_type: str = ""
    summary: str = ""
    pending_tasks: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set message type to HANDOFF."""
        self.message_type = MessageType.HANDOFF


class InteractionProtocol(ABC):
    """Abstract base class for interaction protocols.

    An interaction protocol defines rules for how agents communicate
    and coordinate actions.

    Example:
        >>> class CustomProtocol(InteractionProtocol):
        ...     def can_send(self, sender, recipient, message_type):
        ...         return True  # Allow all messages
        ...
        ...     def prioritize_messages(self, messages):
        ...         return sorted(messages, key=lambda m: m.priority.value)
        ...
        ...     def handle_message(self, message, environment):
        ...         # Custom handling logic
        ...         pass
    """

    @abstractmethod
    def can_send(
        self, sender: 'Agent', recipient: 'Agent', message_type: MessageType
    ) -> bool:
        """Check if sender can send message to recipient.

        Args:
            sender: Sending agent
            recipient: Receiving agent
            message_type: Type of message

        Returns:
            True if message is allowed
        """
        pass

    @abstractmethod
    def prioritize_messages(self, messages: List[Message]) -> List[Message]:
        """Prioritize messages for processing.

        Args:
            messages: List of messages

        Returns:
            Ordered list of messages by priority
        """
        pass

    @abstractmethod
    def handle_message(self, message: Message, environment: 'HospitalEnvironment'):
        """Handle message processing.

        Args:
            message: Message to handle
            environment: Hospital environment
        """
        pass


class StandardClinicalProtocol(InteractionProtocol):
    """Standard clinical communication protocol.

    Implements typical clinical communication rules:
    - CDSS can alert clinicians
    - Nurses can request decisions from clinicians
    - Clinicians can order interventions
    - Critical alerts have highest priority

    Example:
        >>> protocol = StandardClinicalProtocol()
        >>> can_alert = protocol.can_send(cdss_agent, clinician_agent, MessageType.ALERT)
    """

    def __init__(self):
        """Initialize protocol."""
        # Define allowed communications
        self.allowed_communications = {
            'cdss': {'clinician': [MessageType.ALERT]},
            'nurse': {
                'clinician': [MessageType.REQUEST, MessageType.NOTIFICATION],
                'patient': [MessageType.NOTIFICATION],
            },
            'clinician': {
                'nurse': [MessageType.RESPONSE, MessageType.REQUEST],
                'clinician': [MessageType.HANDOFF],
                'patient': [MessageType.NOTIFICATION],
            },
        }

    def can_send(
        self, sender: 'Agent', recipient: 'Agent', message_type: MessageType
    ) -> bool:
        """Check if communication is allowed."""
        sender_type = sender.agent_type.value
        recipient_type = recipient.agent_type.value

        if sender_type not in self.allowed_communications:
            return False

        if recipient_type not in self.allowed_communications[sender_type]:
            return False

        allowed_types = self.allowed_communications[sender_type][recipient_type]
        return message_type in allowed_types

    def prioritize_messages(self, messages: List[Message]) -> List[Message]:
        """Prioritize by message priority."""
        return sorted(
            messages, key=lambda m: (m.priority.value, m.timestamp), reverse=True
        )

    def handle_message(self, message: Message, environment: 'HospitalEnvironment'):
        """Handle message based on type."""
        if message.message_type == MessageType.ALERT:
            self._handle_alert(message, environment)
        elif message.message_type == MessageType.REQUEST:
            self._handle_request(message, environment)
        elif message.message_type == MessageType.HANDOFF:
            self._handle_handoff(message, environment)
        else:
            # Default: just acknowledge
            message.acknowledge()

    def _handle_alert(self, message: AlertMessage, environment: 'HospitalEnvironment'):
        """Handle CDSS alert."""
        # Send to recipient's alert queue
        environment.send_alert(
            from_agent=message.sender,
            to_agent=message.recipient,
            alert_data=message.content,
        )

    def _handle_request(
        self, message: DecisionRequest, environment: 'HospitalEnvironment'
    ):
        """Handle decision request."""
        # Add to recipient's task queue
        # Implementation depends on environment
        pass

    def _handle_handoff(
        self, message: HandoffMessage, environment: 'HospitalEnvironment'
    ):
        """Handle patient handoff."""
        # Transfer patient assignment
        # Implementation depends on environment
        pass


def perform_interaction(
    sender: 'Agent',
    recipient: 'Agent',
    message: Message,
    protocol: InteractionProtocol,
    environment: 'HospitalEnvironment',
) -> bool:
    """Perform interaction between agents.

    Args:
        sender: Sending agent
        recipient: Receiving agent
        message: Message to send
        protocol: Interaction protocol
        environment: Hospital environment

    Returns:
        True if interaction successful

    Example:
        >>> from basics_cdss.multiagent import (
        ...     perform_interaction, AlertMessage, StandardClinicalProtocol
        ... )
        >>>
        >>> alert = AlertMessage(
        ...     message_id='alert_001',
        ...     sender=cdss.agent_id,
        ...     recipient=clinician.agent_id,
        ...     patient_id='P001',
        ...     alert_type='sepsis_risk',
        ...     risk_score=0.85
        ... )
        >>>
        >>> success = perform_interaction(
        ...     sender=cdss,
        ...     recipient=clinician,
        ...     message=alert,
        ...     protocol=StandardClinicalProtocol(),
        ...     environment=hospital
        ... )
    """
    # Check if communication is allowed
    if not protocol.can_send(sender, recipient, message.message_type):
        return False

    # Handle message
    protocol.handle_message(message, environment)

    return True


def create_alert_from_cdss(
    cdss_agent: 'CDSSAgent',
    clinician_agent: 'ClinicianAgent',
    patient_id: str,
    risk_score: float,
    alert_type: str,
    recommendation: Dict[str, Any],
    confidence: float,
    timestamp: float,
) -> AlertMessage:
    """Helper function to create CDSS alert.

    Args:
        cdss_agent: CDSS agent
        clinician_agent: Target clinician
        patient_id: Patient ID
        risk_score: Predicted risk
        alert_type: Type of alert
        recommendation: Recommended action
        confidence: Model confidence
        timestamp: Current time

    Returns:
        AlertMessage object

    Example:
        >>> alert = create_alert_from_cdss(
        ...     cdss_agent=cdss,
        ...     clinician_agent=clinician,
        ...     patient_id='P001',
        ...     risk_score=0.85,
        ...     alert_type='sepsis_risk',
        ...     recommendation={'action': 'obtain_lactate'},
        ...     confidence=0.90,
        ...     timestamp=12.5
        ... )
    """
    import uuid

    # Determine priority based on risk score
    if risk_score > 0.9:
        priority = MessagePriority.CRITICAL
    elif risk_score > 0.8:
        priority = MessagePriority.HIGH
    elif risk_score > 0.6:
        priority = MessagePriority.NORMAL
    else:
        priority = MessagePriority.LOW

    return AlertMessage(
        message_id=str(uuid.uuid4()),
        sender=cdss_agent.agent_id,
        recipient=clinician_agent.agent_id,
        patient_id=patient_id,
        alert_type=alert_type,
        risk_score=risk_score,
        recommendation=recommendation,
        confidence=confidence,
        priority=priority,
        timestamp=timestamp,
        content={
            'patient_id': patient_id,
            'risk_score': risk_score,
            'alert_type': alert_type,
            'recommendation': recommendation,
        },
    )


def compute_communication_overhead(
    messages: List[Message], agents: List['Agent']
) -> Dict[str, Any]:
    """Compute communication overhead metrics.

    Args:
        messages: List of messages
        agents: List of agents

    Returns:
        Dictionary with overhead metrics:
            - total_messages: Total message count
            - messages_per_agent: Average messages per agent
            - high_priority_rate: Proportion of high priority messages
            - response_times: Average time to acknowledge

    Example:
        >>> overhead = compute_communication_overhead(
        ...     messages=hospital.alerts,
        ...     agents=list(hospital.agents.values())
        ... )
    """
    if not messages:
        return {
            'total_messages': 0,
            'messages_per_agent': 0.0,
            'high_priority_rate': 0.0,
            'response_times': [],
        }

    # Count high priority messages
    high_priority = sum(
        1
        for msg in messages
        if msg.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]
    )

    high_priority_rate = high_priority / len(messages)

    # Messages per agent
    messages_per_agent = len(messages) / len(agents) if agents else 0

    return {
        'total_messages': len(messages),
        'messages_per_agent': messages_per_agent,
        'high_priority_rate': high_priority_rate,
        'acknowledged_rate': sum(1 for m in messages if m.acknowledged) / len(messages),
    }
