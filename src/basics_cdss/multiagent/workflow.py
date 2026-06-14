"""Clinical workflow modeling for multi-agent simulation.

This module models clinical workflows as sequences of tasks with
dependencies, timing constraints, and decision points.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np


class TaskStatus(Enum):
    """Status of a workflow task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class WorkflowState(Enum):
    """State of clinical workflow."""

    NOT_STARTED = "not_started"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class Task:
    """Represents a clinical task within a workflow.

    Attributes:
        task_id: Unique identifier
        name: Task name
        description: Task description
        required_agent_type: Type of agent that can perform task
        duration_minutes: Expected duration in minutes
        dependencies: Task IDs that must complete before this task
        status: Current task status
        assigned_agent: Agent assigned to task
        start_time: When task started
        end_time: When task completed

    Example:
        >>> task = Task(
        ...     task_id='vitals_1',
        ...     name='Measure vital signs',
        ...     required_agent_type='nurse',
        ...     duration_minutes=5,
        ...     dependencies=[]
        ... )
    """

    task_id: str
    name: str
    description: str = ""
    required_agent_type: Optional[str] = None
    duration_minutes: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to start.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            True if all dependencies are satisfied
        """
        if self.status != TaskStatus.PENDING:
            return False

        # Check dependencies
        for dep in self.dependencies:
            if dep not in completed_tasks:
                return False

        return True

    def start(self, agent_id: str, current_time: float):
        """Start task execution.

        Args:
            agent_id: Agent performing task
            current_time: Current simulation time
        """
        self.status = TaskStatus.IN_PROGRESS
        self.assigned_agent = agent_id
        self.start_time = current_time

    def complete(self, current_time: float):
        """Mark task as completed.

        Args:
            current_time: Current simulation time
        """
        self.status = TaskStatus.COMPLETED
        self.end_time = current_time

    @property
    def duration_hours(self) -> float:
        """Task duration in hours."""
        return self.duration_minutes / 60.0


class ClinicalWorkflow:
    """Clinical workflow as a directed acyclic graph of tasks.

    A workflow represents a clinical protocol or care pathway,
    consisting of tasks with dependencies and timing constraints.

    Example:
        >>> from basics_cdss.multiagent import ClinicalWorkflow, Task
        >>>
        >>> # Create sepsis workflow
        >>> workflow = ClinicalWorkflow(
        ...     workflow_id='sepsis_bundle',
        ...     name='Sepsis 3-Hour Bundle'
        ... )
        >>>
        >>> # Add tasks
        >>> workflow.add_task(Task(
        ...     task_id='measure_lactate',
        ...     name='Measure lactate',
        ...     required_agent_type='nurse',
        ...     duration_minutes=10
        ... ))
        >>>
        >>> workflow.add_task(Task(
        ...     task_id='obtain_cultures',
        ...     name='Obtain blood cultures',
        ...     required_agent_type='nurse',
        ...     duration_minutes=15,
        ...     dependencies=['measure_lactate']
        ... ))
        >>>
        >>> # Start workflow
        >>> workflow.start(current_time=0.0)
    """

    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str = "",
        time_limit_hours: Optional[float] = None,
    ):
        """Initialize workflow.

        Args:
            workflow_id: Unique identifier
            name: Workflow name
            description: Workflow description
            time_limit_hours: Time limit for workflow completion
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.time_limit_hours = time_limit_hours

        self.state = WorkflowState.NOT_STARTED
        self.tasks: Dict[str, Task] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def add_task(self, task: Task):
        """Add task to workflow.

        Args:
            task: Task to add
        """
        self.tasks[task.task_id] = task

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to start.

        Returns:
            List of tasks that can be started
        """
        completed_task_ids = {
            task_id
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }

        ready_tasks = [
            task for task in self.tasks.values() if task.is_ready(completed_task_ids)
        ]

        return ready_tasks

    def get_active_tasks(self) -> List[Task]:
        """Get tasks currently in progress.

        Returns:
            List of active tasks
        """
        return [
            task
            for task in self.tasks.values()
            if task.status == TaskStatus.IN_PROGRESS
        ]

    def start(self, current_time: float):
        """Start workflow execution.

        Args:
            current_time: Current simulation time
        """
        self.state = WorkflowState.ACTIVE
        self.start_time = current_time

    def update(self, current_time: float):
        """Update workflow state.

        Args:
            current_time: Current simulation time
        """
        if self.state != WorkflowState.ACTIVE:
            return

        # Check if all tasks completed
        all_completed = all(
            task.status == TaskStatus.COMPLETED for task in self.tasks.values()
        )

        if all_completed:
            self.complete(current_time)

        # Check time limit
        if self.time_limit_hours and self.start_time:
            elapsed = current_time - self.start_time
            if elapsed > self.time_limit_hours:
                self.state = WorkflowState.SUSPENDED

    def complete(self, current_time: float):
        """Mark workflow as completed.

        Args:
            current_time: Current simulation time
        """
        self.state = WorkflowState.COMPLETED
        self.end_time = current_time

    @property
    def completion_rate(self) -> float:
        """Proportion of completed tasks."""
        if not self.tasks:
            return 0.0

        n_completed = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED
        )

        return n_completed / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Whether workflow is complete."""
        return self.state == WorkflowState.COMPLETED

    @property
    def total_duration_hours(self) -> Optional[float]:
        """Total workflow duration in hours."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export workflow to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'state': self.state.value,
            'completion_rate': self.completion_rate,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_duration_hours': self.total_duration_hours,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'name': task.name,
                    'status': task.status.value,
                    'duration_minutes': task.duration_minutes,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                }
                for task in self.tasks.values()
            ],
        }


def create_sepsis_workflow() -> ClinicalWorkflow:
    """Create workflow for sepsis 3-hour bundle.

    Based on Surviving Sepsis Campaign guidelines.

    Returns:
        ClinicalWorkflow for sepsis management

    Example:
        >>> workflow = create_sepsis_workflow()
        >>> workflow.start(current_time=0.0)
    """
    workflow = ClinicalWorkflow(
        workflow_id='sepsis_3hr',
        name='Sepsis 3-Hour Bundle',
        description='Surviving Sepsis Campaign 3-hour bundle',
        time_limit_hours=3.0,
    )

    # Task 1: Measure lactate
    workflow.add_task(
        Task(
            task_id='measure_lactate',
            name='Measure serum lactate',
            description='Obtain blood sample and measure lactate level',
            required_agent_type='nurse',
            duration_minutes=10,
            dependencies=[],
        )
    )

    # Task 2: Obtain blood cultures
    workflow.add_task(
        Task(
            task_id='obtain_cultures',
            name='Obtain blood cultures',
            description='Obtain blood cultures before antibiotic administration',
            required_agent_type='nurse',
            duration_minutes=15,
            dependencies=[],
        )
    )

    # Task 3: Administer broad-spectrum antibiotics
    workflow.add_task(
        Task(
            task_id='administer_antibiotics',
            name='Administer broad-spectrum antibiotics',
            description='Give IV antibiotics within 1 hour of recognition',
            required_agent_type='nurse',
            duration_minutes=20,
            dependencies=['obtain_cultures'],  # After cultures
        )
    )

    # Task 4: Administer crystalloid fluids
    workflow.add_task(
        Task(
            task_id='administer_fluids',
            name='Administer 30 mL/kg crystalloid',
            description='Rapid IV fluid resuscitation for hypotension/lactate ≥4',
            required_agent_type='nurse',
            duration_minutes=60,
            dependencies=['measure_lactate'],  # After lactate measurement
        )
    )

    # Task 5: Reassess hemodynamics
    workflow.add_task(
        Task(
            task_id='reassess',
            name='Reassess hemodynamics',
            description='Reassess volume status and tissue perfusion',
            required_agent_type='clinician',
            duration_minutes=10,
            dependencies=['administer_fluids'],
        )
    )

    return workflow


def create_acs_workflow() -> ClinicalWorkflow:
    """Create workflow for acute coronary syndrome.

    Returns:
        ClinicalWorkflow for ACS management

    Example:
        >>> workflow = create_acs_workflow()
        >>> workflow.start(current_time=0.0)
    """
    workflow = ClinicalWorkflow(
        workflow_id='acs_stemi',
        name='STEMI Management',
        description='ST-elevation myocardial infarction protocol',
        time_limit_hours=2.0,  # Door-to-balloon time
    )

    # Task 1: ECG
    workflow.add_task(
        Task(
            task_id='obtain_ecg',
            name='Obtain 12-lead ECG',
            description='ECG within 10 minutes of arrival',
            required_agent_type='nurse',
            duration_minutes=10,
            dependencies=[],
        )
    )

    # Task 2: Aspirin
    workflow.add_task(
        Task(
            task_id='administer_aspirin',
            name='Administer aspirin',
            description='Give 325mg aspirin PO',
            required_agent_type='nurse',
            duration_minutes=5,
            dependencies=['obtain_ecg'],
        )
    )

    # Task 3: Cardiac biomarkers
    workflow.add_task(
        Task(
            task_id='obtain_troponin',
            name='Obtain troponin',
            description='Draw blood for troponin measurement',
            required_agent_type='nurse',
            duration_minutes=10,
            dependencies=[],
        )
    )

    # Task 4: Activate cath lab
    workflow.add_task(
        Task(
            task_id='activate_cathlab',
            name='Activate cardiac catheterization lab',
            description='Call interventional cardiology for STEMI',
            required_agent_type='clinician',
            duration_minutes=15,
            dependencies=['obtain_ecg'],
        )
    )

    # Task 5: Transfer to cath lab
    workflow.add_task(
        Task(
            task_id='transfer_cathlab',
            name='Transfer to cath lab',
            description='Transport patient for PCI',
            required_agent_type='nurse',
            duration_minutes=20,
            dependencies=['activate_cathlab', 'administer_aspirin'],
        )
    )

    return workflow


def create_respiratory_distress_workflow() -> ClinicalWorkflow:
    """Create workflow for respiratory distress management.

    Returns:
        ClinicalWorkflow for respiratory distress

    Example:
        >>> workflow = create_respiratory_distress_workflow()
    """
    workflow = ClinicalWorkflow(
        workflow_id='respiratory_distress',
        name='Respiratory Distress Management',
        description='Acute respiratory failure protocol',
        time_limit_hours=1.0,
    )

    # Task 1: Oxygen therapy
    workflow.add_task(
        Task(
            task_id='apply_oxygen',
            name='Apply supplemental oxygen',
            description='Administer oxygen to maintain SpO2 > 90%',
            required_agent_type='nurse',
            duration_minutes=5,
            dependencies=[],
        )
    )

    # Task 2: ABG
    workflow.add_task(
        Task(
            task_id='obtain_abg',
            name='Obtain arterial blood gas',
            description='Draw ABG for pH, PaO2, PaCO2',
            required_agent_type='nurse',
            duration_minutes=15,
            dependencies=[],
        )
    )

    # Task 3: CXR
    workflow.add_task(
        Task(
            task_id='obtain_cxr',
            name='Obtain chest X-ray',
            description='Portable CXR',
            required_agent_type='nurse',
            duration_minutes=20,
            dependencies=[],
        )
    )

    # Task 4: Assess for intubation
    workflow.add_task(
        Task(
            task_id='assess_intubation',
            name='Assess need for intubation',
            description='Evaluate airway and ventilatory status',
            required_agent_type='clinician',
            duration_minutes=10,
            dependencies=['obtain_abg', 'obtain_cxr'],
        )
    )

    return workflow
