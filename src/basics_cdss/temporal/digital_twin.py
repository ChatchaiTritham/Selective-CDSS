"""Digital Twin implementation for temporal patient simulation.

This module provides the core PatientDigitalTwin class that extends
static SynDX archetypes into time-evolving patient models.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class PatientState:
    """Represents patient state at a single time point.

    Attributes:
        timestamp: Time in hours from initial state (t=0)
        features: Dictionary of clinical features (vitals, labs, symptoms)
        metadata: Additional information (uncertainty, data quality)
    """

    timestamp: float
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to features."""
        return self.features[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get feature with default value."""
        return self.features.get(key, default)

    def copy(self) -> 'PatientState':
        """Create deep copy of patient state."""
        return PatientState(
            timestamp=self.timestamp,
            features=copy.deepcopy(self.features),
            metadata=copy.deepcopy(self.metadata),
        )


class PatientDigitalTwin:
    """Digital twin for temporal patient simulation.

    A digital twin represents a virtual patient whose state evolves over
    time according to physiological disease progression models. It enables:

    1. Temporal scenario generation
    2. What-if analysis (counterfactuals)
    3. Intervention effect simulation
    4. Time-varying CDSS evaluation

    Example:
        >>> from basics_cdss.temporal import PatientDigitalTwin, SepsisModel
        >>>
        >>> # Initialize from archetype
        >>> initial_state = {
        ...     'temperature': 38.5,
        ...     'heart_rate': 110,
        ...     'white_blood_cell_count': 15000,
        ...     'blood_pressure_sys': 95
        ... }
        >>>
        >>> twin = PatientDigitalTwin(
        ...     archetype_id="A001",
        ...     initial_state=initial_state,
        ...     disease_model=SepsisModel(),
        ...     seed=42
        ... )
        >>>
        >>> # Simulate 24 hours
        >>> trajectory = twin.simulate(horizon_hours=24, dt=1.0)
        >>> print(f"States generated: {len(trajectory)}")  # 25 states (t=0 to t=24)
        >>>
        >>> # Apply intervention at t=12
        >>> twin.reset()
        >>> for t in range(12):
        ...     twin.step(dt=1.0)
        >>> twin.apply_intervention({'antibiotic': True, 'fluid_bolus': 1000})
        >>> for t in range(12, 24):
        ...     twin.step(dt=1.0)

    Attributes:
        archetype_id: Identifier for source archetype (from SynDX)
        current_state: Current patient state
        history: List of all past states
        disease_model: Physiological progression model
        seed: Random seed for reproducibility
        interventions: List of interventions applied
    """

    def __init__(
        self,
        archetype_id: str,
        initial_state: Dict[str, Any],
        disease_model: 'DiseaseModel',
        seed: Optional[int] = None,
        protected_features: Optional[List[str]] = None,
    ):
        """Initialize digital twin.

        Args:
            archetype_id: Identifier for source archetype
            initial_state: Initial patient features at t=0
            disease_model: Model for physiological progression
            seed: Random seed for reproducibility
            protected_features: Features that should never be modified
                (e.g., 'archetype_id', 'patient_id')
        """
        self.archetype_id = archetype_id
        self.disease_model = disease_model
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Protected features
        self.protected_features = protected_features or ['archetype_id', 'patient_id']

        # Initialize state
        self.initial_state = PatientState(
            timestamp=0.0,
            features=copy.deepcopy(initial_state),
            metadata={'archetype_id': archetype_id},
        )
        self.current_state = self.initial_state.copy()
        self.history: List[PatientState] = [self.initial_state.copy()]

        # Track interventions
        self.interventions: List[Dict[str, Any]] = []

    def step(
        self,
        dt: float = 1.0,
        interventions: Optional[Dict[str, Any]] = None,
        stochastic: bool = True,
    ) -> PatientState:
        """Advance patient state by dt hours.

        Args:
            dt: Time step in hours
            interventions: Dictionary of interventions applied at this step
                e.g., {'antibiotic': True, 'fluid_bolus': 1000}
            stochastic: Whether to include random noise in evolution

        Returns:
            New patient state after time step
        """
        # Record intervention
        if interventions:
            self.interventions.append(
                {
                    'timestamp': self.current_state.timestamp,
                    'interventions': interventions,
                }
            )

        # Evolve state using disease model
        next_features = self.disease_model.evolve(
            current_state=self.current_state.features,
            dt=dt,
            interventions=interventions,
            rng=self.rng if stochastic else None,
        )

        # Create new state
        next_state = PatientState(
            timestamp=self.current_state.timestamp + dt,
            features=next_features,
            metadata={
                'archetype_id': self.archetype_id,
                'interventions': interventions or {},
                'stochastic': stochastic,
            },
        )

        # Update current state and history
        self.current_state = next_state
        self.history.append(next_state.copy())

        return next_state

    def simulate(
        self,
        horizon_hours: float,
        dt: float = 1.0,
        intervention_schedule: Optional[Dict[float, Dict[str, Any]]] = None,
        stochastic: bool = True,
    ) -> List[PatientState]:
        """Simulate patient trajectory over time horizon.

        Args:
            horizon_hours: Total simulation time in hours
            dt: Time step size in hours
            intervention_schedule: Dictionary mapping timestamps to interventions
                e.g., {6.0: {'antibiotic': True}, 12.0: {'fluid_bolus': 500}}
            stochastic: Whether to include stochastic noise

        Returns:
            List of patient states from t=0 to t=horizon_hours
        """
        self.reset()

        n_steps = int(horizon_hours / dt)
        for step_idx in range(n_steps):
            current_time = step_idx * dt

            # Check for scheduled interventions
            interventions = None
            if intervention_schedule:
                interventions = intervention_schedule.get(current_time)

            self.step(dt=dt, interventions=interventions, stochastic=stochastic)

        return self.history

    def apply_intervention(self, interventions: Dict[str, Any]) -> None:
        """Apply intervention at current time (convenience method).

        Args:
            interventions: Dictionary of interventions to apply
        """
        self.step(dt=0.0, interventions=interventions, stochastic=False)

    def reset(self) -> None:
        """Reset twin to initial state."""
        self.current_state = self.initial_state.copy()
        self.history = [self.initial_state.copy()]
        self.interventions = []
        self.rng = np.random.RandomState(self.seed)

    def clone(self) -> 'PatientDigitalTwin':
        """Create independent copy of this digital twin.

        Returns:
            New PatientDigitalTwin with same configuration but independent state
        """
        cloned = PatientDigitalTwin(
            archetype_id=self.archetype_id,
            initial_state=copy.deepcopy(self.initial_state.features),
            disease_model=self.disease_model,
            seed=self.seed,
            protected_features=self.protected_features.copy(),
        )

        # Copy current state and history
        cloned.current_state = self.current_state.copy()
        cloned.history = [state.copy() for state in self.history]
        cloned.interventions = copy.deepcopy(self.interventions)

        return cloned

    def get_trajectory_dataframe(self) -> 'pd.DataFrame':
        """Export trajectory as pandas DataFrame.

        Returns:
            DataFrame with columns: timestamp, feature1, feature2, ...
        """
        import pandas as pd

        records = []
        for state in self.history:
            record = {'timestamp': state.timestamp}
            record.update(state.features)
            records.append(record)

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"PatientDigitalTwin(archetype_id='{self.archetype_id}', "
            f"current_time={self.current_state.timestamp:.1f}h, "
            f"n_states={len(self.history)})"
        )


class DigitalTwinFactory:
    """Factory for creating digital twins from archetypes.

    Simplifies creation of multiple digital twins from SynDX archetypes.

    Example:
        >>> from basics_cdss.temporal import DigitalTwinFactory, SepsisModel
        >>> import pandas as pd
        >>>
        >>> # Load archetypes
        >>> archetypes_df = pd.read_csv("syndx_archetypes.csv")
        >>>
        >>> # Create factory
        >>> factory = DigitalTwinFactory(disease_model=SepsisModel(), seed=42)
        >>>
        >>> # Generate digital twins
        >>> twins = factory.create_from_dataframe(
        ...     archetypes_df,
        ...     n_per_archetype=10
        ... )
        >>> print(f"Created {len(twins)} digital twins")
    """

    def __init__(
        self,
        disease_model: 'DiseaseModel',
        seed: Optional[int] = None,
        protected_features: Optional[List[str]] = None,
    ):
        """Initialize factory.

        Args:
            disease_model: Disease progression model to use for all twins
            seed: Base random seed (will be incremented for each twin)
            protected_features: Features to protect across all twins
        """
        self.disease_model = disease_model
        self.seed = seed
        self.protected_features = protected_features
        self.rng = np.random.RandomState(seed)

    def create_from_archetype(
        self, archetype_id: str, features: Dict[str, Any], n_twins: int = 1
    ) -> List[PatientDigitalTwin]:
        """Create digital twins from single archetype.

        Args:
            archetype_id: Archetype identifier
            features: Initial patient features
            n_twins: Number of twins to create (with different seeds)

        Returns:
            List of digital twins
        """
        twins = []
        for i in range(n_twins):
            twin_seed = None if self.seed is None else self.seed + i

            twin = PatientDigitalTwin(
                archetype_id=archetype_id,
                initial_state=copy.deepcopy(features),
                disease_model=self.disease_model,
                seed=twin_seed,
                protected_features=self.protected_features,
            )
            twins.append(twin)

        return twins

    def create_from_dataframe(
        self,
        archetypes_df: 'pd.DataFrame',
        n_per_archetype: int = 1,
        archetype_id_col: str = 'archetype_id',
    ) -> List[PatientDigitalTwin]:
        """Create digital twins from DataFrame of archetypes.

        Args:
            archetypes_df: DataFrame with archetype data
            n_per_archetype: Number of twins per archetype
            archetype_id_col: Column name containing archetype IDs

        Returns:
            List of all digital twins
        """
        import pandas as pd

        all_twins = []

        for idx, row in archetypes_df.iterrows():
            archetype_id = row[archetype_id_col]
            features = row.to_dict()

            twins = self.create_from_archetype(
                archetype_id=archetype_id, features=features, n_twins=n_per_archetype
            )
            all_twins.extend(twins)

        return all_twins
