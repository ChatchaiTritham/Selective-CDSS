"""Temporal perturbation operators for digital twin trajectories.

This module extends static perturbation operators (from basics_cdss.scenario)
to the temporal domain, enabling simulation of time-varying uncertainty:

- Missing data over time (intermittent sensor failures)
- Correlated measurement noise (systematic bias)
- Evolving contradictions (changing information)
- Delayed observations (real-world reporting lag)
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
from basics_cdss.temporal.digital_twin import PatientState


class TemporalPerturbationOperator:
    """Base class for temporal perturbation operators.

    Temporal perturbations modify entire patient trajectories rather than
    single snapshots, enabling simulation of realistic time-varying uncertainty.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize temporal perturbation operator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def apply(self, trajectory: List[PatientState]) -> List[PatientState]:
        """Apply temporal perturbation to trajectory.

        Args:
            trajectory: List of patient states over time

        Returns:
            Perturbed trajectory
        """
        raise NotImplementedError

    def reset_rng(self):
        """Reset random number generator to initial seed."""
        self.rng = np.random.RandomState(self.seed)


class TemporalMaskOperator(TemporalPerturbationOperator):
    """Simulate intermittent missing data over time.

    Real-world scenarios:
    - Sensor/device failures (e.g., pulse oximeter disconnected)
    - Delayed lab results (waiting for culture results)
    - Nurse workload (missed vital sign checks during busy periods)
    - Intermittent connectivity (telemetry dropout)

    Masking pattern can be:
    - Random: Each time point independently
    - Bursty: Consecutive missing observations
    - Feature-specific: Some sensors fail more often

    Example:
        >>> from basics_cdss.temporal import TemporalMaskOperator
        >>>
        >>> # Create operator with 20% missingness
        >>> masker = TemporalMaskOperator(p_mask=0.2, burst_length=3, seed=42)
        >>>
        >>> # Apply to trajectory
        >>> perturbed = masker.apply(trajectory)
        >>>
        >>> # Check missingness profile
        >>> profile = masker.compute_uncertainty_profile(perturbed)
        >>> print(f"Missingness: {profile['missingness']:.2%}")
    """

    def __init__(
        self,
        p_mask: float = 0.15,
        burst_length: int = 1,
        feature_specific_rates: Optional[Dict[str, float]] = None,
        protected_features: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize temporal masking operator.

        Args:
            p_mask: Probability of masking at each time point
            burst_length: Average length of consecutive missing periods
            feature_specific_rates: Different masking rates per feature
                e.g., {'oxygen_saturation': 0.3, 'temperature': 0.1}
            protected_features: Features that should never be masked
                (e.g., 'archetype_id', 'patient_id')
            seed: Random seed
        """
        super().__init__(seed)
        self.p_mask = p_mask
        self.burst_length = burst_length
        self.feature_specific_rates = feature_specific_rates or {}
        self.protected_features = protected_features or [
            'archetype_id',
            'patient_id',
            'timestamp',
        ]

    def apply(self, trajectory: List[PatientState]) -> List[PatientState]:
        """Apply temporal masking to trajectory."""
        perturbed_trajectory = []

        # Identify maskable features from first state
        maskable_features = [
            f for f in trajectory[0].features.keys() if f not in self.protected_features
        ]

        for state in trajectory:
            perturbed_state = state.copy()

            # For each feature, decide whether to mask
            for feature in maskable_features:
                p_feature = self.feature_specific_rates.get(feature, self.p_mask)

                if self.rng.rand() < p_feature:
                    # Mask feature (remove it)
                    if feature in perturbed_state.features:
                        del perturbed_state.features[feature]

                        # Add to metadata
                        if 'masked_features' not in perturbed_state.metadata:
                            perturbed_state.metadata['masked_features'] = []
                        perturbed_state.metadata['masked_features'].append(feature)

            perturbed_trajectory.append(perturbed_state)

        return perturbed_trajectory

    def compute_uncertainty_profile(
        self, trajectory: List[PatientState]
    ) -> Dict[str, float]:
        """Compute missingness statistics for trajectory.

        Args:
            trajectory: Patient trajectory

        Returns:
            Dictionary with missingness metrics
        """
        total_features = 0
        total_masked = 0

        for state in trajectory:
            masked = state.metadata.get('masked_features', [])
            total_masked += len(masked)
            total_features += len(state.features) + len(masked)

        missingness = total_masked / total_features if total_features > 0 else 0.0

        return {
            'missingness': missingness,
            'ambiguity': 0.0,
            'conflict': 0.0,
            'degradation': 0.0,
        }


class TemporalNoiseOperator(TemporalPerturbationOperator):
    """Simulate correlated measurement noise over time.

    Real-world scenarios:
    - Systematic measurement bias (miscalibrated device)
    - Observer variability (different nurses measuring BP)
    - Environmental factors (temperature sensor near heater)
    - Natural biological variation

    Noise can be:
    - White noise: Independent at each time point
    - AR(1) process: Auto-regressive (correlated over time)
    - Random walk: Cumulative drift

    Example:
        >>> from basics_cdss.temporal import TemporalNoiseOperator
        >>>
        >>> # Create AR(1) noise with correlation 0.7
        >>> noiser = TemporalNoiseOperator(
        ...     noise_sigma=0.1,
        ...     temporal_correlation=0.7,
        ...     seed=42
        ... )
        >>>
        >>> # Apply to trajectory
        >>> perturbed = noiser.apply(trajectory)
    """

    def __init__(
        self,
        noise_sigma: float = 0.1,
        temporal_correlation: float = 0.5,
        feature_specific_sigma: Optional[Dict[str, float]] = None,
        protected_features: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize temporal noise operator.

        Args:
            noise_sigma: Standard deviation of noise (relative to feature range)
            temporal_correlation: AR(1) correlation coefficient (0=white, 1=random walk)
            feature_specific_sigma: Different noise levels per feature
            protected_features: Features not to perturb
            seed: Random seed
        """
        super().__init__(seed)
        self.noise_sigma = noise_sigma
        self.temporal_correlation = temporal_correlation
        self.feature_specific_sigma = feature_specific_sigma or {}
        self.protected_features = protected_features or [
            'archetype_id',
            'patient_id',
            'timestamp',
        ]

    def apply(self, trajectory: List[PatientState]) -> List[PatientState]:
        """Apply temporal noise to trajectory."""
        # Identify numeric features
        numeric_features = []
        for key, value in trajectory[0].features.items():
            if key not in self.protected_features:
                if isinstance(value, (int, float)):
                    numeric_features.append(key)

        # Generate AR(1) noise process for each feature
        T = len(trajectory)
        noise_processes = {}

        for feature in numeric_features:
            sigma = self.feature_specific_sigma.get(feature, self.noise_sigma)

            # Generate AR(1): x_t = rho * x_{t-1} + epsilon_t
            rho = self.temporal_correlation
            noise = np.zeros(T)
            noise[0] = self.rng.normal(0, sigma)

            for t in range(1, T):
                noise[t] = rho * noise[t - 1] + self.rng.normal(
                    0, sigma * np.sqrt(1 - rho**2)
                )

            noise_processes[feature] = noise

        # Apply noise to trajectory
        perturbed_trajectory = []
        for t, state in enumerate(trajectory):
            perturbed_state = state.copy()

            for feature in numeric_features:
                if feature in perturbed_state.features:
                    original_value = perturbed_state.features[feature]

                    # Add noise (as percentage of value)
                    noise_value = noise_processes[feature][t] * abs(original_value)
                    perturbed_value = original_value + noise_value

                    perturbed_state.features[feature] = type(original_value)(
                        perturbed_value
                    )

            # Record noise magnitude in metadata
            perturbed_state.metadata['noise_applied'] = True
            perturbed_trajectory.append(perturbed_state)

        return perturbed_trajectory

    def compute_uncertainty_profile(
        self, trajectory: List[PatientState]
    ) -> Dict[str, float]:
        """Compute noise statistics."""
        return {
            'missingness': 0.0,
            'ambiguity': self.noise_sigma,
            'conflict': 0.0,
            'degradation': 0.0,
        }


class TemporalConflictOperator(TemporalPerturbationOperator):
    """Simulate evolving contradictions over time.

    Real-world scenarios:
    - Changing information (initial report vs follow-up)
    - Different data sources (nursing notes vs physician assessment)
    - Temporal inconsistency (fever reported then normal temp)

    Example:
        >>> from basics_cdss.temporal import TemporalConflictOperator
        >>>
        >>> # Create conflict operator
        >>> conflictor = TemporalConflictOperator(p_conflict=0.1, seed=42)
        >>>
        >>> # Apply to trajectory
        >>> perturbed = conflictor.apply(trajectory)
    """

    def __init__(
        self,
        p_conflict: float = 0.1,
        conflict_pairs: Optional[List[tuple]] = None,
        protected_features: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize temporal conflict operator.

        Args:
            p_conflict: Probability of introducing conflict at each time
            conflict_pairs: List of (feature1, feature2) pairs that conflict
                e.g., [('fever_present', 'temperature')]
            protected_features: Features not to modify
            seed: Random seed
        """
        super().__init__(seed)
        self.p_conflict = p_conflict
        self.conflict_pairs = conflict_pairs or []
        self.protected_features = protected_features or [
            'archetype_id',
            'patient_id',
            'timestamp',
        ]

    def apply(self, trajectory: List[PatientState]) -> List[PatientState]:
        """Apply temporal conflicts to trajectory."""
        perturbed_trajectory = []

        for state in trajectory:
            perturbed_state = state.copy()

            # Randomly introduce conflicts
            if self.rng.rand() < self.p_conflict and self.conflict_pairs:
                pair = self.conflict_pairs[self.rng.randint(len(self.conflict_pairs))]
                feature1, feature2 = pair

                if (
                    feature1 in perturbed_state.features
                    and feature2 in perturbed_state.features
                ):
                    # Introduce contradiction (implementation depends on feature types)
                    # For now, just flip boolean or reverse numeric trend
                    if isinstance(perturbed_state.features[feature1], bool):
                        perturbed_state.features[feature1] = (
                            not perturbed_state.features[feature1]
                        )

                    perturbed_state.metadata['conflict_introduced'] = True

            perturbed_trajectory.append(perturbed_state)

        return perturbed_trajectory

    def compute_uncertainty_profile(
        self, trajectory: List[PatientState]
    ) -> Dict[str, float]:
        """Compute conflict statistics."""
        n_conflicts = sum(
            1
            for state in trajectory
            if state.metadata.get('conflict_introduced', False)
        )

        conflict_rate = n_conflicts / len(trajectory) if trajectory else 0.0

        return {
            'missingness': 0.0,
            'ambiguity': 0.0,
            'conflict': conflict_rate,
            'degradation': 0.0,
        }


class CompositeTemporalPerturbation(TemporalPerturbationOperator):
    """Combine multiple temporal perturbations.

    Simulates realistic scenarios where multiple uncertainty types occur
    simultaneously (e.g., missing data + measurement noise).

    Example:
        >>> from basics_cdss.temporal import (
        ...     CompositeTemporalPerturbation,
        ...     TemporalMaskOperator,
        ...     TemporalNoiseOperator
        ... )
        >>>
        >>> # Create composite perturbation
        >>> composite = CompositeTemporalPerturbation([
        ...     TemporalMaskOperator(p_mask=0.15, seed=42),
        ...     TemporalNoiseOperator(noise_sigma=0.1, seed=43)
        ... ])
        >>>
        >>> # Apply all perturbations
        >>> perturbed = composite.apply(trajectory)
    """

    def __init__(
        self, operators: List[TemporalPerturbationOperator], seed: Optional[int] = None
    ):
        """Initialize composite perturbation.

        Args:
            operators: List of perturbation operators to apply in sequence
            seed: Random seed
        """
        super().__init__(seed)
        self.operators = operators

    def apply(self, trajectory: List[PatientState]) -> List[PatientState]:
        """Apply all perturbations sequentially."""
        perturbed = trajectory

        for operator in self.operators:
            perturbed = operator.apply(perturbed)

        return perturbed

    def compute_uncertainty_profile(
        self, trajectory: List[PatientState]
    ) -> Dict[str, float]:
        """Aggregate uncertainty profiles from all operators."""
        profile = {
            'missingness': 0.0,
            'ambiguity': 0.0,
            'conflict': 0.0,
            'degradation': 0.0,
        }

        for operator in self.operators:
            op_profile = operator.compute_uncertainty_profile(trajectory)
            for key in profile:
                profile[key] = max(profile[key], op_profile[key])

        return profile


def create_default_temporal_perturbation(
    perturbation_type: str, seed: Optional[int] = None, **kwargs
) -> TemporalPerturbationOperator:
    """Factory function for creating temporal perturbations.

    Args:
        perturbation_type: Type of perturbation ('mask', 'noise', 'conflict', 'composite')
        seed: Random seed
        **kwargs: Additional parameters for specific perturbation types

    Returns:
        Temporal perturbation operator

    Example:
        >>> # Create masking perturbation
        >>> masker = create_default_temporal_perturbation('mask', p_mask=0.2, seed=42)
        >>>
        >>> # Create composite
        >>> composite = create_default_temporal_perturbation('composite', seed=42)
    """
    if perturbation_type == 'mask':
        return TemporalMaskOperator(p_mask=kwargs.get('p_mask', 0.15), seed=seed)

    elif perturbation_type == 'noise':
        return TemporalNoiseOperator(
            noise_sigma=kwargs.get('noise_sigma', 0.1),
            temporal_correlation=kwargs.get('temporal_correlation', 0.5),
            seed=seed,
        )

    elif perturbation_type == 'conflict':
        return TemporalConflictOperator(
            p_conflict=kwargs.get('p_conflict', 0.1), seed=seed
        )

    elif perturbation_type == 'composite':
        return CompositeTemporalPerturbation(
            operators=[
                TemporalMaskOperator(p_mask=0.15, seed=seed),
                TemporalNoiseOperator(noise_sigma=0.1, seed=seed + 1 if seed else None),
            ],
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
