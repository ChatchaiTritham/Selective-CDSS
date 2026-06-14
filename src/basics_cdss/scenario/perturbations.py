"""
Perturbation operators for controlled uncertainty injection in scenario instantiation.

This module implements the perturbation operators described in Table 1 of the manuscript:
- Mask: Information missingness (remove features with probability p_mask)
- Noise: Ambiguity (add Gaussian noise to continuous features)
- Conflict: Internal inconsistency (introduce contradictory findings)
- Degrade: Reduced specificity (replace specific terms with vague descriptors)

All operators preserve clinical plausibility and are deterministic given a random seed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class PerturbationConfig:
    """Configuration for perturbation operators.

    Attributes:
        p_mask: Probability of masking each feature (0.0 to 1.0)
        noise_sigma: Standard deviation for Gaussian noise injection
        protected_features: Features that must not be removed (e.g., patient_id, archetype_id)
        continuous_features: List of feature names considered continuous for noise injection
        categorical_features: List of feature names considered categorical
        conflict_pairs: Dict mapping features to contradictory value mappings
        degrade_map: Dict mapping specific terms to vague descriptors
    """

    p_mask: float = 0.2
    noise_sigma: float = 0.1
    protected_features: Set[str] = field(
        default_factory=lambda: {"archetype_id", "triage_tier", "action", "patient_id"}
    )
    continuous_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    conflict_pairs: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    degrade_map: Dict[str, str] = field(default_factory=dict)


class PerturbationOperator:
    """Base class for perturbation operators."""

    def __init__(
        self, config: Optional[PerturbationConfig] = None, seed: Optional[int] = None
    ):
        """Initialize perturbation operator.

        Args:
            config: Perturbation configuration. If None, uses default.
            seed: Random seed for deterministic perturbation. If None, uses current state.
        """
        self.config = config or PerturbationConfig()
        self.rng = np.random.default_rng(seed)

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply perturbation to features.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (perturbed_features, uncertainty_metrics)
            where uncertainty_metrics contains quantitative measures of perturbation applied.
        """
        raise NotImplementedError("Subclasses must implement apply()")


class MaskOperator(PerturbationOperator):
    """Information missingness operator.

    Randomly removes features with probability p_mask to simulate incomplete
    symptom reporting or missing documentation.

    Implementation follows manuscript Table 1:
    - Operator: Mask
    - Uncertainty type: Information missingness
    - Validation: Clinical plausibility check; retain key safety signals
    """

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply masking perturbation.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (masked_features, uncertainty_profile) where:
            - masked_features: Features with some removed
            - uncertainty_profile: {"missingness": fraction_masked}
        """
        perturbed = copy.deepcopy(features)
        maskable_keys = [
            k for k in features.keys() if k not in self.config.protected_features
        ]

        if not maskable_keys:
            return perturbed, {"missingness": 0.0}

        # Determine which features to mask
        mask_decisions = self.rng.random(len(maskable_keys)) < self.config.p_mask
        masked_count = 0

        for key, should_mask in zip(maskable_keys, mask_decisions):
            if should_mask:
                del perturbed[key]
                masked_count += 1

        missingness = masked_count / len(maskable_keys) if maskable_keys else 0.0

        return perturbed, {"missingness": missingness}


class NoiseOperator(PerturbationOperator):
    """Ambiguity operator via additive Gaussian noise.

    Adds Gaussian noise to continuous features to simulate measurement uncertainty
    or ambiguous symptom intensity reporting.

    Implementation follows manuscript Table 1:
    - Operator: Noise
    - Uncertainty type: Ambiguity
    - Implementation: x' = x + ε, ε ~ N(0, σ²)
    - Validation: Bounded within clinically reasonable ranges
    """

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply noise perturbation to continuous features.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (noisy_features, uncertainty_profile) where:
            - noisy_features: Features with noise added to continuous values
            - uncertainty_profile: {"ambiguity": mean_abs_noise}
        """
        perturbed = copy.deepcopy(features)

        # Identify continuous features to perturb
        if self.config.continuous_features:
            target_features = [
                k for k in self.config.continuous_features if k in features
            ]
        else:
            # Auto-detect: numeric values that aren't protected
            target_features = [
                k
                for k, v in features.items()
                if k not in self.config.protected_features
                and isinstance(v, (int, float, np.number))
            ]

        if not target_features:
            return perturbed, {"ambiguity": 0.0}

        noise_applied = []
        for key in target_features:
            original_value = perturbed[key]
            noise = self.rng.normal(0, self.config.noise_sigma)
            perturbed[key] = original_value + noise
            noise_applied.append(abs(noise))

        ambiguity = np.mean(noise_applied) if noise_applied else 0.0

        return perturbed, {"ambiguity": float(ambiguity)}


class ConflictOperator(PerturbationOperator):
    """Internal inconsistency operator.

    Introduces contradictory findings between related features to simulate
    conflicting clinical information or inconsistent documentation.

    Implementation follows manuscript Table 1:
    - Operator: Conflict
    - Uncertainty type: Internal inconsistency
    - Validation: Must not violate hard clinical constraints
    """

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply conflict perturbation.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (conflicted_features, uncertainty_profile) where:
            - conflicted_features: Features with introduced conflicts
            - uncertainty_profile: {"conflict": fraction_of_conflicts_introduced}
        """
        perturbed = copy.deepcopy(features)

        if not self.config.conflict_pairs:
            return perturbed, {"conflict": 0.0}

        # Identify applicable conflict pairs
        applicable_pairs = {
            k: v for k, v in self.config.conflict_pairs.items() if k in features
        }

        if not applicable_pairs:
            return perturbed, {"conflict": 0.0}

        conflicts_introduced = 0
        for feature_key, conflict_map in applicable_pairs.items():
            current_value = features[feature_key]

            # Check if current value has a defined conflict
            if current_value in conflict_map:
                # Introduce conflict with 50% probability
                if self.rng.random() < 0.5:
                    perturbed[feature_key] = conflict_map[current_value]
                    conflicts_introduced += 1

        conflict_rate = (
            conflicts_introduced / len(applicable_pairs) if applicable_pairs else 0.0
        )

        return perturbed, {"conflict": conflict_rate}


class DegradeOperator(PerturbationOperator):
    """Reduced specificity operator.

    Replaces specific clinical terms with vague descriptors to simulate
    imprecise documentation or non-expert symptom reporting.

    Implementation follows manuscript Table 1:
    - Operator: Degrade
    - Uncertainty type: Reduced specificity
    - Validation: Semantic coherence preserved
    """

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply degradation perturbation.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (degraded_features, uncertainty_profile) where:
            - degraded_features: Features with reduced specificity
            - uncertainty_profile: {"degradation": fraction_degraded}
        """
        perturbed = copy.deepcopy(features)

        if not self.config.degrade_map:
            return perturbed, {"degradation": 0.0}

        degraded_count = 0
        applicable_count = 0

        for key, value in features.items():
            if key in self.config.protected_features:
                continue

            # Check if value matches a degradation mapping
            if isinstance(value, str) and value in self.config.degrade_map:
                applicable_count += 1
                # Apply degradation with 50% probability
                if self.rng.random() < 0.5:
                    perturbed[key] = self.config.degrade_map[value]
                    degraded_count += 1

        degradation_rate = (
            degraded_count / applicable_count if applicable_count > 0 else 0.0
        )

        return perturbed, {"degradation": degradation_rate}


class CompositePerturbation(PerturbationOperator):
    """Composite perturbation applying multiple operators in sequence.

    This operator applies a combination of perturbations to simulate
    realistic multi-dimensional uncertainty as described in the manuscript.
    """

    def __init__(
        self,
        operators: List[PerturbationOperator],
        config: Optional[PerturbationConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize composite perturbation.

        Args:
            operators: List of perturbation operators to apply in sequence
            config: Shared configuration (optional)
            seed: Random seed for deterministic behavior
        """
        super().__init__(config, seed)
        self.operators = operators

    def apply(
        self, features: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply all operators in sequence.

        Args:
            features: Input feature dictionary

        Returns:
            Tuple of (perturbed_features, combined_uncertainty_profile)
        """
        current_features = features
        combined_uncertainty = {}

        for operator in self.operators:
            current_features, uncertainty = operator.apply(current_features)
            combined_uncertainty.update(uncertainty)

        return current_features, combined_uncertainty


def create_default_perturbation(
    perturbation_type: str,
    config: Optional[PerturbationConfig] = None,
    seed: Optional[int] = None,
) -> PerturbationOperator:
    """Factory function to create perturbation operators.

    Args:
        perturbation_type: Type of perturbation ("mask", "noise", "conflict", "degrade", "composite")
        config: Configuration for perturbation
        seed: Random seed

    Returns:
        Initialized perturbation operator

    Raises:
        ValueError: If perturbation_type is not recognized
    """
    operator_map = {
        "mask": MaskOperator,
        "noise": NoiseOperator,
        "conflict": ConflictOperator,
        "degrade": DegradeOperator,
    }

    if perturbation_type == "composite":
        # Create all basic operators
        operators = [
            MaskOperator(config, seed),
            NoiseOperator(config, seed + 1 if seed else None),
            ConflictOperator(config, seed + 2 if seed else None),
        ]
        return CompositePerturbation(operators, config, seed)

    if perturbation_type not in operator_map:
        raise ValueError(
            f"Unknown perturbation type '{perturbation_type}'. "
            f"Must be one of: {list(operator_map.keys())} or 'composite'"
        )

    return operator_map[perturbation_type](config, seed)
