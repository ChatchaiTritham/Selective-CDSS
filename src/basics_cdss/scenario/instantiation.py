"""
Scenario instantiation from archetypes with controlled uncertainty perturbations.

This module implements Algorithm 2 from the manuscript:
Archetype-to-scenario instantiation with controlled uncertainty.

Each archetype is expanded into multiple scenarios through perturbation operators
that simulate realistic sources of uncertainty while preserving clinical plausibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .perturbations import (PerturbationConfig, PerturbationOperator,
                            create_default_perturbation)


@dataclass(frozen=True)
class Scenario:
    """A concrete clinical scenario instantiated from an archetype.

    Attributes:
        archetype_id: Identifier of the originating archetype
        seed: Random seed used for this scenario's generation (for reproducibility)
        features: Clinical feature dictionary (potentially perturbed)
        targets: Ground truth targets (triage_tier, action, etc.)
        uncertainty_profile: Quantitative uncertainty measures (missingness, ambiguity, conflict, etc.)
        perturbation_log: Record of perturbations applied (for audit trail)
    """

    archetype_id: str
    seed: int
    features: Dict[str, Any]
    targets: Dict[str, Any]
    uncertainty_profile: Dict[str, float]
    perturbation_log: Optional[Dict[str, Any]] = None


def instantiate_scenarios(
    archetypes: pd.DataFrame,
    n_per_archetype: int = 10,
    seed: int = 42,
    perturbation_type: Optional[str] = None,
    perturbation_config: Optional[PerturbationConfig] = None,
    apply_perturbation: bool = True,
) -> List[Scenario]:
    """Deterministic scenario instantiation from archetypes with controlled uncertainty.

    Implements Algorithm 2 from the manuscript:
    1. For each archetype, generate n_per_archetype scenarios
    2. Apply perturbation operators (mask/noise/conflict/degrade) if enabled
    3. Compute uncertainty profiles
    4. Verify clinical plausibility
    5. Return scenario set with audit metadata

    Args:
        archetypes: DataFrame of clinical archetypes
        n_per_archetype: Number of scenarios to generate per archetype
        seed: Master random seed for reproducibility
        perturbation_type: Type of perturbation to apply
            ("mask", "noise", "conflict", "degrade", "composite", or None)
        perturbation_config: Configuration for perturbation operators
        apply_perturbation: If False, create scenarios without perturbation (baseline)

    Returns:
        List of Scenario objects with features, targets, and uncertainty profiles

    Example:
        >>> archetypes = load_archetypes_csv("data/archetypes.csv")
        >>> scenarios = instantiate_scenarios(
        ...     archetypes,
        ...     n_per_archetype=5,
        ...     seed=42,
        ...     perturbation_type="composite"
        ... )
        >>> len(scenarios)  # 5 scenarios per archetype
    """
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []

    # Identify ID column
    id_col = (
        "archetype_id"
        if "archetype_id" in archetypes.columns
        else archetypes.columns[0]
    )

    # Identify target columns to exclude from features
    target_cols = {"triage_tier", "action", "urgency", "diagnosis"}

    for _, row in archetypes.iterrows():
        archetype_id = str(row[id_col])

        # Extract targets (ground truth)
        targets = {
            "triage_tier": row.get("triage_tier", None),
            "action": row.get("action", None),
        }
        # Add any additional target columns if present
        for col in target_cols:
            if col in row and col not in targets:
                targets[col] = row[col]

        # Extract baseline features (exclude targets and metadata)
        baseline_features = {
            k: v
            for k, v in row.to_dict().items()
            if k not in target_cols and k != id_col
        }

        # Generate n_per_archetype scenarios
        for i in range(n_per_archetype):
            scenario_seed = int(rng.integers(0, 2**31 - 1))

            if apply_perturbation and perturbation_type:
                # Apply perturbation operator
                operator = create_default_perturbation(
                    perturbation_type,
                    config=perturbation_config,
                    seed=scenario_seed,
                )
                perturbed_features, uncertainty_profile = operator.apply(
                    baseline_features
                )

                perturbation_log = {
                    "perturbation_type": perturbation_type,
                    "archetype_id": archetype_id,
                    "scenario_index": i,
                }
            else:
                # No perturbation: baseline scenario
                perturbed_features = baseline_features.copy()
                uncertainty_profile = {
                    "missingness": 0.0,
                    "ambiguity": 0.0,
                    "conflict": 0.0,
                    "degradation": 0.0,
                }
                perturbation_log = {"perturbation_type": "none"}

            # Create scenario
            scenario = Scenario(
                archetype_id=archetype_id,
                seed=scenario_seed,
                features=perturbed_features,
                targets=targets,
                uncertainty_profile=uncertainty_profile,
                perturbation_log=perturbation_log,
            )
            scenarios.append(scenario)

    return scenarios


def instantiate_stratified_scenarios(
    archetypes: pd.DataFrame,
    n_per_archetype: int = 10,
    seed: int = 42,
    perturbation_config: Optional[PerturbationConfig] = None,
) -> Dict[str, List[Scenario]]:
    """Generate scenarios stratified by perturbation type.

    Creates multiple scenario sets, one per perturbation type, for systematic comparison.
    Useful for ablation studies and perturbation sensitivity analysis.

    Args:
        archetypes: DataFrame of clinical archetypes
        n_per_archetype: Number of scenarios per archetype per perturbation type
        seed: Master random seed
        perturbation_config: Shared configuration for all perturbation operators

    Returns:
        Dictionary mapping perturbation_type -> List[Scenario]
        Keys: "baseline", "mask", "noise", "conflict", "degrade", "composite"

    Example:
        >>> stratified = instantiate_stratified_scenarios(archetypes, n_per_archetype=5)
        >>> len(stratified["baseline"])  # Baseline scenarios (no perturbation)
        >>> len(stratified["composite"])  # Composite perturbation scenarios
    """
    perturbation_types = ["mask", "noise", "conflict", "degrade", "composite"]
    stratified_scenarios = {}

    # Baseline (no perturbation)
    stratified_scenarios["baseline"] = instantiate_scenarios(
        archetypes,
        n_per_archetype=n_per_archetype,
        seed=seed,
        perturbation_type=None,
        perturbation_config=perturbation_config,
        apply_perturbation=False,
    )

    # Each perturbation type
    for i, ptype in enumerate(perturbation_types):
        stratified_scenarios[ptype] = instantiate_scenarios(
            archetypes,
            n_per_archetype=n_per_archetype,
            seed=seed + i + 1,  # Different seed per perturbation type
            perturbation_type=ptype,
            perturbation_config=perturbation_config,
            apply_perturbation=True,
        )

    return stratified_scenarios
