"""BASICS-CDSS Temporal Module: Digital Twin Simulation.

This module extends BASICS-CDSS with temporal evaluation capabilities
using digital twin simulation. It enables:

1. Time-evolving patient states
2. Physiological disease progression models
3. Temporal perturbations (time-varying uncertainty)
4. Counterfactual CDSS evaluation

Key Components:
    - PatientDigitalTwin: Base class for temporal patient simulation
    - DiseaseModel: Physiological progression models (ODE/SDE-based)
    - TemporalPerturbation: Time-evolving uncertainty operators
    - CounterfactualEvaluator: What-if analysis for CDSS decisions

Example:
    >>> from basics_cdss.temporal import PatientDigitalTwin, SepsisModel
    >>>
    >>> # Create digital twin from archetype
    >>> twin = PatientDigitalTwin(
    ...     archetype_id="A001",
    ...     initial_state={'temperature': 37.5, 'heart_rate': 85},
    ...     disease_model=SepsisModel()
    ... )
    >>>
    >>> # Simulate 24-hour trajectory
    >>> trajectory = twin.simulate(horizon_hours=24, dt=1.0)
    >>>
    >>> # Evaluate CDSS with counterfactuals
    >>> from basics_cdss.temporal import CounterfactualEvaluator
    >>> evaluator = CounterfactualEvaluator()
    >>> results = evaluator.evaluate(cdss_model, [twin])
"""

from basics_cdss.temporal.counterfactual import (CounterfactualEvaluator,
                                                 CounterfactualResult)
from basics_cdss.temporal.digital_twin import (DigitalTwinFactory,
                                               PatientDigitalTwin)
from basics_cdss.temporal.disease_models import (CardiacEventModel,
                                                 DiseaseModel,
                                                 RespiratoryDistressModel,
                                                 SepsisModel)
from basics_cdss.temporal.metrics import (counterfactual_regret,
                                          delayed_intervention_risk,
                                          temporal_consistency_score,
                                          trajectory_calibration_error)
from basics_cdss.temporal.temporal_perturbations import (
    TemporalConflictOperator, TemporalMaskOperator, TemporalNoiseOperator,
    TemporalPerturbationOperator)

__all__ = [
    # Digital Twin
    "PatientDigitalTwin",
    "DigitalTwinFactory",
    # Disease Models
    "DiseaseModel",
    "SepsisModel",
    "RespiratoryDistressModel",
    "CardiacEventModel",
    # Perturbations
    "TemporalPerturbationOperator",
    "TemporalMaskOperator",
    "TemporalNoiseOperator",
    "TemporalConflictOperator",
    # Counterfactual
    "CounterfactualEvaluator",
    "CounterfactualResult",
    # Metrics
    "temporal_consistency_score",
    "delayed_intervention_risk",
    "counterfactual_regret",
    "trajectory_calibration_error",
]
