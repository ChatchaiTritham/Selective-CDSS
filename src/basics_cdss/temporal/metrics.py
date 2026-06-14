"""Temporal-specific metrics for CDSS evaluation.

This module provides metrics that assess CDSS performance over time,
complementing the static metrics in basics_cdss.metrics.

Key metrics:
- Temporal consistency: Does CDSS maintain coherent recommendations?
- Delayed intervention risk: Cost of waiting vs acting immediately
- Counterfactual regret: How much better could outcomes have been?
- Trajectory calibration: Does confidence match temporal outcome accuracy?
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from basics_cdss.temporal.digital_twin import PatientState


def temporal_consistency_score(
    predictions: List[Dict[str, Any]],
    window_size: int = 3,
    intervention_keys: Optional[List[str]] = None,
) -> float:
    """Measure consistency of CDSS recommendations over time.

    A temporally consistent CDSS should not drastically change recommendations
    for small changes in patient state (avoid "flip-flopping").

    Args:
        predictions: List of CDSS predictions at each time point
            Each prediction is dict of interventions
        window_size: Size of sliding window for consistency check
        intervention_keys: Keys to check (if None, check all)

    Returns:
        Consistency score in [0, 1] (1 = perfectly consistent)

    Example:
        >>> predictions = [
        ...     {'antibiotic': True, 'fluid': 1000},
        ...     {'antibiotic': True, 'fluid': 1000},  # Consistent
        ...     {'antibiotic': False, 'fluid': 0},    # Inconsistent change
        ...     {'antibiotic': True, 'fluid': 500},   # Another change
        ... ]
        >>> score = temporal_consistency_score(predictions, window_size=2)
        >>> print(f"Consistency: {score:.2f}")
    """
    if len(predictions) < 2:
        return 1.0

    # Determine intervention keys
    if intervention_keys is None:
        intervention_keys = list(predictions[0].keys())

    # Count changes within windows
    n_windows = len(predictions) - window_size + 1
    changes = 0
    total_comparisons = 0

    for i in range(n_windows):
        window = predictions[i : i + window_size]

        # Count intervention changes within window
        for key in intervention_keys:
            values = [pred.get(key) for pred in window]

            # Count transitions
            for j in range(len(values) - 1):
                if values[j] != values[j + 1]:
                    changes += 1
                total_comparisons += 1

    if total_comparisons == 0:
        return 1.0

    # Consistency = 1 - (change rate)
    consistency = 1.0 - (changes / total_comparisons)
    return np.clip(consistency, 0.0, 1.0)


def delayed_intervention_risk(
    trajectory_immediate: List[PatientState],
    trajectory_delayed: List[PatientState],
    harm_function: Optional[callable] = None,
    delay_hours: float = 6.0,
) -> Dict[str, float]:
    """Quantify risk of delaying intervention.

    Compares patient outcomes when intervening immediately vs waiting.

    Args:
        trajectory_immediate: Trajectory with immediate intervention
        trajectory_delayed: Trajectory with delayed intervention
        harm_function: Function to compute harm from patient state
        delay_hours: How long intervention was delayed

    Returns:
        Dictionary with delay risk metrics

    Example:
        >>> # Simulate immediate antibiotic
        >>> twin1 = patient_twin.clone()
        >>> traj_immediate = twin1.simulate(24, intervention_schedule={0: {'antibiotic': True}})
        >>>
        >>> # Simulate 6-hour delayed antibiotic
        >>> twin2 = patient_twin.clone()
        >>> traj_delayed = twin2.simulate(24, intervention_schedule={6: {'antibiotic': True}})
        >>>
        >>> # Assess delay risk
        >>> risk = delayed_intervention_risk(traj_immediate, traj_delayed, delay_hours=6)
        >>> print(f"Additional harm from delay: {risk['harm_increase']:.2f}")
    """
    if harm_function is None:
        # Default harm: infection severity
        harm_function = lambda state: state.features.get('_infection_severity', 0.0)

    # Compute cumulative harm over trajectory
    harm_immediate = sum(harm_function(state) for state in trajectory_immediate)
    harm_delayed = sum(harm_function(state) for state in trajectory_delayed)

    # Peak harm during trajectory
    peak_harm_immediate = max(harm_function(state) for state in trajectory_immediate)
    peak_harm_delayed = max(harm_function(state) for state in trajectory_delayed)

    return {
        'cumulative_harm_immediate': harm_immediate,
        'cumulative_harm_delayed': harm_delayed,
        'harm_increase': harm_delayed - harm_immediate,
        'harm_increase_percent': (
            (harm_delayed - harm_immediate) / harm_immediate * 100
            if harm_immediate > 0
            else 0.0
        ),
        'peak_harm_immediate': peak_harm_immediate,
        'peak_harm_delayed': peak_harm_delayed,
        'delay_hours': delay_hours,
        'harm_per_hour_delay': (
            (harm_delayed - harm_immediate) / delay_hours if delay_hours > 0 else 0.0
        ),
    }


def counterfactual_regret(
    factual_trajectory: List[PatientState],
    counterfactual_trajectories: List[List[PatientState]],
    harm_function: Optional[callable] = None,
) -> Dict[str, float]:
    """Compute regret from counterfactual analysis.

    Regret = difference between actual outcome and best possible outcome.

    Args:
        factual_trajectory: What actually happened
        counterfactual_trajectories: What could have happened under alternatives
        harm_function: Function to compute harm

    Returns:
        Dictionary with regret metrics

    Example:
        >>> # Factual: CDSS recommended conservative treatment
        >>> factual = twin.simulate(24, intervention_schedule={0: cdss_action})
        >>>
        >>> # Counterfactuals: other possible treatments
        >>> cf1 = twin.clone().simulate(24, intervention_schedule={0: {'antibiotic': True}})
        >>> cf2 = twin.clone().simulate(24, intervention_schedule={0: {'fluid': 1000}})
        >>>
        >>> regret = counterfactual_regret(factual, [cf1, cf2])
        >>> if regret['regret'] < 0:
        ...     print("CDSS was suboptimal - could have achieved better outcome")
    """
    if harm_function is None:
        harm_function = lambda state: state.features.get('_infection_severity', 0.0)

    # Compute harm for factual
    factual_harm = sum(harm_function(state) for state in factual_trajectory)

    # Compute harm for each counterfactual
    cf_harms = []
    for cf_traj in counterfactual_trajectories:
        cf_harm = sum(harm_function(state) for state in cf_traj)
        cf_harms.append(cf_harm)

    # Best alternative (lowest harm)
    best_cf_harm = min(cf_harms) if cf_harms else factual_harm

    # Regret = best_alternative - factual (negative = factual was suboptimal)
    regret = best_cf_harm - factual_harm

    return {
        'factual_harm': factual_harm,
        'best_alternative_harm': best_cf_harm,
        'worst_alternative_harm': max(cf_harms) if cf_harms else factual_harm,
        'mean_alternative_harm': np.mean(cf_harms) if cf_harms else factual_harm,
        'regret': regret,
        'regret_percent': (regret / factual_harm * 100) if factual_harm > 0 else 0.0,
        'n_alternatives': len(cf_harms),
        'factual_rank': sorted(cf_harms + [factual_harm]).index(factual_harm) + 1,
    }


def trajectory_calibration_error(
    trajectories: List[List[PatientState]],
    predictions: List[List[Dict[str, Any]]],
    outcomes: List[int],
    confidence_key: str = 'confidence',
    n_bins: int = 10,
) -> float:
    """Compute calibration error for temporal predictions.

    Extends Expected Calibration Error (ECE) to temporal domain by
    considering entire trajectories.

    Args:
        trajectories: List of patient trajectories
        predictions: CDSS predictions at each time point for each trajectory
        outcomes: Binary outcomes (1 = adverse event, 0 = good outcome)
        confidence_key: Key in prediction dict containing confidence score
        n_bins: Number of calibration bins

    Returns:
        Temporal calibration error (0 = perfect calibration)

    Example:
        >>> # Collect predictions and outcomes over multiple patients
        >>> trajectories = [twin.simulate(24) for twin in digital_twins]
        >>> predictions = [[cdss.predict_with_confidence(state) for state in traj]
        ...               for traj in trajectories]
        >>> outcomes = [1 if traj[-1]['_infection_severity'] > 0.7 else 0
        ...            for traj in trajectories]
        >>>
        >>> tce = trajectory_calibration_error(trajectories, predictions, outcomes)
        >>> print(f"Temporal Calibration Error: {tce:.4f}")
    """
    # Extract confidence scores (use maximum confidence from trajectory)
    confidences = []
    for pred_traj in predictions:
        max_conf = max(pred.get(confidence_key, 0.5) for pred in pred_traj)
        confidences.append(max_conf)

    confidences = np.array(confidences)
    outcomes = np.array(outcomes)

    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute ECE
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue

        bin_confidences = confidences[mask]
        bin_outcomes = outcomes[mask]

        avg_confidence = np.mean(bin_confidences)
        avg_accuracy = np.mean(bin_outcomes)
        bin_weight = np.sum(mask) / len(confidences)

        ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return ece


def temporal_harm_trajectory(
    trajectory: List[PatientState], harm_function: Optional[callable] = None
) -> np.ndarray:
    """Compute harm at each time point in trajectory.

    Args:
        trajectory: Patient trajectory
        harm_function: Function to compute harm from state

    Returns:
        Array of harm values over time

    Example:
        >>> trajectory = twin.simulate(24, dt=1.0)
        >>> harm_over_time = temporal_harm_trajectory(trajectory)
        >>>
        >>> # Plot harm trajectory
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(harm_over_time)
        >>> plt.xlabel('Time (hours)')
        >>> plt.ylabel('Harm Score')
        >>> plt.title('Patient Deterioration Over Time')
    """
    if harm_function is None:
        harm_function = lambda state: state.features.get('_infection_severity', 0.0)

    harm_values = np.array([harm_function(state) for state in trajectory])

    return harm_values


def intervention_timing_analysis(
    trajectories_by_timing: Dict[float, List[List[PatientState]]],
    harm_function: Optional[callable] = None,
) -> pd.DataFrame:
    """Analyze impact of intervention timing on outcomes.

    Args:
        trajectories_by_timing: Dict mapping intervention time (hours) to
            list of trajectories with intervention at that time
        harm_function: Function to compute harm

    Returns:
        DataFrame with timing analysis results

    Example:
        >>> # Simulate interventions at different times
        >>> timings = {
        ...     0: [twin.clone().simulate(24, {0: intervention}) for twin in twins],
        ...     3: [twin.clone().simulate(24, {3: intervention}) for twin in twins],
        ...     6: [twin.clone().simulate(24, {6: intervention}) for twin in twins],
        ... }
        >>>
        >>> analysis = intervention_timing_analysis(timings)
        >>> print(analysis)
        #   intervention_time  mean_harm  std_harm  ...
        # 0               0.0       2.45      0.82
        # 1               3.0       3.12      0.95
        # 2               6.0       4.38      1.21
    """
    if harm_function is None:
        harm_function = lambda state: state.features.get('_infection_severity', 0.0)

    results = []

    for timing, trajectories in trajectories_by_timing.items():
        # Compute final harm for each trajectory
        final_harms = [harm_function(traj[-1]) for traj in trajectories]

        # Compute cumulative harm
        cumulative_harms = [
            sum(harm_function(state) for state in traj) for traj in trajectories
        ]

        results.append(
            {
                'intervention_time': timing,
                'mean_final_harm': np.mean(final_harms),
                'std_final_harm': np.std(final_harms),
                'mean_cumulative_harm': np.mean(cumulative_harms),
                'std_cumulative_harm': np.std(cumulative_harms),
                'n_trajectories': len(trajectories),
            }
        )

    return pd.DataFrame(results).sort_values('intervention_time')
