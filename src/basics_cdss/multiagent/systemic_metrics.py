"""System-level metrics for multi-agent CDSS evaluation.

This module provides metrics for evaluating emergent system-level effects
of CDSS deployment, including alert fatigue, workflow disruption, and
coordination efficiency.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_alert_fatigue(
    results: Dict[str, Any], lookback_window_hours: float = 4.0
) -> Dict[str, float]:
    """Compute alert fatigue metrics.

    Alert fatigue occurs when clinicians are overwhelmed by excessive
    alerts, leading to decreased response rates and alert overrides.

    Args:
        results: Simulation results dictionary
        lookback_window_hours: Time window for computing fatigue

    Returns:
        Dictionary with alert fatigue metrics:
            - total_alerts: Total number of alerts
            - alert_rate: Alerts per hour
            - override_rate: Proportion of alerts overridden
            - response_time_mean: Mean time to respond to alerts (minutes)
            - response_time_trend: Trend in response times (slope)
            - fatigue_score: Composite fatigue score (0-1, higher = more fatigue)

    Example:
        >>> from basics_cdss.multiagent import compute_alert_fatigue
        >>>
        >>> results = hospital.simulate(duration_hours=24)
        >>> fatigue = compute_alert_fatigue(results)
        >>> print(f"Alert fatigue score: {fatigue['fatigue_score']:.2f}")
    """
    alerts = results.get('alerts', [])

    if not alerts:
        return {
            'total_alerts': 0,
            'alert_rate': 0.0,
            'override_rate': 0.0,
            'response_time_mean': 0.0,
            'response_time_trend': 0.0,
            'fatigue_score': 0.0,
        }

    # Total alerts
    total_alerts = len(alerts)

    # Alert rate
    duration = results.get('duration_hours', 1.0)
    alert_rate = total_alerts / duration

    # Override rate (alerts acknowledged but not followed)
    overridden = sum(
        1
        for alert in alerts
        if alert.get('acknowledged', False) and not alert.get('followed', False)
    )
    override_rate = overridden / total_alerts if total_alerts > 0 else 0.0

    # Response times
    response_times = []
    for alert in alerts:
        if 'response_time' in alert:
            response_times.append(alert['response_time'])

    if response_times:
        response_time_mean = np.mean(response_times)

        # Compute trend (are response times increasing over time?)
        timestamps = [
            alert.get('timestamp', 0) for alert in alerts if 'response_time' in alert
        ]
        if len(timestamps) > 1:
            # Linear regression: response_time ~ timestamp
            slope, _, _, _, _ = stats.linregress(timestamps, response_times)
            response_time_trend = slope
        else:
            response_time_trend = 0.0
    else:
        response_time_mean = 0.0
        response_time_trend = 0.0

    # Compute composite fatigue score
    # Factors:
    # 1. High alert rate (> 2 per hour)
    # 2. High override rate (> 50%)
    # 3. Increasing response times (positive trend)

    fatigue_components = []

    # Alert rate component (normalized to 0-1)
    alert_rate_component = min(alert_rate / 5.0, 1.0)  # 5+ per hour = max fatigue
    fatigue_components.append(alert_rate_component)

    # Override rate component
    fatigue_components.append(override_rate)

    # Response time trend component (positive trend = fatigue)
    if response_time_trend > 0:
        trend_component = min(response_time_trend / 10.0, 1.0)  # Normalize
        fatigue_components.append(trend_component)

    fatigue_score = np.mean(fatigue_components)

    return {
        'total_alerts': total_alerts,
        'alert_rate': alert_rate,
        'override_rate': override_rate,
        'response_time_mean': response_time_mean,
        'response_time_trend': response_time_trend,
        'fatigue_score': fatigue_score,
    }


def compute_override_rate(
    results: Dict[str, Any], by_clinician: bool = False
) -> Dict[str, Any]:
    """Compute alert override rates.

    Override rate measures how often clinicians ignore or override
    CDSS recommendations.

    Args:
        results: Simulation results
        by_clinician: If True, compute per-clinician rates

    Returns:
        Dictionary with override metrics

    Example:
        >>> override_metrics = compute_override_rate(results, by_clinician=True)
        >>> print(f"Overall override rate: {override_metrics['overall']:.2%}")
    """
    alerts = results.get('alerts', [])

    if not alerts:
        return {'overall': 0.0, 'by_clinician': {}, 'by_alert_type': {}}

    # Overall override rate
    total_alerts = len(alerts)
    overridden = sum(
        1
        for alert in alerts
        if alert.get('acknowledged', False) and not alert.get('followed', False)
    )
    overall_override_rate = overridden / total_alerts

    # By clinician
    clinician_overrides = {}
    if by_clinician:
        for alert in alerts:
            clinician_id = alert.get('to', 'unknown')
            if clinician_id not in clinician_overrides:
                clinician_overrides[clinician_id] = {'total': 0, 'overridden': 0}

            clinician_overrides[clinician_id]['total'] += 1
            if alert.get('acknowledged', False) and not alert.get('followed', False):
                clinician_overrides[clinician_id]['overridden'] += 1

        # Compute rates
        for clinician_id in clinician_overrides:
            total = clinician_overrides[clinician_id]['total']
            overridden = clinician_overrides[clinician_id]['overridden']
            clinician_overrides[clinician_id]['rate'] = (
                overridden / total if total > 0 else 0.0
            )

    # By alert type
    alert_type_overrides = {}
    for alert in alerts:
        alert_type = alert.get('alert_type', 'unknown')
        if alert_type not in alert_type_overrides:
            alert_type_overrides[alert_type] = {'total': 0, 'overridden': 0}

        alert_type_overrides[alert_type]['total'] += 1
        if alert.get('acknowledged', False) and not alert.get('followed', False):
            alert_type_overrides[alert_type]['overridden'] += 1

    # Compute rates by type
    for alert_type in alert_type_overrides:
        total = alert_type_overrides[alert_type]['total']
        overridden = alert_type_overrides[alert_type]['overridden']
        alert_type_overrides[alert_type]['rate'] = (
            overridden / total if total > 0 else 0.0
        )

    return {
        'overall': overall_override_rate,
        'by_clinician': clinician_overrides,
        'by_alert_type': alert_type_overrides,
        'total_alerts': total_alerts,
        'total_overridden': overridden,
    }


def compute_workflow_disruption(
    results: Dict[str, Any], baseline_task_times: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Compute workflow disruption metrics.

    Workflow disruption measures how much CDSS alerts interrupt
    normal clinical workflows, causing delays and task switching.

    Args:
        results: Simulation results
        baseline_task_times: Expected task completion times without CDSS

    Returns:
        Dictionary with disruption metrics:
            - task_switching_rate: Number of task switches per hour
            - task_completion_delay: Average delay in task completion (%)
            - concurrent_alerts_rate: Rate of multiple simultaneous alerts
            - disruption_score: Composite disruption score (0-1)

    Example:
        >>> disruption = compute_workflow_disruption(results)
        >>> print(f"Disruption score: {disruption['disruption_score']:.2f}")
    """
    event_log = results.get('event_log', [])
    alerts = results.get('alerts', [])
    duration = results.get('duration_hours', 1.0)

    if not event_log:
        return {
            'task_switching_rate': 0.0,
            'task_completion_delay': 0.0,
            'concurrent_alerts_rate': 0.0,
            'disruption_score': 0.0,
        }

    # Task switching rate
    # Count transitions between different task types
    task_switches = 0
    prev_task = None

    for event in event_log:
        if event.get('type') == 'agent_action':
            current_task = event.get('action_type')
            if prev_task and current_task != prev_task:
                task_switches += 1
            prev_task = current_task

    task_switching_rate = task_switches / duration

    # Concurrent alerts
    # Find time windows with multiple alerts
    alert_times = [alert.get('timestamp', 0) for alert in alerts]
    concurrent_alerts = 0

    time_window = 0.5  # 30-minute window
    for t in alert_times:
        # Count alerts within window
        alerts_in_window = sum(1 for t2 in alert_times if abs(t2 - t) <= time_window)
        if alerts_in_window > 1:
            concurrent_alerts += 1

    concurrent_alerts_rate = concurrent_alerts / duration if duration > 0 else 0.0

    # Task completion delay
    # Compare actual vs baseline completion times
    task_completion_delay = 0.0
    if baseline_task_times:
        actual_times = {}
        for event in event_log:
            if (
                event.get('type') == 'agent_action'
                and event.get('action_type') == 'monitor'
            ):
                task_type = event.get('action_type')
                if task_type not in actual_times:
                    actual_times[task_type] = []
                actual_times[task_type].append(event.get('timestamp', 0))

        # Compute delays
        delays = []
        for task_type, times in actual_times.items():
            if task_type in baseline_task_times:
                baseline = baseline_task_times[task_type]
                actual_mean = np.mean(np.diff(times)) if len(times) > 1 else 0
                if baseline > 0:
                    delay_pct = (actual_mean - baseline) / baseline * 100
                    delays.append(max(delay_pct, 0))  # Only positive delays

        task_completion_delay = np.mean(delays) if delays else 0.0

    # Composite disruption score
    disruption_components = [
        min(task_switching_rate / 10.0, 1.0),  # Normalize
        min(concurrent_alerts_rate / 2.0, 1.0),
        min(task_completion_delay / 50.0, 1.0),  # 50% delay = max disruption
    ]

    disruption_score = np.mean(disruption_components)

    return {
        'task_switching_rate': task_switching_rate,
        'task_completion_delay': task_completion_delay,
        'concurrent_alerts_rate': concurrent_alerts_rate,
        'disruption_score': disruption_score,
    }


def compute_time_to_action(
    results: Dict[str, Any], action_type: str = 'intervention'
) -> Dict[str, float]:
    """Compute time from alert to clinical action.

    Args:
        results: Simulation results
        action_type: Type of action to measure

    Returns:
        Dictionary with time-to-action metrics:
            - mean: Mean time to action (hours)
            - median: Median time to action
            - std: Standard deviation
            - p90: 90th percentile
            - compliance_rate: Proportion acted upon within threshold

    Example:
        >>> time_to_action = compute_time_to_action(results, action_type='intervention')
        >>> print(f"Mean time to action: {time_to_action['mean']:.1f} hours")
    """
    event_log = results.get('event_log', [])
    alerts = results.get('alerts', [])

    if not alerts:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'p90': 0.0,
            'compliance_rate': 0.0,
        }

    # Match alerts to actions
    times_to_action = []

    for alert in alerts:
        alert_time = alert.get('timestamp', 0)
        patient_id = alert.get('patient_id')

        # Find next action for this patient
        for event in event_log:
            if (
                event.get('type') == 'agent_action'
                and event.get('action_type') == action_type
                and event.get('target') == patient_id
                and event.get('timestamp', 0) > alert_time
            ):

                time_to_action = event.get('timestamp', 0) - alert_time
                times_to_action.append(time_to_action)
                break

    if not times_to_action:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'p90': 0.0,
            'compliance_rate': 0.0,
            'n_samples': 0,
        }

    # Compute metrics
    mean_time = np.mean(times_to_action)
    median_time = np.median(times_to_action)
    std_time = np.std(times_to_action)
    p90_time = np.percentile(times_to_action, 90)

    # Compliance rate (within 1 hour threshold)
    threshold = 1.0  # 1 hour
    compliant = sum(1 for t in times_to_action if t <= threshold)
    compliance_rate = compliant / len(times_to_action)

    return {
        'mean': mean_time,
        'median': median_time,
        'std': std_time,
        'p90': p90_time,
        'compliance_rate': compliance_rate,
        'n_samples': len(times_to_action),
    }


def compute_coordination_efficiency(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute efficiency of coordination between agents.

    Measures how well agents coordinate actions, avoiding duplication
    and ensuring timely handoffs.

    Args:
        results: Simulation results

    Returns:
        Dictionary with coordination metrics:
            - handoff_completeness: Proportion of successful handoffs
            - task_duplication_rate: Rate of duplicated tasks
            - communication_overhead: Messages per action
            - coordination_score: Composite coordination score (0-1)

    Example:
        >>> coordination = compute_coordination_efficiency(results)
        >>> print(f"Coordination score: {coordination['coordination_score']:.2f}")
    """
    event_log = results.get('event_log', [])
    alerts = results.get('alerts', [])

    if not event_log:
        return {
            'handoff_completeness': 0.0,
            'task_duplication_rate': 0.0,
            'communication_overhead': 0.0,
            'coordination_score': 0.0,
        }

    # Count actions by type
    actions = [e for e in event_log if e.get('type') == 'agent_action']
    n_actions = len(actions)

    # Task duplication
    # Count how many times same task performed for same patient
    task_signatures = {}
    duplicates = 0

    for action in actions:
        signature = (
            action.get('action_type'),
            action.get('target'),
            int(action.get('timestamp', 0)),  # Round to hour
        )

        if signature in task_signatures:
            duplicates += 1
        else:
            task_signatures[signature] = True

    task_duplication_rate = duplicates / n_actions if n_actions > 0 else 0.0

    # Communication overhead
    n_messages = len(alerts)
    communication_overhead = n_messages / n_actions if n_actions > 0 else 0.0

    # Handoff completeness
    # (Simplified: just check if handoff messages exist)
    handoffs = [a for a in alerts if a.get('message_type') == 'handoff']
    handoff_completeness = 1.0 if handoffs else 0.5  # Simplified

    # Composite coordination score
    # Higher is better: low duplication, appropriate communication
    coordination_components = [
        1.0 - min(task_duplication_rate, 1.0),  # Low duplication = good
        min(communication_overhead / 2.0, 1.0),  # Some communication needed
        handoff_completeness,
    ]

    coordination_score = np.mean(coordination_components)

    return {
        'handoff_completeness': handoff_completeness,
        'task_duplication_rate': task_duplication_rate,
        'communication_overhead': communication_overhead,
        'coordination_score': coordination_score,
        'n_actions': n_actions,
        'n_messages': n_messages,
    }


def compute_system_resilience(
    results: Dict[str, Any], perturbations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """Compute system resilience to perturbations.

    Resilience measures how well the system maintains performance
    under stress (high workload, agent unavailability, etc.).

    Args:
        results: Simulation results
        perturbations: List of perturbations applied

    Returns:
        Dictionary with resilience metrics:
            - performance_degradation: Decrease in performance (%)
            - recovery_time: Time to recover from perturbation
            - robustness_score: Overall robustness (0-1)

    Example:
        >>> resilience = compute_system_resilience(
        ...     results,
        ...     perturbations=[{'type': 'agent_unavailable', 'duration': 2.0}]
        ... )
    """
    # Simplified implementation
    # In full version, would compare performance before/after perturbations

    metrics = results.get('metrics', {})
    alert_rate = metrics.get('alert_rate', 0)

    # Assume baseline alert rate of 1 per hour
    baseline_rate = 1.0
    performance_degradation = abs(alert_rate - baseline_rate) / baseline_rate * 100

    # Robustness score (higher = more robust)
    robustness_score = max(0, 1.0 - performance_degradation / 100)

    return {
        'performance_degradation': performance_degradation,
        'recovery_time': 0.0,  # Would need temporal analysis
        'robustness_score': robustness_score,
    }


def generate_systemic_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive systemic evaluation report.

    Args:
        results: Simulation results

    Returns:
        Dictionary with all systemic metrics

    Example:
        >>> report = generate_systemic_report(results)
        >>> print(f"Alert Fatigue: {report['alert_fatigue']['fatigue_score']:.2f}")
        >>> print(f"Override Rate: {report['override']['overall']:.2%}")
        >>> print(f"Workflow Disruption: {report['disruption']['disruption_score']:.2f}")
    """
    return {
        'alert_fatigue': compute_alert_fatigue(results),
        'override': compute_override_rate(results, by_clinician=True),
        'disruption': compute_workflow_disruption(results),
        'time_to_action': compute_time_to_action(results),
        'coordination': compute_coordination_efficiency(results),
        'resilience': compute_system_resilience(results),
    }
