"""
Temporal Visualization Module for BASICS-CDSS Tier 1 (Digital Twin)

Publication-quality plots for temporal trajectory analysis, disease progression,
and counterfactual evaluation.

Compliant with:
- IEEE publication standards
- Nature/JAMA figure requirements
- Journal of Biomedical Informatics guidelines
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats

# Publication-quality styling
STYLE_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
}

# Colorblind-friendly palette
COLORS = {
    'critical': '#CC3311',  # Red
    'high_risk': '#EE7733',  # Orange
    'moderate': '#EE9955',  # Light orange
    'low_risk': '#0077BB',  # Blue
    'safe': '#33BB55',  # Green
    'intervention': '#9933CC',  # Purple
    'baseline': '#666666',  # Gray
}


def plot_temporal_trajectory(
    time_points: np.ndarray,
    vitals: Dict[str, np.ndarray],
    interventions: Optional[List[Tuple[float, str]]] = None,
    risk_tiers: Optional[np.ndarray] = None,
    title: str = "Patient Temporal Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 11.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot temporal trajectory of vital signs with interventions and risk tiers.

    Args:
        time_points: Time points (hours)
        vitals: Dictionary of vital sign trajectories {name: values}
        interventions: List of (time, intervention_name) tuples
        risk_tiers: Array of risk tier labels over time (1-5)
        title: Plot title
        save_path: Path to save figure (PDF/EPS/PNG)
        figsize: Figure size in inches

    Returns:
        Figure and axes array

    Example:
        >>> time = np.linspace(0, 24, 100)
        >>> vitals = {
        ...     'heart_rate': 120 + 20*np.sin(time/4),
        ...     'systolic_bp': 80 + 10*np.random.randn(100),
        ...     'spo2': 88 + 3*np.random.randn(100)
        ... }
        >>> fig, axes = plot_temporal_trajectory(time, vitals)
    """
    plt.rcParams.update(STYLE_CONFIG)

    n_vitals = len(vitals)
    fig, axes = plt.subplots(n_vitals + 1, 1, figsize=figsize)

    if n_vitals == 0:
        axes = [axes]

    # Plot vital signs
    for idx, (vital_name, values) in enumerate(vitals.items()):
        ax = axes[idx]

        # Plot trajectory
        ax.plot(
            time_points,
            values,
            color=COLORS['high_risk'],
            linewidth=2.0,
            label=vital_name.replace('_', ' ').title(),
        )

        # Add reference lines for normal ranges
        if 'heart_rate' in vital_name.lower():
            ax.axhspan(60, 100, alpha=0.2, color='green', label='Normal Range')
        elif 'systolic' in vital_name.lower():
            ax.axhspan(90, 140, alpha=0.2, color='green')
        elif 'spo2' in vital_name.lower():
            ax.axhspan(95, 100, alpha=0.2, color='green')

        # Mark interventions
        if interventions:
            for interv_time, interv_name in interventions:
                ax.axvline(
                    interv_time,
                    color=COLORS['intervention'],
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax.text(
                    interv_time,
                    ax.get_ylim()[1] * 0.95,
                    interv_name,
                    rotation=90,
                    va='top',
                    fontsize=9,
                    color=COLORS['intervention'],
                )

        ax.set_ylabel(vital_name.replace('_', ' ').title(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

        if idx == 0:
            ax.set_title(title, fontweight='bold', pad=10)

    # Plot risk tier evolution
    if risk_tiers is not None:
        ax_risk = axes[-1]

        # Convert risk tiers to color map
        colors_risk = [
            COLORS['critical'],
            COLORS['high_risk'],
            COLORS['moderate'],
            COLORS['low_risk'],
            COLORS['safe'],
        ]

        for i in range(len(time_points) - 1):
            tier = int(risk_tiers[i]) - 1
            ax_risk.fill_between(
                [time_points[i], time_points[i + 1]],
                0,
                1,
                color=colors_risk[tier],
                alpha=0.7,
            )

        ax_risk.set_ylabel('Risk Tier', fontweight='bold')
        ax_risk.set_ylim(0, 1)
        ax_risk.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax_risk.set_yticklabels(['R1', 'R2', 'R3', 'R4', 'R5'])
        ax_risk.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax_risk.set_axisbelow(True)

    # Common x-axis
    axes[-1].set_xlabel('Time (hours)', fontweight='bold')

    # Adjust layout
    plt.subplots_adjust(hspace=0.35, left=0.12, right=0.95, top=0.97, bottom=0.05)

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, axes


def plot_disease_progression(
    time_points: np.ndarray,
    biomarker_trajectories: Dict[str, np.ndarray],
    disease_stages: Optional[List[Tuple[float, float, str]]] = None,
    title: str = "Disease Progression Model",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 9.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot disease progression with biomarker evolution and disease stages.

    Args:
        time_points: Time points (hours)
        biomarker_trajectories: Dictionary of biomarker trajectories
        disease_stages: List of (start_time, end_time, stage_name) tuples
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> time = np.linspace(0, 48, 200)
        >>> biomarkers = {
        ...     'Lactate': 2.0 * np.exp(time/24),
        ...     'Procalcitonin': 0.5 + time/10,
        ...     'WBC': 12 + 5*np.sin(time/6)
        ... }
        >>> stages = [(0, 12, 'Early Sepsis'), (12, 36, 'Severe Sepsis'),
        ...           (36, 48, 'Septic Shock')]
        >>> fig, axes = plot_disease_progression(time, biomarkers, stages)
    """
    plt.rcParams.update(STYLE_CONFIG)

    n_biomarkers = len(biomarker_trajectories)
    fig, axes = plt.subplots(n_biomarkers, 1, figsize=figsize, sharex=True)

    if n_biomarkers == 1:
        axes = [axes]

    # Define colors for biomarkers
    biomarker_colors = list(COLORS.values())[:n_biomarkers]

    for idx, (biomarker_name, values) in enumerate(biomarker_trajectories.items()):
        ax = axes[idx]

        # Plot biomarker trajectory
        ax.plot(
            time_points,
            values,
            color=biomarker_colors[idx],
            linewidth=2.5,
            label=biomarker_name,
        )

        # Shade disease stages
        if disease_stages and idx == 0:
            stage_colors = ['#FFCCCC', '#FFAAAA', '#FF8888']
            for stage_idx, (start, end, stage_name) in enumerate(disease_stages):
                ax.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=stage_colors[stage_idx % len(stage_colors)],
                )
                # Add stage label
                mid_time = (start + end) / 2
                ax.text(
                    mid_time,
                    ax.get_ylim()[1] * 0.95,
                    stage_name,
                    ha='center',
                    va='top',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8,
                    ),
                )

        ax.set_ylabel(biomarker_name, fontweight='bold')
        ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

        if idx == 0:
            ax.set_title(title, fontweight='bold', pad=10)

    axes[-1].set_xlabel('Time (hours)', fontweight='bold')

    plt.subplots_adjust(hspace=0.30, left=0.12, right=0.95, top=0.97, bottom=0.05)

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, axes


def plot_counterfactual_analysis(
    time_points: np.ndarray,
    factual_trajectory: np.ndarray,
    counterfactual_trajectories: Dict[str, np.ndarray],
    intervention_time: float,
    outcome_metric: str = "Survival Probability",
    title: str = "Counterfactual Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot counterfactual analysis comparing factual vs counterfactual outcomes.

    Args:
        time_points: Time points (hours)
        factual_trajectory: Observed trajectory (what actually happened)
        counterfactual_trajectories: Dict of alternative trajectories
        intervention_time: Time when intervention occurred
        outcome_metric: Name of outcome being measured
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> time = np.linspace(0, 24, 100)
        >>> factual = 0.9 - 0.3 * (time/24)
        >>> counterfactuals = {
        ...     'Early Intervention': 0.9 - 0.1 * (time/24),
        ...     'No Intervention': 0.9 - 0.5 * (time/24),
        ...     'Delayed Intervention': 0.9 - 0.4 * (time/24)
        ... }
        >>> fig, ax = plot_counterfactual_analysis(time, factual,
        ...                                        counterfactuals, 6.0)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot factual trajectory
    ax.plot(
        time_points,
        factual_trajectory,
        color=COLORS['critical'],
        linewidth=3.0,
        label='Factual (Observed)',
        linestyle='-',
        marker='o',
        markersize=3,
        markevery=10,
    )

    # Plot counterfactual trajectories
    cf_styles = ['-', '--', '-.', ':']
    cf_colors = [
        COLORS['high_risk'],
        COLORS['moderate'],
        COLORS['low_risk'],
        COLORS['safe'],
    ]

    for idx, (cf_name, cf_trajectory) in enumerate(counterfactual_trajectories.items()):
        ax.plot(
            time_points,
            cf_trajectory,
            color=cf_colors[idx % len(cf_colors)],
            linestyle=cf_styles[idx % len(cf_styles)],
            linewidth=2.5,
            label=f'CF: {cf_name}',
            marker='s',
            markersize=2,
            markevery=10,
        )

    # Mark intervention time
    ax.axvline(
        intervention_time,
        color=COLORS['intervention'],
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Intervention Time',
    )
    ax.axvspan(
        intervention_time - 0.5,
        intervention_time + 0.5,
        alpha=0.1,
        color=COLORS['intervention'],
    )

    # Add intervention label
    ax.text(
        intervention_time,
        ax.get_ylim()[1] * 0.95,
        'Intervention',
        ha='center',
        va='top',
        fontsize=11,
        fontweight='bold',
        color=COLORS['intervention'],
    )

    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel(outcome_metric, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.9, fontsize=10, ncol=1)

    # Add text box with causal effect
    factual_outcome = factual_trajectory[-1]
    best_cf_outcome = max([cf[-1] for cf in counterfactual_trajectories.values()])
    causal_effect = best_cf_outcome - factual_outcome

    textstr = f'Causal Effect\n(Best CF vs Factual):\n{causal_effect:+.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black')
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=props,
        fontweight='bold',
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_intervention_timing_analysis(
    intervention_times: np.ndarray,
    outcomes: np.ndarray,
    optimal_window: Tuple[float, float],
    title: str = "Intervention Timing Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot relationship between intervention timing and outcomes.

    Args:
        intervention_times: Array of intervention times (hours)
        outcomes: Array of outcome values (e.g., survival probability)
        optimal_window: (start, end) of optimal intervention window
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> times = np.random.uniform(0, 24, 100)
        >>> outcomes = 0.9 - 0.3 * np.abs(times - 6) / 24 + np.random.normal(0, 0.05, 100)
        >>> fig, ax = plot_intervention_timing_analysis(times, outcomes, (3, 9))
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plot
    ax.scatter(
        intervention_times,
        outcomes,
        alpha=0.6,
        s=50,
        color=COLORS['high_risk'],
        edgecolors='black',
        linewidth=0.5,
        label='Observed Cases',
    )

    # Fit smoothing curve
    from scipy.interpolate import UnivariateSpline

    sorted_idx = np.argsort(intervention_times)
    spl = UnivariateSpline(intervention_times[sorted_idx], outcomes[sorted_idx], s=0.5)
    time_smooth = np.linspace(intervention_times.min(), intervention_times.max(), 200)
    outcome_smooth = spl(time_smooth)

    ax.plot(
        time_smooth,
        outcome_smooth,
        color=COLORS['critical'],
        linewidth=3.0,
        label='Smoothed Trend',
        linestyle='-',
    )

    # Highlight optimal window
    ax.axvspan(
        optimal_window[0],
        optimal_window[1],
        alpha=0.3,
        color=COLORS['safe'],
        label='Optimal Window',
    )

    # Add vertical lines for window boundaries
    ax.axvline(
        optimal_window[0],
        color=COLORS['safe'],
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
    )
    ax.axvline(
        optimal_window[1],
        color=COLORS['safe'],
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
    )

    ax.set_xlabel('Intervention Time (hours)', fontweight='bold')
    ax.set_ylabel('Outcome (Survival Probability)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.9, fontsize=11)

    # Add statistics text
    window_mask = (intervention_times >= optimal_window[0]) & (
        intervention_times <= optimal_window[1]
    )
    window_outcomes = outcomes[window_mask]
    non_window_outcomes = outcomes[~window_mask]

    mean_in = np.mean(window_outcomes) if len(window_outcomes) > 0 else 0
    mean_out = np.mean(non_window_outcomes) if len(non_window_outcomes) > 0 else 0

    textstr = f'Mean Outcome:\nIn window: {mean_in:.3f}\nOut window: {mean_out:.3f}\nDifference: {mean_in - mean_out:+.3f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
    ax.text(
        0.98,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props,
        fontweight='bold',
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


__all__ = [
    'plot_temporal_trajectory',
    'plot_disease_progression',
    'plot_counterfactual_analysis',
    'plot_intervention_timing_analysis',
]
