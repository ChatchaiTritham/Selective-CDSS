"""
Multi-Agent Visualization Module for BASICS-CDSS Tier 3 (Multi-Agent Simulation)

Publication-quality plots for agent interactions, workflow analysis, alert fatigue,
override patterns, and system-level dynamics.

Compliant with:
- JAMIA publication standards
- npj Digital Medicine requirements
- IEEE EMBC guidelines
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, Wedge)
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
}

# Colorblind-friendly palette
COLORS = {
    'patient': '#CC3311',  # Red
    'clinician': '#0077BB',  # Blue
    'cdss': '#33BB55',  # Green
    'nurse': '#EE7733',  # Orange
    'alert': '#9933CC',  # Purple
    'override': '#BBBBBB',  # Gray
}


def plot_agent_interaction_network(
    interactions: List[Tuple[str, str, str, float]],
    agent_types: Dict[str, str],
    title: str = "Agent Interaction Network",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot agent interaction network with weighted edges.

    Args:
        interactions: List of (source, target, interaction_type, weight) tuples
        agent_types: Dictionary mapping agent names to types
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> interactions = [
        ...     ('CDSS_1', 'Clinician_A', 'alert', 15),
        ...     ('Clinician_A', 'Patient_1', 'intervention', 8),
        ...     ('Nurse_1', 'Patient_1', 'monitoring', 20)
        ... ]
        >>> agent_types = {'CDSS_1': 'cdss', 'Clinician_A': 'clinician',
        ...                'Patient_1': 'patient', 'Nurse_1': 'nurse'}
        >>> fig, ax = plot_agent_interaction_network(interactions, agent_types)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create network graph
    G = nx.DiGraph()
    for source, target, int_type, weight in interactions:
        G.add_edge(source, target, interaction_type=int_type, weight=weight)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors based on agent type
    node_colors = [
        COLORS.get(agent_types.get(node, 'patient'), '#BBBBBB') for node in G.nodes()
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=1500,
        edgecolors='black',
        linewidths=2.0,
        alpha=0.9,
    )

    # Draw edges with varying widths based on weight
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [5 * (w / max_weight) for w in weights]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color='black',
        alpha=0.6,
        arrowsize=20,
        arrowstyle='->',
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=10, font_weight='bold', font_family='serif'
    )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Patient',
            markerfacecolor=COLORS['patient'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Clinician',
            markerfacecolor=COLORS['clinician'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='CDSS',
            markerfacecolor=COLORS['cdss'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Nurse',
            markerfacecolor=COLORS['nurse'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        framealpha=0.95,
        fontsize=10,
        edgecolor='black',
    )

    ax.set_title(title, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_workflow_timeline(
    tasks: List[Dict[str, Union[str, float]]],
    title: str = "Clinical Workflow Timeline",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 8.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot clinical workflow as Gantt chart.

    Args:
        tasks: List of task dictionaries with keys:
              'name', 'start_time', 'duration', 'agent', 'status'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> tasks = [
        ...     {'name': 'Triage', 'start_time': 0, 'duration': 15,
        ...      'agent': 'Nurse', 'status': 'completed'},
        ...     {'name': 'Alert Generated', 'start_time': 10, 'duration': 2,
        ...      'agent': 'CDSS', 'status': 'completed'},
        ...     {'name': 'Assessment', 'start_time': 15, 'duration': 20,
        ...      'agent': 'Clinician', 'status': 'in_progress'}
        ... ]
        >>> fig, ax = plot_workflow_timeline(tasks)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Sort tasks by start time
    tasks_sorted = sorted(tasks, key=lambda x: x['start_time'])

    # Agent color mapping
    agent_colors = {
        'Patient': COLORS['patient'],
        'Clinician': COLORS['clinician'],
        'CDSS': COLORS['cdss'],
        'Nurse': COLORS['nurse'],
    }

    # Status patterns
    status_hatches = {
        'completed': '',
        'in_progress': '///',
        'pending': '...',
        'failed': 'xxx',
    }

    y_pos = np.arange(len(tasks_sorted))

    for i, task in enumerate(tasks_sorted):
        start = task['start_time']
        duration = task['duration']
        agent = task.get('agent', 'Unknown')
        status = task.get('status', 'completed')

        color = agent_colors.get(agent, '#BBBBBB')
        hatch = status_hatches.get(status, '')

        # Draw bar
        ax.barh(
            i,
            duration,
            left=start,
            height=0.6,
            color=color,
            edgecolor='black',
            linewidth=1.5,
            hatch=hatch,
            alpha=0.8,
        )

        # Add task name
        ax.text(
            start + duration / 2,
            i,
            task['name'],
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='white' if status == 'completed' else 'black',
        )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Task {i+1}" for i in range(len(tasks_sorted))], fontsize=10)
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=COLORS['patient'], ec='black', label='Patient'),
        Rectangle((0, 0), 1, 1, fc=COLORS['clinician'], ec='black', label='Clinician'),
        Rectangle((0, 0), 1, 1, fc=COLORS['cdss'], ec='black', label='CDSS'),
        Rectangle((0, 0), 1, 1, fc=COLORS['nurse'], ec='black', label='Nurse'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        framealpha=0.95,
        fontsize=10,
        edgecolor='black',
        ncol=2,
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_alert_fatigue_dynamics(
    time_points: np.ndarray,
    alert_counts: np.ndarray,
    override_rates: np.ndarray,
    response_times: np.ndarray,
    title: str = "Alert Fatigue Dynamics Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 9.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot alert fatigue dynamics with multiple metrics over time.

    Args:
        time_points: Time points (hours)
        alert_counts: Number of alerts per time period
        override_rates: Override rate (0-1) per time period
        response_times: Average response time (minutes) per time period
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes array

    Example:
        >>> time = np.arange(0, 24, 1)
        >>> alerts = 10 + 5*np.sin(time/4) + np.random.randint(0, 3, len(time))
        >>> overrides = 0.2 + 0.3*time/24 + np.random.normal(0, 0.05, len(time))
        >>> response = 5 + 10*time/24 + np.random.normal(0, 1, len(time))
        >>> fig, axes = plot_alert_fatigue_dynamics(time, alerts, overrides, response)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot 1: Alert counts
    ax1 = axes[0]
    ax1.bar(
        time_points,
        alert_counts,
        width=0.8,
        color=COLORS['alert'],
        edgecolor='black',
        linewidth=1.0,
        alpha=0.8,
        label='Alert Count',
    )
    ax1.set_ylabel('Alerts per Hour', fontweight='bold')
    ax1.set_title('(a) Alert Frequency', fontweight='bold', pad=10, loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Plot 2: Override rates
    ax2 = axes[1]
    ax2.plot(
        time_points,
        override_rates * 100,
        color=COLORS['override'],
        linewidth=2.5,
        marker='o',
        markersize=5,
        label='Override Rate',
    )
    ax2.axhline(
        y=50,
        color='red',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Critical Threshold (50%)',
    )
    ax2.fill_between(
        time_points,
        0,
        override_rates * 100,
        where=(override_rates * 100 > 50),
        alpha=0.3,
        color='red',
        label='Fatigue Zone',
    )
    ax2.set_ylabel('Override Rate (%)', fontweight='bold')
    ax2.set_title('(b) Alert Override Pattern', fontweight='bold', pad=10, loc='left')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)

    # Plot 3: Response times
    ax3 = axes[2]
    ax3.plot(
        time_points,
        response_times,
        color=COLORS['clinician'],
        linewidth=2.5,
        marker='s',
        markersize=5,
        label='Response Time',
    )
    ax3.axhline(
        y=15,
        color='orange',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Target (15 min)',
    )
    ax3.fill_between(
        time_points,
        15,
        response_times,
        where=(response_times > 15),
        alpha=0.3,
        color='orange',
        label='Delayed Response',
    )
    ax3.set_xlabel('Time (hours)', fontweight='bold')
    ax3.set_ylabel('Response Time (min)', fontweight='bold')
    ax3.set_title(
        '(c) Response Time Degradation', fontweight='bold', pad=10, loc='left'
    )
    ax3.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(loc='upper left', framealpha=0.9, fontsize=9)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(hspace=0.30, left=0.12, right=0.95, top=0.94, bottom=0.06)

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, axes


def plot_override_rates_comparison(
    clinician_names: List[str],
    override_rates: List[float],
    alert_counts: List[int],
    experience_levels: List[str],
    title: str = "Clinician-Specific Override Patterns",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot override rates comparison across clinicians.

    Args:
        clinician_names: List of clinician identifiers
        override_rates: Override rate (0-1) for each clinician
        alert_counts: Total alerts received by each clinician
        experience_levels: Experience level ('junior', 'senior', 'attending')
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> clinicians = ['Dr. A', 'Dr. B', 'Dr. C', 'Dr. D', 'Dr. E']
        >>> overrides = [0.65, 0.45, 0.30, 0.55, 0.25]
        >>> alerts = [120, 95, 110, 88, 105]
        >>> experience = ['junior', 'senior', 'attending', 'junior', 'attending']
        >>> fig, ax = plot_override_rates_comparison(clinicians, overrides,
        ...                                          alerts, experience)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color by experience level
    exp_colors = {'junior': '#CC3311', 'senior': '#EE7733', 'attending': '#0077BB'}
    colors = [exp_colors.get(exp, '#BBBBBB') for exp in experience_levels]

    # Bubble size based on alert count
    sizes = [count * 3 for count in alert_counts]

    y_pos = np.arange(len(clinician_names))

    # Scatter plot
    for i, (y, rate, size, color) in enumerate(
        zip(y_pos, override_rates, sizes, colors)
    ):
        ax.scatter(
            rate * 100,
            y,
            s=size,
            color=color,
            alpha=0.7,
            edgecolors='black',
            linewidth=2.0,
            zorder=3,
        )

        # Add alert count label
        ax.text(
            rate * 100 + 2,
            y,
            f'n={alert_counts[i]}',
            va='center',
            fontsize=9,
            color='black',
        )

    # Add threshold lines
    ax.axvline(
        x=40,
        color='orange',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Warning (40%)',
    )
    ax.axvline(
        x=60,
        color='red',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Critical (60%)',
    )
    ax.axvspan(60, 100, alpha=0.1, color='red', label='Fatigue Zone')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clinician_names, fontsize=11)
    ax.set_xlabel('Override Rate (%)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Junior',
            markerfacecolor=exp_colors['junior'],
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Senior',
            markerfacecolor=exp_colors['senior'],
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Attending',
            markerfacecolor=exp_colors['attending'],
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        framealpha=0.95,
        fontsize=10,
        edgecolor='black',
        title='Experience',
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_system_resilience(
    time_points: np.ndarray,
    workload: np.ndarray,
    performance: np.ndarray,
    stress_events: Optional[List[Tuple[float, str]]] = None,
    title: str = "System Resilience Under Varying Workload",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot system performance resilience under varying workload.

    Args:
        time_points: Time points (hours)
        workload: Workload intensity (0-100%)
        performance: System performance metric (0-100%)
        stress_events: List of (time, event_name) tuples marking stress events
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> time = np.linspace(0, 24, 100)
        >>> workload = 50 + 30*np.sin(time/6) + np.random.normal(0, 5, 100)
        >>> performance = 95 - 0.3*workload + np.random.normal(0, 3, 100)
        >>> events = [(6, 'Mass Casualty'), (18, 'Staff Shortage')]
        >>> fig, ax = plot_system_resilience(time, workload, performance, events)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create twin axis
    ax2 = ax.twinx()

    # Plot workload
    line1 = ax.plot(
        time_points,
        workload,
        color=COLORS['alert'],
        linewidth=2.5,
        label='Workload',
        alpha=0.8,
    )
    ax.fill_between(time_points, 0, workload, alpha=0.2, color=COLORS['alert'])

    # Plot performance
    line2 = ax2.plot(
        time_points,
        performance,
        color=COLORS['cdss'],
        linewidth=2.5,
        label='Performance',
        alpha=0.8,
    )

    # Mark stress events
    if stress_events:
        for event_time, event_name in stress_events:
            ax.axvline(
                event_time, color='red', linestyle='--', linewidth=2.0, alpha=0.7
            )
            ax.text(
                event_time,
                ax.get_ylim()[1] * 0.95,
                event_name,
                rotation=90,
                va='top',
                ha='right',
                fontsize=9,
                color='red',
                fontweight='bold',
            )

    # Add performance threshold
    ax2.axhline(
        y=80,
        color='orange',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Target Performance (80%)',
    )

    # Formatting
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Workload (%)', fontweight='bold', color=COLORS['alert'])
    ax2.set_ylabel('Performance (%)', fontweight='bold', color=COLORS['cdss'])
    ax.set_title(title, fontweight='bold', pad=10)
    ax.tick_params(axis='y', labelcolor=COLORS['alert'])
    ax2.tick_params(axis='y', labelcolor=COLORS['cdss'])
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(
        lines, labels, loc='lower left', framealpha=0.95, fontsize=11, edgecolor='black'
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


__all__ = [
    'plot_agent_interaction_network',
    'plot_workflow_timeline',
    'plot_alert_fatigue_dynamics',
    'plot_override_rates_comparison',
    'plot_system_resilience',
]
