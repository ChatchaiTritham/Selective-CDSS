"""
Causal Visualization Module for BASICS-CDSS Tier 2 (Causal Simulation)

Publication-quality plots for causal DAGs, intervention effects, confounding analysis,
and causal inference results.

Compliant with:
- Nature Machine Intelligence standards
- JMLR figure requirements
- Journal of Causal Inference guidelines
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)
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
    'treatment': '#0077BB',  # Blue
    'outcome': '#CC3311',  # Red
    'confounder': '#EE7733',  # Orange
    'mediator': '#33BB55',  # Green
    'instrumental': '#9933CC',  # Purple
    'collider': '#BBBBBB',  # Gray
}


def plot_causal_dag(
    graph: nx.DiGraph,
    node_types: Optional[Dict[str, str]] = None,
    highlight_path: Optional[List[str]] = None,
    title: str = "Causal Directed Acyclic Graph",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot causal DAG with proper node coloring and edge styles.

    Args:
        graph: NetworkX DiGraph representing causal structure
        node_types: Dictionary mapping node names to types
                   ('treatment', 'outcome', 'confounder', 'mediator')
        highlight_path: List of nodes forming a causal path to highlight
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([('X', 'Y'), ('Z', 'X'), ('Z', 'Y')])
        >>> node_types = {'X': 'treatment', 'Y': 'outcome', 'Z': 'confounder'}
        >>> fig, ax = plot_causal_dag(G, node_types)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use hierarchical layout for better DAG visualization
    try:
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(graph)

    # Assign colors based on node types
    if node_types is None:
        node_types = {node: 'confounder' for node in graph.nodes()}

    node_colors = [
        COLORS.get(node_types.get(node, 'confounder'), '#BBBBBB')
        for node in graph.nodes()
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2000,
        edgecolors='black',
        linewidths=2.0,
        alpha=0.9,
    )

    # Draw edges
    if highlight_path:
        # Draw regular edges first
        regular_edges = [
            (u, v)
            for u, v in graph.edges()
            if not (u in highlight_path and v in highlight_path)
        ]
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=regular_edges,
            edge_color='black',
            width=2.0,
            alpha=0.5,
            arrowsize=20,
            arrowstyle='->',
        )

        # Draw highlighted path edges
        highlight_edges = [
            (highlight_path[i], highlight_path[i + 1])
            for i in range(len(highlight_path) - 1)
            if graph.has_edge(highlight_path[i], highlight_path[i + 1])
        ]
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=highlight_edges,
            edge_color='red',
            width=4.0,
            alpha=0.9,
            arrowsize=25,
            arrowstyle='->',
        )
    else:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color='black',
            width=2.0,
            alpha=0.6,
            arrowsize=20,
            arrowstyle='->',
        )

    # Draw labels
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=12, font_weight='bold', font_family='serif'
    )

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Treatment',
            markerfacecolor=COLORS['treatment'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Outcome',
            markerfacecolor=COLORS['outcome'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Confounder',
            markerfacecolor=COLORS['confounder'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Mediator',
            markerfacecolor=COLORS['mediator'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        framealpha=0.95,
        fontsize=11,
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


def plot_intervention_effects(
    ate_results: Dict[str, Dict[str, float]],
    title: str = "Average Treatment Effects (ATE)",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Average Treatment Effects (ATE) with confidence intervals.

    Args:
        ate_results: Dictionary with intervention names as keys, each containing:
                    {'ate': float, 'ci_lower': float, 'ci_upper': float, 'p_value': float}
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> ate_results = {
        ...     'Antibiotic (Early)': {'ate': -0.15, 'ci_lower': -0.22, 'ci_upper': -0.08, 'p_value': 0.001},
        ...     'Fluid Resuscitation': {'ate': -0.08, 'ci_lower': -0.14, 'ci_upper': -0.02, 'p_value': 0.012},
        ...     'Vasopressor': {'ate': 0.02, 'ci_lower': -0.05, 'ci_upper': 0.09, 'p_value': 0.542}
        ... }
        >>> fig, ax = plot_intervention_effects(ate_results)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    interventions = list(ate_results.keys())
    ates = [ate_results[i]['ate'] for i in interventions]
    ci_lowers = [ate_results[i]['ci_lower'] for i in interventions]
    ci_uppers = [ate_results[i]['ci_upper'] for i in interventions]
    p_values = [ate_results[i]['p_value'] for i in interventions]

    y_pos = np.arange(len(interventions))

    # Color by significance
    colors = [COLORS['treatment'] if p < 0.05 else COLORS['collider'] for p in p_values]

    # Plot error bars
    for i, (y, ate, ci_l, ci_u, p, color) in enumerate(
        zip(y_pos, ates, ci_lowers, ci_uppers, p_values, colors)
    ):
        # Draw error bar
        ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=2.5, alpha=0.8)
        # Draw point estimate
        marker = 'D' if p < 0.05 else 'o'
        ax.scatter(
            ate,
            y,
            s=150,
            color=color,
            edgecolors='black',
            linewidth=2,
            marker=marker,
            zorder=3,
            alpha=0.9,
        )

        # Add significance asterisks
        if p < 0.001:
            sig_text = '***'
        elif p < 0.01:
            sig_text = '**'
        elif p < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        ax.text(
            ci_u + 0.01,
            y,
            sig_text,
            va='center',
            fontweight='bold',
            fontsize=12,
            color=color,
        )

    # Add zero line
    ax.axvline(
        x=0,
        color='black',
        linestyle='--',
        linewidth=2.0,
        alpha=0.7,
        label='Null Effect',
    )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(interventions, fontsize=12)
    ax.set_xlabel('Average Treatment Effect (ATE)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='D',
            color='w',
            label='Significant (p<0.05)',
            markerfacecolor=COLORS['treatment'],
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Not Significant',
            markerfacecolor=COLORS['collider'],
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=2,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        framealpha=0.95,
        fontsize=11,
        edgecolor='black',
    )

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_cate_heterogeneity(
    subgroups: List[str],
    cates: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    title: str = "Conditional Average Treatment Effects (CATE)",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Conditional Average Treatment Effects across subgroups.

    Args:
        subgroups: List of subgroup names
        cates: List of CATE values
        ci_lowers: Lower confidence interval bounds
        ci_uppers: Upper confidence interval bounds
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> subgroups = ['Age <65', 'Age 65-75', 'Age >75', 'Male', 'Female',
        ...              'Comorbidity 0-1', 'Comorbidity 2+']
        >>> cates = [-0.10, -0.15, -0.22, -0.12, -0.16, -0.08, -0.20]
        >>> ci_l = [-0.15, -0.22, -0.30, -0.18, -0.22, -0.13, -0.28]
        >>> ci_u = [-0.05, -0.08, -0.14, -0.06, -0.10, -0.03, -0.12]
        >>> fig, ax = plot_cate_heterogeneity(subgroups, cates, ci_l, ci_u)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_pos = np.arange(len(subgroups))

    # Plot forest plot style
    for i, (y, cate, ci_l, ci_u) in enumerate(zip(y_pos, cates, ci_lowers, ci_uppers)):
        # Error bar
        ax.plot(
            [ci_l, ci_u], [y, y], color=COLORS['treatment'], linewidth=3.0, alpha=0.7
        )
        # Point estimate
        ax.scatter(
            cate,
            y,
            s=200,
            color=COLORS['outcome'],
            edgecolors='black',
            linewidth=2,
            marker='s',
            zorder=3,
        )

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2.0, alpha=0.7)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroups, fontsize=11)
    ax.set_xlabel('CATE (Effect Size)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, ax


def plot_confounding_analysis(
    confounders: List[str],
    bias_estimates: List[float],
    adjusted_effects: List[float],
    unadjusted_effect: float,
    title: str = "Confounding Bias Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot confounding analysis showing bias from each confounder.

    Args:
        confounders: List of confounder variable names
        bias_estimates: Estimated bias from each confounder
        adjusted_effects: Treatment effects after adjusting for each confounder
        unadjusted_effect: Unadjusted treatment effect
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes array

    Example:
        >>> confounders = ['Age', 'Comorbidity', 'Severity', 'Hospital Type']
        >>> bias_est = [0.05, 0.12, 0.18, 0.03]
        >>> adjusted = [-0.15, -0.22, -0.28, -0.12]
        >>> unadj = -0.10
        >>> fig, axes = plot_confounding_analysis(confounders, bias_est,
        ...                                       adjusted, unadj)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Bias magnitude
    y_pos = np.arange(len(confounders))
    bars = ax1.barh(
        y_pos,
        bias_estimates,
        color=COLORS['confounder'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels
    for i, (bar, bias) in enumerate(zip(bars, bias_estimates)):
        width = bar.get_width()
        ax1.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{bias:.3f}',
            ha='left',
            va='center',
            fontweight='bold',
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(confounders, fontsize=11)
    ax1.set_xlabel('Estimated Bias', fontweight='bold')
    ax1.set_title('(a) Bias from Each Confounder', fontweight='bold', pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Plot 2: Treatment effect comparison
    effects = [unadjusted_effect] + adjusted_effects
    labels = ['Unadjusted'] + [f'Adjusted for {c}' for c in confounders]
    y_pos2 = np.arange(len(labels))

    colors_effects = [COLORS['collider']] + [COLORS['treatment']] * len(confounders)

    for i, (y, effect, color) in enumerate(zip(y_pos2, effects, colors_effects)):
        # Draw error bar (simplified, no actual CI here)
        ci_width = abs(effect) * 0.1
        ax2.plot(
            [effect - ci_width, effect + ci_width],
            [y, y],
            color=color,
            linewidth=2.5,
            alpha=0.7,
        )
        # Draw point
        marker = 'o' if i == 0 else 's'
        ax2.scatter(
            effect,
            y,
            s=150,
            color=color,
            edgecolors='black',
            linewidth=2,
            marker=marker,
            zorder=3,
        )

    # Add zero line
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2.0, alpha=0.7)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('Treatment Effect Estimate', fontweight='bold')
    ax2.set_title(
        '(b) Effect Estimates: Unadjusted vs Adjusted', fontweight='bold', pad=10
    )
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.subplots_adjust(hspace=0.35, left=0.20, right=0.95, top=0.95, bottom=0.08)

    if save_path:
        for ext in ['pdf', 'eps', 'png']:
            fig.savefig(
                save_path.replace('.png', f'.{ext}'), dpi=300, bbox_inches='tight'
            )

    return fig, np.array([ax1, ax2])


def plot_backdoor_adjustment(
    treatment: str,
    outcome: str,
    confounders: List[str],
    graph: nx.DiGraph,
    title: str = "Backdoor Adjustment Strategy",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize backdoor adjustment by highlighting backdoor paths and adjustment set.

    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        confounders: List of variables in adjustment set
        graph: Causal DAG
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure and axes

    Example:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([('X', 'Y'), ('Z', 'X'), ('Z', 'Y'), ('W', 'Z')])
        >>> fig, ax = plot_backdoor_adjustment('X', 'Y', ['Z'], G)
    """
    plt.rcParams.update(STYLE_CONFIG)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Layout
    try:
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(graph)

    # Node colors
    node_colors = []
    for node in graph.nodes():
        if node == treatment:
            node_colors.append(COLORS['treatment'])
        elif node == outcome:
            node_colors.append(COLORS['outcome'])
        elif node in confounders:
            node_colors.append(COLORS['confounder'])
        else:
            node_colors.append(COLORS['collider'])

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2500,
        edgecolors='black',
        linewidths=2.5,
        alpha=0.9,
    )

    # Draw edges (backdoor paths in red, others in black)
    for u, v in graph.edges():
        # Check if edge is part of backdoor path
        if u in confounders or v in confounders:
            edge_color = 'red'
            edge_width = 3.5
            edge_alpha = 0.9
        else:
            edge_color = 'black'
            edge_width = 2.0
            edge_alpha = 0.5

        nx.draw_networkx_edges(
            graph,
            pos,
            [(u, v)],
            ax=ax,
            edge_color=edge_color,
            width=edge_width,
            alpha=edge_alpha,
            arrowsize=20,
            arrowstyle='->',
        )

    # Draw labels
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=12, font_weight='bold', font_family='serif'
    )

    # Add adjustment set box
    textstr = 'Adjustment Set:\n' + '\n'.join([f'  • {c}' for c in confounders])
    props = dict(
        boxstyle='round,pad=0.8',
        facecolor='wheat',
        alpha=0.9,
        edgecolor='black',
        linewidth=2,
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=props,
        fontweight='bold',
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


__all__ = [
    'plot_causal_dag',
    'plot_intervention_effects',
    'plot_cate_heterogeneity',
    'plot_confounding_analysis',
    'plot_backdoor_adjustment',
]
