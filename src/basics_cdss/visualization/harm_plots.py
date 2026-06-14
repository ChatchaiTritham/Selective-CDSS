"""
Harm-aware visualization functions for safety-critical analysis.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_harm_by_tier(
    harm_by_tier: Dict[str, float],
    ax: Optional[Axes] = None,
    title: str = "Harm Distribution by Risk Tier",
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[Figure, Axes]:
    """Plot harm distribution across risk tiers.

    Args:
        harm_by_tier: Dict mapping tier -> harm value
        ax: Matplotlib axes
        title: Plot title
        figsize: Figure size
        colors: Dict mapping tier -> color

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import harm_by_risk_tier
        >>> harm = harm_by_risk_tier(y_true, y_pred, risk_tiers)
        >>> fig, ax = plot_harm_by_tier(harm)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if colors is None:
        colors = {
            "high": "#E63946",
            "medium": "#F77F00",
            "low": "#06A77D",
            "urgent": "#E63946",
            "non-urgent": "#06A77D",
        }

    tiers = list(harm_by_tier.keys())
    harm_values = list(harm_by_tier.values())
    tier_colors = [colors.get(tier.lower(), "#2E86AB") for tier in tiers]

    # Bar plot
    bars = ax.bar(
        tiers, harm_values, color=tier_colors, edgecolor='black', linewidth=2, alpha=0.8
    )

    # Add value labels on bars
    for bar, value in zip(bars, harm_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
        )

    # Styling
    ax.set_xlabel("Risk Tier", fontsize=13, fontweight='bold')
    ax.set_ylabel("Weighted Harm", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Total harm annotation
    total_harm = sum(harm_values)
    ax.text(
        0.95,
        0.95,
        f"Total Harm: {total_harm:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='top',
        horizontalalignment='right',
    )

    plt.tight_layout()
    return fig, ax


def plot_escalation_analysis(
    escalation_failures: int,
    false_escalations: int,
    high_risk_samples: int,
    low_risk_samples: int,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[Figure, List[Axes]]:
    """Visualize escalation failure analysis.

    Args:
        escalation_failures: Count of missed escalations (high-risk false negatives)
        false_escalations: Count of unnecessary escalations (low-risk false positives)
        high_risk_samples: Total high-risk samples
        low_risk_samples: Total low-risk samples
        figsize: Figure size

    Returns:
        Tuple of (figure, list of axes)

    Example:
        >>> from basics_cdss.metrics import escalation_failure_analysis
        >>> analysis = escalation_failure_analysis(y_true, y_pred, risk_tiers)
        >>> fig, axes = plot_escalation_analysis(
        ...     analysis['escalation_failures'],
        ...     analysis['false_escalations'],
        ...     analysis['high_risk_samples'],
        ...     analysis['low_risk_samples']
        ... )
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Calculate rates
    escalation_failure_rate = (
        escalation_failures / high_risk_samples if high_risk_samples > 0 else 0
    )
    false_escalation_rate = (
        false_escalations / low_risk_samples if low_risk_samples > 0 else 0
    )

    # Plot 1: Escalation failures (high-risk missed)
    categories = ['Correctly\nEscalated', 'Missed\n(FAILURE)']
    correct_escalations = high_risk_samples - escalation_failures
    values = [correct_escalations, escalation_failures]
    colors_fail = ['#06A77D', '#E63946']

    bars1 = ax1.bar(
        categories, values, color=colors_fail, edgecolor='black', linewidth=2
    )

    # Add value labels
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            (
                f'{value}\n({value/high_risk_samples*100:.1f}%)'
                if high_risk_samples > 0
                else f'{value}'
            ),
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax1.set_ylabel("Count (High-Risk Cases)", fontsize=12, fontweight='bold')
    ax1.set_title(
        f"Escalation Failures\n(Failure Rate: {escalation_failure_rate:.1%})",
        fontsize=14,
        fontweight='bold',
    )
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: False escalations (low-risk over-escalated)
    categories2 = ['Correctly\nDeferred', 'Over-Escalated\n(False Alarm)']
    correct_deferrals = low_risk_samples - false_escalations
    values2 = [correct_deferrals, false_escalations]
    colors_false = ['#06A77D', '#F77F00']

    bars2 = ax2.bar(
        categories2, values2, color=colors_false, edgecolor='black', linewidth=2
    )

    # Add value labels
    for bar, value in zip(bars2, values2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            (
                f'{value}\n({value/low_risk_samples*100:.1f}%)'
                if low_risk_samples > 0
                else f'{value}'
            ),
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax2.set_ylabel("Count (Low-Risk Cases)", fontsize=12, fontweight='bold')
    ax2.set_title(
        f"False Escalations\n(False Alarm Rate: {false_escalation_rate:.1%})",
        fontsize=14,
        fontweight='bold',
    )
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, [ax1, ax2]


def plot_harm_concentration(
    harm_by_tier: Dict[str, float],
    concentration_index: float,
    figsize: Tuple[int, int] = (10, 7),
    high_risk_tiers: Optional[List[str]] = None,
) -> Tuple[Figure, Axes]:
    """Visualize harm concentration in high-risk tier.

    Args:
        harm_by_tier: Dict mapping tier -> harm value
        concentration_index: Fraction of harm in high-risk tier [0, 1]
        figsize: Figure size
        high_risk_tiers: List of tier labels considered high-risk

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import harm_concentration_index, harm_by_risk_tier
        >>> harm = harm_by_risk_tier(y_true, y_pred, risk_tiers)
        >>> concentration = harm_concentration_index(y_true, y_pred, risk_tiers)
        >>> fig, ax = plot_harm_concentration(harm, concentration)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if high_risk_tiers is None:
        high_risk_tiers = ["high", "urgent", "critical", "emergency"]

    # Separate high-risk and other tiers
    high_risk_harm = sum(
        harm for tier, harm in harm_by_tier.items() if tier.lower() in high_risk_tiers
    )
    other_harm = sum(
        harm
        for tier, harm in harm_by_tier.items()
        if tier.lower() not in high_risk_tiers
    )

    # Pie chart
    sizes = [high_risk_harm, other_harm]
    labels = [
        f'High-Risk Tiers\n({high_risk_harm:.3f})',
        f'Other Tiers\n({other_harm:.3f})',
    ]
    colors_pie = ['#E63946', '#2E86AB']
    explode = (0.1, 0)  # Explode high-risk slice

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
    )

    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)

    # Title with concentration index
    ax.set_title(
        f"Harm Concentration\n(Index: {concentration_index:.2%} in High-Risk)",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )

    # Add interpretation
    interpretation = ""
    if concentration_index > 0.7:
        interpretation = "⚠️ HIGH concentration in high-risk tier"
        interp_color = '#E63946'
    elif concentration_index > 0.4:
        interpretation = "⚡ MODERATE concentration"
        interp_color = '#F77F00'
    else:
        interpretation = "✓ Balanced distribution"
        interp_color = '#06A77D'

    ax.text(
        0.5,
        -0.15,
        interpretation,
        transform=ax.transAxes,
        ha='center',
        fontsize=14,
        fontweight='bold',
        color=interp_color,
        bbox=dict(
            boxstyle='round', facecolor='white', edgecolor=interp_color, linewidth=2
        ),
    )

    plt.tight_layout()
    return fig, ax
