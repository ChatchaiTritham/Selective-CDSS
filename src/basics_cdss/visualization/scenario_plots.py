"""
Scenario and perturbation visualization functions.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_uncertainty_distribution(
    scenarios: List,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Uncertainty Profile Distribution",
) -> Tuple[Figure, List[Axes]]:
    """Plot distribution of uncertainty profiles across scenarios.

    Args:
        scenarios: List of Scenario objects with uncertainty_profile
        figsize: Figure size
        title: Overall title

    Returns:
        Tuple of (figure, list of axes)

    Example:
        >>> from basics_cdss.scenario import instantiate_scenarios
        >>> scenarios = instantiate_scenarios(
        ...     archetypes, perturbation_type="composite"
        ... )
        >>> fig, axes = plot_uncertainty_distribution(scenarios)
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)

    # Extract uncertainty metrics
    missingness = [s.uncertainty_profile.get("missingness", 0) for s in scenarios]
    ambiguity = [s.uncertainty_profile.get("ambiguity", 0) for s in scenarios]
    conflict = [s.uncertainty_profile.get("conflict", 0) for s in scenarios]
    degradation = [s.uncertainty_profile.get("degradation", 0) for s in scenarios]

    # Plot histograms
    ax1.hist(missingness, bins=20, color='#E63946', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Missingness", fontweight='bold')
    ax1.set_ylabel("Count", fontweight='bold')
    ax1.set_title(f"Missingness\n(μ={np.mean(missingness):.3f})", fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.hist(ambiguity, bins=20, color='#F77F00', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Ambiguity", fontweight='bold')
    ax2.set_ylabel("Count", fontweight='bold')
    ax2.set_title(f"Ambiguity\n(μ={np.mean(ambiguity):.3f})", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3.hist(conflict, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Conflict", fontweight='bold')
    ax3.set_ylabel("Count", fontweight='bold')
    ax3.set_title(f"Conflict\n(μ={np.mean(conflict):.3f})", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4.hist(degradation, bins=20, color='#06A77D', edgecolor='black', alpha=0.7)
    ax4.set_xlabel("Degradation", fontweight='bold')
    ax4.set_ylabel("Count", fontweight='bold')
    ax4.set_title(f"Degradation\n(μ={np.mean(degradation):.3f})", fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, [ax1, ax2, ax3, ax4]


def plot_perturbation_effects(
    baseline_scenarios: List,
    perturbed_scenarios: Dict[str, List],
    metric_fn: callable,
    metric_name: str = "Metric",
    figsize: Tuple[int, int] = (12, 7),
) -> Tuple[Figure, Axes]:
    """Compare metric values across perturbation types.

    Args:
        baseline_scenarios: List of baseline scenarios (no perturbation)
        perturbed_scenarios: Dict mapping perturbation_type -> scenarios
        metric_fn: Function to compute metric from scenarios
        metric_name: Name of metric for labeling
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> def count_features(scenarios):
        ...     return np.mean([len(s.features) for s in scenarios])
        >>>
        >>> fig, ax = plot_perturbation_effects(
        ...     baseline, perturbed, count_features, "Avg Features"
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    perturbation_types = ['baseline'] + list(perturbed_scenarios.keys())
    metric_values = []

    # Compute metrics
    metric_values.append(metric_fn(baseline_scenarios))

    for ptype in perturbed_scenarios.keys():
        metric_values.append(metric_fn(perturbed_scenarios[ptype]))

    # Bar plot
    colors = ['#2E86AB', '#E63946', '#F77F00', '#06A77D', '#A23B72']
    bars = ax.bar(
        perturbation_types,
        metric_values,
        color=colors[: len(perturbation_types)],
        edgecolor='black',
        linewidth=2,
        alpha=0.8,
    )

    # Value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax.set_xlabel("Perturbation Type", fontsize=13, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
    ax.set_title(
        f"Effect of Perturbations on {metric_name}", fontsize=15, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    return fig, ax


def plot_scenario_summary(
    scenarios: List, figsize: Tuple[int, int] = (14, 10)
) -> Tuple[Figure, List[Axes]]:
    """Create comprehensive summary visualization of scenario set.

    Args:
        scenarios: List of Scenario objects
        figsize: Figure size

    Returns:
        Tuple of (figure, list of axes)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract data
    archetype_ids = [s.archetype_id for s in scenarios]
    risk_tiers = [s.targets.get("triage_tier", "unknown") for s in scenarios]
    missingness = [s.uncertainty_profile.get("missingness", 0) for s in scenarios]
    ambiguity = [s.uncertainty_profile.get("ambiguity", 0) for s in scenarios]

    # 1. Archetype distribution
    ax1 = fig.add_subplot(gs[0, :2])
    unique_archetypes, archetype_counts = np.unique(archetype_ids, return_counts=True)
    ax1.barh(
        range(len(unique_archetypes)), archetype_counts, color='#2E86AB', alpha=0.8
    )
    ax1.set_yticks(range(len(unique_archetypes)))
    ax1.set_yticklabels(unique_archetypes, fontsize=9)
    ax1.set_xlabel("Scenario Count", fontweight='bold')
    ax1.set_title("Scenarios per Archetype", fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. Risk tier distribution
    ax2 = fig.add_subplot(gs[0, 2])
    unique_tiers, tier_counts = np.unique(risk_tiers, return_counts=True)
    colors_tier = {'high': '#E63946', 'medium': '#F77F00', 'low': '#06A77D'}
    tier_colors = [colors_tier.get(t.lower(), '#2E86AB') for t in unique_tiers]
    ax2.pie(
        tier_counts,
        labels=unique_tiers,
        colors=tier_colors,
        autopct='%1.1f%%',
        startangle=90,
    )
    ax2.set_title("Risk Tier Distribution", fontweight='bold')

    # 3. Uncertainty scatter
    ax3 = fig.add_subplot(gs[1, :])
    scatter = ax3.scatter(
        missingness,
        ambiguity,
        c=range(len(scenarios)),
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black',
    )
    ax3.set_xlabel("Missingness", fontweight='bold')
    ax3.set_ylabel("Ambiguity", fontweight='bold')
    ax3.set_title("Uncertainty Profile (Missingness vs Ambiguity)", fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label="Scenario Index")

    # 4. Summary stats
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    summary_text = f"""
    SCENARIO SUMMARY
    {'='*50}
    Total Scenarios: {len(scenarios)}
    Unique Archetypes: {len(unique_archetypes)}
    Risk Tiers: {', '.join(unique_tiers)}

    Uncertainty Statistics:
    - Missingness: μ={np.mean(missingness):.3f}, σ={np.std(missingness):.3f}
    - Ambiguity: μ={np.mean(ambiguity):.3f}, σ={np.std(ambiguity):.3f}
    """
    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        family='monospace',
        verticalalignment='center',
    )

    fig.suptitle("Scenario Set Comprehensive Summary", fontsize=16, fontweight='bold')

    return fig, [ax1, ax2, ax3, ax4]
