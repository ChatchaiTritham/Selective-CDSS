"""
Model comparison and dashboard visualization functions.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_metric_comparison(
    models: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Model Performance Comparison",
) -> Tuple[Figure, Axes]:
    """Compare multiple metrics across models using grouped bar chart.

    Args:
        models: Dict mapping model_name -> {metric_name: value}
        metrics_to_plot: List of metric names to include (plots all if None)
        figsize: Figure size
        title: Plot title

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     "Baseline": {"ECE": 0.15, "AURC": 0.25, "Harm": 2.5},
        ...     "TRI-X": {"ECE": 0.08, "AURC": 0.18, "Harm": 1.2}
        ... }
        >>> fig, ax = plot_metric_comparison(models)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine metrics to plot
    if metrics_to_plot is None:
        all_metrics = set()
        for model_metrics in models.values():
            all_metrics.update(model_metrics.keys())
        metrics_to_plot = sorted(list(all_metrics))

    model_names = list(models.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ['#E63946', '#2E86AB', '#06A77D', '#F77F00', '#A23B72']

    # Plot bars for each model
    for idx, model_name in enumerate(model_names):
        values = [models[model_name].get(metric, 0) for metric in metrics_to_plot]
        offset = (idx - n_models / 2 + 0.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=colors[idx % len(colors)],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
        )

        # Add value labels
        for i, v in enumerate(values):
            ax.text(
                x[i] + offset,
                v,
                f'{v:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                rotation=0,
            )

    # Styling
    ax.set_xlabel("Metrics", fontsize=13, fontweight='bold')
    ax.set_ylabel("Value", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    return fig, ax


def plot_model_comparison_radar(
    models: Dict[str, Dict[str, float]],
    metrics: List[str],
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Model Performance Radar",
    normalize: bool = True,
) -> Tuple[Figure, Axes]:
    """Create radar chart comparing models across multiple metrics.

    Args:
        models: Dict mapping model_name -> {metric_name: value}
        metrics: List of metric names to include (must be in all models)
        figsize: Figure size
        title: Plot title
        normalize: If True, normalize metrics to [0, 1] range

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     "Baseline": {"ECE": 0.15, "Accuracy": 0.85, "F1": 0.82},
        ...     "TRI-X": {"ECE": 0.08, "Accuracy": 0.90, "F1": 0.88}
        ... }
        >>> fig, ax = plot_model_comparison_radar(
        ...     models, ["ECE", "Accuracy", "F1"]
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Number of variables
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ['#E63946', '#2E86AB', '#06A77D', '#F77F00', '#A23B72']

    # Extract and optionally normalize values
    for idx, (model_name, model_metrics) in enumerate(models.items()):
        values = [model_metrics.get(metric, 0) for metric in metrics]

        if normalize:
            # Normalize to [0, 1] for each metric
            max_vals = [
                max(m.get(metric, 0) for m in models.values()) for metric in metrics
            ]
            values = [
                v / max_val if max_val > 0 else 0
                for v, max_val in zip(values, max_vals)
            ]

        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2.5,
            markersize=8,
            label=model_name,
            color=colors[idx % len(colors)],
        )
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1 if normalize else None)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def create_evaluation_dashboard(
    calibration_data: Dict,
    coverage_risk_data: Dict,
    harm_data: Dict,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (18, 12),
) -> Figure:
    """Create comprehensive evaluation dashboard with all key plots.

    Args:
        calibration_data: Dict with 'confidences', 'accuracies', 'counts'
        coverage_risk_data: Dict with 'coverages', 'risks'
        harm_data: Dict with 'harm_by_tier', 'concentration_index'
        model_name: Name of model being evaluated
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> dashboard = create_evaluation_dashboard(
        ...     calibration_data={'confidences': ..., 'accuracies': ..., 'counts': ...},
        ...     coverage_risk_data={'coverages': ..., 'risks': ...},
        ...     harm_data={'harm_by_tier': {...}, 'concentration_index': 0.75}
        ... )
        >>> dashboard.savefig("dashboard.png", dpi=300)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Title
    fig.suptitle(
        f"BASICS-CDSS Evaluation Dashboard: {model_name}",
        fontsize=20,
        fontweight='bold',
        y=0.98,
    )

    # 1. Reliability Diagram (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    confs = calibration_data.get('confidences', np.array([]))
    accs = calibration_data.get('accuracies', np.array([]))

    if len(confs) > 0:
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
        ax1.plot(confs, accs, 'o-', color='#2E86AB', markersize=7, linewidth=2.5)
        ece = np.mean(np.abs(confs - accs)) if len(confs) > 0 else 0
        ax1.text(
            0.05,
            0.95,
            f"ECE: {ece:.4f}",
            transform=ax1.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat'),
        )

    ax1.set_xlabel("Confidence", fontweight='bold')
    ax1.set_ylabel("Accuracy", fontweight='bold')
    ax1.set_title("Calibration Reliability", fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # 2. Coverage-Risk Curve (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    coverages = coverage_risk_data.get('coverages', np.array([]))
    risks = coverage_risk_data.get('risks', np.array([]))

    if len(coverages) > 0:
        valid_mask = ~np.isnan(risks)
        cov_clean = coverages[valid_mask]
        risk_clean = risks[valid_mask]

        ax2.plot(cov_clean, risk_clean, 'o-', color='#E63946', linewidth=2.5)
        ax2.fill_between(cov_clean, 0, risk_clean, alpha=0.2, color='#E63946')

        try:
            aurc = np.trapezoid(risk_clean, cov_clean)
        except AttributeError:
            aurc = np.trapz(risk_clean, cov_clean)

        ax2.text(
            0.95,
            0.95,
            f"AURC: {aurc:.4f}",
            transform=ax2.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue'),
            ha='right',
            va='top',
        )

    ax2.set_xlabel("Coverage", fontweight='bold')
    ax2.set_ylabel("Risk", fontweight='bold')
    ax2.set_title("Coverage-Risk Trade-off", fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. Harm by Tier (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    harm_by_tier = harm_data.get('harm_by_tier', {})

    if harm_by_tier:
        tiers = list(harm_by_tier.keys())
        harms = list(harm_by_tier.values())
        colors_harm = {'high': '#E63946', 'medium': '#F77F00', 'low': '#06A77D'}
        tier_colors = [colors_harm.get(t.lower(), '#2E86AB') for t in tiers]

        ax3.bar(tiers, harms, color=tier_colors, edgecolor='black', alpha=0.8)

    ax3.set_xlabel("Risk Tier", fontweight='bold')
    ax3.set_ylabel("Weighted Harm", fontweight='bold')
    ax3.set_title("Harm by Risk Tier", fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Harm Concentration Pie (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    concentration = harm_data.get('concentration_index', 0)

    if harm_by_tier:
        high_harm = sum(
            h for t, h in harm_by_tier.items() if t.lower() in ['high', 'urgent']
        )
        other_harm = sum(
            h for t, h in harm_by_tier.items() if t.lower() not in ['high', 'urgent']
        )

        ax4.pie(
            [high_harm, other_harm],
            labels=['High-Risk', 'Other'],
            colors=['#E63946', '#2E86AB'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0),
        )

    ax4.set_title(
        f"Harm Concentration\n({concentration:.1%} in High-Risk)",
        fontweight='bold',
        fontsize=14,
    )

    # 5-7. Summary Statistics (middle-middle to bottom)
    ax5 = fig.add_subplot(gs[1:, 1:])
    ax5.axis('off')

    # Format metric values with proper conditionals
    ece_str = f"{ece:.4f}" if len(confs) > 0 else "N/A"
    aurc_str = f"{aurc:.4f}" if len(coverages) > 0 else "N/A"
    total_harm_str = f"{sum(harm_by_tier.values()):.3f}" if harm_by_tier else "N/A"

    # Interpretation messages
    interp_high = (
        "⚠️ High harm concentration in high-risk tier" if concentration > 0.7 else ""
    )
    interp_low = "✓ Balanced harm distribution" if concentration < 0.4 else ""
    interp_mod = (
        "⚡ Moderate concentration - review high-risk handling"
        if 0.4 <= concentration <= 0.7
        else ""
    )

    summary = f"""
    {'='*60}
    EVALUATION SUMMARY
    {'='*60}

    CALIBRATION METRICS:
      • Expected Calibration Error (ECE): {ece_str}
      • Lower is better (perfect = 0.0)

    SELECTIVE PREDICTION:
      • Area Under Risk-Coverage Curve: {aurc_str}
      • Lower is better (ideal selective prediction)

    HARM-AWARE METRICS:
      • Harm Concentration Index: {concentration:.2%}
      • Total Weighted Harm: {total_harm_str}

    INTERPRETATION:
      {interp_high}
      {interp_low}
      {interp_mod}
    """

    ax5.text(
        0.1, 0.5, summary, fontsize=12, family='monospace', verticalalignment='center'
    )

    return fig
