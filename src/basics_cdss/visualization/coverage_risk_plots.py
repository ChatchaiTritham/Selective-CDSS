"""
Coverage-risk visualization functions for selective prediction analysis.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_coverage_risk_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "Coverage-Risk Curve",
    color: str = "#2E86AB",
    fill_alpha: float = 0.2,
    highlight_points: Optional[List[Tuple[float, str]]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot coverage-risk curve for selective prediction.

    Args:
        coverages: Coverage values (fraction retained)
        risks: Conditional risk values
        ax: Matplotlib axes
        title: Plot title
        color: Curve color
        fill_alpha: Alpha for area under curve
        highlight_points: List of (coverage, label) to highlight
        **kwargs: Additional matplotlib kwargs

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import coverage_risk_curve
        >>> covs, risks, _ = coverage_risk_curve(y_true, y_prob)
        >>> fig, ax = plot_coverage_risk_curve(
        ...     covs, risks,
        ...     highlight_points=[(0.8, "Target coverage")]
        ... )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Remove NaN values
    valid_mask = ~np.isnan(risks)
    coverages_clean = coverages[valid_mask]
    risks_clean = risks[valid_mask]

    if len(coverages_clean) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', fontsize=14)
        return fig, ax

    # Plot curve
    ax.plot(
        coverages_clean,
        risks_clean,
        'o-',
        color=color,
        linewidth=2.5,
        markersize=6,
        label="Coverage-Risk curve",
        **kwargs,
    )

    # Fill area under curve
    ax.fill_between(
        coverages_clean,
        0,
        risks_clean,
        alpha=fill_alpha,
        color=color,
        label=f"AURC (area under curve)",
    )

    # Highlight specific coverage points
    if highlight_points:
        for target_cov, label in highlight_points:
            # Find closest coverage point
            idx = np.argmin(np.abs(coverages_clean - target_cov))
            cov_point = coverages_clean[idx]
            risk_point = risks_clean[idx]

            ax.plot(cov_point, risk_point, 'r*', markersize=15, zorder=10)
            ax.axvline(cov_point, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(
                cov_point,
                ax.get_ylim()[1] * 0.95,
                f"{label}\n({cov_point:.2f}, {risk_point:.3f})",
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            )

    # Calculate AURC
    try:
        aurc = np.trapezoid(risks_clean, coverages_clean)
    except AttributeError:
        aurc = np.trapz(risks_clean, coverages_clean)

    # Styling
    ax.set_xlabel(
        "Coverage (fraction of predictions retained)", fontsize=13, fontweight='bold'
    )
    ax.set_ylabel("Conditional Risk", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(risks_clean) * 1.1])

    # AURC annotation
    ax.text(
        0.95,
        0.95,
        f"AURC = {aurc:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        verticalalignment='top',
        horizontalalignment='right',
    )

    ax.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()

    return fig, ax


def plot_selective_prediction_comparison(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (12, 7),
    title: str = "Selective Prediction Comparison",
    colors: Optional[List[str]] = None,
) -> Tuple[Figure, Axes]:
    """Compare coverage-risk curves across multiple models.

    Args:
        models: Dict mapping model_name -> (coverages, risks)
        figsize: Figure size
        title: Plot title
        colors: List of colors

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     "Baseline": (cov_base, risk_base),
        ...     "TRI-X": (cov_trix, risk_trix)
        ... }
        >>> fig, ax = plot_selective_prediction_comparison(models)
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = ["#E63946", "#2E86AB", "#06A77D", "#F77F00", "#A23B72"]

    for idx, (model_name, (coverages, risks)) in enumerate(models.items()):
        color = colors[idx % len(colors)]

        # Remove NaN
        valid_mask = ~np.isnan(risks)
        cov_clean = coverages[valid_mask]
        risk_clean = risks[valid_mask]

        if len(cov_clean) == 0:
            continue

        # Calculate AURC
        try:
            aurc = np.trapezoid(risk_clean, cov_clean)
        except AttributeError:
            aurc = np.trapz(risk_clean, cov_clean)

        ax.plot(
            cov_clean,
            risk_clean,
            'o-',
            color=color,
            linewidth=2.5,
            markersize=5,
            label=f"{model_name} (AURC={aurc:.3f})",
            alpha=0.8,
        )

    ax.set_xlabel("Coverage", fontsize=13, fontweight='bold')
    ax.set_ylabel("Conditional Risk", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    return fig, ax


def plot_abstention_analysis(
    confidence_scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[Figure, List[Axes]]:
    """Analyze abstention behavior across confidence thresholds.

    Args:
        confidence_scores: Model confidence scores
        y_true: Ground truth labels
        thresholds: Confidence thresholds to evaluate
        figsize: Figure size

    Returns:
        Tuple of (figure, list of axes)

    Example:
        >>> fig, axes = plot_abstention_analysis(y_prob, y_true)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    abstention_rates = []
    retained_accuracies = []
    retained_counts = []

    for tau in thresholds:
        # Predictions above threshold
        retained_mask = confidence_scores >= tau
        n_retained = retained_mask.sum()

        abstention_rate = 1.0 - (n_retained / len(confidence_scores))
        abstention_rates.append(abstention_rate)
        retained_counts.append(n_retained)

        if n_retained > 0:
            # Accuracy on retained predictions
            y_pred = (confidence_scores[retained_mask] >= 0.5).astype(int)
            accuracy = (y_pred == y_true[retained_mask]).mean()
            retained_accuracies.append(accuracy)
        else:
            retained_accuracies.append(np.nan)

    # Plot 1: Abstention rate vs threshold
    ax1.plot(thresholds, abstention_rates, 'o-', color='#E63946', linewidth=2.5)
    ax1.set_xlabel("Confidence Threshold", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Abstention Rate", fontsize=12, fontweight='bold')
    ax1.set_title("Abstention Rate", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Retained accuracy vs threshold
    ax2.plot(thresholds, retained_accuracies, 'o-', color='#06A77D', linewidth=2.5)
    ax2.set_xlabel("Confidence Threshold", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Accuracy (on retained)", fontsize=12, fontweight='bold')
    ax2.set_title("Retained Accuracy", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Sample count vs threshold
    ax3.plot(thresholds, retained_counts, 'o-', color='#2E86AB', linewidth=2.5)
    ax3.set_xlabel("Confidence Threshold", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Samples Retained", fontsize=12, fontweight='bold')
    ax3.set_title("Retention Count", fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, [ax1, ax2, ax3]
