"""
Calibration visualization functions for BASICS-CDSS.

Manuscript-preparation plots for calibration analysis including:
- Reliability diagrams
- Stratified calibration by risk tier
- Calibration comparison across models
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from basics_cdss.constants import DEFAULT_FIGURE_SIZE

DEFAULT_RELIABILITY_COLOR = "#2E86AB"
DEFAULT_TIER_COLORS = {
    "high": "#E63946",
    "medium": "#F77F00",
    "low": "#06A77D",
    "urgent": "#E63946",
    "non-urgent": "#06A77D",
}
DEFAULT_MODEL_COMPARISON_COLORS = [
    "#E63946",
    "#2E86AB",
    "#06A77D",
    "#F77F00",
    "#A23B72",
]
DEFAULT_AXIS_LIMITS = [0, 1]
DEFAULT_RELIABILITY_FIGSIZE = (8, 6)
DEFAULT_HISTOGRAM_RELIABILITY_FIGSIZE = (8, 8)
DEFAULT_STRATIFIED_FIGSIZE = (14, 5)


def plot_reliability_diagram(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = "Calibration Reliability Diagram",
    show_histogram: bool = True,
    color: str = DEFAULT_RELIABILITY_COLOR,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot calibration reliability diagram.

    Args:
        bin_confidences: Average confidence per bin
        bin_accuracies: Empirical accuracy per bin
        bin_counts: Number of samples per bin (for histogram)
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        show_histogram: Show confidence histogram below
        color: Line/marker color
        **kwargs: Additional matplotlib kwargs

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import reliability_curve
        >>> confs, accs, counts = reliability_curve(y_true, y_prob)
        >>> fig, ax = plot_reliability_diagram(confs, accs, counts)
        >>> plt.savefig("calibration.png", dpi=300)
    """
    if ax is None:
        if show_histogram and bin_counts is not None:
            fig, (ax, ax_hist) = plt.subplots(
                2,
                1,
                figsize=DEFAULT_HISTOGRAM_RELIABILITY_FIGSIZE,
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(figsize=DEFAULT_RELIABILITY_FIGSIZE)
            ax_hist = None
    else:
        fig = ax.figure
        ax_hist = None

    assert ax is not None

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration", linewidth=2, alpha=0.7)

    # Actual calibration
    ax.plot(
        bin_confidences,
        bin_accuracies,
        'o-',
        color=color,
        label="Model calibration",
        markersize=8,
        linewidth=2.5,
        **kwargs,
    )

    # Styling
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="upper left", fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(DEFAULT_AXIS_LIMITS)
    ax.set_ylim(DEFAULT_AXIS_LIMITS)
    ax.set_aspect('equal')

    # Add ECE annotation
    if len(bin_confidences) > 0:
        ece = np.mean(np.abs(bin_confidences - bin_accuracies))
        ax.text(
            0.05,
            0.95,
            f"ECE = {ece:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top',
        )

    # Histogram
    if show_histogram and bin_counts is not None and ax_hist is not None:
        ax_hist.bar(
            bin_confidences,
            bin_counts,
            width=0.08,
            color=color,
            alpha=0.6,
            edgecolor='black',
        )
        ax_hist.set_xlabel("Confidence", fontsize=14, fontweight='bold')
        ax_hist.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax_hist.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, ax


def plot_stratified_calibration(
    calibration_by_tier: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    figsize: Tuple[float, float] = DEFAULT_STRATIFIED_FIGSIZE,
    title: str = "Calibration by Risk Tier",
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[Figure, List[Axes]]:
    """Plot calibration reliability diagrams stratified by risk tier.

    Args:
        calibration_by_tier: Dict mapping tier -> (confidences, accuracies, counts)
        figsize: Figure size
        title: Overall title
        colors: Dict mapping tier -> color (uses defaults if None)

    Returns:
        Tuple of (figure, list of axes)

    Example:
        >>> from basics_cdss.metrics import stratified_calibration_metrics
        >>> cal_metrics = stratified_calibration_metrics(y_true, y_prob, risk_tiers)
        >>> tier_curves = {
        ...     tier: metrics.reliability_curve
        ...     for tier, metrics in cal_metrics.items()
        ... }
        >>> fig, axes = plot_stratified_calibration(tier_curves)
    """
    if colors is None:
        colors = DEFAULT_TIER_COLORS

    n_tiers = len(calibration_by_tier)
    fig, axes = plt.subplots(1, n_tiers, figsize=figsize, sharey=True)

    if n_tiers == 1:
        axes = [axes]

    for idx, (tier, (confs, accs, counts)) in enumerate(calibration_by_tier.items()):
        ax = axes[idx]

        # Perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)

        # Tier calibration
        tier_color = colors.get(tier.lower(), "#2E86AB")
        ax.plot(
            confs,
            accs,
            'o-',
            color=tier_color,
            markersize=8,
            linewidth=2.5,
            label=f"{tier.capitalize()} tier",
        )

        # ECE annotation
        if len(confs) > 0:
            ece = np.mean(np.abs(confs - accs))
            ax.text(
                0.05,
                0.95,
                f"ECE = {ece:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top',
            )

        # Styling
        ax.set_xlabel("Confidence", fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title(f"{tier.capitalize()} Risk", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(DEFAULT_AXIS_LIMITS)
        ax.set_ylim(DEFAULT_AXIS_LIMITS)
        ax.set_aspect('equal')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_calibration_comparison(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: Optional[Axes] = None,
    title: str = "Calibration Comparison",
    figsize: Tuple[float, float] = DEFAULT_FIGURE_SIZE,
    colors: Optional[List[str]] = None,
) -> Tuple[Figure, Axes]:
    """Compare calibration curves across multiple models.

    Args:
        models: Dict mapping model_name -> (confidences, accuracies)
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        figsize: Figure size (only used if ax is None)
        colors: List of colors for models

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     "Baseline": (confs_base, accs_base),
        ...     "TRI-X": (confs_trix, accs_trix),
        ...     "CDSS-A": (confs_a, accs_a)
        ... }
        >>> fig, ax = plot_calibration_comparison(models)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    assert ax is not None

    if colors is None:
        colors = DEFAULT_MODEL_COMPARISON_COLORS

    # Perfect calibration
    ax.plot(
        [0, 1], [0, 1], 'k--', label="Perfect calibration", linewidth=2.5, alpha=0.7
    )

    # Plot each model
    for idx, (model_name, (confs, accs)) in enumerate(models.items()):
        color = colors[idx % len(colors)]

        ax.plot(
            confs,
            accs,
            'o-',
            color=color,
            label=model_name,
            markersize=7,
            linewidth=2.5,
            alpha=0.8,
        )

        # Calculate ECE for legend
        if len(confs) > 0:
            ece = np.mean(np.abs(confs - accs))
            # Update label with ECE
            lines = ax.get_lines()
            lines[-1].set_label(f"{model_name} (ECE={ece:.3f})")

    # Styling
    ax.set_xlabel("Confidence", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(DEFAULT_AXIS_LIMITS)
    ax.set_ylim(DEFAULT_AXIS_LIMITS)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig, ax
