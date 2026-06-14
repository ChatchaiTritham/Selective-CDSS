"""
Clinical Metrics Visualization Module

Provides manuscript-preparation 2D and 3D visualization functions for:
- Clinical Utility Metrics (Decision Curves, Net Benefit, NNT)
- Fairness Metrics (Demographic Parity, Equalized Odds, Calibration)
- Conformal Prediction (Prediction Sets, Coverage, Intervals)

All plots follow publication standards:
- 300 DPI resolution
- Times New Roman font
- Colorblind-friendly palettes (Paul Tol's schemes)
- IEEE/Nature/JAMA compliant formatting
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Publication style settings
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
}

# Colorblind-friendly palette (Paul Tol's bright scheme)
COLORS = {
    'blue': '#4477AA',
    'cyan': '#66CCEE',
    'green': '#228833',
    'yellow': '#CCBB44',
    'red': '#EE6677',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
}

# Clinical palette
CLINICAL_COLORS = {
    'model': COLORS['blue'],
    'treat_all': COLORS['red'],
    'treat_none': COLORS['grey'],
    'positive': COLORS['red'],
    'negative': COLORS['blue'],
    'uncertain': COLORS['yellow'],
}


# ============================================================================
# Clinical Utility Metrics Visualization
# ============================================================================


def plot_decision_curve(
    dca_result,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Decision Curve Analysis (DCA) showing net benefit vs threshold.

    The decision curve shows the clinical usefulness of a prediction model
    across a range of probability thresholds, compared to "treat all" and
    "treat none" strategies.

    Args:
        dca_result: DecisionCurveResult object from decision_curve_analysis()
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
        save_path: Optional path to save the figure
        title: Optional custom title

    Returns:
        Tuple of (figure, axes)

    Clinical Interpretation:
        - Model curve above both alternatives: Model is clinically useful
        - Useful threshold range: Where model outperforms alternatives
        - Higher net benefit: Better clinical value
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot curves
    ax.plot(
        dca_result.thresholds,
        dca_result.net_benefit_model,
        label='Prediction Model',
        color=CLINICAL_COLORS['model'],
        linewidth=2,
        zorder=3,
    )
    ax.plot(
        dca_result.thresholds,
        dca_result.net_benefit_all,
        label='Treat All',
        color=CLINICAL_COLORS['treat_all'],
        linestyle='--',
        linewidth=1.5,
        zorder=2,
    )
    ax.plot(
        dca_result.thresholds,
        dca_result.net_benefit_none,
        label='Treat None',
        color=CLINICAL_COLORS['treat_none'],
        linestyle=':',
        linewidth=1.5,
        zorder=1,
    )

    # Highlight useful threshold range
    if not np.isnan(dca_result.threshold_range[0]):
        t_min, t_max = dca_result.threshold_range
        ax.axvspan(
            t_min, t_max, alpha=0.1, color=COLORS['green'], label='Model Useful Range'
        )

    # Formatting
    ax.set_xlabel('Probability Threshold', fontweight='bold')
    ax.set_ylabel('Net Benefit', fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')
    else:
        ax.set_title('Decision Curve Analysis', fontweight='bold')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_standardized_net_benefit(
    dca_result,
    threshold: float = 0.3,
    figsize: Tuple[float, float] = (6, 4),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot standardized net benefit (per 100 patients) at a specific threshold.

    Args:
        dca_result: DecisionCurveResult object
        threshold: Specific threshold to highlight
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    # Find closest threshold
    idx = np.argmin(np.abs(dca_result.thresholds - threshold))
    thresh_actual = dca_result.thresholds[idx]

    # Net benefit at this threshold
    nb_model = dca_result.standardized_net_benefit[idx]
    nb_all_val = dca_result.net_benefit_all[idx] * 100
    nb_none_val = 0.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    strategies = ['Model', 'Treat All', 'Treat None']
    net_benefits = [nb_model, nb_all_val, nb_none_val]
    colors = [
        CLINICAL_COLORS['model'],
        CLINICAL_COLORS['treat_all'],
        CLINICAL_COLORS['treat_none'],
    ]

    bars = ax.bar(
        strategies, net_benefits, color=colors, edgecolor='black', linewidth=1
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    ax.set_ylabel('Net Benefit (per 100 patients)', fontweight='bold')
    ax.set_title(
        f'Standardized Net Benefit at Threshold = {thresh_actual:.2f}',
        fontweight='bold',
    )
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_nnt_comparison(
    nnt_results: Dict[str, 'NNTResult'],
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Number Needed to Treat (NNT) comparison across models/strategies.

    Lower NNT is better (fewer patients need treatment to prevent one event).

    Args:
        nnt_results: Dictionary mapping model names to NNTResult objects
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    models = list(nnt_results.keys())
    nnts = [nnt_results[m].nnt for m in models]
    arrs = [nnt_results[m].arr for m in models]

    # Color bars by NNT value (green = good, yellow = moderate, red = poor)
    colors = []
    for nnt in nnts:
        if nnt < 10:
            colors.append(COLORS['green'])
        elif nnt < 20:
            colors.append(COLORS['yellow'])
        else:
            colors.append(COLORS['red'])

    bars = ax.barh(models, nnts, color=colors, edgecolor='black', linewidth=1)

    # Add NNT values
    for i, (bar, nnt) in enumerate(zip(bars, nnts)):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f'NNT = {nnt:.1f}',
            va='center',
            fontweight='bold',
        )

    ax.set_xlabel('Number Needed to Treat (NNT)', fontweight='bold')
    ax.set_title('NNT Comparison (Lower is Better)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # Add reference lines
    ax.axvline(
        x=10,
        color=COLORS['green'],
        linestyle='--',
        alpha=0.5,
        label='Excellent (NNT < 10)',
    )
    ax.axvline(
        x=20,
        color=COLORS['yellow'],
        linestyle='--',
        alpha=0.5,
        label='Moderate (NNT 10-20)',
    )

    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_clinical_impact(
    impact_result,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot clinical impact assessment showing classification breakdown.

    Args:
        impact_result: ClinicalImpactResult object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Left plot: Classification breakdown
    categories = [
        'True\nPositive',
        'False\nPositive',
        'True\nNegative',
        'False\nNegative',
    ]
    counts = [
        impact_result.n_true_positives,
        impact_result.n_false_positives,
        impact_result.n_true_negatives,
        impact_result.n_false_negatives,
    ]
    colors_cm = [COLORS['green'], COLORS['red'], COLORS['blue'], COLORS['purple']]

    bars = ax1.bar(categories, counts, color=colors_cm, edgecolor='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    ax1.set_ylabel('Number of Patients', fontweight='bold')
    ax1.set_title('Classification Breakdown', fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # Right plot: Key metrics
    metrics = [
        'PPV\n(Precision)',
        'NPV',
        f'NNS\n({impact_result.number_needed_to_screen:.1f})',
    ]
    values = [impact_result.ppv, impact_result.npv, 1.0]  # Normalize NNS for display
    colors_met = [COLORS['green'], COLORS['blue'], COLORS['yellow']]

    bars2 = ax2.bar(metrics, values, color=colors_met, edgecolor='black', linewidth=1)

    for i, (bar, val) in enumerate(zip(bars2, values)):
        if i < 2:  # PPV and NPV
            label_text = f'{val:.3f}'
        else:  # NNS
            label_text = f'{impact_result.number_needed_to_screen:.1f}'
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2,
            label_text,
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=11,
        )

    ax2.set_ylabel('Value', fontweight='bold')
    ax2.set_title('Key Performance Metrics', fontweight='bold')
    ax2.set_ylim([0, 1.2])

    plt.suptitle(
        f'Clinical Impact at Threshold = {impact_result.threshold:.2f}',
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, (ax1, ax2)


def plot_clinical_impact_3d(
    impact_results: List['ClinicalImpactResult'],
    thresholds: np.ndarray,
    figsize: Tuple[float, float] = (10, 7),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Axes3D]:
    """Plot 3D surface of clinical impact metrics across thresholds.

    Args:
        impact_results: List of ClinicalImpactResult objects
        thresholds: Array of threshold values
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, 3d axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Extract metrics
    ppvs = [r.ppv for r in impact_results]
    npvs = [r.npv for r in impact_results]
    pct_high_risk = [r.percent_high_risk for r in impact_results]

    # Create surface
    X = thresholds
    Y = np.arange(3)  # PPV, NPV, % High Risk
    X, Y = np.meshgrid(X, Y)

    Z = np.array([ppvs, npvs, pct_high_risk])

    # Plot surfaces
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    ax.set_xlabel('Probability Threshold', fontweight='bold', labelpad=10)
    ax.set_ylabel('Metric', fontweight='bold', labelpad=10)
    ax.set_zlabel('Value', fontweight='bold', labelpad=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['PPV', 'NPV', '% High Risk'])
    ax.set_title('Clinical Impact 3D Analysis', fontweight='bold', pad=20)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


# ============================================================================
# Fairness Metrics Visualization
# ============================================================================


def plot_demographic_parity(
    dp_result,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot demographic parity comparison across groups.

    Args:
        dp_result: DemographicParityResult object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    groups = list(dp_result.group_positive_rates.keys())
    rates = list(dp_result.group_positive_rates.values())

    # Color by fairness
    colors = [COLORS['green'] if dp_result.is_fair else COLORS['red']] * len(groups)
    colors[groups.index(dp_result.reference_group)] = COLORS['blue']

    bars = ax.bar(
        groups, rates, color=colors, edgecolor='black', linewidth=1, alpha=0.7
    )

    # Add rate labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    # Add parity line (mean rate)
    mean_rate = np.mean(rates)
    ax.axhline(
        y=mean_rate,
        color=COLORS['purple'],
        linestyle='--',
        linewidth=2,
        label=f'Mean Rate: {mean_rate:.3f}',
    )

    ax.set_ylabel('Positive Prediction Rate', fontweight='bold')
    ax.set_xlabel('Protected Group', fontweight='bold')
    ax.set_title(
        f'Demographic Parity Assessment (ฮ” = {dp_result.parity_difference:.3f})',
        fontweight='bold',
    )
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add fairness annotation
    fairness_text = 'FAIR' if dp_result.is_fair else 'UNFAIR'
    fairness_color = COLORS['green'] if dp_result.is_fair else COLORS['red']
    ax.text(
        0.95,
        0.95,
        fairness_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color=fairness_color,
        ha='right',
        va='top',
        bbox=dict(
            boxstyle='round', facecolor='white', edgecolor=fairness_color, linewidth=2
        ),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_equalized_odds(
    eo_result,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot equalized odds showing TPR and FPR across groups.

    Args:
        eo_result: EqualizedOddsResult object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    groups = list(eo_result.group_tpr.keys())
    tprs = list(eo_result.group_tpr.values())
    fprs = list(eo_result.group_fpr.values())

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        tprs,
        width,
        label='TPR (Sensitivity)',
        color=COLORS['green'],
        edgecolor='black',
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        fprs,
        width,
        label='FPR (1 - Specificity)',
        color=COLORS['red'],
        edgecolor='black',
        linewidth=1,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
            )

    ax.set_ylabel('Rate', fontweight='bold')
    ax.set_xlabel('Protected Group', fontweight='bold')
    ax.set_title(
        f'Equalized Odds (TPR ฮ” = {eo_result.tpr_difference:.3f}, '
        f'FPR ฮ” = {eo_result.fpr_difference:.3f})',
        fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Fairness annotation
    fairness_text = 'FAIR' if eo_result.is_fair else 'UNFAIR'
    fairness_color = COLORS['green'] if eo_result.is_fair else COLORS['red']
    ax.text(
        0.95,
        0.95,
        fairness_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color=fairness_color,
        ha='right',
        va='top',
        bbox=dict(
            boxstyle='round', facecolor='white', edgecolor=fairness_color, linewidth=2
        ),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_disparate_impact(
    di_results: List['DisparateImpactResult'],
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot disparate impact ratios with 80% rule reference line.

    Args:
        di_results: List of DisparateImpactResult objects
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    comparisons = [
        f"{r.unprivileged_group}\nvs\n{r.privileged_group}" for r in di_results
    ]
    ratios = [r.disparate_impact_ratio for r in di_results]

    # Color by fairness (0.8 โค DI โค 1.25 is fair)
    colors = []
    for r in di_results:
        if r.is_fair:
            colors.append(COLORS['green'])
        else:
            colors.append(COLORS['red'])

    bars = ax.barh(comparisons, ratios, color=colors, edgecolor='black', linewidth=1)

    # Add ratio labels
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f'{ratio:.3f}',
            va='center',
            fontweight='bold',
        )

    # Add reference lines
    ax.axvline(
        x=0.8,
        color=COLORS['purple'],
        linestyle='--',
        linewidth=2,
        label='80% Rule (4/5ths)',
        alpha=0.7,
    )
    ax.axvline(
        x=1.0,
        color='black',
        linestyle='-',
        linewidth=1,
        alpha=0.5,
        label='Perfect Parity',
    )
    ax.axvline(x=1.25, color=COLORS['purple'], linestyle='--', linewidth=2, alpha=0.7)

    # Shade fair region
    ax.axvspan(0.8, 1.25, alpha=0.1, color=COLORS['green'], label='Fair Range')

    ax.set_xlabel('Disparate Impact Ratio', fontweight='bold')
    ax.set_title('Disparate Impact Analysis', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim([0, max(2.0, max(ratios) + 0.2)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_calibration_by_group(
    calib_result,
    figsize: Tuple[float, float] = (7, 7),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot calibration curves for each protected group.

    Args:
        calib_result: CalibrationResult object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration', alpha=0.5)

    # Plot each group
    color_list = list(COLORS.values())
    for i, (group, (pred, obs)) in enumerate(calib_result.group_calibration.items()):
        color = color_list[i % len(color_list)]
        ece = calib_result.calibration_error[group]
        ax.plot(
            pred,
            obs,
            'o-',
            color=color,
            linewidth=2,
            markersize=6,
            label=f'{group} (ECE={ece:.3f})',
            alpha=0.8,
        )

    ax.set_xlabel('Predicted Probability', fontweight='bold')
    ax.set_ylabel('Observed Frequency', fontweight='bold')
    ax.set_title('Calibration by Protected Group', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    # Fairness annotation
    fairness_text = 'CALIBRATED' if calib_result.is_calibrated else 'MISCALIBRATED'
    fairness_color = COLORS['green'] if calib_result.is_calibrated else COLORS['red']
    ax.text(
        0.95,
        0.05,
        fairness_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        color=fairness_color,
        ha='right',
        va='bottom',
        bbox=dict(
            boxstyle='round', facecolor='white', edgecolor=fairness_color, linewidth=2
        ),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_fairness_radar(
    fairness_report,
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot radar chart showing all fairness metrics.

    Args:
        fairness_report: FairnessReport object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(
        figsize=figsize, dpi=dpi, subplot_kw=dict(projection='polar')
    )

    # Metrics (convert to scores where 1 = perfect fairness)
    metrics = [
        'Demographic\nParity',
        'Equalized\nOdds',
        'Equal\nOpportunity',
        'Calibration',
    ]
    scores = [
        1 - fairness_report.demographic_parity.parity_difference,
        1 - fairness_report.equalized_odds.avg_odds_difference,
        1 - fairness_report.equal_opportunity.tpr_difference,
        1 - fairness_report.calibration.max_calibration_error,
    ]

    # If disparate impact available
    if fairness_report.disparate_impact:
        metrics.append('Disparate\nImpact')
        di_score = min(
            fairness_report.disparate_impact.disparate_impact_ratio,
            1 / fairness_report.disparate_impact.disparate_impact_ratio,
        )
        scores.append(di_score)

    # Normalize scores to [0, 1]
    scores = np.clip(scores, 0, 1).tolist()

    # Number of variables
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores = scores + scores[:1]  # Complete the circle
    angles = angles + angles[:1]

    ax.plot(angles, scores, 'o-', linewidth=2, color=COLORS['blue'], markersize=8)
    ax.fill(angles, scores, alpha=0.25, color=COLORS['blue'])

    # Reference circle at 0.8 (good fairness threshold)
    ax.plot(
        angles,
        [0.8] * len(angles),
        'r--',
        linewidth=1,
        alpha=0.5,
        label='Fairness Threshold',
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)

    ax.set_title(
        'Fairness Assessment Radar Chart', fontweight='bold', pad=20, fontsize=14
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


# ============================================================================
# Conformal Prediction Visualization
# ============================================================================


def plot_prediction_set_sizes(
    conf_result,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histogram of prediction set sizes from conformal prediction.

    Args:
        conf_result: ConformalPredictionSet object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        conf_result.set_sizes,
        bins=np.arange(conf_result.set_sizes.max() + 2) - 0.5,
        color=COLORS['blue'],
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
    )

    ax.set_xlabel('Prediction Set Size', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(
        f'Conformal Prediction Set Sizes (Avg = {conf_result.efficiency:.2f})',
        fontweight='bold',
    )
    ax.grid(True, axis='y', alpha=0.3)

    # Add stats text
    stats_text = f'Target Coverage: {conf_result.target_coverage:.1%}\n'
    stats_text += f'Average Set Size: {conf_result.efficiency:.2f}\n'
    stats_text += f'Singleton Sets: {(conf_result.set_sizes == 1).sum()}/{len(conf_result.set_sizes)}'

    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_conformal_intervals(
    conf_interval,
    y_true: Optional[np.ndarray] = None,
    max_samples: int = 50,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot prediction intervals from conformal regression.

    Args:
        conf_interval: ConformalInterval object
        y_true: True target values (optional, for coverage visualization)
        max_samples: Maximum number of samples to plot
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n_samples = min(len(conf_interval.point_predictions), max_samples)
    x = np.arange(n_samples)

    # Plot intervals
    ax.errorbar(
        x,
        conf_interval.point_predictions[:n_samples],
        yerr=[
            conf_interval.point_predictions[:n_samples]
            - conf_interval.lower_bounds[:n_samples],
            conf_interval.upper_bounds[:n_samples]
            - conf_interval.point_predictions[:n_samples],
        ],
        fmt='o',
        color=COLORS['blue'],
        ecolor=COLORS['grey'],
        elinewidth=2,
        capsize=3,
        markersize=4,
        label='Predicted ยฑ Interval',
    )

    # Plot true values if provided
    if y_true is not None:
        ax.scatter(
            x,
            y_true[:n_samples],
            color=COLORS['red'],
            s=50,
            zorder=5,
            marker='x',
            linewidths=2,
            label='True Value',
        )

        # Check coverage
        covered = (y_true[:n_samples] >= conf_interval.lower_bounds[:n_samples]) & (
            y_true[:n_samples] <= conf_interval.upper_bounds[:n_samples]
        )
        coverage = covered.mean()
    else:
        coverage = np.nan

    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel('Target Value', fontweight='bold')
    title = (
        f'Conformal Prediction Intervals ({conf_interval.target_coverage:.0%} Coverage)'
    )
    if not np.isnan(coverage):
        title += f'\nEmpirical Coverage: {coverage:.1%}'
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_coverage_vs_alpha(
    alphas: np.ndarray,
    coverages: np.ndarray,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot empirical coverage vs target coverage (1 - alpha).

    Args:
        alphas: Array of miscoverage rates
        coverages: Array of empirical coverage rates
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    target_coverages = 1 - alphas

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Coverage', alpha=0.5)

    # Empirical coverage
    ax.plot(
        target_coverages,
        coverages,
        'o-',
        color=COLORS['blue'],
        linewidth=2,
        markersize=6,
        label='Empirical Coverage',
    )

    ax.set_xlabel('Target Coverage (1 - ฮฑ)', fontweight='bold')
    ax.set_ylabel('Empirical Coverage', fontweight='bold')
    ax.set_title('Conformal Prediction Coverage Guarantee', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_adaptive_efficiency_3d(
    adaptive_result,
    figsize: Tuple[float, float] = (10, 7),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Axes3D]:
    """Plot 3D visualization of adaptive conformal prediction efficiency.

    Args:
        adaptive_result: AdaptiveConformalResult object
        figsize: Figure size
        dpi: Resolution
        save_path: Optional save path

    Returns:
        Tuple of (figure, 3d axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: difficulty vs set size vs sample index
    n_samples = len(adaptive_result.difficulty_scores)
    x = np.arange(n_samples)
    y = adaptive_result.difficulty_scores
    z = adaptive_result.set_sizes

    scatter = ax.scatter(
        x, y, z, c=z, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5
    )

    ax.set_xlabel('Sample Index', fontweight='bold', labelpad=10)
    ax.set_ylabel('Difficulty Score', fontweight='bold', labelpad=10)
    ax.set_zlabel('Prediction Set Size', fontweight='bold', labelpad=10)
    ax.set_title('Adaptive Conformal Prediction Efficiency', fontweight='bold', pad=20)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Set Size', fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax
