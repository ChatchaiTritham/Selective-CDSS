"""
XAI (Explainable AI) Visualization Plots

Visualization functions for SHAP values and counterfactual explanations.
All plots follow publication standards (IEEE/Nature/JAMA) with 300 DPI resolution,
Times New Roman font, and colorblind-friendly color schemes.

Author: Chatchai Tritham
Affiliation: Department of Computer Science and Information Technology,
             Faculty of Science, Naresuan University
Date: 2026-01-25
Version: 2.0.0 (XAI Enhancement)
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..xai.counterfactual import (CounterfactualExample, CounterfactualSet,
                                  InterventionSuggestion)
# Import XAI modules
from ..xai.shap_analysis import (FeatureImportance, SHAPInteractionValues,
                                 SHAPValues)

# Publication-quality settings
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
}

# Paul Tol's colorblind-friendly palette
COLORS_VIBRANT = [
    '#EE7733',
    '#0077BB',
    '#33BBEE',
    '#EE3377',
    '#CC3311',
    '#009988',
    '#BBBBBB',
]
COLORS_DIVERGING = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377']


def plot_shap_waterfall(
    shap_values: SHAPValues,
    sample_idx: int = 0,
    max_display: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot SHAP waterfall chart showing feature contributions.

    Waterfall plots show how each feature contributes to moving the prediction
    from the base value (expected value) to the final prediction.

    Parameters:
        shap_values: SHAP values object
        sample_idx: Index of sample to explain
        max_display: Maximum number of features to display
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> fig, ax = plot_shap_waterfall(shap_vals, sample_idx=0)
        >>> plt.show()
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if not SHAP_AVAILABLE:
        warnings.warn("SHAP library required for waterfall plots")
        return _create_placeholder_plot(
            figsize, "SHAP Waterfall Plot\n(SHAP library not installed)"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create SHAP Explanation object
    # Handle multi-output case (binary classification gives 2 outputs)
    values = shap_values.values[sample_idx]
    base_val = shap_values.base_value

    # If values is 2D (multi-output), select positive class
    if len(values.shape) > 1 and values.shape[0] > 1:
        values = values[1]  # Positive class
        if isinstance(base_val, np.ndarray):
            base_val = base_val[1]

    explanation = shap.Explanation(
        values=values,
        base_values=base_val,
        data=shap_values.data[sample_idx],
        feature_names=shap_values.feature_names,
    )

    # Generate waterfall plot
    shap.plots.waterfall(explanation, max_display=max_display, show=False)

    if title:
        plt.title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        plt.title(
            f"SHAP Waterfall Plot (Sample {sample_idx})",
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_shap_summary(
    shap_values: SHAPValues,
    plot_type: str = "dot",
    max_display: int = 20,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 7.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot SHAP summary plot (beeswarm or bar).

    Summary plots show the distribution of SHAP values for each feature across
    all samples, providing a global view of feature importance and effects.

    Parameters:
        shap_values: SHAP values object
        plot_type: Type of summary plot ('dot' for beeswarm, 'bar' for bar chart)
        max_display: Maximum number of features to display
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> fig, ax = plot_shap_summary(shap_vals, plot_type='dot')
        >>> plt.show()
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if not SHAP_AVAILABLE:
        warnings.warn("SHAP library required for summary plots")
        return _create_placeholder_plot(
            figsize, "SHAP Summary Plot\n(SHAP library not installed)"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Handle multi-output SHAP values
    values = shap_values.values
    if len(values.shape) == 3:
        values = values[:, :, 1]  # Positive class for binary classification

    # Generate summary plot
    shap.summary_plot(
        values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )

    if title:
        plt.title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        plot_type_name = "Beeswarm" if plot_type == "dot" else "Bar"
        plt.title(
            f"SHAP {plot_type_name} Summary Plot",
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_shap_bar(
    feature_importance: FeatureImportance,
    max_display: int = 15,
    title: Optional[str] = None,
    highlight_critical: bool = True,
    figsize: Tuple[float, float] = (8.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot bar chart of SHAP-based feature importance.

    Bar chart showing mean absolute SHAP values for each feature,
    highlighting critical vs non-critical features.

    Parameters:
        feature_importance: FeatureImportance object
        max_display: Maximum number of features to display
        title: Plot title (optional)
        highlight_critical: Whether to highlight critical features
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    # Get top features
    top_indices = np.argsort(-feature_importance.importance_scores)[:max_display]
    features = [feature_importance.feature_names[i] for i in top_indices.tolist()]
    scores = feature_importance.importance_scores[top_indices]

    # Determine colors
    if highlight_critical:
        colors = [
            (
                COLORS_VIBRANT[0]
                if feat in feature_importance.critical_features
                else COLORS_VIBRANT[1]
            )
            for feat in features
        ]
    else:
        colors = [COLORS_VIBRANT[0]] * len(features)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|', fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax.set_ylabel('Features', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            'Feature Importance (SHAP)',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    # Add legend if highlighting
    if highlight_critical:
        critical_patch = mpatches.Patch(
            color=COLORS_VIBRANT[0], label='Critical features'
        )
        noncritical_patch = mpatches.Patch(
            color=COLORS_VIBRANT[1], label='Non-critical features'
        )
        ax.legend(handles=[critical_patch, noncritical_patch], loc='lower right')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_shap_dependence(
    shap_values: SHAPValues,
    feature_name: str,
    interaction_feature: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 5.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot SHAP dependence plot showing feature vs SHAP value relationship.

    Dependence plots show how a feature's value affects its SHAP value,
    with optional coloring by interaction feature.

    Parameters:
        shap_values: SHAP values object
        feature_name: Feature to plot on x-axis
        interaction_feature: Feature to color by (optional)
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if not SHAP_AVAILABLE:
        warnings.warn("SHAP library required for dependence plots")
        return _create_placeholder_plot(
            figsize, "SHAP Dependence Plot\n(SHAP library not installed)"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Handle multi-output SHAP values (binary classification gives 2 outputs)
    values = shap_values.values
    # If values is 3D (n_samples, n_features, n_outputs), select positive class
    if len(values.shape) == 3:
        values = values[:, :, 1]  # Positive class

    # Generate dependence plot
    shap.dependence_plot(
        feature_name,
        values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        interaction_index=interaction_feature,
        show=False,
        ax=ax,
    )

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        if interaction_feature:
            ax.set_title(
                f'SHAP Dependence: {feature_name} (colored by {interaction_feature})',
                fontsize=PUBLICATION_STYLE['figure.titlesize'],
                pad=10,
            )
        else:
            ax.set_title(
                f'SHAP Dependence: {feature_name}',
                fontsize=PUBLICATION_STYLE['figure.titlesize'],
                pad=10,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_shap_heatmap(
    shap_values: SHAPValues,
    max_features: int = 15,
    max_samples: int = 50,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot heatmap of SHAP values.

    Heatmap showing SHAP values for top features across samples.

    Parameters:
        shap_values: SHAP values object
        max_features: Maximum number of features to display
        max_samples: Maximum number of samples to display
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    # Handle multi-output SHAP values
    values = shap_values.values
    if len(values.shape) == 3:
        values = values[:, :, 1]  # Positive class

    # Select top features by mean absolute SHAP value
    mean_abs_shap = np.abs(values).mean(axis=0)
    # Handle if mean_abs_shap is still multi-dimensional
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=-1)

    top_feature_indices = np.argsort(-mean_abs_shap)[:max_features]
    top_features = [shap_values.feature_names[i] for i in top_feature_indices.tolist()]

    # Select samples
    num_samples = min(max_samples, values.shape[0])
    sample_indices = np.linspace(0, values.shape[0] - 1, num_samples, dtype=int)

    # Extract data
    shap_subset = values[np.ix_(sample_indices, top_feature_indices)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    im = ax.imshow(
        shap_subset.T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-np.abs(shap_subset).max(),
        vmax=np.abs(shap_subset).max(),
    )

    # Set ticks and labels
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Samples', fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax.set_ylabel('Features', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            'SHAP Values Heatmap',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SHAP value', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_shap_interaction_heatmap(
    interaction_values: SHAPInteractionValues,
    top_k: int = 15,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 9.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot heatmap of SHAP interaction values.

    Shows pairwise feature interactions as a heatmap.

    Parameters:
        interaction_values: SHAP interaction values
        top_k: Number of top features to include
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    # Average interaction values across samples
    interaction_vals = interaction_values.values

    # Handle multi-output (binary classification)
    if (
        len(interaction_vals.shape) == 4
    ):  # (n_samples, n_features, n_features, n_outputs)
        interaction_vals = interaction_vals[:, :, :, 1]  # Select positive class

    mean_interactions = np.abs(interaction_vals).mean(axis=0)

    # Get top features by total interaction strength
    total_interaction = mean_interactions.sum(axis=1)

    # Flatten if needed
    if len(total_interaction.shape) > 1:
        total_interaction = total_interaction.flatten()

    top_indices = np.argsort(-total_interaction)[:top_k]
    top_features = [
        interaction_values.feature_names[int(top_indices[i])]
        for i in range(len(top_indices))
    ]

    # Extract subset
    interaction_subset = mean_interactions[np.ix_(top_indices, top_indices)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    im = ax.imshow(interaction_subset, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_yticklabels(top_features)

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=15)
    else:
        ax.set_title(
            'SHAP Interaction Values',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=15,
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        'Mean |interaction value|', fontsize=PUBLICATION_STYLE['axes.labelsize']
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


# ============================================================================
# Counterfactual Visualization Functions
# ============================================================================


def plot_counterfactual_comparison(
    counterfactual: CounterfactualExample,
    max_features: int = 15,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot comparison of original vs counterfactual features.

    Side-by-side comparison showing which features changed and by how much.

    Parameters:
        counterfactual: Counterfactual explanation
        max_features: Maximum number of features to display
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    # Get features that changed
    changed_features = list(counterfactual.feature_changes.keys())[:max_features]

    if len(changed_features) == 0:
        return _create_placeholder_plot(figsize, "No feature changes in counterfactual")

    original_values = [counterfactual.feature_changes[f][0] for f in changed_features]
    cf_values = [counterfactual.feature_changes[f][1] for f in changed_features]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(len(changed_features))
    width = 0.35

    ax.barh(
        x - width / 2,
        original_values,
        width,
        label='Original',
        color=COLORS_VIBRANT[1],
        edgecolor='black',
        linewidth=0.5,
    )
    ax.barh(
        x + width / 2,
        cf_values,
        width,
        label='Counterfactual',
        color=COLORS_VIBRANT[0],
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_yticks(x)
    ax.set_yticklabels(changed_features)
    ax.set_xlabel('Feature Value', fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax.legend(loc='best')

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            f'Counterfactual Comparison\n'
            f'Original: Class {counterfactual.original_prediction} → '
            f'Counterfactual: Class {counterfactual.counterfactual_prediction}',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_feature_changes(
    counterfactual: CounterfactualExample,
    show_percentage: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot magnitude of feature changes.

    Bar chart showing how much each feature needs to change.

    Parameters:
        counterfactual: Counterfactual explanation
        show_percentage: Whether to show percentage change
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if len(counterfactual.feature_changes) == 0:
        return _create_placeholder_plot(figsize, "No feature changes")

    features = list(counterfactual.feature_changes.keys())
    changes = []

    for feat in features:
        old, new = counterfactual.feature_changes[feat]
        if show_percentage and old != 0:
            change = abs((new - old) / old) * 100
        else:
            change = abs(new - old)
        changes.append(change)

    # Sort by magnitude
    sorted_indices = np.argsort(-np.array(changes))
    features = [features[i] for i in sorted_indices]
    changes = [changes[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    y_pos = np.arange(len(features))
    ax.barh(y_pos, changes, color=COLORS_VIBRANT[0], edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()

    xlabel = 'Change (%)' if show_percentage else 'Absolute Change'
    ax.set_xlabel(xlabel, fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax.set_ylabel('Features', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            'Feature Changes for Counterfactual',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_intervention_priority(
    interventions: List[InterventionSuggestion],
    max_display: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot clinical intervention suggestions ranked by priority.

    Shows recommended interventions with their magnitudes and priorities.

    Parameters:
        interventions: List of intervention suggestions
        max_display: Maximum number to display
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if len(interventions) == 0:
        return _create_placeholder_plot(figsize, "No interventions")

    # Limit display
    interventions = interventions[:max_display]

    features = [interv.feature_name for interv in interventions]
    magnitudes = [interv.change_magnitude for interv in interventions]
    priorities = [interv.priority for interv in interventions]

    # Color by priority (lower number = higher priority = brighter color)
    norm_priorities = (
        np.array(priorities) / max(priorities)
        if max(priorities) > 0
        else np.zeros(len(priorities))
    )
    colors = plt.cm.RdYlGn_r(1 - norm_priorities)  # Reverse so high priority is red

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, magnitudes, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p}. {f}" for p, f in zip(priorities, features)])
    ax.invert_yaxis()
    ax.set_xlabel('Change Magnitude', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            'Clinical Intervention Priority',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    # Add value labels
    for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
        ax.text(
            mag,
            bar.get_y() + bar.get_height() / 2,
            f' {mag:.2f}',
            va='center',
            fontsize=8,
        )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_whatif_curve(
    whatif_df: pd.DataFrame,
    feature_name: str,
    prediction_column: str = 'prediction_proba',
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 5.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot what-if analysis curve.

    Shows how prediction changes as a feature varies.

    Parameters:
        whatif_df: DataFrame from whatif_analysis()
        feature_name: Name of varied feature
        prediction_column: Column name for predictions
        threshold: Decision threshold to visualize (optional)
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot curve
    ax.plot(
        whatif_df[feature_name],
        whatif_df[prediction_column],
        color=COLORS_VIBRANT[0],
        linewidth=2,
        label='Prediction',
    )

    # Add threshold line if provided
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color=COLORS_VIBRANT[3],
            linestyle='--',
            linewidth=1.5,
            label=f'Threshold ({threshold:.2f})',
        )

    ax.set_xlabel(feature_name, fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax.set_ylabel('Predicted Probability', fontsize=PUBLICATION_STYLE['axes.labelsize'])

    if title:
        ax.set_title(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], pad=10)
    else:
        ax.set_title(
            f'What-If Analysis: {feature_name}',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            pad=10,
        )

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, ax


def plot_counterfactual_diversity(
    cf_set: CounterfactualSet,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 7.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot diversity of counterfactual set.

    Visualizes how diverse the counterfactual explanations are.

    Parameters:
        cf_set: CounterfactualSet object
        title: Plot title (optional)
        figsize: Figure size in inches
        save_path: Path to save figure (optional)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes)
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    if cf_set.num_counterfactuals == 0:
        return _create_placeholder_plot(figsize, "No counterfactuals in set")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Plot 1: Number of changed features per counterfactual
    num_changes = [len(cf.feature_changes) for cf in cf_set.counterfactuals]

    ax1.bar(
        range(cf_set.num_counterfactuals),
        num_changes,
        color=COLORS_VIBRANT[0],
        edgecolor='black',
        linewidth=0.5,
    )
    ax1.set_xlabel('Counterfactual Index', fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax1.set_ylabel(
        'Number of Changed Features', fontsize=PUBLICATION_STYLE['axes.labelsize']
    )
    ax1.set_title(
        'Features Changed per Counterfactual',
        fontsize=PUBLICATION_STYLE['axes.titlesize'],
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Plot 2: Distance from original
    distances = [cf.distance for cf in cf_set.counterfactuals]

    ax2.bar(
        range(cf_set.num_counterfactuals),
        distances,
        color=COLORS_VIBRANT[1],
        edgecolor='black',
        linewidth=0.5,
    )
    ax2.set_xlabel('Counterfactual Index', fontsize=PUBLICATION_STYLE['axes.labelsize'])
    ax2.set_ylabel(
        'Distance from Original', fontsize=PUBLICATION_STYLE['axes.labelsize']
    )
    ax2.set_title('Distance Diversity', fontsize=PUBLICATION_STYLE['axes.titlesize'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    if title:
        fig.suptitle(title, fontsize=PUBLICATION_STYLE['figure.titlesize'], y=1.02)
    else:
        fig.suptitle(
            f'Counterfactual Diversity (Score: {cf_set.diversity_score:.3f})',
            fontsize=PUBLICATION_STYLE['figure.titlesize'],
            y=1.02,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')

    return fig, (ax1, ax2)


def _create_placeholder_plot(
    figsize: Tuple[float, float], message: str
) -> Tuple[plt.Figure, plt.Axes]:
    """Create placeholder plot with message."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.text(
        0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=12
    )
    ax.axis('off')
    return fig, ax
