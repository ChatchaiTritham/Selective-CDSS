"""
Performance Visualization Module

Manuscript-preparation 2D visualizations for classification performance metrics:
- Confusion matrix heatmaps
- ROC curves
- Precision-Recall curves
- Sensitivity-Specificity curves
- Threshold analysis plots
- Multi-model comparison

IEEE/Nature/JAMA compliant (300 DPI, colorblind-friendly, Times New Roman)

Author: Chatchai Tritham
Affiliation: Naresuan University
Date: 2026-01-25
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Colorblind-friendly palette (Paul Tol's vibrant scheme)
COLORS = {
    'blue': '#0077BB',
    'cyan': '#33BBEE',
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
}

# Publication settings
plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (7.0, 6.0),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix as heatmap.

    Parameters:
        confusion_matrix: 2D array of confusion matrix (TN, FP, FN, TP)
        class_names: Names for classes (default: ["Negative", "Positive"])
        normalize: If True, normalize to show proportions
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import confusion_matrix
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> fig, ax = plot_confusion_matrix(cm.to_array())
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Normalize if requested
    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
        vmin, vmax = 0, 1
    else:
        fmt = 'd'
        vmin, vmax = None, None

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), rotation=0, va='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "ROC Curve",
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ROC (Receiver Operating Characteristic) curve.

    Parameters:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        roc_auc: Area Under ROC Curve
        title: Plot title
        label: Legend label (default: "Model (AUC = X.XX)")
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure
        ax: Existing axes to plot on (optional)

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import compute_roc_curve
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        >>> roc_auc = np.trapz(tpr, fpr)
        >>> fig, ax = plot_roc_curve(fpr, tpr, roc_auc)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if label is None:
        label = f"Model (AUC = {roc_auc:.3f})"

    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['blue'], lw=2.5, label=label)

    # Plot diagonal (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        color=COLORS['grey'],
        lw=2,
        linestyle='--',
        label='Random (AUC = 0.500)',
    )

    # Styling
    ax.set_xlabel(
        'False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path and ax is None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    baseline_prevalence: Optional[float] = None,
    title: str = "Precision-Recall Curve",
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Precision-Recall curve.

    Parameters:
        precision: Precision array
        recall: Recall array
        pr_auc: Area Under PR Curve (Average Precision)
        baseline_prevalence: Class prevalence for baseline (optional)
        title: Plot title
        label: Legend label (default: "Model (AP = X.XX)")
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure
        ax: Existing axes to plot on (optional)

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import compute_pr_curve
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> precision, recall, _ = compute_pr_curve(y_true, y_prob)
        >>> pr_auc = np.trapz(precision[:-1], recall[:-1])
        >>> fig, ax = plot_pr_curve(precision, recall, pr_auc, baseline_prevalence=0.6)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if label is None:
        label = f"Model (AP = {pr_auc:.3f})"

    # Plot PR curve
    ax.plot(recall, precision, color=COLORS['blue'], lw=2.5, label=label)

    # Plot baseline (random classifier at prevalence)
    if baseline_prevalence is not None:
        ax.axhline(
            y=baseline_prevalence,
            color=COLORS['grey'],
            lw=2,
            linestyle='--',
            label=f'Baseline (Prevalence = {baseline_prevalence:.3f})',
        )

    # Styling
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path and ax is None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_sensitivity_specificity_curve(
    thresholds: np.ndarray,
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    optimal_threshold: Optional[float] = None,
    title: str = "Sensitivity-Specificity Tradeoff",
    figsize: Tuple[float, float] = (7.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot sensitivity and specificity vs. threshold.

    Parameters:
        thresholds: Threshold values
        sensitivity: Sensitivity (recall) at each threshold
        specificity: Specificity at each threshold
        optimal_threshold: Optimal threshold to highlight (optional)
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from basics_cdss.metrics import sensitivity_specificity_analysis
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        >>> df = sensitivity_specificity_analysis(y_true, y_prob)
        >>> fig, ax = plot_sensitivity_specificity_curve(
        ...     df['threshold'].values,
        ...     df['sensitivity'].values,
        ...     df['specificity'].values,
        ...     optimal_threshold=0.5
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot curves
    ax.plot(
        thresholds,
        sensitivity,
        color=COLORS['blue'],
        lw=2.5,
        marker='o',
        markersize=6,
        label='Sensitivity (TPR)',
    )
    ax.plot(
        thresholds,
        specificity,
        color=COLORS['orange'],
        lw=2.5,
        marker='s',
        markersize=6,
        label='Specificity (TNR)',
    )

    # Highlight optimal threshold
    if optimal_threshold is not None:
        # Find index closest to optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))

        ax.axvline(
            x=optimal_threshold,
            color=COLORS['red'],
            lw=2,
            linestyle='--',
            alpha=0.7,
            label=f'Optimal (ฯ = {optimal_threshold:.2f})',
        )

        # Mark points
        ax.plot(
            optimal_threshold,
            sensitivity[idx],
            'o',
            color=COLORS['blue'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        )
        ax.plot(
            optimal_threshold,
            specificity[idx],
            's',
            color=COLORS['orange'],
            markersize=12,
            markeredgecolor='black',
            markeredgewidth=2,
        )

    # Styling
    ax.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([thresholds.min() - 0.05, thresholds.max() + 0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_threshold_analysis(
    df_threshold: pd.DataFrame,
    title: str = "Threshold Analysis",
    figsize: Tuple[float, float] = (7.0, 8.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot comprehensive threshold analysis (3 panels).

    Parameters:
        df_threshold: DataFrame from sensitivity_specificity_analysis()
        title: Main title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, array of axes)

    Example:
        >>> from basics_cdss.metrics import sensitivity_specificity_analysis
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9, 0.2, 0.85])
        >>> df = sensitivity_specificity_analysis(y_true, y_prob)
        >>> fig, axes = plot_threshold_analysis(df)
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)

    # Panel (a): Sensitivity and Specificity
    axes[0].plot(
        df_threshold['threshold'],
        df_threshold['sensitivity'],
        color=COLORS['blue'],
        lw=2.5,
        marker='o',
        markersize=6,
        label='Sensitivity',
    )
    axes[0].plot(
        df_threshold['threshold'],
        df_threshold['specificity'],
        color=COLORS['orange'],
        lw=2.5,
        marker='s',
        markersize=6,
        label='Specificity',
    )
    axes[0].set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    axes[0].set_title(
        '(a) Sensitivity & Specificity vs. Threshold',
        fontsize=13,
        fontweight='bold',
        loc='left',
    )
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='best', frameon=True)
    axes[0].set_ylim([-0.05, 1.05])

    # Panel (b): Precision and F1-Score
    axes[1].plot(
        df_threshold['threshold'],
        df_threshold['precision'],
        color=COLORS['teal'],
        lw=2.5,
        marker='^',
        markersize=6,
        label='Precision',
    )
    axes[1].plot(
        df_threshold['threshold'],
        df_threshold['f1_score'],
        color=COLORS['magenta'],
        lw=2.5,
        marker='D',
        markersize=6,
        label='F1-Score',
    )
    axes[1].set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    axes[1].set_title(
        '(b) Precision & F1-Score vs. Threshold',
        fontsize=13,
        fontweight='bold',
        loc='left',
    )
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='best', frameon=True)
    axes[1].set_ylim([-0.05, 1.05])

    # Panel (c): Youden's J Statistic
    axes[2].plot(
        df_threshold['threshold'],
        df_threshold['youdens_j'],
        color=COLORS['red'],
        lw=2.5,
        marker='o',
        markersize=6,
        label="Youden's J",
    )

    # Find and mark optimal threshold (max Youden's J)
    optimal_idx = df_threshold['youdens_j'].idxmax()
    optimal_threshold = df_threshold.loc[optimal_idx, 'threshold']
    optimal_j = df_threshold.loc[optimal_idx, 'youdens_j']

    axes[2].axvline(
        x=optimal_threshold,
        color=COLORS['grey'],
        lw=2,
        linestyle='--',
        alpha=0.7,
        label=f'Optimal ฯ = {optimal_threshold:.2f}',
    )
    axes[2].plot(
        optimal_threshold,
        optimal_j,
        'o',
        color=COLORS['red'],
        markersize=12,
        markeredgecolor='black',
        markeredgewidth=2,
    )

    axes[2].set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Youden's J", fontsize=12, fontweight='bold')
    axes[2].set_title(
        "(c) Youden's J Statistic (Optimal Threshold Selection)",
        fontsize=13,
        fontweight='bold',
        loc='left',
    )
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='best', frameon=True)

    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes


def plot_multi_model_roc(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curve Comparison",
    figsize: Tuple[float, float] = (7.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ROC curves for multiple models on same axes.

    Parameters:
        models_data: Dictionary mapping model name to (fpr, tpr, auc)
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     'Model A': (fpr_a, tpr_a, auc_a),
        ...     'Model B': (fpr_b, tpr_b, auc_b),
        ... }
        >>> fig, ax = plot_multi_model_roc(models)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color cycle
    colors = [
        COLORS['blue'],
        COLORS['orange'],
        COLORS['teal'],
        COLORS['red'],
        COLORS['magenta'],
        COLORS['cyan'],
    ]

    # Plot each model
    for idx, (model_name, (fpr, tpr, auc)) in enumerate(models_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {auc:.3f})')

    # Plot diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color=COLORS['grey'],
        lw=2,
        linestyle='--',
        label='Random (AUC = 0.500)',
    )

    # Styling
    ax.set_xlabel(
        'False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_metrics_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Performance Metrics Comparison",
    figsize: Tuple[float, float] = (10.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot grouped bar chart comparing metrics across models.

    Parameters:
        metrics_dict: Dictionary mapping model name to metrics dict
        metrics_to_plot: List of metric names to include (default: all)
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> metrics = {
        ...     'Model A': {'accuracy': 0.85, 'f1_score': 0.82, 'roc_auc': 0.90},
        ...     'Model B': {'accuracy': 0.88, 'f1_score': 0.86, 'roc_auc': 0.92},
        ... }
        >>> fig, ax = plot_metrics_comparison_bar(metrics)
    """
    # Default metrics
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    # Prepare data
    models = list(metrics_dict.keys())
    n_models = len(models)
    n_metrics = len(metrics_to_plot)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = [
        COLORS['blue'],
        COLORS['orange'],
        COLORS['teal'],
        COLORS['red'],
        COLORS['magenta'],
        COLORS['cyan'],
    ]

    # Plot bars for each model
    for idx, model_name in enumerate(models):
        values = [
            metrics_dict[model_name].get(metric, 0.0) for metric in metrics_to_plot
        ]
        offset = (idx - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=colors[idx % len(colors)],
            edgecolor='black',
            linewidth=1.2,
        )

    # Styling
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace('_', ' ').title() for m in metrics_to_plot], rotation=0, ha='center'
    )
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_multi_class_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Multi-Class Confusion Matrix",
    figsize: Tuple[float, float] = (8.0, 7.0),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix for multi-class classification.

    Parameters:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
        normalize: If True, normalize to show proportions
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from sklearn.metrics import confusion_matrix as cm
        >>> y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2])
        >>> conf_mat = cm(y_true, y_pred)
        >>> fig, ax = plot_multi_class_confusion_matrix(
        ...     conf_mat, class_names=['Class A', 'Class B', 'Class C']
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Normalize if requested
    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0, va='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax
