"""
Advanced Charts Module

Advanced 2D/3D visualization for CDSS performance analysis:
- 3D surface plots (threshold-metric spaces)
- Contour plots
- Interactive analysis
- Multi-dimensional performance landscapes
- Stratified performance heatmaps

Manuscript-preparation (IEEE/Nature/JAMA compliant)

Author: Chatchai Tritham
Affiliation: Naresuan University
Date: 2026-01-25
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Colorblind-friendly palette
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
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)


def plot_3d_performance_surface(
    threshold_range: np.ndarray,
    metric_range: np.ndarray,
    performance_values: np.ndarray,
    xlabel: str = "Threshold",
    ylabel: str = "Metric",
    zlabel: str = "Performance",
    title: str = "3D Performance Landscape",
    figsize: Tuple[float, float] = (10.0, 8.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, Axes3D]:
    """Plot 3D surface of performance metric across two parameter dimensions.

    Parameters:
        threshold_range: 1D array of threshold values
        metric_range: 1D array of second parameter values
        performance_values: 2D array of performance values (shape: len(threshold_range) x len(metric_range))
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label (performance metric name)
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, 3D axes)

    Example:
        >>> # Create grid
        >>> thresholds = np.linspace(0.1, 0.9, 20)
        >>> params = np.linspace(0.0, 1.0, 20)
        >>> T, P = np.meshgrid(thresholds, params)
        >>> # Simulate performance (e.g., F1-score)
        >>> performance = np.sin(T * np.pi) * np.cos(P * np.pi)
        >>> fig, ax = plot_3d_performance_surface(thresholds, params, performance.T)
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid
    X, Y = np.meshgrid(metric_range, threshold_range)

    # Plot surface
    surf = ax.plot_surface(
        X,
        Y,
        performance_values.T,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9,
        antialiased=True,
    )

    # Styling
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(zlabel, fontsize=12, fontweight='bold')

    # View angle
    ax.view_init(elev=25, azim=45)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_contour_performance(
    threshold_range: np.ndarray,
    metric_range: np.ndarray,
    performance_values: np.ndarray,
    xlabel: str = "Threshold",
    ylabel: str = "Metric",
    title: str = "Performance Contour Map",
    optimal_point: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8.0, 7.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D contour map of performance metric.

    Parameters:
        threshold_range: 1D array of threshold values
        metric_range: 1D array of second parameter values
        performance_values: 2D array of performance values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        optimal_point: (x, y) coordinates of optimal point to highlight (optional)
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> thresholds = np.linspace(0.1, 0.9, 20)
        >>> params = np.linspace(0.0, 1.0, 20)
        >>> T, P = np.meshgrid(thresholds, params)
        >>> performance = np.sin(T * np.pi) * np.cos(P * np.pi)
        >>> fig, ax = plot_contour_performance(
        ...     thresholds, params, performance.T,
        ...     optimal_point=(0.5, 0.5)
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create meshgrid
    X, Y = np.meshgrid(metric_range, threshold_range)

    # Contour plot
    levels = 15
    contour = ax.contourf(
        X, Y, performance_values.T, levels=levels, cmap='viridis', alpha=0.9
    )

    # Contour lines
    contour_lines = ax.contour(
        X,
        Y,
        performance_values.T,
        levels=levels,
        colors='black',
        alpha=0.3,
        linewidths=0.5,
    )

    # Label contours
    ax.clabel(contour_lines, inline=True, fontsize=9, fmt='%.2f')

    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label('Performance', fontsize=12, fontweight='bold')

    # Mark optimal point
    if optimal_point is not None:
        ax.plot(
            optimal_point[0],
            optimal_point[1],
            'r*',
            markersize=20,
            markeredgecolor='white',
            markeredgewidth=2,
            label='Optimal',
            zorder=10,
        )
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    # Styling
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_stratified_heatmap(
    metrics_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Stratified Performance Heatmap",
    xlabel: str = "Risk Tier",
    ylabel: str = "Model",
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot heatmap of performance metrics stratified by groups.

    Parameters:
        metrics_matrix: 2D array of metric values (rows: models/methods, cols: strata/tiers)
        row_labels: Labels for rows (e.g., model names)
        col_labels: Labels for columns (e.g., risk tiers)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> # Performance of 3 models across 4 risk tiers
        >>> metrics = np.array([
        ...     [0.85, 0.88, 0.90, 0.92],  # Model A
        ...     [0.82, 0.84, 0.87, 0.89],  # Model B
        ...     [0.88, 0.91, 0.93, 0.95],  # Model C
        ... ])
        >>> fig, ax = plot_stratified_heatmap(
        ...     metrics,
        ...     row_labels=['Model A', 'Model B', 'Model C'],
        ...     col_labels=['R1 (Low)', 'R2 (Med)', 'R3 (High)', 'R4 (Critical)']
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap
    sns.heatmap(
        metrics_matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Performance Score'},
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=False,
        linewidths=1,
        linecolor='white',
        ax=ax,
    )

    # Styling
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), rotation=0, va='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_radar_chart(
    metrics_dict: Dict[str, float],
    title: str = "Performance Radar Chart",
    max_value: float = 1.0,
    figsize: Tuple[float, float] = (7.0, 7.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot radar/spider chart for multiple performance metrics.

    Parameters:
        metrics_dict: Dictionary mapping metric name to value
        title: Plot title
        max_value: Maximum value for radar chart (default: 1.0)
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> metrics = {
        ...     'Accuracy': 0.85,
        ...     'Precision': 0.82,
        ...     'Recall': 0.88,
        ...     'F1-Score': 0.85,
        ...     'ROC-AUC': 0.90,
        ...     'PR-AUC': 0.87
        ... }
        >>> fig, ax = plot_radar_chart(metrics)
    """
    # Prepare data
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    N = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(projection='polar'), dpi=dpi
    )

    # Plot data
    ax.plot(
        angles,
        values,
        'o-',
        linewidth=2.5,
        color=COLORS['blue'],
        markersize=8,
        markerfacecolor=COLORS['cyan'],
        markeredgecolor=COLORS['blue'],
        markeredgewidth=2,
    )
    ax.fill(angles, values, alpha=0.25, color=COLORS['blue'])

    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    # Set y-limits
    ax.set_ylim(0, max_value)

    # Set y-ticks
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels(
        [f'{v:.1f}' for v in np.linspace(0, max_value, 5)], fontsize=10, color='grey'
    )

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_multi_radar_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    title: str = "Multi-Model Radar Comparison",
    max_value: float = 1.0,
    figsize: Tuple[float, float] = (8.0, 8.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot radar chart comparing multiple models.

    Parameters:
        models_metrics: Dictionary mapping model name to metrics dict
        title: Plot title
        max_value: Maximum value for radar chart
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     'Model A': {'Accuracy': 0.85, 'Precision': 0.82, 'Recall': 0.88},
        ...     'Model B': {'Accuracy': 0.88, 'Precision': 0.86, 'Recall': 0.84},
        ... }
        >>> fig, ax = plot_multi_radar_comparison(models)
    """
    # Get metric names (use first model's keys)
    first_model = list(models_metrics.keys())[0]
    categories = list(models_metrics[first_model].keys())
    N = len(categories)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(projection='polar'), dpi=dpi
    )

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
    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        values = [metrics[cat] for cat in categories]
        values += values[:1]  # Complete circle

        color = colors[idx % len(colors)]
        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2.5,
            color=color,
            label=model_name,
            markersize=6,
            alpha=0.8,
        )
        ax.fill(angles, values, alpha=0.1, color=color)

    # Fix axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    # Set y-limits
    ax.set_ylim(0, max_value)
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels(
        [f'{v:.1f}' for v in np.linspace(0, max_value, 5)], fontsize=10, color='grey'
    )

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_parallel_coordinates(
    df: pd.DataFrame,
    class_column: str,
    title: str = "Parallel Coordinates Plot",
    figsize: Tuple[float, float] = (10.0, 6.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot parallel coordinates for multi-dimensional data.

    Parameters:
        df: DataFrame with metrics as columns
        class_column: Column name for class/group coloring
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> df = pd.DataFrame({
        ...     'Model': ['A', 'B', 'C'],
        ...     'Accuracy': [0.85, 0.88, 0.82],
        ...     'Precision': [0.82, 0.86, 0.80],
        ...     'Recall': [0.88, 0.84, 0.86],
        ...     'F1': [0.85, 0.85, 0.83]
        ... })
        >>> fig, ax = plot_parallel_coordinates(df, class_column='Model')
    """
    from pandas.plotting import parallel_coordinates

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot parallel coordinates
    parallel_coordinates(
        df,
        class_column,
        ax=ax,
        color=[COLORS['blue'], COLORS['orange'], COLORS['teal']],
        linewidth=2.5,
        alpha=0.7,
    )

    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_3d_scatter_performance(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    zlabel: str = "Performance",
    title: str = "3D Performance Scatter",
    figsize: Tuple[float, float] = (10.0, 8.0),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, Axes3D]:
    """Plot 3D scatter plot for performance analysis.

    Parameters:
        x: X-coordinates
        y: Y-coordinates
        z: Z-coordinates (performance values)
        colors: Color values for each point (optional)
        labels: Text labels for points (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, 3D axes)

    Example:
        >>> x = np.random.rand(50)
        >>> y = np.random.rand(50)
        >>> z = x + y + np.random.rand(50) * 0.1
        >>> fig, ax = plot_3d_scatter_performance(x, y, z)
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Default colors
    if colors is None:
        colors = z  # Color by performance value

    # Scatter plot
    scatter = ax.scatter(
        x,
        y,
        z,
        c=colors,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=1,
    )

    # Add labels
    if labels is not None:
        for i, label in enumerate(labels):
            ax.text(x[i], y[i], z[i], label, fontsize=9)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(zlabel, fontsize=12, fontweight='bold')

    # Styling
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # View angle
    ax.view_init(elev=20, azim=45)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax
