"""
Report generation and artifact export for governance and reproducibility.

This module provides:
- Metric table export (CSV format)
- Plot generation (calibration curves, coverage-risk curves)
- Reproducibility manifests
- Audit-ready report packages

All artifacts are designed for regulatory review and independent replication.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvaluationReport:
    """Container for complete evaluation results and artifacts.

    Attributes:
        overall_metrics: Overall evaluation metrics
        stratified_metrics: Metrics stratified by risk tier
        plots: Dictionary of plot file paths
        tables: Dictionary of table file paths
        manifest: Reproducibility manifest
    """

    overall_metrics: Dict[str, float]
    stratified_metrics: Dict[str, Dict[str, float]]
    plots: Dict[str, Path] = None
    tables: Dict[str, Path] = None
    manifest: Dict[str, Any] = None


def export_metrics_table(
    metrics: Dict[str, Any], output_path: str | Path, include_stratified: bool = True
) -> Path:
    """Export metrics to CSV table for regulatory review.

    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Output CSV path
        include_stratified: Include risk-tier stratified metrics

    Returns:
        Path to created CSV file

    Example:
        >>> metrics = {
        ...     "overall": {"ece": 0.05, "brier": 0.12},
        ...     "by_tier": {"high": {"ece": 0.03}, "low": {"ece": 0.07}}
        ... }
        >>> export_metrics_table(metrics, "results/metrics.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Overall metrics
    if "overall" in metrics:
        for metric_name, value in metrics["overall"].items():
            if metric_name == "reliability_curve":
                continue  # Skip complex objects
            rows.append({"tier": "overall", "metric": metric_name, "value": value})

    # Stratified metrics
    if include_stratified and "by_risk_tier" in metrics:
        for tier, tier_metrics in metrics["by_risk_tier"].items():
            if hasattr(tier_metrics, "__dict__"):
                # If it's a dataclass, convert to dict
                tier_metrics = asdict(tier_metrics)

            for metric_name, value in tier_metrics.items():
                if metric_name == "reliability_curve":
                    continue
                rows.append({"tier": tier, "metric": metric_name, "value": value})

    # Write CSV
    if rows:
        fieldnames = ["tier", "metric", "value"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return output_path


def export_calibration_plot(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    output_path: str | Path,
    title: str = "Reliability Diagram",
    stratified: Optional[Dict[str, tuple]] = None,
) -> Path:
    """Export calibration reliability curve plot.

    Args:
        bin_confidences: Bin-wise average confidences
        bin_accuracies: Bin-wise empirical accuracies
        output_path: Output plot path (*.png or *.pdf)
        title: Plot title
        stratified: Optional dict of tier -> (confidences, accuracies) for stratification

    Returns:
        Path to created plot file

    Example:
        >>> confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> accs = np.array([0.12, 0.28, 0.52, 0.68, 0.88])
        >>> export_calibration_plot(confs, accs, "plots/calibration.png")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    # Overall calibration
    if len(bin_confidences) > 0:
        ax.plot(bin_confidences, bin_accuracies, "o-", label="Overall", linewidth=2)

    # Stratified calibration
    if stratified:
        for tier, (confs, accs) in stratified.items():
            if len(confs) > 0:
                ax.plot(confs, accs, "o-", label=tier, alpha=0.7)

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def export_coverage_risk_plot(
    coverages: np.ndarray,
    risks: np.ndarray,
    output_path: str | Path,
    title: str = "Coverage-Risk Curve",
    stratified: Optional[Dict[str, tuple]] = None,
) -> Path:
    """Export coverage-risk curve plot.

    Args:
        coverages: Coverage values (fraction of predictions retained)
        risks: Conditional risk values
        output_path: Output plot path
        title: Plot title
        stratified: Optional dict of tier -> (coverages, risks) for stratification

    Returns:
        Path to created plot file

    Example:
        >>> coverages = np.array([0.0, 0.5, 1.0])
        >>> risks = np.array([0.0, 0.1, 0.2])
        >>> export_coverage_risk_plot(coverages, risks, "plots/coverage_risk.png")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Overall curve
    if len(coverages) > 0:
        # Remove NaN values
        valid_mask = ~np.isnan(risks)
        valid_cov = coverages[valid_mask]
        valid_risk = risks[valid_mask]

        if len(valid_cov) > 0:
            ax.plot(valid_cov, valid_risk, "o-", label="Overall", linewidth=2)

    # Stratified curves
    if stratified:
        for tier, (cov, risk) in stratified.items():
            valid_mask = ~np.isnan(risk)
            if valid_mask.sum() > 0:
                ax.plot(cov[valid_mask], risk[valid_mask], "o-", label=tier, alpha=0.7)

    ax.set_xlabel("Coverage (fraction retained)", fontsize=12)
    ax.set_ylabel("Conditional Risk", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def create_reproducibility_manifest(
    config_path: str | Path,
    data_info: Dict[str, Any],
    results_files: Dict[str, Path],
    output_path: str | Path,
) -> Path:
    """Create reproducibility manifest for independent verification.

    Args:
        config_path: Path to configuration file
        data_info: Information about data sources (SynDX version, archetypes)
        results_files: Dictionary of result file paths
        output_path: Output manifest path (*.yaml or *.json)

    Returns:
        Path to created manifest

    Example:
        >>> manifest = create_reproducibility_manifest(
        ...     "config.yaml",
        ...     {"syndx_version": "1.0", "n_archetypes": 50},
        ...     {"metrics": Path("metrics.csv"), "plots": Path("plots/")},
        ...     "REPRODUCIBILITY.yaml"
        ... )
    """
    import yaml

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_version": "1.0",
        "created_timestamp": datetime.utcnow().isoformat() + "Z",
        "framework": "BASICS-CDSS",
        "configuration": str(config_path),
        "data_sources": data_info,
        "result_files": {key: str(path) for key, path in results_files.items()},
        "reproduction_instructions": {
            "step_1": "Install environment: conda env create -f environment.yml",
            "step_2": "Load configuration: config = load_config('config.yaml')",
            "step_3": "Run evaluation pipeline as specified in config",
            "step_4": "Compare outputs against provided result files",
        },
    }

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        with open(output_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return output_path


def generate_evaluation_report(
    calibration_metrics: Dict,
    coverage_risk_metrics: Dict,
    harm_metrics: Dict,
    output_dir: str | Path,
    config_path: Optional[str | Path] = None,
    data_info: Optional[Dict] = None,
    generate_plots: bool = True,
) -> EvaluationReport:
    """Generate comprehensive evaluation report with all artifacts.

    Args:
        calibration_metrics: Calibration evaluation results
        coverage_risk_metrics: Coverage-risk evaluation results
        harm_metrics: Harm-aware evaluation results
        output_dir: Directory to save all report artifacts
        config_path: Optional path to evaluation configuration
        data_info: Optional information about data sources
        generate_plots: Whether to generate plots

    Returns:
        EvaluationReport with paths to all generated artifacts

    Example:
        >>> report = generate_evaluation_report(
        ...     calibration_metrics=cal_metrics,
        ...     coverage_risk_metrics=cr_metrics,
        ...     harm_metrics=harm_metrics,
        ...     output_dir="results/eval_001"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all metrics
    overall_metrics = {}

    if "overall" in calibration_metrics:
        overall_metrics["ece"] = calibration_metrics["overall"]["ece"]
        overall_metrics["brier_score"] = calibration_metrics["overall"]["brier_score"]

    if hasattr(coverage_risk_metrics, "aurc"):
        overall_metrics["aurc"] = coverage_risk_metrics.aurc

    if hasattr(harm_metrics, "weighted_harm_loss"):
        overall_metrics["weighted_harm_loss"] = harm_metrics.weighted_harm_loss
        overall_metrics["harm_concentration"] = harm_metrics.harm_concentration

    # Export metric tables
    tables = {}

    calibration_table = output_dir / "calibration_metrics.csv"
    export_metrics_table(calibration_metrics, calibration_table)
    tables["calibration"] = calibration_table

    # Export plots
    plots = {}

    if generate_plots:
        if "overall" in calibration_metrics:
            rel_curve = calibration_metrics["overall"]["reliability_curve"]
            if len(rel_curve[0]) > 0:
                cal_plot = output_dir / "calibration_curve.png"
                export_calibration_plot(
                    rel_curve[0],
                    rel_curve[1],
                    cal_plot,
                    title="Calibration Reliability Diagram",
                )
                plots["calibration"] = cal_plot

        if hasattr(coverage_risk_metrics, "coverage_curve"):
            if coverage_risk_metrics.coverage_curve is not None:
                cr_plot = output_dir / "coverage_risk_curve.png"
                export_coverage_risk_plot(
                    coverage_risk_metrics.coverage_curve,
                    coverage_risk_metrics.risk_curve,
                    cr_plot,
                    title="Selective Prediction: Coverage vs Risk",
                )
                plots["coverage_risk"] = cr_plot

    # Create reproducibility manifest
    if config_path and data_info:
        manifest_path = output_dir / "REPRODUCIBILITY.yaml"
        create_reproducibility_manifest(
            config_path, data_info, {**tables, **plots}, manifest_path
        )
        manifest = manifest_path
    else:
        manifest = None

    return EvaluationReport(
        overall_metrics=overall_metrics,
        stratified_metrics={},
        plots=plots,
        tables=tables,
        manifest=manifest,
    )
