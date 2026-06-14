"""
Configuration logging and execution tracking for reproducible evaluation.

This module provides:
- Configuration management (YAML/JSON export)
- Execution metadata capture
- Audit trail generation

All evaluation runs must be logged for governance compliance and reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EvaluationConfig:
    """Configuration for BASICS-CDSS evaluation run.

    Attributes:
        seed: Master random seed for reproducibility
        n_per_archetype: Scenarios generated per archetype
        perturbation_type: Type of perturbation applied
        perturbation_params: Perturbation operator parameters
        calibration_bins: Number of bins for calibration metrics
        coverage_thresholds: Number of thresholds for coverage-risk curves
        harm_weights: Harm weight mapping by risk tier
        metadata: Additional metadata (e.g., experiment name, author)
    """

    seed: int = 42
    n_per_archetype: int = 10
    perturbation_type: Optional[str] = "composite"
    perturbation_params: Dict[str, Any] = field(default_factory=dict)
    calibration_bins: int = 10
    coverage_thresholds: int = 100
    harm_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "EvaluationConfig":
        """Load config from dictionary."""
        return cls(**config_dict)


@dataclass
class ExecutionLog:
    """Log of evaluation execution for audit trail.

    Attributes:
        timestamp: ISO format timestamp of execution
        config: Evaluation configuration used
        n_scenarios: Total scenarios evaluated
        n_archetypes: Number of archetypes loaded
        execution_time_seconds: Total execution time
        errors: Any errors encountered
        warnings: Any warnings generated
        results_summary: High-level results summary
    """

    timestamp: str
    config: EvaluationConfig
    n_scenarios: int
    n_archetypes: int
    execution_time_seconds: float
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    results_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert log to dictionary."""
        log_dict = asdict(self)
        log_dict["config"] = self.config.to_dict()
        return log_dict

    @classmethod
    def from_dict(cls, log_dict: Dict) -> "ExecutionLog":
        """Load log from dictionary."""
        config = EvaluationConfig.from_dict(log_dict["config"])
        log_dict["config"] = config
        return cls(**log_dict)


def save_config(
    config: EvaluationConfig, output_path: str | Path, format: str = "yaml"
) -> None:
    """Save evaluation configuration to file.

    Args:
        config: Configuration to save
        output_path: Path to save config (*.yaml or *.json)
        format: Output format ("yaml" or "json")

    Example:
        >>> config = EvaluationConfig(seed=42, n_per_archetype=10)
        >>> save_config(config, "config.yaml")
    """
    output_path = Path(output_path)
    config_dict = config.to_dict()

    if format == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'yaml' or 'json'.")


def load_config(config_path: str | Path) -> EvaluationConfig:
    """Load evaluation configuration from file.

    Args:
        config_path: Path to config file (*.yaml or *.json)

    Returns:
        Loaded EvaluationConfig

    Example:
        >>> config = load_config("config.yaml")
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return EvaluationConfig.from_dict(config_dict)


def log_evaluation_run(
    config: EvaluationConfig,
    n_scenarios: int,
    n_archetypes: int,
    execution_time: float,
    results_summary: Optional[Dict] = None,
    errors: Optional[list] = None,
    warnings: Optional[list] = None,
    output_path: Optional[str | Path] = None,
    format: str = "yaml",
) -> ExecutionLog:
    """Log evaluation run with complete metadata.

    Args:
        config: Evaluation configuration
        n_scenarios: Number of scenarios evaluated
        n_archetypes: Number of archetypes used
        execution_time: Execution time in seconds
        results_summary: Optional summary of results
        errors: Optional list of errors
        warnings: Optional list of warnings
        output_path: Optional path to save log
        format: Output format ("yaml" or "json")

    Returns:
        ExecutionLog object

    Example:
        >>> config = EvaluationConfig(seed=42)
        >>> log = log_evaluation_run(
        ...     config, n_scenarios=100, n_archetypes=10,
        ...     execution_time=5.2,
        ...     output_path="logs/run_001.yaml"
        ... )
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    log = ExecutionLog(
        timestamp=timestamp,
        config=config,
        n_scenarios=n_scenarios,
        n_archetypes=n_archetypes,
        execution_time_seconds=execution_time,
        errors=errors or [],
        warnings=warnings or [],
        results_summary=results_summary or {},
    )

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log_dict = log.to_dict()

        if format == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(log_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(log_dict, f, indent=2)
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'yaml' or 'json'.")

    return log


def create_audit_trail(
    config_path: str | Path, log_path: str | Path, output_dir: str | Path
) -> Dict[str, Path]:
    """Create audit trail package with config and log.

    Args:
        config_path: Path to configuration file
        log_path: Path to execution log
        output_dir: Directory to save audit trail

    Returns:
        Dictionary of created audit files

    Example:
        >>> files = create_audit_trail(
        ...     "config.yaml", "execution.yaml", "audit/"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    config = load_config(config_path)
    config_out = output_dir / "config.yaml"
    save_config(config, config_out, format="yaml")

    # Copy log
    with open(log_path, "r") as f:
        if Path(log_path).suffix in [".yaml", ".yml"]:
            log_dict = yaml.safe_load(f)
        else:
            log_dict = json.load(f)

    log_out = output_dir / "execution_log.yaml"
    with open(log_out, "w") as f:
        yaml.dump(log_dict, f, default_flow_style=False)

    # Create manifest
    manifest = {
        "audit_trail_version": "1.0",
        "created_timestamp": datetime.utcnow().isoformat() + "Z",
        "files": {
            "config": str(config_out.name),
            "execution_log": str(log_out.name),
        },
    }

    manifest_path = output_dir / "MANIFEST.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    return {
        "config": config_out,
        "log": log_out,
        "manifest": manifest_path,
    }
