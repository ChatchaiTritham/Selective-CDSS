"""Intervention operations and causal effect estimation.

This module implements do-calculus interventions and causal effect estimation,
including Average Treatment Effect (ATE) and Conditional ATE (CATE).

Theoretical Foundation:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Hernán, M. A., & Robins, J. M. (2020). Causal Inference: What If.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DoIntervention:
    """Represents a do-intervention in causal inference.

    A do-intervention sets variables to specific values, breaking
    incoming causal edges: do(X=x) represents P(Y | do(X=x)).

    Attributes:
        target: Variable being intervened on
        value: Value to set (for atomic intervention)
        values: Distribution of values (for soft intervention)
        is_soft: Whether this is a soft intervention (probabilistic)

    Example:
        >>> # Hard intervention: set antibiotic=True
        >>> intervention = DoIntervention(target='antibiotic', value=True)
        >>>
        >>> # Soft intervention: set dose ~ Normal(10, 2)
        >>> intervention = DoIntervention(
        ...     target='dose',
        ...     values={'distribution': 'normal', 'mean': 10, 'std': 2},
        ...     is_soft=True
        ... )
    """

    target: str
    value: Optional[Any] = None
    values: Optional[Dict[str, Any]] = None
    is_soft: bool = False

    def __post_init__(self):
        """Validate intervention specification."""
        if not self.is_soft and self.value is None:
            raise ValueError("Hard intervention requires value")
        if self.is_soft and self.values is None:
            raise ValueError("Soft intervention requires values distribution")


def perform_do_intervention(
    scm: 'StructuralCausalModel',
    intervention: Union[Dict[str, Any], DoIntervention],
    n_samples: int = 1000,
    return_dataframe: bool = True,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Perform do-intervention on SCM and sample interventional distribution.

    Args:
        scm: Structural Causal Model
        intervention: Intervention specification (dict or DoIntervention)
        n_samples: Number of samples
        return_dataframe: If True, return pandas DataFrame

    Returns:
        Interventional data P(X | do(Y=y))

    Example:
        >>> from basics_cdss.causal import StructuralCausalModel, perform_do_intervention
        >>>
        >>> # Create SCM
        >>> scm = StructuralCausalModel(graph)
        >>>
        >>> # Perform intervention
        >>> data = perform_do_intervention(
        ...     scm=scm,
        ...     intervention={'antibiotic': True},
        ...     n_samples=1000
        ... )
    """
    # Convert dict to intervention object
    if isinstance(intervention, dict):
        if len(intervention) != 1:
            raise ValueError("Only single-variable interventions supported")
        target = list(intervention.keys())[0]
        value = intervention[target]
        intervention_obj = DoIntervention(target=target, value=value)
    else:
        intervention_obj = intervention

    # Use SCM's do_intervention method
    return scm.do_intervention(
        interventions={intervention_obj.target: intervention_obj.value},
        n=n_samples,
        return_dataframe=return_dataframe,
    )


def compute_ate(
    scm: 'StructuralCausalModel',
    treatment: str,
    outcome: str,
    treatment_values: Optional[List[Any]] = None,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """Compute Average Treatment Effect (ATE).

    ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

    Measures the average causal effect of treatment on outcome.

    Args:
        scm: Structural Causal Model
        treatment: Treatment variable
        outcome: Outcome variable
        treatment_values: Values to compare (default: [0, 1])
        n_samples: Number of samples per intervention

    Returns:
        Dictionary with ATE statistics:
            - ate: Average treatment effect
            - control_mean: Mean outcome under control
            - treatment_mean: Mean outcome under treatment
            - relative_effect: Relative effect size

    Example:
        >>> # Compute ATE of antibiotic on mortality
        >>> result = compute_ate(
        ...     scm=scm,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     treatment_values=[False, True]
        ... )
        >>> print(f"ATE: {result['ate']:.3f}")
    """
    if treatment_values is None:
        treatment_values = [0, 1]

    if len(treatment_values) != 2:
        raise ValueError("ATE requires exactly 2 treatment values")

    control_value, treatment_value = treatment_values

    # Sample under control
    control_data = scm.do_intervention(
        interventions={treatment: control_value}, n=n_samples, return_dataframe=True
    )
    control_mean = control_data[outcome].mean()

    # Sample under treatment
    treatment_data = scm.do_intervention(
        interventions={treatment: treatment_value}, n=n_samples, return_dataframe=True
    )
    treatment_mean = treatment_data[outcome].mean()

    # Compute ATE
    ate = treatment_mean - control_mean

    # Relative effect
    if control_mean != 0:
        relative_effect = ate / abs(control_mean)
    else:
        relative_effect = np.nan

    return {
        'ate': ate,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'relative_effect': relative_effect,
        'treatment_values': treatment_values,
    }


def compute_cate(
    scm: 'StructuralCausalModel',
    treatment: str,
    outcome: str,
    conditioning_vars: List[str],
    conditioning_values: Dict[str, Any],
    treatment_values: Optional[List[Any]] = None,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """Compute Conditional Average Treatment Effect (CATE).

    CATE = E[Y | do(X=1), Z=z] - E[Y | do(X=0), Z=z]

    Measures treatment effect conditional on covariates Z.

    Args:
        scm: Structural Causal Model
        treatment: Treatment variable
        outcome: Outcome variable
        conditioning_vars: Variables to condition on
        conditioning_values: Values for conditioning variables
        treatment_values: Values to compare (default: [0, 1])
        n_samples: Number of samples per intervention

    Returns:
        Dictionary with CATE statistics

    Example:
        >>> # CATE of antibiotic on mortality, conditional on age > 65
        >>> result = compute_cate(
        ...     scm=scm,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     conditioning_vars=['age'],
        ...     conditioning_values={'age': 70},
        ...     treatment_values=[False, True]
        ... )
    """
    if treatment_values is None:
        treatment_values = [0, 1]

    control_value, treatment_value = treatment_values

    # Create intervention with conditioning
    control_intervention = {treatment: control_value, **conditioning_values}
    treatment_intervention = {treatment: treatment_value, **conditioning_values}

    # Sample under control
    control_data = scm.do_intervention(
        interventions=control_intervention, n=n_samples, return_dataframe=True
    )
    control_mean = control_data[outcome].mean()

    # Sample under treatment
    treatment_data = scm.do_intervention(
        interventions=treatment_intervention, n=n_samples, return_dataframe=True
    )
    treatment_mean = treatment_data[outcome].mean()

    # Compute CATE
    cate = treatment_mean - control_mean

    return {
        'cate': cate,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'conditioning_vars': conditioning_vars,
        'conditioning_values': conditioning_values,
        'treatment_values': treatment_values,
    }


def estimate_ate_from_data(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    method: str = 'regression',
    confounders: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Estimate ATE from observational data.

    Uses adjustment methods to estimate causal effects from non-experimental data.

    Args:
        data: Observational dataset
        treatment: Treatment variable
        outcome: Outcome variable
        method: Estimation method ('regression', 'matching', 'ipw')
        confounders: Confounding variables to adjust for

    Returns:
        Dictionary with ATE estimate and confidence interval

    Example:
        >>> # Estimate ATE from observational data
        >>> result = estimate_ate_from_data(
        ...     data=obs_data,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     method='regression',
        ...     confounders=['age', 'comorbidities']
        ... )
    """
    if confounders is None:
        confounders = []

    if method == 'regression':
        # Regression adjustment
        from sklearn.linear_model import LinearRegression

        # Fit model: Y ~ T + confounders
        X = data[[treatment] + confounders].values
        y = data[outcome].values

        model = LinearRegression()
        model.fit(X, y)

        # ATE is coefficient on treatment
        ate = model.coef_[0]

        # Bootstrap confidence interval
        n_bootstrap = 1000
        ate_bootstrap = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(data), len(data), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            model_boot = LinearRegression()
            model_boot.fit(X_boot, y_boot)
            ate_bootstrap.append(model_boot.coef_[0])

        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)

        return {
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'regression',
            'confounders': confounders,
        }

    elif method == 'matching':
        # Simple matching estimator
        treated = data[data[treatment] == 1]
        control = data[data[treatment] == 0]

        ate = treated[outcome].mean() - control[outcome].mean()

        # Standard error
        se = np.sqrt(
            treated[outcome].var() / len(treated)
            + control[outcome].var() / len(control)
        )

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return {
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'matching',
            'se': se,
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_intervention_curve(
    scm: 'StructuralCausalModel',
    treatment: str,
    outcome: str,
    treatment_range: np.ndarray,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """Compute dose-response curve for treatment.

    Args:
        scm: Structural Causal Model
        treatment: Treatment variable (continuous)
        outcome: Outcome variable
        treatment_range: Range of treatment values
        n_samples: Number of samples per treatment level

    Returns:
        DataFrame with treatment values and mean outcomes

    Example:
        >>> # Dose-response curve for antibiotic dose
        >>> curve = compute_intervention_curve(
        ...     scm=scm,
        ...     treatment='antibiotic_dose',
        ...     outcome='infection_clearance',
        ...     treatment_range=np.linspace(0, 100, 20)
        ... )
        >>> plt.plot(curve['treatment'], curve['outcome_mean'])
    """
    results = []

    for treatment_value in treatment_range:
        # Intervene at this treatment level
        data = scm.do_intervention(
            interventions={treatment: treatment_value},
            n=n_samples,
            return_dataframe=True,
        )

        results.append(
            {
                'treatment': treatment_value,
                'outcome_mean': data[outcome].mean(),
                'outcome_std': data[outcome].std(),
                'outcome_q25': data[outcome].quantile(0.25),
                'outcome_q75': data[outcome].quantile(0.75),
            }
        )

    return pd.DataFrame(results)


def test_intervention_effect(
    scm: 'StructuralCausalModel',
    treatment: str,
    outcome: str,
    treatment_values: List[Any],
    n_samples: int = 1000,
    test: str = 'ttest',
) -> Dict[str, Any]:
    """Statistical test for intervention effect.

    Args:
        scm: Structural Causal Model
        treatment: Treatment variable
        outcome: Outcome variable
        treatment_values: Treatment values to compare
        n_samples: Number of samples per group
        test: Test type ('ttest', 'mannwhitney')

    Returns:
        Dictionary with test results

    Example:
        >>> result = test_intervention_effect(
        ...     scm=scm,
        ...     treatment='antibiotic',
        ...     outcome='mortality',
        ...     treatment_values=[False, True]
        ... )
        >>> print(f"p-value: {result['p_value']:.4f}")
    """
    if len(treatment_values) != 2:
        raise ValueError("Test requires exactly 2 treatment values")

    # Sample under both interventions
    data_0 = scm.do_intervention(
        interventions={treatment: treatment_values[0]},
        n=n_samples,
        return_dataframe=True,
    )

    data_1 = scm.do_intervention(
        interventions={treatment: treatment_values[1]},
        n=n_samples,
        return_dataframe=True,
    )

    outcomes_0 = data_0[outcome].values
    outcomes_1 = data_1[outcome].values

    # Perform test
    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(outcomes_1, outcomes_0)
        test_name = "Independent t-test"
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(outcomes_1, outcomes_0)
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test: {test}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((outcomes_0.var() + outcomes_1.var()) / 2)
    cohens_d = (outcomes_1.mean() - outcomes_0.mean()) / pooled_std

    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_0': outcomes_0.mean(),
        'mean_1': outcomes_1.mean(),
        'significant': p_value < 0.05,
    }
