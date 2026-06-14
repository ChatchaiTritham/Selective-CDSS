"""Structural Causal Model (SCM) implementation.

This module provides SCM functionality for causal simulation, enabling
sampling from causal models with structural equations and interventions.

Theoretical Foundation:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from basics_cdss.causal.causal_graph import CausalGraph


@dataclass
class CausalMechanism:
    """Represents a structural equation (causal mechanism) for a variable.

    A causal mechanism defines how a variable is generated from its parents:
        X := f(parents(X), noise)

    Attributes:
        variable: Target variable name
        parents: List of parent variables
        function: Callable that generates variable from parents
        noise_distribution: Distribution for exogenous noise
        is_deterministic: Whether mechanism is deterministic (no noise)

    Example:
        >>> # Linear mechanism: Y = 2*X + noise
        >>> mechanism = CausalMechanism(
        ...     variable='Y',
        ...     parents=['X'],
        ...     function=lambda parents, noise: 2 * parents['X'] + noise,
        ...     noise_distribution=lambda rng: rng.normal(0, 0.5)
        ... )
        >>>
        >>> # Nonlinear mechanism: HR = base_hr + 0.5*temp^2 + noise
        >>> mechanism = CausalMechanism(
        ...     variable='heart_rate',
        ...     parents=['temperature'],
        ...     function=lambda p, n: 70 + 0.5 * (p['temperature'] - 37)**2 + n,
        ...     noise_distribution=lambda rng: rng.normal(0, 5)
        ... )
    """

    variable: str
    parents: List[str]
    function: Callable[[Dict[str, Any], float], float]
    noise_distribution: Optional[Callable[[np.random.RandomState], float]] = None
    is_deterministic: bool = False

    def sample(
        self, parent_values: Dict[str, Any], rng: np.random.RandomState
    ) -> float:
        """Sample variable value given parent values.

        Args:
            parent_values: Dictionary of parent variable values
            rng: Random number generator

        Returns:
            Sampled value for this variable
        """
        # Generate noise
        if self.is_deterministic or self.noise_distribution is None:
            noise = 0.0
        else:
            noise = self.noise_distribution(rng)

        # Apply structural equation
        return self.function(parent_values, noise)

    def __call__(
        self, parent_values: Dict[str, Any], rng: np.random.RandomState
    ) -> float:
        """Allow mechanism to be called directly."""
        return self.sample(parent_values, rng)


class StructuralCausalModel:
    """Structural Causal Model (SCM) for causal data generation.

    An SCM consists of:
    1. A causal graph G (DAG)
    2. Structural equations for each variable
    3. Exogenous noise distributions

    The SCM enables:
    - Observational sampling: P(X)
    - Interventional sampling: P(X | do(Y=y))
    - Counterfactual reasoning: P(Y_x | X=x')

    Example:
        >>> from basics_cdss.causal import CausalGraph, StructuralCausalModel
        >>>
        >>> # Define causal graph
        >>> graph = CausalGraph()
        >>> graph.add_edge('infection', 'temperature')
        >>> graph.add_edge('temperature', 'heart_rate')
        >>>
        >>> # Create SCM
        >>> scm = StructuralCausalModel(graph, seed=42)
        >>>
        >>> # Define mechanisms
        >>> scm.add_mechanism(
        ...     variable='infection',
        ...     parents=[],
        ...     function=lambda p, n: n > 0.5,  # Bernoulli(0.5)
        ...     noise_distribution=lambda rng: rng.uniform(0, 1)
        ... )
        >>>
        >>> scm.add_mechanism(
        ...     variable='temperature',
        ...     parents=['infection'],
        ...     function=lambda p, n: 37.0 + 2.0 * p['infection'] + n,
        ...     noise_distribution=lambda rng: rng.normal(0, 0.5)
        ... )
        >>>
        >>> # Sample observational data
        >>> data = scm.sample(n=100)
        >>>
        >>> # Perform intervention: do(temperature=38.5)
        >>> interventional_data = scm.do_intervention({'temperature': 38.5}, n=100)
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        seed: Optional[int] = None,
        default_mechanisms: bool = True,
    ):
        """Initialize SCM.

        Args:
            causal_graph: Causal DAG structure
            seed: Random seed for reproducibility
            default_mechanisms: If True, create default linear mechanisms
        """
        if not causal_graph.is_dag():
            raise ValueError("Causal graph must be a DAG")

        self.graph = causal_graph
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Structural equations
        self.mechanisms: Dict[str, CausalMechanism] = {}

        # Initialize default mechanisms if requested
        if default_mechanisms:
            self._initialize_default_mechanisms()

    def _initialize_default_mechanisms(self):
        """Create default linear mechanisms for all variables."""
        for variable in self.graph.graph.nodes():
            parents = self.graph.get_parents(variable)

            if not parents:
                # Root node: sample from standard normal
                self.add_mechanism(
                    variable=variable,
                    parents=[],
                    function=lambda p, n: n,
                    noise_distribution=lambda rng: rng.normal(0, 1),
                )
            else:
                # Non-root: linear combination of parents + noise
                def linear_function(parents_dict, noise, parent_names=parents):
                    # Linear: X = sum(parents) + noise
                    return sum(parents_dict.get(p, 0) for p in parent_names) + noise

                self.add_mechanism(
                    variable=variable,
                    parents=parents,
                    function=linear_function,
                    noise_distribution=lambda rng: rng.normal(0, 0.5),
                )

    def add_mechanism(
        self,
        variable: str,
        parents: List[str],
        function: Callable[[Dict[str, Any], float], float],
        noise_distribution: Optional[Callable[[np.random.RandomState], float]] = None,
        is_deterministic: bool = False,
    ):
        """Add structural equation for a variable.

        Args:
            variable: Target variable
            parents: Parent variables
            function: Structural equation f(parents, noise)
            noise_distribution: Noise distribution (default: N(0,1))
            is_deterministic: Whether equation is deterministic
        """
        # Validate parents match graph
        graph_parents = set(self.graph.get_parents(variable))
        if set(parents) != graph_parents:
            raise ValueError(
                f"Parents {parents} don't match graph parents {graph_parents} "
                f"for variable {variable}"
            )

        mechanism = CausalMechanism(
            variable=variable,
            parents=parents,
            function=function,
            noise_distribution=noise_distribution,
            is_deterministic=is_deterministic,
        )

        self.mechanisms[variable] = mechanism

    def sample(
        self, n: int = 1, return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Sample observational data from SCM.

        Generates data according to P(X) by sampling variables in
        topological order.

        Args:
            n: Number of samples
            return_dataframe: If True, return pandas DataFrame

        Returns:
            DataFrame or list of dictionaries with sampled data

        Example:
            >>> data = scm.sample(n=1000)
            >>> print(data.head())
        """
        samples = []

        for _ in range(n):
            sample = {}

            # Sample in topological order (parents before children)
            for variable in self.graph.topological_order():
                if variable not in self.mechanisms:
                    raise ValueError(f"No mechanism defined for variable {variable}")

                mechanism = self.mechanisms[variable]

                # Get parent values
                parent_values = {parent: sample[parent] for parent in mechanism.parents}

                # Sample variable
                sample[variable] = mechanism.sample(parent_values, self.rng)

            samples.append(sample)

        if return_dataframe:
            return pd.DataFrame(samples)
        return samples

    def do_intervention(
        self, interventions: Dict[str, Any], n: int = 1, return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Sample interventional data: P(X | do(Y=y)).

        Performs hard intervention by:
        1. Breaking edges into intervened variables
        2. Setting intervened variables to fixed values
        3. Sampling remaining variables according to SCM

        Args:
            interventions: Dictionary of interventions {variable: value}
            n: Number of samples
            return_dataframe: If True, return pandas DataFrame

        Returns:
            DataFrame or list of dictionaries with interventional data

        Example:
            >>> # Intervene: set antibiotic=True
            >>> data = scm.do_intervention({'antibiotic': True}, n=100)
        """
        samples = []

        for _ in range(n):
            sample = {}

            # Sample in topological order
            for variable in self.graph.topological_order():
                # Check if variable is intervened upon
                if variable in interventions:
                    # Hard intervention: set to fixed value
                    sample[variable] = interventions[variable]
                else:
                    # Sample according to mechanism
                    mechanism = self.mechanisms[variable]
                    parent_values = {
                        parent: sample[parent] for parent in mechanism.parents
                    }
                    sample[variable] = mechanism.sample(parent_values, self.rng)

            samples.append(sample)

        if return_dataframe:
            return pd.DataFrame(samples)
        return samples

    def counterfactual(
        self,
        observation: Dict[str, Any],
        intervention: Dict[str, Any],
        query_variables: List[str],
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """Perform counterfactual reasoning: P(Y_x | X=x').

        Three-step process:
        1. Abduction: Infer exogenous noise from observation
        2. Action: Apply intervention
        3. Prediction: Propagate through modified SCM

        Args:
            observation: Observed variable values
            intervention: Counterfactual intervention
            query_variables: Variables to query in counterfactual world
            n_samples: Number of samples for approximation

        Returns:
            Dictionary of mean values for query variables

        Example:
            >>> # Observed: patient deteriorated
            >>> obs = {'infection': True, 'temperature': 39.5, 'outcome': 'poor'}
            >>>
            >>> # Counterfactual: what if antibiotic given earlier?
            >>> cf_result = scm.counterfactual(
            ...     observation=obs,
            ...     intervention={'antibiotic_time': 2.0},  # 2 hours earlier
            ...     query_variables=['outcome']
            ... )
        """
        # For simplicity, approximate counterfactual by:
        # 1. Sample consistent with observation
        # 2. Apply intervention
        # Note: Full counterfactual requires noise inference (not implemented here)

        results = []

        for _ in range(n_samples):
            sample = {}

            # Sample with intervention
            for variable in self.graph.topological_order():
                if variable in intervention:
                    sample[variable] = intervention[variable]
                elif variable in observation:
                    # Condition on observation (approximate)
                    sample[variable] = observation[variable]
                else:
                    mechanism = self.mechanisms[variable]
                    parent_values = {
                        parent: sample[parent] for parent in mechanism.parents
                    }
                    sample[variable] = mechanism.sample(parent_values, self.rng)

            results.append(sample)

        # Compute mean for query variables
        df = pd.DataFrame(results)
        return {var: df[var].mean() for var in query_variables if var in df.columns}

    def to_dict(self) -> Dict[str, Any]:
        """Export SCM to dictionary format."""
        return {
            'graph': self.graph.to_dict(),
            'seed': self.seed,
            'mechanisms': {
                var: {
                    'variable': mech.variable,
                    'parents': mech.parents,
                    'is_deterministic': mech.is_deterministic,
                }
                for var, mech in self.mechanisms.items()
            },
        }


def create_linear_mechanism(
    variable: str,
    parents: List[str],
    coefficients: Optional[Dict[str, float]] = None,
    intercept: float = 0.0,
    noise_std: float = 1.0,
) -> CausalMechanism:
    """Create linear causal mechanism: X = intercept + sum(coef * parent) + noise.

    Args:
        variable: Target variable
        parents: Parent variables
        coefficients: Coefficients for each parent (default: 1.0)
        intercept: Intercept term
        noise_std: Standard deviation of Gaussian noise

    Returns:
        CausalMechanism with linear function

    Example:
        >>> # heart_rate = 70 + 0.5 * temperature + noise
        >>> mech = create_linear_mechanism(
        ...     variable='heart_rate',
        ...     parents=['temperature'],
        ...     coefficients={'temperature': 0.5},
        ...     intercept=70.0,
        ...     noise_std=5.0
        ... )
    """
    if coefficients is None:
        coefficients = {p: 1.0 for p in parents}

    def linear_function(parent_values, noise):
        return (
            intercept
            + sum(coefficients.get(p, 1.0) * parent_values.get(p, 0.0) for p in parents)
            + noise
        )

    return CausalMechanism(
        variable=variable,
        parents=parents,
        function=linear_function,
        noise_distribution=lambda rng: rng.normal(0, noise_std),
        is_deterministic=False,
    )


def create_nonlinear_mechanism(
    variable: str,
    parents: List[str],
    function_expr: Callable[[Dict[str, Any]], float],
    noise_std: float = 1.0,
) -> CausalMechanism:
    """Create nonlinear causal mechanism.

    Args:
        variable: Target variable
        parents: Parent variables
        function_expr: Function of parents (excluding noise)
        noise_std: Standard deviation of noise

    Returns:
        CausalMechanism with nonlinear function

    Example:
        >>> # Sigmoid response: Y = sigmoid(X) + noise
        >>> mech = create_nonlinear_mechanism(
        ...     variable='response',
        ...     parents=['dose'],
        ...     function_expr=lambda p: 1 / (1 + np.exp(-p['dose'])),
        ...     noise_std=0.1
        ... )
    """

    def nonlinear_function(parent_values, noise):
        return function_expr(parent_values) + noise

    return CausalMechanism(
        variable=variable,
        parents=parents,
        function=nonlinear_function,
        noise_distribution=lambda rng: rng.normal(0, noise_std),
        is_deterministic=False,
    )
