"""Counterfactual evaluation for CDSS using digital twins.

This module enables "what-if" analysis: comparing CDSS recommendations
against alternative actions to identify potential decision regret.

Key capabilities:
- Simulate patient outcomes under different interventions
- Compute regret (difference between CDSS action and optimal action)
- Identify cases where CDSS decisions lead to worse outcomes
- Support safety-critical deployment decisions
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from basics_cdss.temporal.digital_twin import PatientDigitalTwin, PatientState


@dataclass
class CounterfactualResult:
    """Results from counterfactual evaluation.

    Attributes:
        twin_id: Identifier for digital twin
        factual_action: Action actually taken by CDSS
        factual_outcome: Resulting patient state/outcome
        factual_harm: Harm score for factual trajectory
        counterfactuals: List of alternative scenarios
        best_alternative: Best alternative action found
        regret: Harm difference (best_alternative - factual)
        regret_percent: Regret as percentage of factual harm
    """

    twin_id: str
    factual_action: Dict[str, Any]
    factual_outcome: PatientState
    factual_harm: float
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    best_alternative: Optional[Dict[str, Any]] = None
    regret: float = 0.0
    regret_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame export."""
        return {
            'twin_id': self.twin_id,
            'factual_harm': self.factual_harm,
            'best_alternative_harm': (
                self.best_alternative['harm'] if self.best_alternative else None
            ),
            'regret': self.regret,
            'regret_percent': self.regret_percent,
            'n_alternatives': len(self.counterfactuals),
        }


class CounterfactualEvaluator:
    """Evaluate CDSS decisions using counterfactual reasoning.

    This evaluator simulates what would have happened under different
    interventions and compares outcomes to identify potential regret.

    Example:
        >>> from basics_cdss.temporal import (
        ...     CounterfactualEvaluator,
        ...     PatientDigitalTwin,
        ...     SepsisModel
        ... )
        >>>
        >>> # Create evaluator
        >>> evaluator = CounterfactualEvaluator(
        ...     horizon_hours=24,
        ...     harm_function=lambda state: state['_infection_severity']
        ... )
        >>>
        >>> # Evaluate CDSS model
        >>> results = evaluator.evaluate(cdss_model, digital_twins)
        >>>
        >>> # Analyze regret
        >>> summary = evaluator.summarize_results(results)
        >>> print(f"Mean regret: {summary['mean_regret']:.3f}")
        >>> print(f"High-regret cases: {summary['high_regret_count']}")
    """

    def __init__(
        self,
        horizon_hours: float = 24.0,
        dt: float = 1.0,
        harm_function: Optional[Callable[[PatientState], float]] = None,
        alternative_generator: Optional[Callable] = None,
        intervention_time: float = 0.0,
    ):
        """Initialize counterfactual evaluator.

        Args:
            horizon_hours: Simulation time horizon
            dt: Time step for simulation
            harm_function: Function to compute harm from patient state
                Default: infection severity (for sepsis)
            alternative_generator: Function to generate alternative actions
                Default: standard intervention variations
            intervention_time: When to apply intervention (hours from start)
        """
        self.horizon_hours = horizon_hours
        self.dt = dt
        self.intervention_time = intervention_time

        # Default harm function (for sepsis)
        self.harm_function = harm_function or self._default_harm_function

        # Alternative action generator
        self.alternative_generator = alternative_generator or self._default_alternatives

    def _default_harm_function(self, state: PatientState) -> float:
        """Default harm function based on infection severity."""
        # Higher values = worse outcome
        infection = state.features.get('_infection_severity', 0.0)
        temp_deviation = abs(state.features.get('temperature', 37.0) - 37.0)
        bp_deficit = max(0, 90 - state.features.get('blood_pressure_sys', 120))

        harm = 10.0 * infection + 2.0 * temp_deviation + 0.5 * bp_deficit
        return harm

    def _default_alternatives(
        self, current_action: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default alternative actions.

        For sepsis: try different combinations of interventions.
        """
        alternatives = []

        # Baseline interventions
        base_options = {
            'antibiotic': [False, True],
            'fluid_bolus': [0, 500, 1000],
            'vasopressor': [False, True],
        }

        # Generate combinations (limited to avoid combinatorial explosion)
        alternatives.append({})  # No intervention

        alternatives.append({'antibiotic': True})
        alternatives.append({'fluid_bolus': 1000})
        alternatives.append({'vasopressor': True})

        alternatives.append({'antibiotic': True, 'fluid_bolus': 1000})
        alternatives.append({'antibiotic': True, 'vasopressor': True})
        alternatives.append({'fluid_bolus': 1000, 'vasopressor': True})

        alternatives.append(
            {'antibiotic': True, 'fluid_bolus': 1000, 'vasopressor': True}
        )

        return alternatives

    def evaluate_single_twin(
        self,
        cdss_model: Any,
        twin: PatientDigitalTwin,
        factual_action: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualResult:
        """Evaluate counterfactuals for single digital twin.

        Args:
            cdss_model: CDSS model with .predict() method
            twin: Digital twin to evaluate
            factual_action: CDSS-recommended action (if None, query model)

        Returns:
            Counterfactual evaluation results
        """
        # Get CDSS recommendation if not provided
        if factual_action is None:
            # CDSS model should have predict method returning intervention dict
            factual_action = cdss_model.predict(twin.current_state.features)

        # Simulate factual trajectory (what actually happened)
        factual_twin = twin.clone()
        factual_twin.reset()

        intervention_schedule = {self.intervention_time: factual_action}
        factual_trajectory = factual_twin.simulate(
            horizon_hours=self.horizon_hours,
            dt=self.dt,
            intervention_schedule=intervention_schedule,
            stochastic=False,  # Deterministic for fair comparison
        )

        # Compute factual outcome harm
        factual_outcome = factual_trajectory[-1]
        factual_harm = self.harm_function(factual_outcome)

        # Generate alternative actions
        alternatives = self.alternative_generator(factual_action)

        # Evaluate each counterfactual
        counterfactuals = []
        for alt_action in alternatives:
            cf_twin = twin.clone()
            cf_twin.reset()

            alt_schedule = {self.intervention_time: alt_action}
            cf_trajectory = cf_twin.simulate(
                horizon_hours=self.horizon_hours,
                dt=self.dt,
                intervention_schedule=alt_schedule,
                stochastic=False,
            )

            cf_outcome = cf_trajectory[-1]
            cf_harm = self.harm_function(cf_outcome)

            counterfactuals.append(
                {
                    'action': alt_action,
                    'outcome': cf_outcome,
                    'harm': cf_harm,
                    'trajectory': cf_trajectory,
                }
            )

        # Find best alternative
        best_cf = min(counterfactuals, key=lambda x: x['harm'])

        # Compute regret
        regret = best_cf['harm'] - factual_harm  # Negative = CDSS was suboptimal
        regret_percent = (regret / factual_harm * 100) if factual_harm > 0 else 0.0

        return CounterfactualResult(
            twin_id=twin.archetype_id,
            factual_action=factual_action,
            factual_outcome=factual_outcome,
            factual_harm=factual_harm,
            counterfactuals=counterfactuals,
            best_alternative=best_cf,
            regret=regret,
            regret_percent=regret_percent,
        )

    def evaluate(
        self,
        cdss_model: Any,
        digital_twins: List[PatientDigitalTwin],
        factual_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CounterfactualResult]:
        """Evaluate CDSS across multiple digital twins.

        Args:
            cdss_model: CDSS model with .predict() method
            digital_twins: List of digital twins
            factual_actions: Optional list of CDSS actions (if None, query model)

        Returns:
            List of counterfactual results
        """
        results = []

        for i, twin in enumerate(digital_twins):
            factual_action = None
            if factual_actions and i < len(factual_actions):
                factual_action = factual_actions[i]

            result = self.evaluate_single_twin(cdss_model, twin, factual_action)
            results.append(result)

        return results

    def summarize_results(self, results: List[CounterfactualResult]) -> Dict[str, Any]:
        """Compute summary statistics from counterfactual results.

        Args:
            results: List of counterfactual evaluation results

        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {}

        regrets = [r.regret for r in results]
        factual_harms = [r.factual_harm for r in results]

        # Identify high-regret cases (CDSS was significantly suboptimal)
        high_regret_threshold = np.percentile(regrets, 10)  # Bottom 10%
        high_regret_cases = [r for r in results if r.regret < high_regret_threshold]

        summary = {
            'n_cases': len(results),
            'mean_regret': np.mean(regrets),
            'median_regret': np.median(regrets),
            'std_regret': np.std(regrets),
            'min_regret': np.min(regrets),
            'max_regret': np.max(regrets),
            'mean_factual_harm': np.mean(factual_harms),
            'high_regret_count': len(high_regret_cases),
            'high_regret_threshold': high_regret_threshold,
            'fraction_suboptimal': np.mean([r.regret < 0 for r in results]),
        }

        return summary

    def to_dataframe(self, results: List[CounterfactualResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Args:
            results: List of counterfactual results

        Returns:
            DataFrame with one row per twin
        """
        records = [r.to_dict() for r in results]
        return pd.DataFrame(records)

    def identify_critical_cases(
        self, results: List[CounterfactualResult], regret_threshold: float = -1.0
    ) -> List[CounterfactualResult]:
        """Identify cases where CDSS decisions led to high regret.

        Args:
            results: Counterfactual results
            regret_threshold: Threshold for "critical" regret (negative = worse than alternative)

        Returns:
            List of critical cases requiring review
        """
        critical_cases = [r for r in results if r.regret < regret_threshold]

        # Sort by regret (most critical first)
        critical_cases.sort(key=lambda r: r.regret)

        return critical_cases


class MockCDSSModel:
    """Mock CDSS model for testing counterfactual evaluation.

    This is a simple rule-based model for demonstration purposes.
    In practice, replace with actual ML-based CDSS.
    """

    def __init__(self, strategy: str = 'conservative'):
        """Initialize mock CDSS.

        Args:
            strategy: Decision strategy ('conservative', 'aggressive', 'balanced')
        """
        self.strategy = strategy

    def predict(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict intervention based on patient state.

        Args:
            patient_state: Current patient features

        Returns:
            Dictionary of interventions
        """
        temp = patient_state.get('temperature', 37.0)
        hr = patient_state.get('heart_rate', 80)
        bp = patient_state.get('blood_pressure_sys', 120)

        interventions = {}

        if self.strategy == 'conservative':
            # Only intervene if clearly abnormal
            if temp > 38.5:
                interventions['antibiotic'] = True
            if bp < 90:
                interventions['fluid_bolus'] = 1000

        elif self.strategy == 'aggressive':
            # Intervene early
            if temp > 37.5:
                interventions['antibiotic'] = True
            if bp < 100 or hr > 100:
                interventions['fluid_bolus'] = 1000
            if bp < 85:
                interventions['vasopressor'] = True

        elif self.strategy == 'balanced':
            # Moderate intervention threshold
            if temp > 38.0:
                interventions['antibiotic'] = True
            if bp < 95:
                interventions['fluid_bolus'] = 500

        return interventions
