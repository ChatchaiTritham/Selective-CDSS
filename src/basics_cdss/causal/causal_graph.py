"""Causal graph construction for clinical domains.

This module provides tools for defining and manipulating causal graphs
(Directed Acyclic Graphs) representing causal relationships in clinical
scenarios.

Theoretical foundation:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class CausalEdge:
    """Represents a causal edge in the graph.

    Attributes:
        cause: Parent variable (cause)
        effect: Child variable (effect)
        strength: Optional edge weight (causal strength)
        mechanism: Optional description of causal mechanism
    """

    cause: str
    effect: str
    strength: Optional[float] = None
    mechanism: Optional[str] = None


class CausalGraph:
    """Directed Acyclic Graph (DAG) representing causal relationships.

    A causal graph encodes assumptions about cause-effect relationships
    in a clinical domain. Edges represent direct causal influence.

    Example:
        >>> # Create causal graph for sepsis
        >>> graph = CausalGraph()
        >>> graph.add_edge('infection', 'temperature')
        >>> graph.add_edge('infection', 'white_blood_cell_count')
        >>> graph.add_edge('temperature', 'heart_rate')
        >>>
        >>> # Check if graph is valid DAG
        >>> assert graph.is_dag()
        >>>
        >>> # Find parents and children
        >>> print(graph.get_parents('temperature'))  # ['infection']
        >>> print(graph.get_children('infection'))   # ['temperature', 'wbc']
    """

    def __init__(self):
        """Initialize empty causal graph."""
        self.graph = nx.DiGraph()
        self.edge_metadata: Dict[Tuple[str, str], CausalEdge] = {}

    def add_node(self, variable: str, **attributes):
        """Add variable to graph.

        Args:
            variable: Variable name
            **attributes: Additional node attributes (e.g., type='continuous')
        """
        self.graph.add_node(variable, **attributes)

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: Optional[float] = None,
        mechanism: Optional[str] = None,
    ):
        """Add causal edge to graph.

        Args:
            cause: Parent variable (cause)
            effect: Child variable (effect)
            strength: Optional causal strength
            mechanism: Optional description of mechanism

        Raises:
            ValueError: If adding edge would create a cycle
        """
        # Check if edge would create cycle
        if effect in self.get_ancestors(cause):
            raise ValueError(f"Adding edge {cause} → {effect} would create a cycle")

        self.graph.add_edge(cause, effect)

        # Store metadata
        edge = CausalEdge(cause, effect, strength, mechanism)
        self.edge_metadata[(cause, effect)] = edge

    def remove_edge(self, cause: str, effect: str):
        """Remove causal edge."""
        if self.graph.has_edge(cause, effect):
            self.graph.remove_edge(cause, effect)
            if (cause, effect) in self.edge_metadata:
                del self.edge_metadata[(cause, effect)]

    def get_parents(self, variable: str) -> List[str]:
        """Get direct causes (parents) of variable."""
        return list(self.graph.predecessors(variable))

    def get_children(self, variable: str) -> List[str]:
        """Get direct effects (children) of variable."""
        return list(self.graph.successors(variable))

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors (causes) of variable."""
        return nx.ancestors(self.graph, variable)

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendants (effects) of variable."""
        return nx.descendants(self.graph, variable)

    def is_dag(self) -> bool:
        """Check if graph is a valid DAG (no cycles)."""
        return nx.is_directed_acyclic_graph(self.graph)

    def topological_order(self) -> List[str]:
        """Return variables in topological order (causes before effects).

        Returns:
            List of variables ordered such that parents come before children

        Raises:
            ValueError: If graph contains cycles
        """
        if not self.is_dag():
            raise ValueError("Graph contains cycles")
        return list(nx.topological_sort(self.graph))

    def d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """Check if X and Y are d-separated given Z.

        D-separation determines conditional independence in causal graphs.

        Args:
            X: First set of variables
            Y: Second set of variables
            Z: Conditioning set

        Returns:
            True if X ⊥⊥ Y | Z (X independent of Y given Z)
        """
        return nx.d_separated(self.graph, X, Y, Z)

    def get_markov_blanket(self, variable: str) -> Set[str]:
        """Get Markov blanket of variable.

        Markov blanket = parents + children + parents of children

        Args:
            variable: Target variable

        Returns:
            Set of variables in Markov blanket
        """
        parents = set(self.get_parents(variable))
        children = set(self.get_children(variable))

        # Parents of children (spouses)
        spouses = set()
        for child in children:
            spouses.update(self.get_parents(child))
        spouses.discard(variable)

        return parents | children | spouses

    def visualize(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_color: str = '#87CEEB',
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize causal graph.

        Args:
            figsize: Figure size
            node_color: Color for nodes
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(self.graph, seed=42, k=2)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_color, node_size=3000, alpha=0.9, ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=2,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, font_size=10, font_weight='bold', ax=ax
        )

        ax.set_title('Causal Graph', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': [
                {
                    'cause': edge.cause,
                    'effect': edge.effect,
                    'strength': edge.strength,
                    'mechanism': edge.mechanism,
                }
                for edge in self.edge_metadata.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """Create graph from dictionary."""
        graph = cls()

        # Add nodes
        for node in data['nodes']:
            graph.add_node(node)

        # Add edges
        for edge_data in data['edges']:
            graph.add_edge(
                edge_data['cause'],
                edge_data['effect'],
                strength=edge_data.get('strength'),
                mechanism=edge_data.get('mechanism'),
            )

        return graph


def create_sepsis_causal_graph() -> CausalGraph:
    """Create causal graph for sepsis domain.

    Based on clinical knowledge of sepsis pathophysiology:
    - Infection drives inflammatory response
    - Temperature, WBC, HR respond to infection
    - Hemodynamics affected by inflammation

    Returns:
        CausalGraph for sepsis

    Example:
        >>> graph = create_sepsis_causal_graph()
        >>> graph.visualize()
    """
    graph = CausalGraph()

    # Add nodes
    nodes = [
        'age',
        'comorbidities',
        'infection',
        'infection_severity',
        'temperature',
        'heart_rate',
        'respiratory_rate',
        'white_blood_cell_count',
        'blood_pressure_sys',
        'lactate',
        'antibiotic',
        'fluid_bolus',
        'outcome',
    ]

    for node in nodes:
        graph.add_node(node)

    # Baseline factors
    graph.add_edge(
        'age', 'comorbidities', mechanism='Older patients have more comorbidities'
    )
    graph.add_edge(
        'comorbidities',
        'infection_severity',
        mechanism='Comorbidities worsen infection',
    )
    graph.add_edge('age', 'baseline_heart_rate')

    # Infection dynamics
    graph.add_edge(
        'infection', 'infection_severity', mechanism='Infection drives severity'
    )
    graph.add_edge(
        'infection_severity', 'temperature', mechanism='Inflammation causes fever'
    )
    graph.add_edge(
        'infection_severity', 'white_blood_cell_count', mechanism='Immune response'
    )
    graph.add_edge('infection_severity', 'lactate', mechanism='Tissue hypoperfusion')

    # Hemodynamic cascade
    graph.add_edge(
        'temperature', 'heart_rate', mechanism='Fever increases metabolic demand'
    )
    graph.add_edge(
        'infection_severity',
        'blood_pressure_sys',
        mechanism='Vasodilation from inflammation',
    )
    graph.add_edge(
        'infection_severity', 'heart_rate', mechanism='Compensatory tachycardia'
    )
    graph.add_edge(
        'infection_severity',
        'respiratory_rate',
        mechanism='Metabolic acidosis compensation',
    )

    # Interventions
    graph.add_edge(
        'antibiotic',
        'infection_severity',
        mechanism='Antibiotics reduce bacterial load',
    )
    graph.add_edge('fluid_bolus', 'blood_pressure_sys', mechanism='Volume expansion')
    graph.add_edge('fluid_bolus', 'lactate', mechanism='Improved perfusion')

    # Outcome
    graph.add_edge(
        'infection_severity', 'outcome', mechanism='Disease severity determines outcome'
    )
    graph.add_edge('age', 'outcome', mechanism='Age affects mortality')
    graph.add_edge('blood_pressure_sys', 'outcome', mechanism='Hemodynamic stability')
    graph.add_edge('lactate', 'outcome', mechanism='Perfusion marker')

    return graph


def create_respiratory_causal_graph() -> CausalGraph:
    """Create causal graph for respiratory distress (ARDS).

    Returns:
        CausalGraph for ARDS
    """
    graph = CausalGraph()

    # Add nodes
    for node in [
        'age',
        'lung_injury',
        'oxygen_saturation',
        'respiratory_rate',
        'pf_ratio',
        'heart_rate',
        'oxygen_therapy',
        'peep',
        'prone_positioning',
        'outcome',
    ]:
        graph.add_node(node)

    # Causal structure
    graph.add_edge('lung_injury', 'oxygen_saturation')
    graph.add_edge('lung_injury', 'pf_ratio')
    graph.add_edge('lung_injury', 'respiratory_rate')
    graph.add_edge('oxygen_saturation', 'heart_rate')

    # Interventions
    graph.add_edge('oxygen_therapy', 'oxygen_saturation')
    graph.add_edge('peep', 'pf_ratio')
    graph.add_edge('peep', 'oxygen_saturation')
    graph.add_edge('prone_positioning', 'pf_ratio')

    # Outcome
    graph.add_edge('pf_ratio', 'outcome')
    graph.add_edge('oxygen_saturation', 'outcome')
    graph.add_edge('age', 'outcome')

    return graph


def create_cardiac_causal_graph() -> CausalGraph:
    """Create causal graph for cardiac events (MI/ACS).

    Returns:
        CausalGraph for cardiac events
    """
    graph = CausalGraph()

    # Add nodes
    for node in [
        'age',
        'risk_factors',
        'coronary_occlusion',
        'ischemia',
        'troponin',
        'st_elevation',
        'chest_pain',
        'heart_rate',
        'blood_pressure_sys',
        'aspirin',
        'nitrate',
        'beta_blocker',
        'pci',
        'outcome',
    ]:
        graph.add_node(node)

    # Risk factors
    graph.add_edge('age', 'risk_factors')
    graph.add_edge('risk_factors', 'coronary_occlusion')

    # Ischemic cascade
    graph.add_edge('coronary_occlusion', 'ischemia')
    graph.add_edge('ischemia', 'troponin')
    graph.add_edge('ischemia', 'st_elevation')
    graph.add_edge('ischemia', 'chest_pain')
    graph.add_edge('ischemia', 'heart_rate')
    graph.add_edge('ischemia', 'blood_pressure_sys')

    # Interventions
    graph.add_edge('aspirin', 'coronary_occlusion')
    graph.add_edge('nitrate', 'chest_pain')
    graph.add_edge('nitrate', 'blood_pressure_sys')
    graph.add_edge('beta_blocker', 'heart_rate')
    graph.add_edge('beta_blocker', 'blood_pressure_sys')
    graph.add_edge('pci', 'coronary_occlusion')

    # Outcome
    graph.add_edge('ischemia', 'outcome')
    graph.add_edge('age', 'outcome')

    return graph
