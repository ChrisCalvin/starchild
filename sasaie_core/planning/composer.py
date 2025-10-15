"""
Defines the GraphComposer for dynamically building a factor graph from a policy.
"""

from typing import List

from sasaie_core.planning.graph import FactorGraph, Node, Factor
from sasaie_core.evolution.registry import GraphComponentRegistry
from sasaie_core.evolution.morphisms import RGMorphism

class GraphComposer:
    """
    Dynamically instantiates a Forney Factor Graph by chaining RG-Morphisms
    based on a high-level policy representation.
    """

    def __init__(self, registry: GraphComponentRegistry):
        """
        Initializes the GraphComposer.

        Args:
            registry: The registry containing all available RG-Morphisms.
        """
        self.registry = registry

    def compose_graph(self, policy_representation: List[str], initial_state_dim: int) -> FactorGraph:
        """
        Builds a factor graph by chaining RG-Morphisms.

        Args:
            policy_representation: A list of names of RG-Morphisms to compose.
            initial_state_dim: The dimensionality of the initial state node.

        Returns:
            A FactorGraph instance representing the composed policy.
        """
        graph = FactorGraph()
        
        if not policy_representation:
            return graph

        # Create the initial state node
        current_state_node = graph.add_node(name="state_0", dim=initial_state_dim)

        for i, morphism_name in enumerate(policy_representation):
            morphism = self.registry.get_morphism(morphism_name)
            if not morphism:
                raise ValueError(f"Morphism '{morphism_name}' not found in registry.")

            # Create the next state node
            next_state_node = graph.add_node(name=f"state_{i+1}", dim=morphism.io_dim)

            # Add the morphism as a factor connecting the current and next states
            # Note: We are treating the morphism itself as the factor for simplicity.
            # In a more complex system, the morphism might be a subgraph of factors.
            graph.add_factor(
                name=morphism.name,
                node_names=[current_state_node.name, next_state_node.name]
            )

            # The next iteration starts from the state produced by this morphism
            current_state_node = next_state_node
            
        return graph
