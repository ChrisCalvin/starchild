"""
Defines the configuration loader for the generative model.
"""

from typing import List, Dict, Any
import yaml
from collections import deque

class GenerativeModelConfigLoader:
    """
    Loads the generative_model.yaml, validates its structure, and provides
    the execution plan for the hierarchical model.
    """

    def __init__(self, config_path: str):
        """
        Initializes the loader with the path to the config file.

        Args:
            config_path: The absolute path to the generative_model.yaml file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scales = self.config.get('model', {}).get('scales', [])
        self._validate_config()
        self.execution_plan = self._generate_execution_plan()

    def _validate_config(self):
        """Validates the structure of the loaded configuration."""
        if not self.scales:
            raise ValueError("Configuration must define at least one scale.")
        scale_names = {scale['name'] for scale in self.scales}
        for scale in self.scales:
            for coupling in scale.get('couplings', []):
                dep = coupling['from_scale']
                if dep not in scale_names:
                    raise ValueError(f"Scale '{scale['name']}' contains an undefined dependency: '{dep}'")

    def _generate_execution_plan(self) -> List[str]:
        """
        Performs a topological sort on the scales based on bottom-up dependencies
        to determine the correct execution order for a single time step.

        Returns:
            A list of scale names in the correct execution order.
        """
        graph = {scale['name']: [] for scale in self.scales}
        for scale in self.scales:
            for coupling in scale.get('couplings', []):
                if 'bottom_up' in coupling['type']:
                    graph[scale['name']].append(coupling['from_scale'])

        in_degree = {name: 0 for name in graph}
        adj = {name: [] for name in graph}

        for name, deps in graph.items():
            for dep in deps:
                adj[dep].append(name)
                in_degree[name] += 1

        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_order = []

        while queue:
            node = queue.popleft()
            sorted_order.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(graph):
            raise ValueError("Circular dependency detected in bottom-up couplings. The model must be a DAG.")

        return sorted_order

    def get_scale_config(self, name: str) -> Dict[str, Any]:
        """Returns the configuration for a specific scale by name."""
        for scale in self.scales:
            if scale['name'] == name:
                return scale
        raise ValueError(f"Scale with name '{name}' not found in configuration.")
