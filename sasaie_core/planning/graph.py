"""
Defines the basic data structures for the Forney Factor Graph.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import torch
from dataclasses import dataclass

@dataclass
class Message:
    """
    Represents a message passed between nodes and factors.
    For now, we model messages as Normal distributions.
    """
    loc: torch.Tensor
    scale: torch.Tensor

class Node:
    """
    Represents a variable in the factor graph (e.g., state, action, observation).
    """
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self.belief: Optional[Message] = None
        self.factors: List['Factor'] = []
        self.incoming_messages: Dict[str, Message] = {}
        self.pruned: bool = False
        self.is_observed: bool = False

    def update(self):
        """
        VMP node update rule.
        If the node is observed, it just sends its fixed belief to all factors.
        Otherwise, it calculates its belief by taking the product of all incoming messages
        and then calculates and sends outgoing messages.
        """
        epsilon = 1e-6 # A small positive value to prevent division by zero or negative variance

        if self.is_observed:
            if self.belief:
                for factor in self.factors:
                    factor.receive_message(self.name, self.belief)
            return

        if not self.incoming_messages:
            return

        # Calculate belief (product of ALL incoming messages)
        total_precision = torch.zeros(self.dim)
        total_weighted_mean = torch.zeros(self.dim)

        for msg in self.incoming_messages.values():
            precision = 1.0 / (msg.scale ** 2 + epsilon)
            total_precision += precision
            total_weighted_mean += msg.loc * precision
        
        # Avoid division by zero if no messages have arrived yet
        if torch.any(total_precision <= epsilon):
            return

        belief_scale = torch.sqrt(1.0 / total_precision)
        belief_loc = total_weighted_mean / total_precision
        self.belief = Message(loc=belief_loc, scale=belief_scale)

        # Calculate and send outgoing messages to each connected factor
        for factor in self.factors:
            # The message to a factor is the product of all messages *except* the one from that factor.
            messages_for_this_factor = [msg for name, msg in self.incoming_messages.items() if name != factor.name]
            
            if not messages_for_this_factor:
                continue

            factor_precision = torch.zeros(self.dim)
            factor_weighted_mean = torch.zeros(self.dim)

            for msg in messages_for_this_factor:
                precision = 1.0 / (msg.scale ** 2 + epsilon)
                factor_precision += precision
                factor_weighted_mean += msg.loc * precision
            
            if torch.any(factor_precision <= epsilon):
                continue

            outgoing_scale = torch.sqrt(1.0 / factor_precision)
            outgoing_loc = factor_weighted_mean / factor_precision
            
            factor.receive_message(self.name, Message(loc=outgoing_loc, scale=outgoing_scale))

    def receive_message(self, from_factor_name: str, message: 'Message'):
        """Receives a message from a connected factor."""
        self.incoming_messages[from_factor_name] = message

    def __repr__(self) -> str:
        return f"Node({self.name})"

class Factor:
    """
    Represents a factor in the graph, defining a probabilistic relationship
    between connected nodes.
    """
    def __init__(self, name: str, nodes: List[Node]):
        self.name = name
        self.nodes = nodes
        self.incoming_messages: Dict[str, Message] = {}
        self.pruned: bool = False
        self.last_updated_iteration: int = -1
        self.total_updates: int = 0
        for node in nodes:
            node.factors.append(self)

    def receive_message(self, from_node_name: str, message: 'Message'):
        """Receives a message from a connected node."""
        self.incoming_messages[from_node_name] = message

    def update(self):
        """Placeholder for the VMP factor update rule."""
        pass

    def __repr__(self) -> str:
        node_names = ", ".join([node.name for node in self.nodes])
        return f"Factor({self.name}, nodes=[{node_names}])"

class GaussianFactor(Factor):
    """
    A factor representing a linear Gaussian relationship: y = Ax + b + noise.
    For simplicity, we'll initially handle a 2-node case: y = x + b + noise (A=I).
    """
    def __init__(self, name: str, nodes: List[Node], b: torch.Tensor, noise_scale: torch.Tensor):
        super().__init__(name, nodes)
        if len(nodes) != 2:
            raise ValueError("GaussianFactor currently supports only 2 nodes (x and y).")
        self.b = b
        self.noise_scale = noise_scale

    def update(self):
        """
        VMP factor update rule for a simple linear Gaussian relationship.
        Calculates the outgoing message to each connected node.
        """
        # Assume nodes are [x, y]
        x_node, y_node = self.nodes

        # --- Calculate message to y_node --- #
        # Message to y is N(y | μ_y, σ_y^2) where μ_y = E[x] + b and σ_y^2 = Var(x) + σ_noise^2
        msg_from_x = self.incoming_messages.get(x_node.name)
        if msg_from_x:
            outgoing_loc_to_y = msg_from_x.loc + self.b
            outgoing_scale_to_y = torch.sqrt(msg_from_x.scale**2 + self.noise_scale**2)
            y_node.receive_message(self.name, Message(loc=outgoing_loc_to_y, scale=outgoing_scale_to_y))

        # --- Calculate message to x_node --- #
        # Message to x is N(x | μ_x, σ_x^2) where μ_x = E[y] - b and σ_x^2 = Var(y) + σ_noise^2
        msg_from_y = self.incoming_messages.get(y_node.name)
        if msg_from_y:
            outgoing_loc_to_x = msg_from_y.loc - self.b
            outgoing_scale_to_x = torch.sqrt(msg_from_y.scale**2 + self.noise_scale**2)
            x_node.receive_message(self.name, Message(loc=outgoing_loc_to_x, scale=outgoing_scale_to_x))

class FactorGraph:
    """
    Represents the complete factor graph, containing all nodes and factors.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.factors: Dict[str, Factor] = {}

    def add_node(self, name: str, dim: int) -> Node:
        """Adds a new node to the graph."""
        if name in self.nodes:
            raise ValueError(f"Node with name '{name}' already exists.")
        node = Node(name, dim)
        self.nodes[name] = node
        return node

    def add_factor(self, name: str, node_names: List[str]) -> Factor:
        """Adds a new factor to the graph, connecting to nodes by their names."""
        if name in self.factors:
            raise ValueError(f"Factor with name '{name}' already exists.")
        
        nodes_to_connect = []
        for node_name in node_names:
            if node_name not in self.nodes:
                raise ValueError(f"Node '{node_name}' not found in graph.")
            nodes_to_connect.append(self.nodes[node_name])
            
        factor = Factor(name, nodes_to_connect)
        self.factors[name] = factor
        return factor

    def get_node(self, name: str) -> Optional[Node]:
        """Retrieves a node by its name."""
        return self.nodes.get(name)

    def get_factor(self, name: str) -> Optional[Factor]:
        """Retrieves a factor by its name."""
        return self.factors.get(name)

    def __repr__(self) -> str:
        return f"FactorGraph(nodes={list(self.nodes.keys())}, factors={list(self.factors.keys())})"
