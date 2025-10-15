"""
Defines the core components for Variational Message Passing (VMP).
"""

import torch
from typing import Dict

from sasaie_core.planning.graph import FactorGraph, Node, Factor, Message

class Scheduler:
    """
    Manages the order of message passing updates in the factor graph.
    """
    def __init__(self, graph: FactorGraph):
        self.graph = graph
        # A simple schedule: update all factors, then all nodes.
        self.schedule = list(graph.factors.values()) + list(graph.nodes.values())
        self._current_absolute_iteration = 0 # New attribute for cumulative iteration count

    def run(self, iterations: int = 10):
        """
        Runs a structured message passing schedule for a given number of iterations.
        This implementation ensures a full update cycle in each iteration.
        """
        # The old schedule logic was flawed. This new logic implements a more standard
        # schedule that ensures messages are passed back and forth correctly.
        print(f"Starting VMP schedule for {iterations} iterations...")
        
        factors = list(self.graph.factors.values())
        nodes = list(self.graph.nodes.values())

        for i in range(iterations):
            # 1. Update all factors. Factors compute messages to send to nodes.
            for factor in factors:
                factor.update()

            # 2. Update all nodes. Nodes receive messages, update their beliefs,
            #    and send new messages back to factors.
            for node in nodes:
                node.update()

        print("VMP schedule complete.")