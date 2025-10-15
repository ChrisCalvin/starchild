"""
Defines the gating mechanism for modulating information flow between hierarchical layers.
"""

import torch
import torch.nn as nn
from typing import Tuple

class GatingMechanism(nn.Module):
    """
    A Feature-wise Linear Modulation (FiLM) layer that uses a gating signal
    to modulate an input tensor.
    """
    def __init__(self, signal_dim: int, tensor_dim: int):
        """
        Initializes the GatingMechanism.

        Args:
            signal_dim: The dimension of the gating signal (e.g., S2 coefficient vector).
            tensor_dim: The dimension of the tensor to be gated (e.g., the belief state z).
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.tensor_dim = tensor_dim

        # A small neural network to map the gating signal to modulation parameters.
        # It outputs 2 * tensor_dim: `tensor_dim` for gamma (scale) and `tensor_dim` for beta (shift).
        self.gating_network = nn.Linear(signal_dim, 2 * tensor_dim)

        # Initialize weights to produce near-identity transformation at the start.
        # This promotes stability during early stages of training.
        nn.init.zeros_(self.gating_network.bias)
        nn.init.zeros_(self.gating_network.weight)

    def forward(self, tensor_to_gate: torch.Tensor, gating_signal: torch.Tensor) -> torch.Tensor:
        """
        Applies the gating mechanism.

        Args:
            tensor_to_gate: The tensor to be modulated (e.g., belief state z).
            gating_signal: The signal used to control the gate (e.g., S2 coefficients).

        Returns:
            The modulated tensor.
        """
        # Generate gamma and beta from the gating signal
        # The weights are initialized to zero, so gamma starts at 1 and beta at 0.
        gamma, beta = self.gating_network(gating_signal).chunk(2, dim=-1)
        gamma = gamma + 1 # Center gamma around 1

        # Apply the feature-wise linear modulation
        return (gamma * tensor_to_gate) + beta
