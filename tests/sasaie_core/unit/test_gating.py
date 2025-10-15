"""
Unit tests for the GatingMechanism module.
"""

import torch
import torch.nn as nn
import pytest
from sasaie_core.components.gating import GatingMechanism

@pytest.fixture
def gating_module():
    """Provides a GatingMechanism instance for testing."""
    return GatingMechanism(signal_dim=2, tensor_dim=10)

class TestGatingMechanism:

    def test_initialization(self, gating_module):
        """Tests that the GatingMechanism initializes correctly."""
        assert isinstance(gating_module, torch.nn.Module)
        assert gating_module.signal_dim == 2
        assert gating_module.tensor_dim == 10
        assert gating_module.gating_network.in_features == 2
        assert gating_module.gating_network.out_features == 20 # 2 * tensor_dim

    def test_forward_pass_shape(self, gating_module):
        """Tests that the output tensor has the same shape as the input tensor_to_gate."""
        tensor_to_gate = torch.randn(1, 10)
        gating_signal = torch.randn(1, 2)
        
        output = gating_module(tensor_to_gate, gating_signal)
        
        assert output.shape == tensor_to_gate.shape

    def test_identity_initialization(self, gating_module):
        """Tests that the initial state of the gate is close to an identity transformation."""
        tensor_to_gate = torch.randn(1, 10)
        gating_signal = torch.randn(1, 2)

        # Because weights and biases are initialized to zero, the initial output
        # of the gating_network should be all zeros. This means gamma will be 1
        # and beta will be 0, so the output should be very close to the input.
        output = gating_module(tensor_to_gate, gating_signal)

        assert torch.allclose(output, tensor_to_gate, atol=1e-6)

    def test_gating_effect(self):
        """Tests that the gate has a real effect on the output when weights are non-zero."""
        # Create a module with non-zero weights
        module = GatingMechanism(signal_dim=2, tensor_dim=10)
        nn.init.xavier_uniform_(module.gating_network.weight)
        nn.init.normal_(module.gating_network.bias)

        tensor_to_gate = torch.randn(1, 10)
        gating_signal = torch.randn(1, 2)

        output = module(tensor_to_gate, gating_signal)

        # The output should now be different from the input
        assert not torch.allclose(output, tensor_to_gate)
