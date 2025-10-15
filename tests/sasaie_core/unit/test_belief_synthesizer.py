"""
Unit tests for the BeliefSynthesizerGAT component.
"""

import pytest
import torch

from sasaie_core.components.global_belief_fusion import BeliefSynthesizerGAT

@pytest.fixture
def synthesizer_config():
    """Provides a sample configuration for the GAT synthesizer."""
    return {
        'scale_dims': {
            'fast': 4,
            'medium': 8,
            'slow': 8
        },
        'common_dim': 16, # New common dimension for projection
        'output_dim': 64,
        'heads': 4
    }

@pytest.fixture
def dummy_belief_states(synthesizer_config):
    """Provides a dummy dictionary of belief states matching the config."""
    return {
        name: torch.randn(1, dim) 
        for name, dim in synthesizer_config['scale_dims'].items()
    }


def test_synthesizer_initialization(synthesizer_config):
    """Tests that the BeliefSynthesizerGAT initializes correctly."""
    synthesizer = BeliefSynthesizerGAT(**synthesizer_config)
    assert synthesizer is not None
    assert len(synthesizer.scale_names) == 3
    # Check if the output layer has the correct dimension
    assert synthesizer.out.out_features == synthesizer_config['output_dim']

def test_synthesizer_forward_pass(synthesizer_config, dummy_belief_states):
    """Tests the forward pass of the synthesizer."""
    synthesizer = BeliefSynthesizerGAT(**synthesizer_config)
    synthesizer.train() # Set to training mode

    # Execute the forward pass
    consolidated_belief = synthesizer(dummy_belief_states)

    # --- Assertions ---
    # Check the output shape
    assert consolidated_belief.shape == (1, synthesizer_config['output_dim'])

    # Check that the output has a gradient function, indicating it's part of a computation graph
    assert consolidated_belief.requires_grad

    # Test a simple backward pass
    try:
        consolidated_belief.sum().backward()
    except Exception as e:
        pytest.fail(f"Backward pass failed with exception: {e}")
