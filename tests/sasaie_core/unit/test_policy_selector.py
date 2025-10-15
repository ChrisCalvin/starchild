"""
Unit tests for the PolicySelectorVAE.
"""

import pytest
import torch
from sasaie_core.planning.policy_selector import PolicySelectorVAE

# Define model parameters for testing
CBS_DIM = 32 # Dimensionality of the ConsolidatedBeliefState
HIDDEN_DIM = 64
POLICY_LATENT_DIM = 16 # Dimensionality of the latent policy space
VOCAB_SIZE = 10 # Number of possible morphisms (skills)
MAX_SEQ_LEN = 5 # Max length of a policy sequence
BATCH_SIZE = 1
CONTEXT_DIM = 8 # New: Dimensionality of the market context
NUM_GOALS = 3 # New: Number of competing goals

@pytest.fixture
def policy_selector_model():
    """Provides a PolicySelectorVAE instance for testing."""
    return PolicySelectorVAE(
        input_dim=CBS_DIM,
        latent_dim=POLICY_LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        context_dim=CONTEXT_DIM, # New
        num_goals=NUM_GOALS # New
    )

@pytest.fixture
def sample_cbs():
    """Provides a sample ConsolidatedBeliefState tensor."""
    return torch.randn(BATCH_SIZE, CBS_DIM)

@pytest.fixture
def sample_context():
    """Provides a sample market context tensor."""
    return torch.randn(BATCH_SIZE, CONTEXT_DIM)

@pytest.fixture
def sample_goal_weights():
    """Provides a sample goal weights tensor."""
    return torch.randn(BATCH_SIZE, NUM_GOALS)

def test_policy_selector_initialization(policy_selector_model):
    """Tests that the PolicySelectorVAE is initialized correctly."""
    assert policy_selector_model is not None
    assert policy_selector_model.latent_dim == POLICY_LATENT_DIM
    assert policy_selector_model.vocab_size == VOCAB_SIZE
    assert policy_selector_model.max_seq_len == MAX_SEQ_LEN
    assert policy_selector_model.context_dim == CONTEXT_DIM # New
    assert policy_selector_model.num_goals == NUM_GOALS # New

def test_policy_selector_forward_pass(policy_selector_model, sample_cbs, sample_context, sample_goal_weights):
    """Tests the forward pass of the VAE returns tensors of correct shape."""
    # Act
    log_probs, mu, logvar = policy_selector_model(sample_cbs, sample_context, sample_goal_weights) # Updated inputs

    # Assert
    assert torch.is_tensor(log_probs)
    assert torch.is_tensor(mu)
    assert torch.is_tensor(logvar)

    # Check shapes
    assert log_probs.shape == (BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE)
    assert mu.shape == (BATCH_SIZE, POLICY_LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, POLICY_LATENT_DIM)

    # Check that log_probs are valid log probabilities (sum to 1 on the log scale)
    assert torch.allclose(torch.logsumexp(log_probs, dim=-1), torch.zeros(BATCH_SIZE, MAX_SEQ_LEN), atol=1e-6)
