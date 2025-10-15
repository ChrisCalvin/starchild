"""
Unit tests for the VAE model.
"""

import pytest
import torch
from sasaie_core.models.vae import VAE

@pytest.fixture
def vae_model():
    """Provides a VAE instance for testing."""
    return VAE(input_dim=10, latent_dim=2, hidden_dim=32)

@pytest.fixture
def sample_input_tensor():
    """Provides a sample input tensor for the VAE."""
    return torch.randn(1, 10)  # Batch size of 1, input_dim of 10

def test_vae_creation(vae_model):
    """Tests that the VAE model is initialized correctly."""
    # Assert
    assert vae_model is not None
    assert isinstance(vae_model, torch.nn.Module)
    assert vae_model.input_dim == 10
    assert vae_model.latent_dim == 2

def test_vae_forward_pass(vae_model, sample_input_tensor):
    """Tests the forward pass of the VAE model."""
    # Act
    reconstructed_x, mu, logvar = vae_model.forward(sample_input_tensor)

    # Assert
    assert torch.is_tensor(reconstructed_x)
    assert torch.is_tensor(mu)
    assert torch.is_tensor(logvar)

    # Check shapes
    assert reconstructed_x.shape == (1, vae_model.input_dim)
    assert mu.shape == (1, vae_model.latent_dim)
    assert logvar.shape == (1, vae_model.latent_dim)

def test_vae_encode_decode(vae_model, sample_input_tensor):
    """Tests the encode and decode methods separately."""
    # Act
    mu, logvar = vae_model.encode(sample_input_tensor)
    z = vae_model.reparameterize(mu, logvar)
    reconstructed_x = vae_model.decode(z)

    # Assert
    assert mu.shape == (1, vae_model.latent_dim)
    assert logvar.shape == (1, vae_model.latent_dim)
    assert z.shape == (1, vae_model.latent_dim)
    assert reconstructed_x.shape == (1, vae_model.input_dim)
