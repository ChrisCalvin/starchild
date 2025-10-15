"""
Unit tests for the VAEBridge.
"""

import pytest
import torch
from sasaie_core.components.vae_bridge import VAEBridge

# Define common parameters for testing
INPUT_DIMS = [5, 3, 2] # Example: 3 input latents with dimensions 5, 3, and 2
FUSED_LATENT_DIM = 4
HIDDEN_DIM = 64
BATCH_SIZE = 10

@pytest.fixture
def vae_bridge():
    """Provides a VAEBridge instance for testing."""
    return VAEBridge(
        input_dims=INPUT_DIMS,
        fused_latent_dim=FUSED_LATENT_DIM,
        hidden_dim=HIDDEN_DIM
    )

@pytest.fixture
def dummy_inputs():
    """Provides dummy input tensors for the VAEBridge."""
    return [
        torch.randn(BATCH_SIZE, INPUT_DIMS[0]),
        torch.randn(BATCH_SIZE, INPUT_DIMS[1]),
        torch.randn(BATCH_SIZE, INPUT_DIMS[2])
    ]

def test_vae_bridge_initialization(vae_bridge):
    """Tests that the VAEBridge is initialized correctly."""
    assert vae_bridge is not None
    assert vae_bridge.total_input_dim == sum(INPUT_DIMS)
    assert vae_bridge.fused_latent_dim == FUSED_LATENT_DIM
    assert isinstance(vae_bridge.encoder, torch.nn.Sequential)
    assert isinstance(vae_bridge.decoder, torch.nn.Sequential)
    assert isinstance(vae_bridge.fc_mu, torch.nn.Linear)
    assert isinstance(vae_bridge.fc_logvar, torch.nn.Linear)

def test_encode_method(vae_bridge, dummy_inputs):
    """Tests the encode method returns correct shapes."""
    concatenated_inputs = torch.cat(dummy_inputs, dim=-1)
    mu, logvar = vae_bridge.encode(concatenated_inputs)
    assert mu.shape == (BATCH_SIZE, FUSED_LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, FUSED_LATENT_DIM)

def test_reparameterize_method(vae_bridge):
    """Tests the reparameterize method returns correct shape."""
    mu = torch.randn(BATCH_SIZE, FUSED_LATENT_DIM)
    logvar = torch.randn(BATCH_SIZE, FUSED_LATENT_DIM)
    z = vae_bridge.reparameterize(mu, logvar)
    assert z.shape == (BATCH_SIZE, FUSED_LATENT_DIM)

def test_decode_method(vae_bridge):
    """Tests the decode method returns correct shape."""
    z = torch.randn(BATCH_SIZE, FUSED_LATENT_DIM)
    reconstruction = vae_bridge.decode(z)
    assert reconstruction.shape == (BATCH_SIZE, sum(INPUT_DIMS))

def test_forward_method(vae_bridge, dummy_inputs):
    """Tests the forward method returns correct shapes."""
    reconstruction, mu, logvar = vae_bridge.forward(dummy_inputs)
    assert reconstruction.shape == (BATCH_SIZE, sum(INPUT_DIMS))
    assert mu.shape == (BATCH_SIZE, FUSED_LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, FUSED_LATENT_DIM)

def test_fuse_method(vae_bridge, dummy_inputs):
    """Tests the fuse method returns correct shape."""
    fused_latent = vae_bridge.fuse(dummy_inputs)
    assert fused_latent.shape == (BATCH_SIZE, FUSED_LATENT_DIM)
