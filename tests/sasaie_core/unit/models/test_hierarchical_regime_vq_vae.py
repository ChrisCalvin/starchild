# Part of the new RegimeVAE architecture as of 2025-10-13

import pytest
import torch
import torch.nn.functional as F

from sasaie_core.models.hierarchical_regime_vq_vae import (
    Encoder,
    Decoder,
    CompositionMatrix,
    ContinualVQVAELayer,
    HierarchicalRegimeVQVAE,
)

# Constants for testing
INPUT_DIM = 100
LATENT_DIM = 128
N_CODES_BELOW = 16
N_CODES_CURRENT = 12
INITIAL_CODEBOOK_SIZE = 8
BATCH_SIZE = 4

# Config for hierarchical model
TEST_SCALES = [10, 50, 200]
TEST_CODEBOOK_SIZES = [16, 12, 8]


@pytest.fixture
def encoder() -> Encoder:
    """Provides a default Encoder instance."""
    return Encoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)


@pytest.fixture
def decoder() -> Decoder:
    """Provides a default Decoder instance."""
    return Decoder(latent_dim=LATENT_DIM, output_dim=INPUT_DIM)


@pytest.fixture
def composition_matrix() -> CompositionMatrix:
    """Provides a default CompositionMatrix instance."""
    return CompositionMatrix(n_codes_below=N_CODES_BELOW, n_codes_current=N_CODES_CURRENT)


@pytest.fixture
def continual_vq_layer() -> ContinualVQVAELayer:
    """Provides a default ContinualVQVAELayer instance."""
    return ContinualVQVAELayer(
        input_dim=INPUT_DIM, 
        latent_dim=LATENT_DIM, 
        initial_codebook_size=INITIAL_CODEBOOK_SIZE
    )

@pytest.fixture
def hierarchical_vq_vae() -> HierarchicalRegimeVQVAE:
    """Provides a default HierarchicalRegimeVQVAE instance."""
    return HierarchicalRegimeVQVAE(
        scales=TEST_SCALES,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        codebook_sizes=TEST_CODEBOOK_SIZES
    )


# --- Test Cases for Encoder --- #

def test_encoder_instantiation(encoder):
    """Test that the Encoder can be instantiated."""
    assert encoder is not None
    assert isinstance(encoder.net, torch.nn.Sequential)

def test_encoder_forward_pass_shape(encoder):
    """Test the output shape of the encoder's forward pass."""
    # Arrange
    input_tensor = torch.randn(BATCH_SIZE, INPUT_DIM)

    # Act
    output_tensor = encoder(input_tensor)

    # Assert
    assert output_tensor.shape == (BATCH_SIZE, LATENT_DIM)


# --- Test Cases for Decoder --- #

def test_decoder_instantiation(decoder):
    """Test that the Decoder can be instantiated."""
    assert decoder is not None
    assert isinstance(decoder.net, torch.nn.Sequential)


def test_decoder_forward_pass_shape(decoder):
    """Test the output shape of the decoder's forward pass."""
    # Arrange
    latent_tensor = torch.randn(BATCH_SIZE, LATENT_DIM)

    # Act
    output_tensor = decoder(latent_tensor)

    # Assert
    assert output_tensor.shape == (BATCH_SIZE, INPUT_DIM)


# --- Test Cases for CompositionMatrix --- #

def test_composition_matrix_instantiation(composition_matrix):
    """Test that the CompositionMatrix can be instantiated."""
    assert composition_matrix is not None
    assert composition_matrix.composition.shape == (N_CODES_CURRENT, N_CODES_BELOW)
    assert composition_matrix.transitions.shape == (N_CODES_CURRENT, N_CODES_CURRENT)


def test_composition_matrix_forward_pass(composition_matrix):
    """Test the forward pass of the composition matrix."""
    # Arrange
    # Simulate a batch of probability distributions from the level below
    codes_below_dist = F.softmax(torch.randn(BATCH_SIZE, N_CODES_BELOW), dim=-1)

    # Act
    output_dist = composition_matrix(codes_below_dist)

    # Assert
    assert output_dist.shape == (BATCH_SIZE, N_CODES_CURRENT)
    # Check if the output is a valid probability distribution
    assert torch.allclose(output_dist.sum(dim=-1), torch.tensor(1.0))


def test_composition_matrix_expand_current_level(composition_matrix):
    """Test the expansion of the matrix for new codes at the current level."""
    # Arrange
    n_new_codes = 2
    original_shape = composition_matrix.composition.shape

    # Act
    composition_matrix.expand_current_level(n_new_codes)

    # Assert
    new_shape = composition_matrix.composition.shape
    assert new_shape[0] == original_shape[0] + n_new_codes
    assert new_shape[1] == original_shape[1]


def test_composition_matrix_expand_below_level(composition_matrix):
    """Test the expansion of the matrix for new codes at the level below."""
    # Arrange
    n_new_codes = 3
    original_shape = composition_matrix.composition.shape

    # Act
    composition_matrix.expand_below_level(n_new_codes)

    # Assert
    new_shape = composition_matrix.composition.shape
    assert new_shape[0] == original_shape[0]
    assert new_shape[1] == original_shape[1] + n_new_codes


# --- Test Cases for ContinualVQVAELayer --- #

class TestContinualVQVAELayer:
    def test_layer_instantiation(self, continual_vq_layer):
        """Test that the layer and its components are instantiated correctly."""
        assert continual_vq_layer is not None
        assert continual_vq_layer.codebook_size == INITIAL_CODEBOOK_SIZE
        assert continual_vq_layer.codebook.num_embeddings == INITIAL_CODEBOOK_SIZE
        assert continual_vq_layer.codebook.embedding_dim == LATENT_DIM

    def test_quantize(self, continual_vq_layer):
        """Test that quantization finds the nearest codebook vector."""
        # Arrange
        # Create a latent vector that is very close to the first codebook entry
        z_e = continual_vq_layer.codebook.weight[0].data.clone() + torch.randn(LATENT_DIM) * 0.01
        
        # Act
        z_q, index, dist = continual_vq_layer.quantize(z_e)
        
        # Assert
        assert index.item() == 0
        assert torch.allclose(z_q, continual_vq_layer.codebook.weight[0].data)

    def test_forward_pass_shapes(self, continual_vq_layer):
        """Test the shapes of tensors in a full forward pass."""
        # Arrange
        input_tensor = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Act
        x_recon, z_q, commitment_loss, indices = continual_vq_layer(input_tensor)

        # Assert
        assert x_recon.shape == (BATCH_SIZE, INPUT_DIM)
        assert z_q.shape == (BATCH_SIZE, LATENT_DIM)
        assert commitment_loss.ndim == 0 # Should be a scalar tensor
        assert indices.shape == (BATCH_SIZE,)

    def test_add_new_code(self, continual_vq_layer):
        """Test the dynamic expansion of the codebook."""
        # Arrange
        original_size = continual_vq_layer.codebook_size
        novel_pattern = torch.randn(1, LATENT_DIM)

        # Act
        new_index = continual_vq_layer.add_new_code(novel_pattern)

        # Assert
        assert new_index == original_size
        assert continual_vq_layer.codebook_size == original_size + 1
        assert continual_vq_layer.codebook.num_embeddings == original_size + 1
        assert continual_vq_layer.code_usage.shape[0] == original_size + 1
        # Check that the new code was added correctly
        assert torch.allclose(continual_vq_layer.codebook.weight[original_size], novel_pattern.squeeze())

    def test_ewc_loss_is_zero_initially(self, continual_vq_layer):
        """Test that EWC loss is zero before the Fisher matrix is computed."""
        # Act
        ewc_loss = continual_vq_layer.ewc_loss()
        # Assert
        assert ewc_loss.item() == 0.0


# --- Test Cases for HierarchicalRegimeVQVAE --- #

class TestHierarchicalRegimeVQVAE:
    def test_orchestrator_instantiation(self, hierarchical_vq_vae):
        """Test that the orchestrator and its layers are created correctly."""
        assert hierarchical_vq_vae is not None
        assert len(hierarchical_vq_vae.layers) == len(TEST_SCALES)
        assert len(hierarchical_vq_vae.composition_matrices) == len(TEST_SCALES) - 1
        assert len(hierarchical_vq_vae.cross_attentions) == len(TEST_SCALES) - 1

    def test_hierarchical_encode_output(self, hierarchical_vq_vae):
        """Test the output structure of the hierarchical_encode method."""
        # Arrange
        mp_features = {
            scale: torch.randn(1, INPUT_DIM) for scale in TEST_SCALES
        }

        # Act
        codes = hierarchical_vq_vae.hierarchical_encode(mp_features)

        # Assert
        assert isinstance(codes, dict)
        assert set(codes.keys()) == set(TEST_SCALES)
        for scale, (code, distance) in codes.items():
            assert isinstance(code, int)
            assert isinstance(distance, float)
            level = TEST_SCALES.index(scale)
            assert 0 <= code < TEST_CODEBOOK_SIZES[level]

    def test_explain_composition(self, hierarchical_vq_vae):
        """Test the output structure of the explain_composition method."""
        # Arrange
        # Run encode first to ensure statistics are not empty
        mp_features = {scale: torch.randn(1, INPUT_DIM) for scale in TEST_SCALES}
        hierarchical_vq_vae.hierarchical_encode(mp_features)
        
        # Act: Explain composition for a code at level 1
        level_to_explain = 1
        code_to_explain = 5 # Arbitrary code index
        explanation = hierarchical_vq_vae.explain_composition(code_to_explain, level_to_explain)

        # Assert
        assert isinstance(explanation, dict)
        assert explanation["level"] == level_to_explain
        assert explanation["code"] == code_to_explain
        assert "composed_from" in explanation
        assert isinstance(explanation["composed_from"], list)
        if explanation["composed_from"]:
            first_comp = explanation["composed_from"][0]
            assert "lower_code" in first_comp
            assert "contribution" in first_comp
