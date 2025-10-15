"""
Defines the VAEBridge for fusing beliefs across hierarchical levels and modalities.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List

class VAEBridge(nn.Module):
    """
    A Variational Autoencoder (VAE) based bridge for fusing multiple latent
    representations (beliefs) into a single, consolidated latent space.
    """

    def __init__(self, input_dims: List[int], fused_latent_dim: int, hidden_dim: int = 128):
        """
        Initializes the VAEBridge.

        Args:
            input_dims: A list of integers, where each integer is the dimensionality
                        of one of the input latent tensors.
            fused_latent_dim: The dimensionality of the fused latent space.
            hidden_dim: The dimensionality of the hidden layers in the encoder and decoder.
        """
        super().__init__()
        self.input_dims = input_dims
        self.fused_latent_dim = fused_latent_dim
        self.total_input_dim = sum(input_dims)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, fused_latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, fused_latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(fused_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_input_dim)
        )

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encodes the concatenated input into mean and log-variance of the latent space.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent sample back into the input space.
        """
        return self.decoder(z)

    def forward(self, inputs: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass through the VAEBridge.

        Args:
            inputs: A list of input latent tensors to be fused.

        Returns:
            A tuple containing:
            - reconstruction: The reconstructed input tensor.
            - mu: The mean of the fused latent distribution.
            - logvar: The log-variance of the fused latent distribution.
        """
        # Concatenate all input tensors along the last dimension
        x = torch.cat(inputs, dim=-1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def fuse(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuses multiple latent inputs into a single, sampled fused latent.

        Args:
            inputs: A list of input latent tensors to be fused.

        Returns:
            The sampled fused latent tensor.
        """
        x = torch.cat(inputs, dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
