"""
Defines a Variational Autoencoder (VAE) model.
"""

import torch
from torch import nn
from typing import Tuple

class VAE(nn.Module):
    """
    A simple Variational Autoencoder (VAE).

    Args:
        input_dim: The dimensionality of the input data.
        latent_dim: The dimensionality of the latent space.
        hidden_dim: The dimensionality of the hidden layers.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)  # for mean
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)  # for log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid() # Removed: Use Sigmoid only if input is normalized to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input by passing it through the encoder network."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent variable z by passing it through the decoder network."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the full forward pass of the VAE.

        Args:
            x: The input tensor.

        Returns:
            A tuple containing: (reconstructed_x, mu, logvar)
        """
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar
