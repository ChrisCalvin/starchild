"""
Defines the RGMorphism, a core component for structural learning and compositional policies.
"""

import torch
from torch import nn

from copy import deepcopy # New import

from sasaie_core.models.nf import ConditionalSplineFlow

class RGMorphism(nn.Module):
    """
    Represents a single, reusable, and evolvable skill or sub-policy, implemented
    as a "micro" Normalizing Flow.
    
    An RG-Morphism learns a transformation from one belief state to another,
    conditioned on some context, representing the outcome of a specific action or
    a sequence of actions.
    """

    def __init__(self, name: str, io_dim: int, context_dim: int, modality: str, num_layers: int = 2):
        """
        Initializes the RGMorphism.

        Args:
            name: The unique name of the morphism (e.g., 'limit_buy', 'observe_volatility').
            io_dim: The dimensionality of the input and output belief state space.
            context_dim: The dimensionality of the conditioning context vector.
            modality: The modality this morphism specializes in (e.g., 'price', 'volume', 'news').
            num_layers: The number of flow layers in the internal micro-NF.
        """
        super().__init__()
        self.name = name
        self.io_dim = io_dim
        self.context_dim = context_dim
        self.modality = modality
        self.num_layers = num_layers # Store num_layers for cloning

        # The core of the morphism is a small conditional normalizing flow.
        self.micro_nf = ConditionalSplineFlow(
            latent_dim=io_dim,
            context_dim=context_dim,
            num_layers=num_layers
        )

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward transformation of the morphism.
        This represents applying the skill/sub-policy.

        Args:
            z: The input belief state (a batch of latent vectors).
            context: The conditioning context.

        Returns:
            The transformed belief state.
        """
        # For a normalizing flow, the forward pass generates a sample.
        # We pass the context to the sample method.
        # The number of samples should match the batch size of z.
        num_samples = z.shape[0]
        return self.micro_nf.sample(num_samples, context=context)

    def inverse(self, z_transformed: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse transformation of the morphism.
        This calculates the log probability of a transformed state, which is essential
        for training and evaluating the flow.

        Args:
            z_transformed: The transformed belief state.
            context: The conditioning context.

        Returns:
            The log probability of the transformed state.
        """
        return self.micro_nf.log_prob(z_transformed, context=context)

    def clone(self, copy_weights: bool = True) -> 'RGMorphism':
        """
        Creates a new RGMorphism instance with the same architecture.

        Args:
            copy_weights: If True, the weights of the micro_nf are copied to the new instance.

        Returns:
            A new RGMorphism instance.
        """
        new_morphism = RGMorphism(
            name=self.name,
            io_dim=self.io_dim,
            context_dim=self.context_dim,
            modality=self.modality,
            num_layers=self.num_layers
        )
        if copy_weights:
            new_morphism.load_state_dict(deepcopy(self.state_dict()))
        return new_morphism

    def __repr__(self) -> str:
        return f"RGMorphism(name='{self.name}', io_dim={self.io_dim}, context_dim={self.context_dim}, modality='{self.modality}')"
