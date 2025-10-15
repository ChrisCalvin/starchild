"""
Defines a conditional normalizing flow for the state-space model.
"""

import torch
from torch import nn
from nflows import transforms, distributions, flows
from nflows.nn import nets # Import nets for ResidualNet

class ContextualNet(nn.Module):
    """
    A small neural network that processes an input and a context vector.
    This network is used within the transform_net_create_fn.
    """
    def __init__(self, in_features, out_features, context_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features + context_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_features),
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs and context before passing to the network
        x = torch.cat([inputs, context], dim=-1)
        return self.network(x)

class ConditionalSplineFlow(nn.Module):
    """
    A conditional normalizing flow using neural splines.

    This flow models the distribution p(z_t | c_t), where z_t is the latent state
    and c_t is the context from the perception layer.
    """

    def __init__(self, latent_dim: int, context_dim: int, num_layers: int = 5):
        """
        Initializes the ConditionalSplineFlow.

        Args:
            latent_dim: The dimensionality of the latent space (z_t).
            context_dim: The dimensionality of the context vector (c_t).
            num_layers: The number of flow layers to stack.
        """
        super().__init__()
        self.latent_dim = latent_dim # Store as attribute
        self.context_dim = context_dim # Store as attribute

        # Define the base distribution (a standard normal distribution)
        base_distribution = distributions.StandardNormal(shape=[latent_dim])

        # Define the transformations
        transformations = []
        
        for _ in range(num_layers):
            transformations.append(
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=torch.ones(latent_dim), # Use mask instead of features
                    transform_net_create_fn=lambda in_features, out_features: ContextualNet(
                        in_features=in_features,
                        out_features=out_features,
                        context_features=context_dim
                    ),
                    tails='linear', 
                    tail_bound=5.0 # A reasonable bound for StandardNormal
                )
            )
            transformations.append(transforms.LULinear(latent_dim))

        # Combine transformations into a single transform object
        transform = transforms.CompositeTransform(transformations)

        # Create the flow object
        self.flow = flows.Flow(transform, base_distribution)

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Calculates the log probability of the inputs given the context."""
        return self.flow.log_prob(inputs, context=context)

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Generates samples from the flow given the context."""
        return self.flow.sample(num_samples, context=context)