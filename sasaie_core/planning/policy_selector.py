import torch
from torch import nn
import torch.nn.functional as F

class PolicySelectorVAE(nn.Module):
    """
    A VAE that takes the ConsolidatedBeliefState and proposes candidate policies.
    Its latent space represents a "policy space". The decoder generates a sequence
    of categorical distributions over the vocabulary of available morphisms.
    It incorporates context and dynamic goal weights into its policy generation.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, vocab_size: int, max_seq_len: int,
                 context_dim: int, num_goals: int):
        """
        Initializes the PolicySelectorVAE.

        Args:
            input_dim: Dimensionality of the input (ConsolidatedBeliefState).
            latent_dim: Dimensionality of the latent policy space.
            hidden_dim: Dimensionality of the hidden layers.
            vocab_size: The number of possible morphisms in the vocabulary.
            max_seq_len: The maximum length of a policy sequence.
            context_dim: Dimensionality of the market context vector.
            num_goals: Number of competing goals.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.context_dim = context_dim
        self.num_goals = num_goals

        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        # Input to decoder_fc1 now includes latent_dim + context_dim + num_goals
        self.decoder_fc1 = nn.Linear(latent_dim + context_dim + num_goals, hidden_dim)
        # The output layer now produces scores for each token in the sequence for the entire vocabulary
        self.decoder_fc2 = nn.Linear(hidden_dim, max_seq_len * vocab_size)

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Encodes the input and returns the mean and log-variance of the latent space."""
        h1 = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mu(h1)
        logvar = self.encoder_fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, context: torch.Tensor, goal_weights: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent policy vector z into log probabilities over policy sequences,
        conditioned on context and goal weights.
        """
        # Concatenate latent vector, context, and goal weights
        decoder_input = torch.cat([z, context, goal_weights], dim=-1)
        h2 = F.relu(self.decoder_fc1(decoder_input))
        logits = self.decoder_fc2(h2)
        # Reshape to (batch_size, max_seq_len, vocab_size)
        logits = logits.view(-1, self.max_seq_len, self.vocab_size)
        # Apply log_softmax to get log probabilities over the vocabulary
        return F.log_softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor, goal_weights: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Performs a forward pass through the VAE.

        Args:
            x: The input tensor (ConsolidatedBeliefState).
            context: The market context tensor.
            goal_weights: Tensor of weights for each goal.

        Returns:
            A tuple of (log_probs, mu, logvar), where log_probs is a tensor of shape
            [batch_size, max_seq_len, vocab_size].
        """
        mu, logvar = self.encode(x.view(-1, x.shape[-1]))
        z = self.reparameterize(mu, logvar)
        log_probs = self.decode(z, context, goal_weights)
        return log_probs, mu, logvar

    def sample(self, num_samples: int, context: torch.Tensor, goal_weights: torch.Tensor) -> torch.Tensor:
        """
        Generates `num_samples` candidate policies from the latent space.

        Args:
            num_samples: The number of policies to generate.
            context: The market context tensor (batch size should be 1 or num_samples).
            goal_weights: Tensor of weights for each goal (batch size should be 1 or num_samples).

        Returns:
            A tensor of shape [num_samples, max_seq_len] containing the sampled policy sequences (indices).
        """
        with torch.no_grad():
            # Sample from the prior (standard normal)
            z = torch.randn(num_samples, self.latent_dim).to(context.device)

            # Expand context and goal_weights to match the number of samples if they have a batch size of 1
            if context.shape[0] == 1 and num_samples > 1:
                context = context.expand(num_samples, -1)
            if goal_weights.shape[0] == 1 and num_samples > 1:
                goal_weights = goal_weights.expand(num_samples, -1)

            # Decode to get log probabilities
            log_probs = self.decode(z, context, goal_weights)
            # Sample from the categorical distribution defined by the log probabilities
            # This creates a distribution for each position in the sequence
            dist = torch.distributions.Categorical(logits=log_probs)
            # Sample one token for each position in the sequence
            sampled_indices = dist.sample()
        return sampled_indices
