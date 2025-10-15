"""
Defines the MultiScaleNFModel for orchestrating parallel Normalizing Flows.
"""

from typing import Dict, Any
import torch
from torch import nn

from sasaie_core.models.nf import ConditionalSplineFlow
from sasaie_core.config.loader import GenerativeModelConfigLoader

class MultiScaleNFModel(nn.Module):
    """
    A container for orchestrating multiple ConditionalSplineFlow instances,
    each representing a different modality or temporal scale.
    """

    def __init__(self, config_loader: GenerativeModelConfigLoader):
        """
        Initializes the MultiScaleNFModel.

        Args:
            config_loader: The configuration loader instance. This is used by the
                           BeliefUpdater to lazily initialize the flows.
        """
        super().__init__()
        self.nfs = nn.ModuleDict() # Flows will be added lazily

    def log_prob(self, inputs_by_scale: Dict[str, torch.Tensor], contexts_by_scale: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates the log probability for each scale.

        Args:
            inputs_by_scale: A dictionary of input tensors, keyed by scale name.
            contexts_by_scale: A dictionary of context tensors, keyed by scale name.

        Returns:
            A dictionary of log probabilities, keyed by scale name.
        """
        log_probs = {}
        for scale_name, nf in self.nfs.items():
            log_probs[scale_name] = nf.log_prob(inputs_by_scale[scale_name], contexts_by_scale[scale_name])
        return log_probs

    def sample(self, num_samples: int, contexts_by_scale: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generates samples from each scale.

        Args:
            num_samples: The number of samples to generate.
            contexts_by_scale: A dictionary of context tensors, keyed by scale name.

        Returns:
            A dictionary of samples, keyed by scale name.
        """
        samples = {}
        for scale_name, nf in self.nfs.items():
            samples[scale_name] = nf.sample(num_samples, contexts_by_scale[scale_name])
        return samples

    def forward(self, inputs_by_scale: Dict[str, torch.Tensor], contexts_by_scale: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        A convenience method that calls log_prob.
        """
        return self.log_prob(inputs_by_scale, contexts_by_scale)
