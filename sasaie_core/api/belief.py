# Part of the new RegimeVAE architecture as of 2025-10-13

from dataclasses import dataclass
import torch

@dataclass
class RegimeBeliefs:
    """Agent's beliefs about current regime structure"""
    current_regime: int
    regime_confidence: float  # How certain are we?
    regime_probabilities: torch.Tensor  # Distribution over regimes
    expected_duration: float  # How long will this regime last?
    transition_imminence: float  # How likely is regime change?
