# Part of the new RegimeVAE architecture as of 2025-10-13

import torch
import numpy as np
from typing import Dict, List, Optional

from sasaie_core.api.policy import Policy
from sasaie_core.api.belief import RegimeBeliefs
from sasaie_core.api.action import ActionType

class HierarchicalEFECalculator:
    """
    Computes Expected Free Energy across multiple timescales, integrating information
    from all hierarchical levels of the world model.
    """
    def __init__(self, scales: List[int], vqvae):
        self.scales = scales
        self.vqvae = vqvae
        
        # Learned preferences at each scale (what's "good")
        self.preferences = {
            scale: torch.zeros(1) for scale in scales
        }
        
        # Scale importance weights (some scales matter more for planning)
        # These could also be learned.
        weights = torch.linspace(0.5, 0.1, len(scales))
        self.scale_weights = {scale: weight for scale, weight in zip(scales, weights / weights.sum())}
    
    def compute_efe(self, 
                    policy: Policy,
                    current_beliefs: Dict[int, RegimeBeliefs],
                    forecasts: Dict[int, np.ndarray]) -> float:
        """
        Compute Expected Free Energy for a policy.
        EFE = Pragmatic Value (expected reward) - Epistemic Value (info gain)
        Lower EFE is better.
        """
        total_efe = 0.0
        
        for scale in self.scales:
            if scale not in current_beliefs:
                continue
            
            beliefs = current_beliefs[scale]
            
            # 1. EPISTEMIC VALUE (information gain)
            epistemic = self._epistemic_value(policy, beliefs, scale)
            
            # 2. PRAGMATIC VALUE (expected reward)
            pragmatic = self._pragmatic_value(policy, beliefs, forecasts.get(scale), scale)
            
            # 3. REGIME STABILITY BONUS/PENALTY
            # If regime is stable, favor exploitation; if unstable, favor exploration.
            stability_factor = 1.0 / (beliefs.regime_confidence + 0.1) # Higher confidence -> lower factor
            
            # Combine, weighted by scale importance
            # Note: We subtract epistemic value from pragmatic value.
            scale_efe = (pragmatic - epistemic) * stability_factor
            total_efe += self.scale_weights.get(scale, 0.1) * scale_efe
        
        # The final EFE is typically negative (as we maximize reward/minimize negative reward)
        # We return the value to be minimized.
        return -total_efe
    
    def _epistemic_value(self, 
                        policy: Policy, 
                        beliefs: RegimeBeliefs,
                        scale: int) -> float:
        """
        Expected information gain from the policy.
        Higher value means the policy is more informative.
        """
        # Entropy of current beliefs is a measure of uncertainty
        p = beliefs.regime_probabilities
        current_entropy = -torch.sum(p * torch.log(p + 1e-9))
        
        # Estimate how much this policy will reduce entropy
        if policy.action_type == ActionType.EXPLORE:
            # Exploratory policies are designed to maximize information gain
            expected_entropy_reduction = current_entropy * 0.7
        elif policy.action_type == ActionType.ADAPT:
            expected_entropy_reduction = current_entropy * 0.4
        else: # EXPLOIT or PREPARE
            # Exploitative policies yield less new information
            expected_entropy_reduction = current_entropy * 0.1
        
        return float(expected_entropy_reduction)
    
    def _pragmatic_value(self,
                        policy: Policy,
                        beliefs: RegimeBeliefs,
                        forecast: Optional[np.ndarray],
                        scale: int) -> float:
        """
        Expected reward under current beliefs, weighted by confidence.
        Higher value means a better expected outcome.
        """
        if forecast is None:
            return 0.0
        
        expected_reward = 0.0
        
        # Simulate policy outcomes based on the expert forecast for this regime
        for t, action in enumerate(policy.actions[:len(forecast)]):
            # A simple model of action outcome
            predicted_state = forecast[t] + action * 0.1
            
            # Reward function is problem-specific. 
            # Example: prefer states near 0.5, penalize extremes.
            reward = -abs(predicted_state - 0.5)
            
            discount = 0.95 ** t
            expected_reward += discount * reward
        
        # Weight the expected reward by our confidence in the current regime
        return expected_reward * beliefs.regime_confidence