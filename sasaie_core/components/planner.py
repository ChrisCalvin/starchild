# Part of the new RegimeVAE architecture as of 2025-10-13

import numpy as np
import torch
from typing import Dict, List, Tuple

from sasaie_core.api.policy import Policy
from sasaie_core.api.action import ActionType
from sasaie_core.api.belief import RegimeBeliefs
from sasaie_core.planning.efe import HierarchicalEFECalculator

class RegimeAwarePlanner:
    """
    Active Inference planner that reasons hierarchically across regimes.
    """
    def __init__(self, 
                 scales: List[int],
                 vqvae, # HierarchicalRegimeVQVAE model
                 expert_bank: Dict[int, any], # Maps regime code to a generative model (e.g., a FactorGraph)
                 action_space_size: int = 5):
        
        self.scales = scales
        self.vqvae = vqvae
        self.expert_bank = expert_bank
        self.action_space_size = action_space_size
        
        self.efe_calculator = HierarchicalEFECalculator(scales, vqvae)
        
        self.action_history = []
        self.regime_history = []
        self.outcome_history = []
        self.regime_policy_cache = {}
    
    def update_beliefs(self, regime_codes: Dict[int, Tuple[int, float]]) -> Dict[int, RegimeBeliefs]:
        """
        Enriches the raw output from the VQ-VAE into a structured belief object.
        """
        beliefs = {}
        for scale in self.scales:
            if scale not in regime_codes:
                continue
            
            code, distance = regime_codes[scale]
            confidence = 1.0 / (1.0 + distance) # Convert distance to confidence
            
            # Get full distribution over regimes for this scale
            scale_idx = self.scales.index(scale)
            codebook_size = self.vqvae.layers[scale_idx].codebook_size
            probs = torch.zeros(codebook_size)
            if code < len(probs):
                probs[code] = 1.0 / (distance + 1e-8)
            regime_probs = torch.softmax(probs, dim=0)
            
            beliefs[scale] = RegimeBeliefs(
                current_regime=code,
                regime_confidence=confidence,
                regime_probabilities=regime_probs,
                expected_duration=self._estimate_regime_duration(scale, code),
                transition_imminence=0.3 # Placeholder
            )
        return beliefs

    def _estimate_regime_duration(self, scale: int, code: int) -> float:
        """Estimates how long this regime typically lasts based on history."""
        durations = []
        current_duration = 0
        for past_regime_state in self.regime_history[-500:]:
            if past_regime_state.get(scale) == code:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        return np.mean(durations) if durations else 20.0 # Default assumption

    def generate_candidate_policies(self, 
                                    current_beliefs: Dict[int, RegimeBeliefs],
                                    horizon: int = 10) -> List[Policy]:
        """Generates candidate policies based on the current belief state."""
        policies = []
        short_scale = self.scales[0]
        if short_scale not in current_beliefs:
            return policies
        
        beliefs = current_beliefs[short_scale]
        
        # 1. EXPLOIT: Use best known action for current regime
        if beliefs.regime_confidence > 0.7:
            policies.append(Policy(
                action_type=ActionType.EXPLOIT,
                actions=self._get_cached_policy(beliefs.current_regime, horizon),
                horizon=horizon
            ))
        
        # 2. EXPLORE: Information-seeking actions
        if beliefs.regime_confidence < 0.6:
            policies.append(Policy(
                action_type=ActionType.EXPLORE,
                actions=[np.sin(t * 0.5) * 0.5 for t in range(horizon)],
                horizon=horizon
            ))

        # 3. PREPARE: Anticipate regime change
        if beliefs.transition_imminence > 0.6:
            policies.append(Policy(
                action_type=ActionType.PREPARE,
                actions=[0.1] * horizon, # Conservative actions
                horizon=horizon
            ))
        
        # Ensure there is always at least one policy
        if not policies:
            policies.append(Policy(
                action_type=ActionType.EXPLOIT,
                actions=[0.0] * horizon, # Neutral action
                horizon=horizon
            ))

        return policies

    def _get_cached_policy(self, regime: int, horizon: int) -> List[float]:
        """Gets a learned policy for a specific regime."""
        return self.regime_policy_cache.get(regime, [0.0] * horizon)

    def select_best_policy(self, 
                           current_beliefs: Dict[int, RegimeBeliefs],
                           forecasts: Dict[int, np.ndarray]) -> Policy:
        """
        Selects the policy that minimizes Expected Free Energy (EFE).
        """
        candidates = self.generate_candidate_policies(current_beliefs)
        
        for policy in candidates:
            policy.efe = self.efe_calculator.compute_efe(policy, current_beliefs, forecasts)
        
        best_policy = min(candidates, key=lambda p: p.efe)
        return best_policy

    def learn_from_outcome(self, 
                           policy: Policy,
                           outcome: float,
                           regime_codes: Dict[int, Tuple[int, float]]):
        """
        Updates internal models based on the outcome of an action.
        """
        self.action_history.append(policy.actions[0])
        self.regime_history.append({s: c for s, (c, _) in regime_codes.items()})
        self.outcome_history.append(outcome)
        
        # Example learning rule: if a policy resulted in a good outcome, cache it.
        reward = -abs(outcome - 0.5) # Example reward
        if reward > -0.1:
            short_scale = self.scales[0]
            if short_scale in regime_codes:
                regime = regime_codes[short_scale][0]
                self.regime_policy_cache[regime] = policy.actions