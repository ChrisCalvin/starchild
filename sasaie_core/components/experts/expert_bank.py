# Part of the new RegimeVAE architecture as of 2025-10-13

import numpy as np
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_expert import BaseExpert, ExpertMetadata
from .ar_forecaster import ARForecaster


# ============================================================================ 
# 3. EXPERT BANK MANAGER
# ============================================================================ 

class ExpertBankManager:
    """
    Manages the lifecycle of expert models.
    
    Responsibilities:
    - Create new experts for novel regimes
    - Route observations to appropriate experts
    - Prune inactive experts
    - Track performance metrics
    """
    
    def __init__(self, 
                 expert_factory: callable,
                 max_experts: int = 100,
                 pruning_threshold: int = 1000):
        """
        Initialize expert bank.
        
        Args:
            expert_factory: Function that creates new experts
            max_experts: Maximum number of concurrent experts
            pruning_threshold: Steps of inactivity before pruning
        """
        self.expert_factory = expert_factory
        self.max_experts = max_experts
        self.pruning_threshold = pruning_threshold
        
        # Expert storage
        self.experts: Dict[int, BaseExpert] = {}
        self.metadata: Dict[int, ExpertMetadata] = {}
        
        # Global tracking
        self.current_timestep = 0
        self.total_forecasts = 0
    
    def get_or_create_expert(self, regime_id: int) -> BaseExpert:
        """
        Get existing expert or create new one.
        
        Args:
            regime_id: Regime code
            
        Returns:
            Expert specialized for this regime
        """
        if regime_id not in self.experts:
            self._create_expert(regime_id)
        
        return self.experts[regime_id]
    
    def _create_expert(self, regime_id: int) -> None: 
        """Create and register new expert"""
        # Check capacity
        if len(self.experts) >= self.max_experts:
            self._prune_least_used()
        
        # Create expert
        expert = self.expert_factory(regime_id)
        self.experts[regime_id] = expert
        
        # Create metadata
        self.metadata[regime_id] = ExpertMetadata(
            regime_id=regime_id,
            created_at=self.current_timestep
        )
        
        print(f"[ExpertBank] Created new expert for regime {regime_id}")
    
    def update_expert(self, 
                     regime_id: int, 
                     observation: float,
                     context: Optional[Dict[str, Any]] = None) -> None:
        """
        Route observation to expert and update.
        
        Args:
            regime_id: Target regime
            observation: New data
            context: Optional additional info
        """
        expert = self.get_or_create_expert(regime_id)
        expert.update(observation, context)
        
        # Update metadata
        meta = self.metadata[regime_id]
        meta.n_updates += 1
        meta.last_active = self.current_timestep
    
    def get_forecast(self, regime_id: int, horizon: int) -> np.ndarray:
        """
        Get forecast from expert.
        
        Args:
            regime_id: Source regime
            horizon: Forecast length
            
        Returns:
            Forecast array
        """
        expert = self.get_or_create_expert(regime_id)
        forecast = expert.predict(horizon)
        
        self.total_forecasts += 1
        self.metadata[regime_id].last_active = self.current_timestep
        
        return forecast
    
    def get_ensemble_forecast(self, 
                             regime_probs: Dict[int, float],
                             horizon: int) -> np.ndarray:
        """
        Weighted ensemble forecast across multiple regimes.
        
        Args:
            regime_probs: Dict mapping regime_id -> probability
            horizon: Forecast length
            
        Returns:
            Weighted average forecast
        """
        ensemble = np.zeros(horizon)
        total_weight = 0.0
        
        for regime_id, prob in regime_probs.items():
            if prob > 0.01:  # Skip negligible contributions
                forecast = self.get_forecast(regime_id, horizon)
                ensemble += prob * forecast
                total_weight += prob
        
        if total_weight > 0:
            ensemble /= total_weight
        
        return ensemble
    
    def prune_inactive_experts(self) -> int:
        """
        Remove experts that haven't been used recently.
        
        Returns:
            Number of experts pruned
        """
        to_prune = []
        
        for regime_id, meta in self.metadata.items():
            inactive_steps = self.current_timestep - meta.last_active
            
            if inactive_steps > self.pruning_threshold and not meta.is_frozen:
                to_prune.append(regime_id)
        
        for regime_id in to_prune:
            del self.experts[regime_id]
            del self.metadata[regime_id]
            print(f"[ExpertBank] Pruned inactive expert {regime_id}")
        
        return len(to_prune)
    
    def _prune_least_used(self) -> None:
        """Prune the expert with fewest updates"""
        if not self.experts:
            return
        
        # Find least used non-frozen expert
        least_used = min(
            (meta for meta in self.metadata.values() if not meta.is_frozen),
            key=lambda m: m.n_updates,
            default=None
        )
        
        if least_used:
            regime_id = least_used.regime_id
            del self.experts[regime_id]
            del self.metadata[regime_id]
            print(f"[ExpertBank] Pruned least-used expert {regime_id}")
    
    def step(self) -> None:
        """Advance timestep (call once per iteration)"""
        self.current_timestep += 1
        
        # Periodic maintenance
        if self.current_timestep % 100 == 0:
            self.prune_inactive_experts()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bank statistics"""
        return {
            'n_experts': len(self.experts),
            'total_forecasts': self.total_forecasts,
            'timestep': self.current_timestep,
            'expert_ages': {
                rid: self.current_timestep - meta.created_at
                for rid, meta in self.metadata.items()
            },
            'expert_updates': {
                rid: meta.n_updates
                for rid, meta in self.metadata.items()
            }
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Serialize entire bank.
        
        Note: This currently only saves metadata and ARForecaster parameters.
        FFG-based experts would require more complex serialization.
        """
        return {
            'experts': {
                rid: expert.get_parameters()
                for rid, expert in self.experts.items()
            },
            'metadata': {
                rid: {
                    'regime_id': meta.regime_id,
                    'created_at': meta.created_at,
                    'n_updates': meta.n_updates,
                    'last_active': meta.last_active,
                    'is_frozen': meta.is_frozen,
                    'fisher_information': {k: v.tolist() for k, v in meta.fisher_information.items()} if meta.fisher_information else None
                }
                for rid, meta in self.metadata.items()
            },
            'current_timestep': self.current_timestep,
            'total_forecasts': self.total_forecasts
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize bank state"""
        self.current_timestep = state['current_timestep']
        self.total_forecasts = state['total_forecasts']
        
        # Recreate experts
        for regime_id, params in state['experts'].items():
            regime_id = int(regime_id)
            expert = self.expert_factory(regime_id)
            expert.set_parameters(params)
            self.experts[regime_id] = expert
        
        # Recreate metadata
        for regime_id, meta_dict in state['metadata'].items():
            regime_id = int(regime_id)
            # Handle fisher_information deserialization
            if meta_dict.get('fisher_information'):
                meta_dict['fisher_information'] = {k: torch.tensor(v) for k, v in meta_dict['fisher_information'].items()}
            self.metadata[regime_id] = ExpertMetadata(**meta_dict)


# ============================================================================ 
# 4. INTEGRATION WITH CONTINUAL LEARNING (EWC)
# ============================================================================ 

class ContinualExpertBank(ExpertBankManager):
    """
    Expert bank with Elastic Weight Consolidation support.
    
    Extends base manager to prevent catastrophic forgetting
    when experts are updated with new data.
    """
    
    def __init__(self, *args, ewc_lambda: float = 1000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
    
    def freeze_expert(self, regime_id: int) -> None:
        """
        Freeze expert parameters (mark as important).
        
        Args:
            regime_id: Expert to freeze
        """
        if regime_id in self.metadata:
            self.metadata[regime_id].is_frozen = True
            print(f"[ExpertBank] Froze expert {regime_id}")
    
    def compute_fisher_information(self, regime_id: int) -> None:
        """
        Compute Fisher Information Matrix for expert.
        
        This identifies which parameters are important for
        maintaining performance on seen data.
        
        Args:
            regime_id: Target expert
        """
        if regime_id not in self.experts:
            return
        
        expert = self.experts[regime_id]
        
        # For AR model, Fisher is proportional to X'X
        if isinstance(expert, ARForecaster) and expert.coefficients is not None:
            # Use inverse of parameter covariance as Fisher approximation
            params = expert.get_parameters()
            
            fisher = {
                'coefficients': torch.ones_like(torch.tensor(expert.coefficients)),
                'intercept': torch.tensor(1.0)
            }
            
            self.metadata[regime_id].fisher_information = fisher
            print(f"[ExpertBank] Computed Fisher information for expert {regime_id}")


# ============================================================================ 
# 5. FFG FOUNDATION (For Future Development)
# ============================================================================ 

class FFGNode:
    """
    Base class for Forney Factor Graph nodes.
    
    This is a foundation for future FFG-based experts.
    """
    
    def __init__(self, node_id: str, dimension: int):
        self.node_id = node_id
        self.dimension = dimension
        self.messages_in: Dict[str, np.ndarray] = {}
        self.messages_out: Dict[str, np.ndarray] = {}
    
    def receive_message(self, from_node: str, message: np.ndarray) -> None:
        """Receive message from connected node"""
        self.messages_in[from_node] = message
    
    def send_message(self, to_node: str) -> np.ndarray:
        """Compute and send message to connected node"""
        raise NotImplementedError("Subclasses must implement")
    
    def compute_marginal(self) -> np.ndarray:
        """Compute marginal distribution at this node"""
        raise NotImplementedError("Subclasses must implement")


class FFGFactor(FFGNode):
    """
    Factor node in FFG (represents constraints/relationships)"""
    
    def __init__(self, node_id: str, factor_type: str):
        super().__init__(node_id, dimension=1)
        self.factor_type = factor_type  # e.g., "gaussian", "ar"
    
    def send_message(self, to_node: str) -> np.ndarray:
        # Placeholder for VMP update
        return np.zeros(self.dimension)


class FFGExpertPlaceholder:
    """
    Placeholder for future FFG-based expert.
    
    This would use Forney Factor Graphs for more sophisticated
    inference and forecasting.
    """
    
    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        self.graph = {}  # Will contain FFGNode instances
        
    # Would implement BaseExpert protocol
    # ...


# ============================================================================ 
# 6. FACTORY FUNCTIONS
# ============================================================================ 

def create_ar_expert_factory(order: int = 10) -> callable:
    """
    Create factory function for AR experts.
    
    Args:
        order: AR model order
        
    Returns:
        Factory function
    """
    def factory(regime_id: int) -> ARForecaster:
        return ARForecaster(
            regime_id=regime_id,
            order=order,
            min_observations=max(20, order * 2)
        )
    
    return factory


def create_expert_bank(expert_type: str = "ar", **kwargs) -> ExpertBankManager:
    """
    Convenience function to create configured expert bank.
    
    Args:
        expert_type: Type of experts ("ar", "ffg", etc.)
        **kwargs: Additional configuration
        
    Returns:
        Configured ExpertBankManager
    """
    if expert_type == "ar":
        factory = create_ar_expert_factory(order=kwargs.get('order', 10))
        return ContinualExpertBank(
            expert_factory=factory,
            max_experts=kwargs.get('max_experts', 100),
            pruning_threshold=kwargs.get('pruning_threshold', 1000),
            ewc_lambda=kwargs.get('ewc_lambda', 1000.0)
        )
    elif expert_type == "ffg":
        raise NotImplementedError("FFG experts not yet implemented")
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


# ============================================================================ 
# 7. USAGE EXAMPLE
# ============================================================================ 

if __name__ == "__main__":
    print("="*70)
    print(" EXPERT BANK DEMONSTRATION")
    print("="*70)
    
    # Create expert bank
    bank = create_expert_bank(expert_type="ar", order=10, max_experts=50)
    
    # Simulate streaming data with regime changes
    np.random.seed(42)
    
    for t in range(500):
        # Determine regime (changes every 100 steps)
        regime = t // 100
        
        # Generate observation
        if regime == 0:
            obs = 0.5 + np.random.randn() * 0.1
        elif regime == 1:
            obs = 0.3 + (t % 100) / 100 * 0.4 + np.random.randn() * 0.05
        else:
            obs = 0.5 + np.sin(t * 0.1) * 0.3 + np.random.randn() * 0.1
        
        # Update expert
        bank.update_expert(regime, obs)
        
        # Periodic forecasting
        if t % 50 == 0 and t > 0:
            forecast = bank.get_forecast(regime, horizon=10)
            print(f"\nStep {t}, Regime {regime}:")
            print(f"  Forecast: {forecast[:5]}")
            print(f"  Bank stats: {bank.get_statistics()}")
        
        bank.step()
    
    print("\n" + "="*70)
    print(" FINAL STATISTICS")
    print("="*70)
    stats = bank.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
