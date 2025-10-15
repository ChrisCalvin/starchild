# Part of the new RegimeVAE architecture as of 2025-10-13

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np # Added import
import torch # Added import

@runtime_checkable
class BaseExpert(Protocol):
    """
    Protocol defining the interface all experts must implement.
    
    This ensures modularity and allows different expert types
    (AR, FFG, neural, etc.) to be used interchangeably.
    """
    
    @property
    def regime_id(self) -> int:
        """The regime this expert specializes in"""
        ...
    
    @property
    def n_observations(self) -> int:
        """Number of observations this expert has processed"""
        ...
    
    def update(self, observation: float, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the expert's internal model with new observation.
        
        Args:
            observation: New data point
            context: Optional context (e.g., other features, metadata)
        """
        ...
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecast for given horizon.
        
        Args:
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions of length horizon
        """
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve current model parameters.
        
        Used for:
        - Serialization (saving/loading)
        - EWC Fisher matrix computation
        - Expert comparison
        """
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary of parameters to set
        """
        ...
    
    def clone(self) -> 'BaseExpert':
        """
        Create a deep copy of this expert.
        
        Used when branching experts for new regimes.
        """
        ...
    
    def get_uncertainty(self) -> float:
        """
        Estimate prediction uncertainty.
        
        Returns:
            Scalar uncertainty measure (e.g., prediction variance)
        """
        ...


@dataclass
class ExpertMetadata:
    """Tracks expert performance and lifecycle"""
    regime_id: int
    created_at: int  # Timestep
    n_updates: int = 0
    total_error: float = 0.0
    last_active: int = 0
    is_frozen: bool = False  # For EWC
    fisher_information: Optional[Dict[str, torch.Tensor]] = None
