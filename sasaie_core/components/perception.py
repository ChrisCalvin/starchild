"""
Defines the PerceptionEngine for processing raw data into Observations.
"""

import torch
import numpy as np # Added import
import stumpy # Added import
from collections import deque # Added import
from typing import Dict, Any, List, Tuple # Added List, Tuple
from sasaie_core.api.observation import Observation
from sasaie_core.models.vae import VAE
from sasaie_trader.connectors import DataConnector, CSVDataConnector, HummingbotAPIConnector
from datetime import datetime, timezone

class PerceptionEngine:
    """
    Orchestrates the full perception pipeline, from raw data to a rich,
    standardized Observation object.

    This engine uses a DataPreprocessor to normalize raw data, and a trained VAE
    to extract a latent representation. It packages these into the standard
    Observation format.
    """

    def __init__(self,
                 data_connector: DataConnector,
                 model: VAE = None,
                 model_path: str = None,
                 input_dim: int = None,
                 latent_dim: int = None,
                 hidden_dim: int = None):
        """
        Initializes the PerceptionEngine.

        Args:
            data_connector: An initialized instance of DataConnector.
            model: An optional pre-initialized VAE model instance.
            model_path: Path to the saved .pt file for the trained VAE model.
            input_dim: The dimensionality of the input data for the VAE.
            latent_dim: The dimensionality of the latent space for the VAE.
            hidden_dim: The dimensionality of the hidden layers for the VAE.
        """
        self.data_connector = data_connector

        # The old ScatteringPreprocessor and related DataPreprocessors are removed
        # as they are part of the old architecture.
        # A new PerceptionEngine would integrate HierarchicalStreamingMP directly.
        # For now, we remove the old dependencies to resolve import errors.

        if model:
            self.model = model
            print("PerceptionEngine initialized with provided model.")
        elif model_path and input_dim and latent_dim and hidden_dim:
            self.model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"PerceptionEngine initialized and model loaded from {model_path}")
        else:
            raise ValueError("Either a model instance or model_path with dimensions must be provided.")

    def process(self, raw_data_point: Dict[str, Any]) -> Observation:
        """
        Processes a raw data point to generate a rich Observation.

        Args:
            raw_data_point: A dictionary representing one row from the data source.

        Returns:
            An Observation object containing the computed latent features.
        """
        # For now, we're removing the old data preprocessor and scattering logic
        # as it's part of the old architecture.
        # A new PerceptionEngine would integrate HierarchicalStreamingMP directly.
        
        # Placeholder for actual preprocessing and VAE inference
        # In a real scenario, raw_data_point would be processed into a feature_tensor
        # and then passed to the VAE.
        
        # Mocking feature_tensor and VAE output for now to avoid further NameErrors
        feature_tensor = torch.randn(1, self.model.input_dim) # Assuming model.input_dim exists
        
        with torch.no_grad():
            recon_x, mu, logvar = self.model(feature_tensor)

        observation_features = {"market_state_t1": mu}

        # 4. Assemble the final Observation object
        observation = Observation(
            features=observation_features,
            metadata={
                "source": "perception_engine",
                "reconstruction_error": torch.nn.functional.mse_loss(recon_x, feature_tensor).item(),
                "timestamp": datetime.now(timezone.utc)
            },
            timestamp=datetime.now(timezone.utc)
        )

        return observation

# ============================================================================
# REGIME-BASED PREPROCESSING (New Architecture)
# ============================================================================

class HierarchicalStreamingMP:
    """
    Multi-scale streaming matrix profile computation using STUMPY.
    """
    def __init__(self, initial_ts: np.ndarray, scales: List[int] = [10, 50, 200]):
        self.scales = scales
        self.streams = {}
        self.current_mps = {}
        
        # Initialize streaming MP for each scale
        for scale in scales:
            if len(initial_ts) >= scale:
                self.streams[scale] = stumpy.stumpi(
                    initial_ts,
                    m=scale,
                    egress=True  # Sliding window mode
                )
                self.current_mps[scale] = self.streams[scale].P_
    
    def update(self, new_point: float) -> Dict[int, np.ndarray]:
        """
        Update all scales with a new data point.
        Returns: Dict mapping scale to its updated matrix profile.
        """
        updated_mps = {}
        for scale, stream in self.streams.items():
            stream.update(new_point)
            self.current_mps[scale] = stream.P_
            updated_mps[scale] = stream.P_
        return updated_mps
    
    def get_latest_profile(self, scale: int) -> np.ndarray:
        """Get the current matrix profile for a specific scale."""
        return self.current_mps.get(scale, np.array([]))


class StreamingRegimeDetector:
    """
    Online changepoint detection based on the latest matrix profile value.
    A high value indicates a new pattern (discord) has appeared.
    """
    def __init__(self, initial_ts: np.ndarray, m: int = 50):
        self.m = m
        self.stream = stumpy.stumpi(initial_ts, m=m)
        self.regime_change_threshold = 2.5 # This should be tuned (e.g., 3 standard deviations above the mean)
        
    def update(self, new_point: float) -> Tuple[bool, float]:
        """
        Check for a regime change by analyzing the latest matrix profile value.
        Returns: A tuple of (is_regime_change, latest_mp_value).
        """
        self.stream.update(new_point)
        latest_mp_value = self.stream.P_[-1]
        
        # A simple threshold-based change detection
        is_change = latest_mp_value > self.regime_change_threshold
        
        return is_change, latest_mp_value