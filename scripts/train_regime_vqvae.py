# Part of the new RegimeVAE architecture as of 2025-10-13

"""
Training script for the HierarchicalRegimeVQVAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import yaml
from collections import deque
from typing import Dict, List, Tuple

from sasaie_core.models.hierarchical_regime_vq_vae import HierarchicalRegimeVQVAE
from sasaie_core.components.perception import HierarchicalStreamingMP

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
def load_config(config_path: str) -> dict:
    """Loads the main YAML configuration file."""
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config

# --- Dataset Definition ---
class TimeSeriesDataset(Dataset):
    def __init__(self, raw_data: np.ndarray, scales: List[int], mp_input_dim: int):
        self.raw_data = raw_data
        self.scales = scales
        self.mp_input_dim = mp_input_dim
        self.max_window_size = max(scales)
        
        # Pre-compute MP features for all possible windows
        self.mp_features_cache = self._precompute_mp_features()

    def _precompute_mp_features(self) -> List[Dict[int, torch.Tensor]]:
        logger.info("Pre-computing Matrix Profile features for dataset...")
        features_list = []
        # Iterate through the raw data with a sliding window of max_window_size
        for i in range(len(self.raw_data) - self.max_window_size + 1):
            window_data = self.raw_data[i : i + self.max_window_size]
            
            # Use HierarchicalStreamingMP to get features for this window
            # Initialize with the full window, then get latest profiles
            temp_hmp = HierarchicalStreamingMP(initial_ts=window_data, scales=self.scales)
            current_mp_features = {scale: temp_hmp.get_latest_profile(scale) for scale in self.scales}
            
            processed_mp_features = {}
            for scale, mp_data in current_mp_features.items():
                if np.isnan(mp_data).any() or np.isinf(mp_data).any():
                    logger.warning(f"NaN or Inf found in mp_data for scale {scale} at index {i}. Replacing with finite values.")
                    # Replace inf with a large number, and nan with the max of the profile (or a large number if all are nan)
                    finite_max = np.max(mp_data[np.isfinite(mp_data)]) if np.any(np.isfinite(mp_data)) else 10.0
                    mp_data = np.nan_to_num(mp_data, nan=finite_max, posinf=finite_max, neginf=0.0)

                if len(mp_data) < self.mp_input_dim:
                    padded = np.pad(mp_data, (0, self.mp_input_dim - len(mp_data)), mode='edge')
                else:
                    indices = np.linspace(0, len(mp_data) - 1, self.mp_input_dim, dtype=int)
                    padded = mp_data[indices]
                processed_mp_features[scale] = torch.tensor(padded, dtype=torch.float32)
            
            features_list.append(processed_mp_features)
        logger.info(f"Finished pre-computing {len(features_list)} samples.")
        return features_list

    def __len__(self):
        return len(self.mp_features_cache)

    def __getitem__(self, idx):
        return self.mp_features_cache[idx]

# --- Main Training Function ---
def train_vqvae(config_path: str = "configs/generative_model.yaml",
                data_path: str = "data/market_data.csv"):

    logger.info("Starting HierarchicalRegimeVQVAE training...")

    # Load configuration
    app_config = load_config(config_path)
    training_config = app_config.get('training', {}) # Use .get for safety

    # Training parameters
    epochs = training_config.get('epochs', 10)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 1e-3)
    commitment_cost = training_config.get('commitment_cost', 0.25)

    # Model parameters
    scales = [s['window_size'] for s in app_config['scales']]
    codebook_sizes = [s['codebook_size'] for s in app_config['scales']]
    mp_input_dim = 100 # This should ideally come from config or be dynamically determined
    latent_dim = 128

    # Load data
    raw_data = pd.read_csv(data_path)['close'].values # Using 'close' price from market_data.csv
    
    # Create a dataset of pre-computed MP features
    ts_dataset = TimeSeriesDataset(raw_data, scales, mp_input_dim)
    ts_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate VQ-VAE model
    model = HierarchicalRegimeVQVAE(
        scales=scales,
        input_dim=mp_input_dim,
        latent_dim=latent_dim,
        codebook_sizes=codebook_sizes
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, mp_features_batch in enumerate(ts_dataloader):
            # mp_features_batch is already a Dict[scale, Tensor(batch_size, input_dim)]
            optimizer.zero_grad()
            
            # Forward pass
            reconstructions, vq_loss, commitment_loss = model(mp_features_batch)

            # Calculate reconstruction loss (e.g., MSE)
            reconstruction_loss = torch.tensor(0.0)
            for scale in scales:
                reconstruction_loss += torch.mean((reconstructions[scale] - mp_features_batch[scale])**2)

            loss = reconstruction_loss + vq_loss + commitment_loss * commitment_cost
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(ts_dataloader):.4f}")

    logger.info("HierarchicalRegimeVQVAE training complete.")
    # Save the trained model
    torch.save(model.state_dict(), "vqvae_model.pt")

if __name__ == "__main__":
    # Run training with parameters from the config file
    train_vqvae()
