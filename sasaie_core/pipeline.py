# Part of the new RegimeVAE architecture as of 2025-10-13

"""
Defines the main processing pipeline for the SASAIE system.
"""

import logging
import json
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Any

from sasaie_core.models.hierarchical_regime_vq_vae import HierarchicalRegimeVQVAE
from sasaie_core.components.planner import RegimeAwarePlanner
from sasaie_core.components.perception import HierarchicalStreamingMP # Changed import path

logger = logging.getLogger(__name__)

class MainPipeline:
    """
    Orchestrates the entire perception-belief-planning loop for the agent.
    This class is initialized with all necessary components and configurations,
    and its `process` method is called for each new data point.
    """
    def __init__(self, 
                 vqvae_model: HierarchicalRegimeVQVAE, 
                 planner: RegimeAwarePlanner, 
                 scales: List[int], 
                 mp_input_dim: int, 
                 initial_buffer_size: int = 500):
        
        self.vqvae = vqvae_model
        self.planner = planner
        self.scales = scales
        self.mp_input_dim = mp_input_dim
        self.initial_buffer_size = initial_buffer_size

        # Internal state
        self.ts_buffer = deque(maxlen=self.initial_buffer_size)
        self.mp_stream: HierarchicalStreamingMP = None
        logger.info("MainPipeline initialized.")

    def _preprocess_mp(self, mp: np.ndarray) -> torch.Tensor:
        """Converts a matrix profile to a fixed-size input tensor."""
        if len(mp) < self.mp_input_dim:
            padded = np.pad(mp, (0, self.mp_input_dim - len(mp)), mode='edge')
        else:
            indices = np.linspace(0, len(mp) - 1, self.mp_input_dim, dtype=int)
            padded = mp[indices]
        return torch.tensor(padded, dtype=torch.float32).unsqueeze(0)

    def process(self, raw_payload: bytes):
        """Main processing method for a single timestep."""
        try:
            payload = json.loads(raw_payload.decode())
            price_val = payload.get('price')
            if price_val is None:
                logger.warning("Received MQTT message without a 'price' field.")
                return
            price = float(price_val)

            self.ts_buffer.append(price)

            # 1. Handle warm-up period for the streaming matrix profile
            if self.mp_stream is None:
                if len(self.ts_buffer) >= self.initial_buffer_size:
                    logger.info("Initial buffer full. Initializing streaming matrix profile...")
                    self.mp_stream = HierarchicalStreamingMP(
                        initial_ts=np.array(self.ts_buffer),
                        scales=self.scales
                    )
                    logger.info("Streaming matrix profile initialized.")
                else:
                    # Not enough data yet, just buffer
                    return
            
            # --- Perception --- #
            # 2. Update matrix profiles with the new data point
            updated_mps = self.mp_stream.update(price)
            mp_features = {scale: self._preprocess_mp(mp) for scale, mp in updated_mps.items()}

            # --- Belief Update --- #
            # 3. Encode with the VQ-VAE to get current regime beliefs
            with torch.no_grad():
                regime_codes = self.vqvae.hierarchical_encode(mp_features)
            logger.debug(f"Perceived Regimes: {regime_codes}")

            # 4. Enrich beliefs with planner's context
            beliefs = self.planner.update_beliefs(regime_codes)
            logger.debug(f"Updated Beliefs: {beliefs}")

            # --- Forecasting --- #
            # 5. Generate forecasts from the planner
            forecasts = self.planner.generate_forecasts(regime_codes, horizon=10)
            logger.debug(f"Generated Forecasts: {forecasts}")

            # 6. Select the best policy by minimizing EFE
            best_policy = self.planner.select_best_policy(beliefs, forecasts)
            logger.info(f"Optimal policy selected: {best_policy.action_type.value} (EFE: {best_policy.efe:.3f})\n")

            # 7. Learn from the outcome (simplified)
            self.planner.learn_from_outcome(best_policy, price, regime_codes)

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from MQTT message.")
        except Exception as e:
            logger.error(f"Error in MainPipeline.process: {e}", exc_info=True)
