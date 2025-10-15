# Part of the new RegimeVAE architecture as of 2025-10-13

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import stumpy
from collections import deque

class DataPreprocessor:
    """Processes raw data from a connector into a model-ready format."""

    def __init__(self, feature_columns: list[str]):
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = feature_columns
        self.mean = None
        self.std = None

    def fit(self, raw_data: List[Dict[str, Any]]):
        """
        Calculates the mean and standard deviation from the entire dataset for normalization.
        """
        print("Fitting preprocessor to data...")
        # Extract the feature data into a list of lists
        feature_data = [[row[col] for col in self.feature_columns] for row in raw_data]
        # Convert to a NumPy array for efficient calculation
        data_array = np.array(feature_data, dtype=np.float32)
        
        # Calculate mean and std using NumPy
        self.mean = torch.from_numpy(data_array.mean(axis=0))
        self.std = torch.from_numpy(data_array.std(axis=0))
        
        # Add a small epsilon to std to avoid division by zero
        self.std[self.std == 0] = 1e-7
        print("Preprocessor fitted.")

    def process(self, raw_data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and normalizes a single raw data point.

        Args:
            raw_data_point: A dictionary representing one row from the data source.

        Returns:
            A dictionary containing the processed tensor and metadata.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        # Convert timestamp string to a datetime object
        timestamp_str = raw_data_point['timestamp']
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        timestamp = datetime.fromisoformat(timestamp_str)

        # Extract and validate features
        features = []
        for col in self.feature_columns:
            if col not in raw_data_point:
                raise KeyError(f"Feature column ''{col}'' not found in raw data.")
            features.append(raw_data_point[col])

        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Normalize the tensor
        normalized_tensor = (feature_tensor - self.mean) / self.std

        return {
            "timestamp": timestamp,
            "features": normalized_tensor,
            "metadata": {k: v for k, v in raw_data_point.items() if k not in self.feature_columns and k != 'timestamp'}
        }

class HummingbotDataPreprocessor(DataPreprocessor):
    """
    Processes raw order book data from Hummingbot API into a model-ready format.
    """
    def __init__(self, feature_columns: list[str], depth: int = 10):
        super().__init__(feature_columns)
        self.depth = depth

    def process(self, raw_data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and normalizes a single raw order book data point.
        Expected raw_data_point: {'trading_pair': 'BTC-USDT', 'bids': [...], 'asks': [...], 'timestamp': ...}
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        # Extract bids and asks
        bids = raw_data_point.get("bids", [])
        asks = raw_data_point.get("asks", [])

        # Extract features: best bid, best ask, mid-price, spread, and volumes at different levels
        features = []

        # Best bid and ask
        best_bid_price = bids[0][0] if bids else 0.0
        best_ask_price = asks[0][0] if asks else 0.0
        features.extend([best_bid_price, best_ask_price])

        # Mid-price and spread
        if best_bid_price > 0 and best_ask_price > 0:
            mid_price = (best_bid_price + best_ask_price) / 2
            spread = (best_ask_price - best_bid_price) / mid_price
        else:
            mid_price = 0.0
            spread = 0.0
        features.extend([mid_price, spread])

        # Volumes at different levels (up to self.depth)
        for i in range(self.depth):
            features.extend([bids[i][1] if i < len(bids) else 0.0, asks[i][1] if i < len(asks) else 0.0])

        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Normalize the tensor
        normalized_tensor = (feature_tensor - self.mean) / self.std

        # Extract timestamp
        timestamp = datetime.fromtimestamp(raw_data_point.get("timestamp", 0.0), tz=timezone.utc)

        return {
            "timestamp": timestamp,
            "features": normalized_tensor,
            "metadata": {
                "trading_pair": raw_data_point.get("trading_pair"),
                "data_type": "order_book_snapshot"
            }
        }

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

