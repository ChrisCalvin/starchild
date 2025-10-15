"""
Defines the PerceptionEngine for processing raw data into Observations.
"""

import torch
from typing import Dict, Any
from sasaie_core.api.observation import Observation
from sasaie_core.models.vae import VAE
from sasaie_trader.preprocessing import ScatteringPreprocessor, CSVDataPreprocessor, HummingbotDataPreprocessor
from sasaie_trader.connectors import DataConnector, CSVDataConnector, HummingbotAPIConnector
from datetime import datetime, timezone

class PerceptionEngine:
    """
    Orchestrates the full perception pipeline, from raw data to a rich,
    standardized Observation object.

    This engine uses a DataPreprocessor to normalize raw data, a
    ScatteringPreprocessor to compute scattering coefficients, and a trained VAE
    to extract a latent representation. It packages these into the standard
    Observation format and also calculates advanced features from the
    scattering coefficients.
    """

    def __init__(self,
                 data_connector: DataConnector,
                 scattering_preprocessor: ScatteringPreprocessor,
                 model: VAE = None,
                 model_path: str = None,
                 input_dim: int = None,
                 latent_dim: int = None,
                 hidden_dim: int = None):
        """
        Initializes the PerceptionEngine.

        Args:
            data_connector: An initialized instance of DataConnector.
            scattering_preprocessor: An initialized instance of ScatteringPreprocessor.
            model: An optional pre-initialized VAE model instance.
            model_path: Path to the saved .pt file for the trained VAE model.
            input_dim: The dimensionality of the input data for the VAE.
            latent_dim: The dimensionality of the latent space for the VAE.
            hidden_dim: The dimensionality of the hidden layers for the VAE.
        """
        self.data_connector = data_connector
        self.scattering_preprocessor = scattering_preprocessor

        if isinstance(data_connector, CSVDataConnector):
            self.data_preprocessor = CSVDataPreprocessor(feature_columns=["value"])
        elif isinstance(data_connector, HummingbotAPIConnector):
            self.data_preprocessor = HummingbotDataPreprocessor(feature_columns=["best_bid_price", "best_ask_price", "mid_price", "spread"])
        else:
            raise ValueError("Unsupported DataConnector type.")

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

    def calculate_scattering_features(self, scattering_coeffs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates advanced features from scattering coefficients.

        Args:
            scattering_coeffs: A dictionary containing scattering coefficients.

        Returns:
            A dictionary of engineered features.
        """
        s0 = scattering_coeffs.get("s0")
        s1 = scattering_coeffs.get("s1")
        s2 = scattering_coeffs.get("s2")

        engineered_features = {}

        if s0 is not None and s1 is not None and s2 is not None:
            s1_total = torch.sum(s1)
            s2_total = torch.sum(s2)
            if s1_total > 1e-6:
                volatility_regime = s2_total / s1_total
                engineered_features["volatility_regime"] = volatility_regime.unsqueeze(0)

            s1_low_freq = torch.sum(s1[:s1.shape[0] // 4])
            if s0 > 1e-6:
                trend_strength = s1_low_freq / s0
                engineered_features["trend_strength"] = trend_strength.unsqueeze(0)

        return engineered_features

    def process(self, raw_data_point: Dict[str, Any], time_series: torch.Tensor = None) -> Observation:
        """
        Processes a raw data point and optional time series to generate a rich Observation.

        Args:
            raw_data_point: A dictionary representing one row from the data source.
            time_series: An optional PyTorch tensor representing a related time series
                         for scattering transform.

        Returns:
            An Observation object containing the computed latent features and
            engineered scattering features.
        """
        # 1. Preprocess the raw data point
        preprocessed_data = self.data_preprocessor.process(raw_data_point)
        feature_tensor = preprocessed_data["features"]

        # Add a batch dimension if the input is a single tensor
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)

        # 2. Run the VAE on the normalized features
        with torch.no_grad():
            recon_x, mu, logvar = self.model(feature_tensor)

        observation_features = {"market_state_t1": mu}

        # 3. Process scattering coefficients if time_series is provided
        if time_series is not None:
            scattering_coeffs = self.scattering_preprocessor.process(time_series)
            engineered_scattering_features = self.calculate_scattering_features(scattering_coeffs)
            observation_features.update(engineered_scattering_features)

        # 4. Assemble the final Observation object
        observation = Observation(
            features=observation_features,
            metadata={
                "source": "perception_engine",
                "reconstruction_error": torch.nn.functional.mse_loss(recon_x, feature_tensor).item(),
                **preprocessed_data.get("metadata", {})
            },
            timestamp=datetime.now(timezone.utc)
        )

        return observation