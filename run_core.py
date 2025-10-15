# Part of the new RegimeVAE architecture as of 2025-10-13

"""
Main entry point for running the SASAIE Core service.
This script assembles and starts the main application pipeline.
"""

import time
import logging
import yaml

# Core application components
from sasaie_core.pipeline import MainPipeline
from sasaie_core.models.hierarchical_regime_vq_vae import HierarchicalRegimeVQVAE
from sasaie_core.components.planner import RegimeAwarePlanner
from sasaie_trader.mqtt_consumer import MarketDataConsumer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads the main YAML configuration file."""
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config


if __name__ == "__main__":
    logger.info("Initializing SASAIE Core Service...")

    # 1. Load Configuration
    CONFIG_PATH = "configs/generative_model.yaml"
    app_config = load_config(CONFIG_PATH)

    # Extract parameters from config
    # Model parameters
    scales = [s['window_size'] for s in app_config['scales']]
    codebook_sizes = [s['codebook_size'] for s in app_config['scales']]
    # These would also be in a more detailed config
    mp_input_dim = 100 
    latent_dim = 128

    # 2. Instantiate Core Components
    logger.info("Instantiating core components...")
    
    # World Model
    vqvae_model = HierarchicalRegimeVQVAE(
        scales=scales,
        input_dim=mp_input_dim,
        latent_dim=latent_dim,
        codebook_sizes=codebook_sizes
    )

    # Planner
    planner = RegimeAwarePlanner(
        scales=scales,
        vqvae=vqvae_model,
        expert_config={'expert_type': 'ar', 'order': 10} # Pass expert_config
    )

    # 3. Instantiate the Main Pipeline
    logger.info("Instantiating main processing pipeline...")
    main_pipeline = MainPipeline(
        vqvae_model=vqvae_model,
        planner=planner,
        scales=scales,
        mp_input_dim=mp_input_dim,
        initial_buffer_size=500 # This could also be in config
    )

    # 4. Instantiate and Start the Data Consumer
    logger.info("Instantiating and starting the data consumer...")
    # These would typically come from environment variables or a config file
    MQTT_HOST = "localhost" # Changed to localhost for local testing
    MQTT_PORT = 1883
    MQTT_TOPIC = "sasaie/market_data"

    consumer = MarketDataConsumer(
        mqtt_host=MQTT_HOST,
        mqtt_port=MQTT_PORT,
        topic=MQTT_TOPIC,
        pipeline=main_pipeline
    )
    
    # The consumer runs a blocking loop, so we don't need the time.sleep loop here.
    # We run it in a try/finally block to ensure cleanup.
    try:
        logger.info("SASAIE Core Service is running. Press Ctrl+C to stop.")
        # This will block until KeyboardInterrupt
        asyncio.run(consumer.start())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        consumer.stop()
        logger.info("SASAIE Core Service stopped.")