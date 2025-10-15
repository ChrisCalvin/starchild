# Part of the new RegimeVAE architecture as of 2025-10-13

"""
MQTT consumer for receiving market data.
This component is domain-specific and acts as the entry point for the core processing pipeline.
"""

import asyncio
import logging
import paho.mqtt.client as mqtt

from sasaie_core.pipeline import MainPipeline

logger = logging.getLogger(__name__)

class MarketDataConsumer:
    """
    Connects to an MQTT broker, subscribes to a topic, and passes received
    messages to the main processing pipeline.
    """

    def __init__(self, mqtt_host: str, mqtt_port: int, topic: str, pipeline: MainPipeline):
        self._mqtt_host = mqtt_host
        self._mqtt_port = mqtt_port
        self._topic = topic
        self.pipeline = pipeline
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the MQTT broker."""
        if rc == 0:
            logger.info(f"MarketDataConsumer: Successfully connected to MQTT broker at {self._mqtt_host}:{self._mqtt_port}")
            client.subscribe(self._topic)
            logger.info(f"MarketDataConsumer: Subscribed to topic: {self._topic}")
        else:
            logger.error(f"MarketDataConsumer: Failed to connect to MQTT broker, return code {rc}")

    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received from the MQTT broker."""
        logger.debug(f"MarketDataConsumer: Received message on topic {msg.topic}")
        try:
            # Pass the raw payload to the pipeline for processing
            self.pipeline.process(msg.payload)
        except Exception as e:
            logger.error(f"Exception during pipeline processing: {e}", exc_info=True)

    async def start(self):
        """Starts the MQTT client's network loop."""
        logger.info("MarketDataConsumer: Starting...")
        try:
            self._client.connect(self._mqtt_host, self._mqtt_port, 60)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.loop_forever)
        except Exception as e:
            logger.error(f"MarketDataConsumer: Failed to start: {e}", exc_info=True)

    def stop(self):
        """Stops the MQTT client's network loop."""
        logger.info("MarketDataConsumer: Stopping...")
        self._client.loop_stop()
