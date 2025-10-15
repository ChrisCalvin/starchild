"""
Centralized manager for interacting with the Hummingbot API.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from hummingbot_api_client import HummingbotAPIClient
from sasaie_trader.config import HummingbotConfig

logger = logging.getLogger(__name__)

class HummingbotManager:
    """
    A centralized manager to handle all interactions with the Hummingbot API.
    This class encapsulates the HummingbotAPIClient and provides a clean interface
    for the rest of the application.
    """

    def __init__(self, config: HummingbotConfig):
        self._config = config
        self._api_client: Optional[HummingbotAPIClient] = None

    async def __aenter__(self):
        """Async context manager to ensure proper initialization and cleanup."""
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager to ensure proper cleanup."""
        await self.close()

    async def init(self):
        """Initializes the Hummingbot API client."""
        if not self._api_client:
            self._api_client = HummingbotAPIClient(
                base_url=self._config.api_url,
                username=self._config.api_username,
                password=self._config.api_password
            )
        logger.info("HummingbotManager: Initializing API client.")
        await self._api_client.init()

    async def close(self):
        """Closes the Hummingbot API client session."""
        if self._api_client:
            logger.info("HummingbotManager: Closing API client session.")
            await self._api_client.close()
            self._api_client = None

    async def start_strategy(self, bot_name: str, strategy_name: str, conf: Optional[str] = None) -> Dict[str, Any]:
        """Starts a strategy on a specified bot."""
        if not self._api_client:
            raise ConnectionError("HummingbotManager is not initialized. Call init() first.")
        
        logger.info(f"Attempting to start strategy '{strategy_name}' on bot '{bot_name}'...")
        try:
            response = await self._api_client.bot_orchestration.start_bot(
                bot_name=bot_name,
                script=strategy_name,
                conf=conf
            )
            if response.get("status") == "success":
                logger.info(f"Successfully sent start command for strategy '{strategy_name}'.")
            else:
                logger.error(f"Failed to start strategy: {response.get('msg')}")
            return response
        except Exception as e:
            logger.error(f"An error occurred while starting strategy: {e}", exc_info=True)
            raise

    async def get_active_bots_status(self) -> Dict[str, Any]:
        """
        Retrieves the status of all active bots.
        """
        if not self._api_client:
            raise ConnectionError("HummingbotManager is not initialized. Call init() first.")
        
        try:
            return await self._api_client.bot_orchestration.get_active_bots_status()
        except Exception as e:
            logger.error(f"An error occurred while getting active bots status: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def get_order_book(self, connector_name: str, trading_pair: str, depth: int = 10) -> Dict[str, Any]:
        """
        Retrieves the order book for a specific market.
        """
        if not self._api_client:
            raise ConnectionError("HummingbotManager is not initialized. Call init() first.")
        
        try:
            return await self._api_client.market_data.get_order_book(
                connector_name=connector_name,
                trading_pair=trading_pair,
                depth=depth
            )
        except Exception as e:
            logger.error(f"An error occurred while getting order book: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
