"""
Core orchestration class for the SASAIE Launcher.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from .docker_manager import DockerManager, DockerError
from sasaie_trader.config import HummingbotConfig

logger = logging.getLogger(__name__)

class SASAIEOrchestrator:
    """
    Main orchestration class for managing the SASAIE system, including the
    containerized Hummingbot instance.
    """

    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.docker_manager = DockerManager()
        self.is_running = False
        self.container_ids: Dict[str, str] = {}
        self.hummingbot_config = HummingbotConfig() # Load config

        # Load environment variables from .env file
        dotenv_path = Path(__file__).parent.parent.parent / 'configs' / 'launcher' / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Loaded environment variables from {dotenv_path}")

    async def launch_system(self, dry_run: bool = False, force_restart: bool = False, hummingbot_password: Optional[str] = None) -> bool:
        """
        Validates the environment and starts all services using Docker Compose.

        Args:
            dry_run: If True, validates setup without starting services.
            force_restart: If True, stops existing services before starting.
            hummingbot_password: The password for the Hummingbot instance.

        Returns:
            True if the launch was successful, False otherwise.
        """
        logger.info("--- Starting SASAIE System --- ")
        try:
            # 1. Pre-flight validation
            logger.info("Running pre-flight validation...")
            await self.docker_manager.verify_docker_daemon()
            logger.info("Docker daemon verified.")

            status = await self.get_system_status()
            if status.get('status') == 'running' and not force_restart:
                logger.warning("System is already running. Use --force-restart to override.")
                return False

            if dry_run:
                logger.info("Dry run validation successful.")
                return True

            # 2. Stop existing services if force_restart is enabled
            if status.get('status') == 'running' and force_restart:
                logger.info("Forcing restart. Stopping existing services...")
                await self.docker_manager.stop_services(self.profile_name)
                await asyncio.sleep(5) # Give time for services to stop

            # 3. Infrastructure Startup
            logger.info("Starting infrastructure services via Docker Compose...")
            self.container_ids = await self.docker_manager.start_services(self.profile_name, hummingbot_password=hummingbot_password)
            if not self.container_ids:
                raise Exception("Docker services failed to start.")

            self.is_running = True
            logger.info(f"System launched successfully with profile '{self.profile_name}'.")
            logger.info(f"Container IDs: {self.container_ids}")
            logger.info("Hummingbot is now running. The native MQTT bridge is configured to start automatically.")

            return True

        except Exception as e:
            logger.error(f"System launch failed: {e}", exc_info=True)
            return False

    async def stop_system(self) -> bool:
        """
        Stops all running services managed by Docker Compose.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        logger.info("--- Stopping SASAIE System ---")
        try:
            await self.docker_manager.stop_services(self.profile_name)
            self.is_running = False
            self.container_ids = {}
            logger.info("System stopped successfully.")
            return True
        except DockerError as e:
            logger.error(f"Error during system shutdown: {e}", exc_info=True)
            return False

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of all managed services.
        """
        try:
            status = await self.docker_manager.get_services_status(self.profile_name)
            return status
        except DockerError as e:
            logger.error(f"Failed to get system status: {e}")
            return {"status": "error", "error": str(e)}