"""
Docker container management for the SASAIE Production Launcher.
"""

import asyncio
import logging
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
import docker.errors

logger = logging.getLogger(__name__) 

class DockerError(Exception):
    """Custom exception for Docker-related errors"""
    pass

class DockerManager:
    """
    Manages Docker container lifecycle and orchestration using Docker Compose.
    Adapted from the thor project for SASAIE.
    """

    def __init__(self):
        self.client = None
        self.network_name = "sasaie-network"
        # Corrected path to docker-compose.yml relative to the project root
        self.compose_file = Path(os.path.join(os.getcwd(), "configs", "launcher", "docker-compose.yml"))
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Docker client with error handling"""
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("Docker client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise DockerError(f"Docker daemon not accessible: {e}")

    async def verify_docker_daemon(self) -> None:
        """Verify Docker daemon is running and accessible."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.ping)
            await self._check_docker_compose()
            logger.info("Docker daemon and Docker Compose verified.")
        except Exception as e:
            raise DockerError(f"Docker daemon not accessible: {e}")

    async def _check_docker_compose(self) -> None:
        """Check if Docker Compose is available."""
        try:
            result = await self._run_command(["docker", "compose", "version"])
            if result["returncode"] != 0:
                raise DockerError("Docker Compose V2 not available.")
        except Exception as e:
            logger.error(f"Docker Compose check failed: {e}")
            raise DockerError("Docker Compose check failed.")

    async def start_services(self, profile_name: str, hummingbot_password: Optional[str] = None) -> Dict[str, str]:
        """
        Start containers using Docker Compose with a specific profile name.
        Returns a dictionary mapping service names to container IDs.
        """
        try:
            if not self.compose_file.exists():
                raise DockerError(f"Docker Compose file not found: {self.compose_file}")

            env_vars = os.environ.copy()
            if hummingbot_password:
                env_vars["HUMMINGBOT_PASSWORD"] = hummingbot_password
            env_vars["COMPOSE_PROJECT_NAME"] = f"sasaie-{profile_name}"
            env_vars["PWD"] = str(Path.cwd()) # Ensure PWD is set for docker compose
            env_vars["PROJECT_ROOT"] = str(Path.cwd()) # Set project root for absolute volume paths

            result = await self._run_compose_command(["up", "-d"], env=env_vars)
            if result["returncode"] != 0:
                raise DockerError(f"Docker Compose up failed: {result['stderr']}")

            container_ids = await self._get_compose_container_ids(profile_name)
            await self._wait_for_services_healthy(container_ids)
            logger.info("All Docker Compose services started and healthy.")
            return container_ids
        except Exception as e:
            raise DockerError(f"Failed to start services with Docker Compose: {e}")

    async def stop_services(self, profile_name: str) -> None:
        """Stop containers using Docker Compose."""
        try:
            result = await self._run_compose_command(["--project-name", f"sasaie-{profile_name}", "stop"])
            if result["returncode"] != 0:
                logger.warning(f"Docker Compose down failed: {result['stderr']}")
            logger.info("Docker Compose services stopped.")
        except Exception as e:
            raise DockerError(f"Failed to stop services with Docker Compose: {e}")

    async def get_services_status(self, profile_name: str) -> Dict[str, Any]:
        """Get the status of Docker Compose services."""
        try:
            result = await self._run_compose_command(["--project-name", f"sasaie-{profile_name}", "ps", "--format", "json"])
            logger.debug(f"Result from docker compose ps command: {result}") # ADD THIS LINE
            if result["returncode"] != 0:
                return {"services": {}, "status": "error", "error": result['stderr']}

            services = {}
            if result["stdout"]:
                logger.debug(f"Raw stdout from docker compose ps: {result['stdout']}") # Add this line
                # Split by the actual delimiter and reconstruct valid JSON lines
                json_parts = result["stdout"].replace('}n{', '}\n{').strip().split('\n')

                for line in json_parts: # Iterate over the correctly split JSON parts
                    if line:
                        try: # Add try-except block
                            service_info = json.loads(line)
                            logger.debug(f"Type of service_info: {type(service_info)}") # ADD THIS LINE
                            service_name = service_info.get("Service", "unknown")

                            # Map Docker container state to a simpler status
                            container_state = service_info.get("State", "unknown").lower()
                            status_map = {
                                "running": "running",
                                "restarting": "restarting",
                                "paused": "paused",
                                "exited": "stopped",
                                "dead": "stopped",
                                "created": "created"
                            }
                            mapped_status = status_map.get(container_state, "unknown")

                            health_status = "unknown" # Default
                            if "Health" in service_info and isinstance(service_info["Health"], dict): # Check if it's a dict
                                health_status = service_info["Health"].get("Status", "unknown")
                            elif mapped_status == "running":
                                health_status = "healthy" # Assume healthy if running and no healthcheck defined

                            services[service_name] = {
                                "status": mapped_status,
                                "health": health_status,
                                "container_id": service_info.get("ID", ""),
                            }
                        except json.JSONDecodeError as e: # Catch JSON decoding errors
                            logger.error(f"Failed to parse JSON line: {line}. Error: {e}")
                            continue # Skip to next line if parsing fails
            return {"services": services, "status": "running" if services else "stopped"}
        except docker.errors.APIError as e: # Catch specific Docker API errors
            logger.error(f"Failed to get compose status (Docker API Error): {e}")
            return {"services": {}, "status": "error", "error": str(e)}
        except Exception as e: # Catch any other general exceptions
            logger.error(f"Failed to get compose status (General Error): {e}", exc_info=True) # Add exc_info=True to get traceback
            return {"services": {}, "status": "error", "error": str(e)}

    async def _get_compose_container_ids(self, profile_name: str) -> Dict[str, str]:
        """Get container IDs for the services defined in the compose file."""
        try:
            result = await self._run_compose_command(["--project-name", f"sasaie-{profile_name}", "ps", "-q"])
            if result["returncode"] != 0 or not result["stdout"]:
                return {}

            ids = result["stdout"].strip().split('\n')
            containers = self.client.containers.list(filters={'id': ids})
            return {c.labels.get('com.docker.compose.service', c.name): c.id for c in containers}
        except Exception as e:
            raise DockerError(f"Failed to get compose container IDs: {e}")

    async def _wait_for_services_healthy(self, container_ids: Dict[str, str], timeout: int = 180) -> None:
        """Wait for all specified containers to report a healthy status."""
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            all_healthy = True
            for service_name, container_id in container_ids.items():
                try:
                    container = self.client.containers.get(container_id)
                    if container.status != "running":
                        all_healthy = False
                        break
                    health = container.attrs.get("State", {}).get("Health", {})
                    if health and health.get("Status") == "unhealthy":
                        raise DockerError(f"Service '{service_name}' is unhealthy.")
                    if health and health.get("Status") == "starting":
                        all_healthy = False
                        break
                except docker.errors.NotFound:
                    all_healthy = False
                    break
            if all_healthy:
                return
            await asyncio.sleep(5)
        raise DockerError(f"Services did not become healthy within {timeout} seconds.")

    async def _run_command(self, command: List[str], env: Dict[str, str] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run a shell command asynchronously."""
        full_command_str = " ".join(command) # ADD THIS LINE
        logger.debug(f"Executing command: {full_command_str} in cwd: {cwd}") # ADD THIS LINE
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env or os.environ,
            cwd=cwd
        )
        stdout, stderr = await process.communicate()
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip(),
        }

    async def _run_compose_command(self, args: List[str], env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run a Docker Compose command."""
        compose_file_dir = self.compose_file.parent # Get the parent directory of the compose file
        command = ["docker", "compose", "-f", str(self.compose_file)] + args
        return await self._run_command(command, env=env, cwd=compose_file_dir)

    async def exec_in_container(self, container_id: str, command: List[str], detached: bool = False) -> Dict[str, Any]:
        """Execute a command inside a running container."""
        detached_flag = ["-d"] if detached else []
        full_command = ["docker", "exec"] + detached_flag + [container_id] + command
        return await self._run_command(full_command)