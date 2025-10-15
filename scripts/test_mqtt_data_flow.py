import json
import logging
import subprocess
import sys
import time
from threading import Event, Thread

import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_HOST = "localhost"
MQTT_PORT = 1883
WILDCARD_TOPIC = "sasaie/#"
ORDERBOOK_TOPIC = "sasaie/hummingbot/events/orderbook"

# --- Test State ---
message_received = Event()
received_payload = None


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("MQTT Client: Connected to broker.")
        client.subscribe(WILDCARD_TOPIC)
        logger.info(f"MQTT Client: Subscribed to {WILDCARD_TOPIC}")
    else:
        logger.error(f"MQTT Client: Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    global received_payload
    logger.info(f"MQTT Client: Received message on topic {msg.topic}")
    if msg.topic == ORDERBOOK_TOPIC:
        received_payload = msg.payload.decode("utf-8")
        message_received.set()  # Signal that a message has been received


def validate_payload(payload: str) -> bool:
    """Validates the structure of the received JSON payload."""
    logger.info("Validating payload...")
    try:
        data = json.loads(payload)
        required_keys = ["timestamp", "trading_pair", "bids", "asks"]
        for key in required_keys:
            if key not in data:
                logger.error(f"Validation failed: Missing key '{key}'")
                return False
        if not isinstance(data["bids"], list) or not isinstance(data["asks"], list):
            logger.error("Validation failed: 'bids' or 'asks' is not a list.")
            return False
        logger.info("Payload validation successful.")
        return True
    except json.JSONDecodeError:
        logger.error("Validation failed: Could not decode JSON.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation: {e}")
        return False

def main():
    """Main integration test function."""
    launch_process = None
    mqtt_client = None
    try:
        # 1. Start the SASAIE system
        logger.info("Starting SASAIE system...")
        # Using a password to avoid interactive prompt
        launch_command = [sys.executable, "launch.py", "start", "-P", "a_password", "--force-restart"]
        launch_process = subprocess.Popen(
            launch_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        logger.info("Waiting for system to initialize... (approx. 45 seconds)")
        time.sleep(45)  # Give time for Docker and Hummingbot to start

        # Check system status
        logger.info("Checking system status...")
        status_command = [sys.executable, "launch.py", "status"]
        status_process = subprocess.run(
            status_command, capture_output=True, text=True
        )
        logger.info(f"Status command stdout:\n{status_process.stdout}")
        if status_process.stderr:
            logger.error(f"Status command stderr:\n{status_process.stderr}")

        # 2. Set up MQTT client
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        mqtt_thread = Thread(target=mqtt_client.loop_forever)
        mqtt_thread.daemon = True
        mqtt_thread.start()

        # 3. Wait for a message
        logger.info("Waiting for MQTT message...")
        received = message_received.wait(timeout=90)  # Wait for up to 90 seconds

        if not received:
            raise Exception("Test Failed: Did not receive an orderbook message on the MQTT topic.")

        logger.info("Orderbook message received!")

        # 4. Validate the payload
        if not validate_payload(received_payload):
            raise Exception("Test Failed: Payload validation failed.")

        print("\n\n*** Data Flow Integration Test PASSED! ***\n\n")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
        print(f"\n\n*** Data Flow Integration Test FAILED: {e} ***\n\n")

    finally:
        # 5. Stop the SASAIE system
        logger.info("Stopping SASAIE system...")
        if mqtt_client:
            mqtt_client.disconnect()
        if launch_process:
            # Check stdout/stderr from the launch process
            stdout, stderr = launch_process.communicate()
            logger.info(f"Launch process stdout:\n{stdout}")
            if stderr:
                logger.error(f"Launch process stderr:\n{stderr}")

        stop_command = [sys.executable, "launch.py", "stop"]
        subprocess.run(stop_command, check=True)
        logger.info("System stopped.")

        # Read the data collector log file
        try:
            log_file_path = "logs/hummingbot_logs/data_collector.log"
            with open(log_file_path, "r") as f:
                logger.info(f"--- Contents of {log_file_path} ---")
                logger.info(f.read())
                logger.info("-------------------------------------")
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_file_path}")
        except Exception as e:
            logger.error(f"Error reading log file: {e}")


if __name__ == "__main__":
    main()