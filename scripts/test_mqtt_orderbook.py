import paho.mqtt.client as mqtt
import json

MQTT_HOST = "localhost"
MQTT_PORT = 1883
# Using the hbot namespace and the sasaie/order_book topic
MQTT_TOPIC = "hbot/sasaie/order_book/#" 

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the MQTT broker."""
    if rc == 0:
        print(f"Successfully connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for when a message is received from the MQTT broker."""
    print(f"Received message on topic: {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        print("Payload (JSON):")
        print(json.dumps(payload, indent=2))
    except json.JSONDecodeError:
        print("Payload (raw):")
        print(msg.payload)
    print("-" * 20)

def main():
    """Starts the MQTT client to listen for order book data."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Attempting to connect to MQTT broker at {MQTT_HOST}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Failed to start MQTT consumer: {e}", exc_info=True)

if __name__ == "__main__":
    main()
