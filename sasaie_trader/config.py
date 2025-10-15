from dataclasses import dataclass

@dataclass
class HummingbotConfig:
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = "admin"
    mqtt_password: str = "admin"
    bot_id: str = "sasaie-hummingbot"
    api_url: str = "http://localhost:8000"
    api_username: str = "admin"
    api_password: str = "admin"
