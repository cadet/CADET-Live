"""Simple data container for MQTT connection settings."""


class Client:
    """Holds MQTT broker connection parameters."""

    def __init__(self):
        self.host: str = "localhost"
        self.port: int = 1883
        self.username: str = ""
        self.password: str = ""
        self.timeout: int = 60
