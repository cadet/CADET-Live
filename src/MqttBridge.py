import json
import logging
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from Provider import MeasurementProvider, ControlProvider

#TODO make an abstract class
#TODO create a experiment and add mqtt topic
logger = logging.getLogger(__name__)

"""Simple data container for MQTT connection settings."""


class Client:
    """Holds MQTT broker connection parameters."""

    def __init__(self):
        self.host: str = "localhost"
        self.port: int = 1883
        self.username: str = ""
        self.password: str = ""
        self.timeout: int = 60


class MqttBridge:
    """Bidirectional bridge between MQTT and the Provider system.

    Subscribes to PioReactor sensor topics and routes incoming data into
    a single MeasurementProvider. Publishes control outputs back via MQTT.

    Each measurement variable is stored as a TimeDependentData inside the
    MeasurementProvider — the Provider handles names, noise, and state_index.
    """

    def __init__(self, client_config, topic_map, subscribe_topic="pioreactor/pioreactor01/#"):
        """
        Parameters
        ----------
        client_config : mqtt.Client (from config.py)
            Object with .host, .port, .username, .password, .timeout.
        topic_map : dict
            Mapping config from config.yaml with 'measurements' and 'controls'.
        subscribe_topic : str
            MQTT topic pattern to subscribe to.
        """
        self._lock = threading.Lock()
        self._subscribe_topic = subscribe_topic
        self._connected = False
        self._t_start = None

        # MeasurementProvider — variables are managed as TimeDependentData
        self._measurement_provider = MeasurementProvider(name="mqtt_measurements")

        # Routing: topic_suffix -> [(variable_name, payload_key, payload_parser)]
        self._measurement_routes: dict[str, list[tuple[str, str | None, str | None]]] = {}

        # Single ControlProvider for outgoing commands
        self._control_provider = ControlProvider("mqtt_controls")
        self._control_routes: dict[str, str] = {}

        self._parse_topic_map(topic_map)

        # --- MQTT client ---
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._client.username_pw_set(client_config.username, client_config.password)

        self._host = client_config.host
        self._port = client_config.port
        self._timeout = client_config.timeout

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_topic_map(self, topic_map):
        """Parse the topic_map config and register variables in the Provider."""
        if topic_map is None:
            return

        for meas_cfg in topic_map.get("measurements", []):
            name = meas_cfg["name"]
            topic_suffix = meas_cfg["topic_suffix"]
            payload_key = meas_cfg.get("payload_key")
            payload_parser = meas_cfg.get("payload_parser")
            noise = meas_cfg.get("noise", 0.01)
            state_index = meas_cfg.get("state_index")

            # TimeDependentData is created inside the Provider
            self._measurement_provider.add_variable(
                name, noise=np.array([[noise]]), state_index=state_index
            )

            # Build routing table (only topic -> variable mapping + parser info)
            if topic_suffix not in self._measurement_routes:
                self._measurement_routes[topic_suffix] = []
            self._measurement_routes[topic_suffix].append(
                (name, payload_key, payload_parser)
            )

        for ctrl_cfg in topic_map.get("controls", []):
            name = ctrl_cfg["name"]
            topic = ctrl_cfg["topic"]
            self._control_routes[name] = topic
            self._control_provider.add_variable(name)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Connect to the MQTT broker and start the background loop."""
        logger.info("Connecting to MQTT broker at %s:%d", self._host, self._port)
        self._client.connect(self._host, self._port, self._timeout)
        self._client.loop_start()
        self._t_start = time.time()

    def disconnect(self):
        """Disconnect from the MQTT broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Disconnected from MQTT broker")

    @property
    def connected(self):
        return self._connected

    @property
    def elapsed_time(self):
        """Seconds since connect() was called."""
        if self._t_start is None:
            return 0.0
        return time.time() - self._t_start

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        logger.info("MQTT connected: %s", reason_code)
        client.subscribe(self._subscribe_topic)
        self._connected = True

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        logger.info("MQTT disconnected: %s", reason_code)
        self._connected = False

    def _on_message(self, client, userdata, message):
        """Route incoming MQTT message to the MeasurementProvider."""
        topic = message.topic
        try:
            payload_str = message.payload.decode()
        except UnicodeDecodeError:
            return

        t = self.elapsed_time

        for suffix, routes in self._measurement_routes.items():
            if topic.endswith(suffix):
                for var_name, payload_key, payload_parser in routes:
                    value = self._extract_value(payload_str, payload_key, payload_parser)
                    if value is not None:
                        with self._lock:
                            self._measurement_provider.add_measurement(var_name, t, value)
                        logger.debug("Measurement %s=%.4f at t=%.2f", var_name, value, t)

    @staticmethod
    def _extract_value(payload_str, payload_key, payload_parser=None):
        """Extract a numeric value from an MQTT payload.

        Supports three modes:
        1. payload_parser: Custom parsing function name (e.g., 'od_channel_2')
        2. payload_key: JSON key extraction
        3. None: Parse as plain number

        Parameters
        ----------
        payload_str : str
            Raw MQTT payload string.
        payload_key : str, optional
            JSON key to extract.
        payload_parser : str, optional
            Custom parser name ('od_channel_2', 'od_channel_1', etc.).

        Returns
        -------
        float or None
            Extracted numeric value.
        """
        try:
            # Custom parsers for PioReactor's nested JSON structures
            if payload_parser == "od_channel_2":
                # Extract from: {"timestamp":"...","ods":{"2":{"od":0.0002...}}}
                data = json.loads(payload_str)
                od_value = data.get("ods", {}).get("2", {}).get("od")
                return float(od_value) if od_value is not None else None

            elif payload_parser == "od_channel_1":
                # Extract from channel 1 instead
                data = json.loads(payload_str)
                od_value = data.get("ods", {}).get("1", {}).get("od")
                return float(od_value) if od_value is not None else None

            # Regular key extraction
            elif payload_key is not None:
                data = json.loads(payload_str)
                raw = data.get(payload_key)
                return float(raw) if raw is not None else None

            # Plain number
            else:
                return float(payload_str)

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Provider access (thread-safe)
    # ------------------------------------------------------------------

    @property
    def measurement_provider(self) -> MeasurementProvider:
        """The MeasurementProvider containing all MQTT measurement variables.

        Each variable is a TimeDependentData with name, noise, and state_index.
        """
        return self._measurement_provider

    @property
    def control_provider(self) -> ControlProvider:
        return self._control_provider

    def get_latest_measurement(self, name: str):
        """Get the most recent measurement value for a variable."""
        var = self._measurement_provider.get_variable(name)
        if var is None or len(var) == 0:
            return None
        with self._lock:
            return var.data_points[-1]  # (time, value)

    # ------------------------------------------------------------------
    # Publishing controls
    # ------------------------------------------------------------------

    def publish_control(self, variable_name: str, value: float, t: float = None):
        """Publish a control value to the PioReactor via MQTT.

        Parameters
        ----------
        variable_name : str
            Name of the control variable (must be in control_routes).
        value : float
            The control value to publish.
        t : float, optional
            Time for logging in the ControlProvider. Uses elapsed_time if None.
        """
        topic = self._control_routes.get(variable_name)
        if topic is None:
            logger.warning("No MQTT topic configured for control '%s'", variable_name)
            return

        if t is None:
            t = self.elapsed_time

        payload = json.dumps({"value": value})
        self._client.publish(topic, payload)

        with self._lock:
            self._control_provider.add_control(variable_name, t, np.array([value]))

        logger.info("Published control %s=%.4f to %s", variable_name, value, topic)
