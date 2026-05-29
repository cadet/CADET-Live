# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


def parse_topic_map(self, topic_map: dict, mqtt_bridge: MqttConnection):
    """Parse the topic_map config and register variables in the Provider."""
    if topic_map is None:
        logger.warning("No topic_map provided")
        return

    for meas_cfg in topic_map.get("measurements", []):
        name = meas_cfg.get("name")
        topic_suffix = meas_cfg.get("topic_suffix")
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