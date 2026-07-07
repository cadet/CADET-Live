# -*- coding: utf-8 -*-
import logging

import mqtt
import yaml

logger = logging.getLogger(__name__)


def get_config(filename: str = "config.yml") -> dict:
    logger.debug("Try to read file ", filename)
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
        logger.debug("Read config file successfully")
    return config


def config_to_source(config: dict) -> mqtt.Client:
    if "source" in config:
        source = config["source"]
        if "mqtt" in source:
            mqtt_client = mqtt.Client()

            all_clients = source["mqtt"]
            first_client = all_clients[0]
            # print("First Client: ", first_client)
            if "username" in first_client:
                mqtt_client.username = first_client["username"]
            if "password" in first_client:
                mqtt_client.password = first_client["password"]
            if "host" in first_client:
                mqtt_client.host = first_client["host"]
            if "port" in first_client:
                mqtt_client.port = first_client["port"]
            if "timeout" in first_client:
                mqtt_client.timeout = first_client["timeout"]
            if "topic" in first_client:
                mqtt_client.topic = first_client["topic"]

            # print("Client: ", vars(mqtt_client))

            return (mqtt_client)
    else:
        print("No source found")


def get_mqtt_client_config(config: dict) -> dict:
    if "source" in config:
        source = config["source"]
        if "mqtt" in source:
            all_clients = source["mqtt"]
            first_client = all_clients[0]

            return (first_client)
    else:
        logger.error("No source found")


def get_control_api(config: dict) -> dict:
    api_config = config.get("control").get("api")
    return api_config


def get_topic_map(config: dict) -> dict:
    """Extract the topic_map section from the config."""
    return config.get("topic_map", None)
