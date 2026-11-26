# -*- coding: utf-8 -*-
import logging

from cadet_live import mqtt
from cadet_live import prometheus
import yaml

logger = logging.getLogger(__name__)


def get_config(filename: str = "../../examples/config.yml") -> dict:
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
        if "prometheus" in source:
            prometheus_client = prometheus.Client()

            all_clients = source["prometheus_client"]
            first_client = all_clients[0]
            # print("First Client: ", first_client)
            if "host" in first_client:
                prometheus_client.host = first_client["host"]
            if "port" in first_client:
                prometheus_client.port = first_client["port"]
            
    else:
        print("No source found")


def get_mqtt_client_config(config: dict) -> dict:
    if "source" in config:
        source = config["source"]
        if "mqtt" in source:
            if "devices" in source.get("mqtt"):
                all_clients = source["mqtt"]["devices"]
                first_client = all_clients[0]

            return (first_client)
    else:
        logger.error("No source found")

def create_mqtt_client(config: dict) -> mqtt.Client:
    client_config = []
    if "source" in config:
        source = config["source"]
        if "mqtt" in source:
            if "devices" in source.get("mqtt"):
                all_clients = source["mqtt"]["devices"]
                client_config = all_clients[0]


    # Find higher-level input_mapping if not set per device
    if not client_config.get("input_mapping"):
        # Try Type-level
        if config.get("source").get("mqtt").get("input_mapping"):
            client_config["input_mapping"] = config.get("source").get("mqtt").get("input_mapping")
        # Try global level
        elif config.get("source").get("input_mapping"):
            client_config["input_mapping"] = config.get("source").get("input_mapping")

    client = mqtt.Client(client_config)
    return client

def create_prometheus_client(config: dict) -> prometheus.Client:
    client_config = []
    if "source" in config:
        source = config["source"]
        if "prometheus" in source:
            if "devices" in source.get("prometheus"):
                all_clients = source["prometheus"]["devices"]
                client_config = all_clients[0]
    # Find higher-level input_mapping if not set per device
    if not client_config.get("input_mapping"):
        # Try Type-level
        if config.get("source").get("prometheus").get("input_mapping"):
            client_config["input_mapping"] = config.get("source").get("prometheus").get("input_mapping")
        # Try global level
        elif config.get("source").get("input_mapping"):
            client_config["input_mapping"] = config.get("source").get("input_mapping")
    client = prometheus.Client(client_config)
    return client

def get_target_api(config: dict) -> dict:
    api_config = config.get("target").get("api")
    return api_config


def get_topic_map(config: dict) -> dict:
    """Extract the topic_map section from the config."""
    return config.get("input_mapping", None)
