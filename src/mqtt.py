#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:57:57 2025

@author: jannisbergmann
"""

import logging
from datetime import datetime

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, config):
        logger.debug("Init Client with", config)
        self.host = config["host"] or "127.0.0.1"
        self.username = config["username"] or ""
        self.password = config["password"] or ""
        self.port = config["port"] or 1883
        self.timeout = config["timeout"] or 60
        self.timestamp_format = config["timestamp_format"] or "%Y-%m-%dT%H:%M:%S.%fZ"
        logger.info("Create Client with info: ", self)


class MqttConnection:
    # Setup
    def __init__(self, client_info: Client, mapping):
        logger.debug("Init MQTT-Connection")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.enable_logger()
        self.client.on_connect = self.__on_connect
        self.client.on_message = self.__on_message
        self.client.on_subscribe = self.__on_subscribe
        self.client.on_unsubscribe = self.__on_unsubscribe
        self.mapping = mapping
        self.timestamp_format = client_info.timestamp_format

        print("[MQTT] Connect to server")
        self.client.username_pw_set(client_info.username, client_info.password)
        self.client.user_data_set({})
#        logger.info("Connecting to MQTT broker: ", self.client.username_pw_set)
        self.client.connect(client_info.host, client_info.port, client_info.timeout)
        print("[MQTT] Server connected")

        # Blocking call that processes network traffic, dispatches callbacks and
        # handles reconnecting.
        # Other loop*() functions are available that give a threaded interface and a
        # manual interface.
        print("[MQTT] Start Loop")
        self.client.loop_start()
        print(f"Received the following message: {self.client.user_data_get()}")
        print(self.client)

    # The callback for when the client receives a CONNACK response from the server.
    def __on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("pioreactor/pioreactor01/#")

    def __on_subscribe(self, client, userdata, mid, reason_code_list, properties):
        # Since we subscribed only for a single channel, reason_code_list contains
        # a single entry
        if reason_code_list[0].is_failure:
            print(f"Broker rejected you subscription: {reason_code_list[0]}")
        else:
            print(f"Broker granted the following QoS: {reason_code_list[0].value}")

    # While Running
    def on_log(client, userdata, paho_log_level, messages):
        if paho_log_level == mqtt.LogLevel.MQTT_LOG_ERR:
            print(messages.payload)

    def __on_message(self, client, userdata, message):
        # userdata is the structure we choose to provide, here it's a list()
    #    print(client, ": ", message.payload)
    #    userdata.append(message.payload)
    #    userdata = []
        # Get LED intensity
        print(message.topic)
        for measurement_mapping in self.mapping:
            if (message.topic.endswith(measurement_mapping.get("topic_suffix"))):
                payload = eval(message.payload.decode())
                
                timestamp = datetime.now()
                value = ""

                print(measurement_mapping.get("name"), payload)
                if(measurement_mapping.get("type") == "nested_with_channel"):
                    value = payload.get(measurement_mapping.get("label")).get(measurement_mapping.get("channel")).get(measurement_mapping.get("name"))
                    timestamp = datetime.strptime(
                        payload.get(measurement_mapping.get("label")).get(measurement_mapping.get("channel")).get("timestamp"),
                        self.timestamp_format)
                elif(measurement_mapping.get("type") == "tuple"):
                    value = payload.get(measurement_mapping.get("label"))
                    timestamp = datetime.strptime(
                        payload.get("timestamp"),
                        self.timestamp_format)

 
                if (measurement_mapping.get("name") in userdata.keys()):
                    userdata[measurement_mapping.get("name")].append([timestamp, value])
                else:
                    userdata[measurement_mapping.get("name")] = [[timestamp, value]]

#                print("Timestamp:", timestamp, "  --  Measurement:", value)

        # We only want to process 10 messages
        if len(userdata) >= 50:
            client.unsubscribe("pioreactor/pioreactor01/#")

    # Disconnect
    def __on_unsubscribe(self, client, userdata, mid, reason_code_list, properties):
        # Be careful, the reason_code_list is only present in MQTTv5.
        # In MQTTv3 it will always be empty
        if len(reason_code_list) == 0 or not reason_code_list[0].is_failure:
            print("unsubscribe succeeded (if SUBACK is received in MQTTv3 it success)")
        else:
            print(f"Broker replied with failure: {reason_code_list[0]}")
        client.disconnect()

    def mqtt_stop(self):
        status = self.client.disconnect()
        print("MQTT Disconnect status: ", status)


# Toni
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
# Toni Ende


# my_client = Client()
# mqtt_setup(my_client)
