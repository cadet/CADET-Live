#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:57:57 2025

@author: jannisbergmann
"""

import logging

import paho.mqtt.client as mqtt
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)



class Client:
    def __init__(self, config):
        self.host = config["host"] or ""
        self.username = config["username"] or ""
        self.password = config["password"] or ""
        self.port = config["port"] or 1883
        self.timeout = config["timeout"] or 60


class MqttConnection:
    ### Setup
    def __init__ (self, client_info: Client):
    #    print("[MQTT] - Client Info: ", vars(client_info))
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.enable_logger()
        self.client.on_connect = self.__on_connect
        self.client.on_message = self.__on_message
        self.client.on_subscribe = self.__on_subscribe
        self.client.on_unsubscribe = self.__on_unsubscribe
        
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
    
    ### While Running
    def on_log(client, userdata, paho_log_level, messages):
        if paho_log_level == mqtt.LogLevel.MQTT_LOG_ERR:
            print(messages.payload)
    
    
    
    def __on_message(self, client, userdata, message):
        # userdata is the structure we choose to provide, here it's a list()
    #    print(client, ": ", message.payload)
    #    userdata.append(message.payload)
    #    userdata = []
        # Get LED intensity
        if(message.topic.endswith("/leds/intensity")):
            payload = eval(message.payload.decode())
            print(datetime.now())
            print("LED A: " + str(payload.get("A")))
            print("LED B: " + str(payload.get("B")))
            print("LED C: " + str(payload.get("C")))
            print("LED D: " + str(payload.get("D")))
    #        print(client)
    
            userdata["A"] = payload.get("A")
            userdata["B"] = payload.get("B")
            userdata["C"] = payload.get("C")
            userdata["D"] = payload.get("D")
    
        
        # We only want to process 10 messages
        if len(userdata) >= 50:
            client.unsubscribe("pioreactor/pioreactor01/#")
    
    
    
    ### Disconnect
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


#my_client = Client()
#mqtt_setup(my_client)