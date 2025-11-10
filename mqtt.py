#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:57:57 2025

@author: jannisbergmann
"""
import asyncio

import paho.mqtt.client as mqtt
from datetime import datetime

class Client:
    username = "pioreactor"
    password = "raspberry"
    host = "pioreactor.local"
    port = 1883
    timeout = 60

### Setup
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("pioreactor/pioreactor01/#")

def on_subscribe(client, userdata, mid, reason_code_list, properties):
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



def on_message(client, userdata, message):
    # userdata is the structure we choose to provide, here it's a list()
#    print(client, ": ", message.payload)
    userdata.append(message.payload)
    
    # Get LED intensity
    if(message.topic.endswith("/leds/intensity")):
        payload = eval(message.payload.decode())
        print(datetime.now())
        print("LED A: " + str(payload.get("A")))
        print("LED B: " + str(payload.get("B")))
        print("LED C: " + str(payload.get("C")))
        print("LED D: " + str(payload.get("D")))
        
    # We only want to process 10 messages
    if len(userdata) >= 50:
        client.unsubscribe("pioreactor/pioreactor01/#")



### Disconnect
def on_unsubscribe(client, userdata, mid, reason_code_list, properties):
    # Be careful, the reason_code_list is only present in MQTTv5.
    # In MQTTv3 it will always be empty
    if len(reason_code_list) == 0 or not reason_code_list[0].is_failure:
        print("unsubscribe succeeded (if SUBACK is received in MQTTv3 it success)")
    else:
        print(f"Broker replied with failure: {reason_code_list[0]}")
    client.disconnect()


def mqtt_setup(client_info):
    print("AAAAA")
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.enable_logger()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_subscribe = on_subscribe
    mqtt_client.on_unsubscribe = on_unsubscribe
        
    mqtt_client.username_pw_set(client_info.username, client_info.password)
    mqtt_client.user_data_set([])
    mqtt_client.connect(client_info.host, client_info.port, client_info.timeout)
    
    
    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
#    mqttc.loop_forever()
    mqtt_client.loop_start()
    print(f"Received the following message: {mqtt_client.user_data_get()}")
    
    return mqtt_client

def mqtt_stop(mqtt_client):
    status = mqtt_client.disconnect()
    print("MQTT Disconnect status: ", status)


#my_client = Client()
#mqtt_setup(my_client)
