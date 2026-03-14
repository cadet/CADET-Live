#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Script zum Überprüfen der MQTT-Verbindung mit dem PioReactor
"""
import os
import sys

import paho.mqtt.client as mqtt
import time
from datetime import datetime

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _SRC)

import config

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"✓ Verbunden! Reason Code: {reason_code}")
    # Alle Topics vom PioReactor abonnieren
    client.subscribe("pioreactor/pioreactor01/#")
    print("✓ Abonniert: pioreactor/pioreactor01/#")

def on_message(client, userdata, message):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Neue Nachricht:")
    print(f"  Topic: {message.topic}")
    print(f"  Payload: {message.payload.decode()}")
    userdata["count"] += 1
    
    if userdata["count"] >= 40:
        print("\n Test erfolgreich! 40 Nachrichten empfangen.")
        client.disconnect()

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"\n✓ Verbindung beendet. Insgesamt {userdata['count']} Nachrichten empfangen.")

# Config laden
print("1. Lade Konfiguration...")
cfg = config.get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
mqtt_config = config.config_to_source(cfg)

print(f"   Host: {mqtt_config.host}")
print(f"   Port: {mqtt_config.port}")
print(f"   Benutzer: {mqtt_config.username}\n")

# MQTT-Client erstellen
print("2. Verbinde mit MQTT Broker...")
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.username_pw_set(mqtt_config.username, mqtt_config.password)
client.user_data_set({"count": 0})

try:
    client.connect(mqtt_config.host, mqtt_config.port, mqtt_config.timeout)
    client.loop_start()
    
    print("3. Warte auf Nachrichten (max. 60 Sekunden)...\n")
    
    # Warten bis Verbindung abgebrochen oder Timeout
    for i in range(60):
        if client.user_data_get()["count"] >= 20:
            break
        time.sleep(1)
    
    print("\n✓ Test abgeschlossen!")
    
except ConnectionRefusedError:
    print(" FEHLER: Verbindung verweigert!")
    print("  Überprüfe:")
    print(f"  - Host '{mqtt_config.host}' ist erreichbar")
    print(f"  - Port {mqtt_config.port} ist offen")
    print(f"  - PioReactor läuft")
except Exception as e:
    print(f"✗ FEHLER: {e}")
finally:
    client.loop_stop()
