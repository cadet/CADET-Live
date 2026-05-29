# -*- coding: utf-8 -*-

import alles

#setup

config = Config.get("Dateiname")

bridge.setup(config)
    # MQTT-Verbindung - Source and sink
    # Mapper
cadet.setup(config)
kalmna.setup(config)
regler.setup(config)

soll # Ziel kommt aus config

ist_data # bekommt daten immer von bridge/time-dependant-data
#============

for t in time:
    processed_data = ist_data.process()
    estimate = kalmann.run(processed_data, cadet) 
    anweisugnen = regler.run(estimate, soll, cadet)
    bridge.control(anweisungen)