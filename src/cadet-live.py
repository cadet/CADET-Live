# -*- coding: utf-8 -*-

import logging
import time

from datetime import datetime

from CADETProcess.simulator import Cadet

import config
import h5
import mqtt
#import conversion


start_time = datetime.now()
print(type(start_time))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Setup MQTT client
raw_config = config.get_config()
mqtt_client_config = config.get_mqtt_client_config(raw_config)

client = mqtt.Client(mqtt_client_config)

data_mapping = config.get_topic_map(raw_config)

mqtt_client = mqtt.MqttConnection(client, data_mapping)


# Setup h5
sim_file = h5.load_h5_file(raw_config["simulation"]["filepath"])
logger.info("Start Loop")
for i in range(0, 1):
    print("Iteration: ", i)

#    print()
    ist_data = mqtt_client.client.user_data_get()
    ist_data_relativ = conversion.time_to_relative(ist_data, start_time)
#    print(ist_data or [])
#    data = sim_file["input"]["model"]["unit_001"]["INIT_C"]
#    print(data[0])
#    current_data = mqtt_client.client.user_data_get()
#    data[0] = current_data.get("A")
#    data[0] = current_data.get("B")
#    print(sim_file["input"]["model"]["unit_001"]["INIT_C"][()])
#    h5.save_h5_file(h5_file_path, sim_file)


#    cadet = Cadet()
#    cadet.check_cadet()
#    cadet.load_from_h5("./data.h5")
#    cadet.run_simulation()
#    print(vars(cadet))
#    result = cadet.run_h5(h5_file_path)
#    print(result.root.output.solution.unit_001.SOLUTION_BULK)
    time.sleep(10)

mqtt_client.mqtt_stop()
