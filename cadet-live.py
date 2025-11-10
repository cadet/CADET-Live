# -*- coding: utf-8 -*-

import time
import random

from CADETProcess.simulator import Cadet

import h5
import mqtt

client = mqtt.Client()

mqtt_client = mqtt.mqtt_setup(client)

sim_file = h5.load_h5_file("./data.h5")

print("Info: Start loop")
for i in range(0, 3):
    print("Iteration: ", i)

    data = sim_file["input"]["model"]["unit_000"]["INIT_C"]
#    print(data[()])
#    data[()] = random.randint(1, 100)
#    print(data[()])
    
     
    cadet = Cadet()
    cadet.check_cadet()
    time.sleep(5)

h5.save_h5_file("./data.h5", sim_file)

mqtt.mqtt_stop(mqtt_client)
# Set the filename for the existing simulation data
#sim.filename = sim_file
#sim.load()
#print(sim.run_simulation())

