# -*- coding: utf-8 -*-

import time

from CADETProcess.simulator import Cadet

import h5
import mqtt
import config


config_file = config.get_config()

configuration = config.config_to_source(config_file)

h5_file_path = "./modelLibrary/cstr_one_inlet_one_mal.h5"

sim_file = h5.load_h5_file(h5_file_path)

client = mqtt.Client()

mqtt_client = mqtt.mqtt_setup(configuration)
time.sleep(1)

print("Info: Start loop")
for i in range(0, 2):
    print("Iteration: ", i)

    data = sim_file["input"]["model"]["unit_001"]["INIT_C"]
    print(data[0])
    current_data = mqtt_client.user_data_get()
    data[0] = current_data["A"]
    print(sim_file["input"]["model"]["unit_001"]["INIT_C"][()])
    h5.save_h5_file(h5_file_path, sim_file)

     
    cadet = Cadet()
#    cadet.check_cadet()
#    cadet.load_from_h5("./data.h5")
#    cadet.run_simulation()
#    print(vars(cadet))
    result = cadet.run_h5(h5_file_path)
    print(result.root.output.solution.unit_001.SOLUTION_BULK)
    time.sleep(10)



h5.save_h5_file(h5_file_path, sim_file)

mqtt.mqtt_stop(mqtt_client)



# Set the filename for the existing simulation data
#sim.filename = sim_file
#sim.load()
#print(sim.run_simulation())

