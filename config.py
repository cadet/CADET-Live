# -*- coding: utf-8 -*-
import yaml
import mqtt

def get_config(filename="config.yaml"):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
def config_to_source(config):
    if "source" in config:
        source = config["source"]
        if "mqtt" in source:
            mqtt_client = mqtt.Client()
            
            all_clients = source["mqtt"]
            first_client = all_clients[0]
            #print("First Client: ", first_client)
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
            
            #print("Client: ", vars(mqtt_client))
                
            return(mqtt_client)
    else:
        print("No source found")

#my_config = get_config()

#my_client = config_to_source(my_config)


#print("my_config: ", my_config)

#print("my_client: ", vars(my_client))


