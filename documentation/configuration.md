# CADET Live Documentation

## Usage
CADET Live is configured via a YAML file. This file "config.yml" and saved in the same location as the python script.

### Sources
Only a single concurrent source is supported at the moment.

#### MQTT Connection
The only data source supported is MQTT with the main focus on Pioreactors though other MQTT providers might also work.

```yaml
source:
  mqtt:
    - host: "127.0.0.1" # hostname or IP adress of mqtt server
      port: 1883 # port of the mqtt server to connect to 
      username: "" # username used for connection to MQTT server
      password: "" # password used for connection to MQTT server
      timeout: 60 # time before stop trying to connect to mqtt server
      timestamp_format: "%Y-%m-%dT%H:%M:%S.%fZ" # timestamp format used in MQTT messages
```

## Data Mapping
Data collected from sources has to be standardized for further usage. These settings enable this by mapping arbitrary information to known structures.
Depending on the information provided, there are multiple ways to process the data. These have to be configured using the type.

### tuple
This Type expects a timestamp and a label

```yaml
topic_map:
    - name: "" # internal name, has to match simulation variable
      type: "tuple" # has to be tuple
      topic_suffix: "" # the suffix of the mqtt topic to match
      label: "" # label used for value
```

### nested_with_channel
```yaml
topic_map:
    - name: "" # internal name, has to match simulation variable
      type: "nested_with_channel" # has to be nested_with_channel
      label: "" # label used for value
      topic_suffix: "/od_reading/ods" # the suffix of the mqtt topic to match
      channel: "" # the channel to get the value from
```

___

## Examples
### Sources
#### MQTT - Pioreactor
```yaml
source:
  mqtt:
    - host: "pioreactor01.local"
      port: 1883
      username: "pioreactor"
      password: "raspberry"
      timeout: 15
      timestamp_format: "%Y-%m-%dT%H:%M:%S.%fZ"
```
### Data Mapping

#### Pioreactor - Stirring RPM
```yaml
topic_map:
    - name: "rpm"
      type: "tuple"
      topic_suffix: "/stirring/measured_rpm"
      label: "measured_rpm"
```

#### Pioreactor - Temperature
```yaml
topic_map:
    - name: "temp"
      topic_suffix: "temperature_automation/temperature"
      type: "tuple"
      label: "temperature"
```

#### Pioreactor - Optical Density
##### tuble
```yaml
topic_map:
    - name: "od2"
      topic_suffix: "od_reading/od2"
      type: "tuple"
      label: "od"
```

##### netsed_with_channel
```yaml
topic_map:
    - name: "od"
      type: "nested_with_channel"
      label: "ods"
      topic_suffix: "/od_reading/ods"
      channel: "2"
```
