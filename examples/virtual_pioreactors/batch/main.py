import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

from cadet_live.config import get_config, create_mqtt_client
from cadet_live.mqtt import MqttConnection

from examples.virtual_pioreactors.batch.batchPioreactor import BatchVirtualPioreacor

reactor = BatchVirtualPioreacor(vmax=0.4, km=0.01, y_xs=0.5, s_in=1.0, x_in=0.01)
CONFIG_PATH = Path(__file__).parent / "batch_config.yml"

config = get_config(CONFIG_PATH)
mqtt_client = create_mqtt_client(config)
mqtt_connection = MqttConnection(mqtt_client)

def main():

    print("Starting virtual Pioreactor")
    print("MQTT connected")

    simulation_dt = 1 / 60  
    wall_clock_dt = 0.01

    times = []
    od_values = []
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], "o-")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("OD")
    ax.set_title("Virtual Pioreactor Batch")
    ax.grid(True)
    plt.show()

    try:
        while True:
            reactor.step(simulation_dt)
            od_message = { "od": reactor.x }

            mqtt_connection.client.publish(
                "virtual_pioreactor/od_reading/od",
                json.dumps(od_message)
            )
            #plot
            times.append(reactor.time)
            od_values.append(reactor.x)
            line.set_data(times, od_values)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(wall_clock_dt)

    except KeyboardInterrupt:
        print("\nStopping virtual Pioreactor")


if __name__ == "__main__":
    main()