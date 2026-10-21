import math
from pathlib import Path

import matplotlib.pyplot as plt
import requests

from cadet_live.config import create_mqtt_client, get_config, get_control_api
from cadet_live.control.pid import PIDController
from cadet_live.mqtt import MqttConnection


CONFIG_PATH = Path(__file__).parent / "tubidostat_config.yml"


def update_plot(ax, line, times, values):
    line.set_data(times, values)
    ax.relim()
    ax.autoscale_view()
    ax.figure.canvas.draw_idle()


def main():
    config = get_config(CONFIG_PATH)
    api_config = get_control_api(config)
    mqtt_connection = MqttConnection(create_mqtt_client(config))

    pid = PIDController(
        kp=-10.0,
        ki=-1.0,
        kd=0.0,
        setpoint=0.025,
        output_limits=(0.0, 0.5),
    )
    plt.style.use("/Users/berger/fzj/projects/SAR-style.mplstyle")
    plt.ion()
    fig, (od_ax, control_ax) = plt.subplots(2, 1, sharex=True)
    od_line, = od_ax.plot([], [], "o-")
    control_line, = control_ax.step([], [], where="post")
    od_ax.set_ylabel("OD")
    control_ax.set_xlabel("Time [h]")
    control_ax.set_ylabel("Dilution rate [1/h]")
    fig.suptitle("Virtual Turbidostat")
    od_ax.grid(True)
    control_ax.grid(True)
    fig.tight_layout()
    plt.show(block=False)

    od_times = []
    od_history = []
    control_times = []
    control_values = []
    first_time_stamp = None
    last_time_stamp = None
    plotted_measurements = 0

    try:
        while plt.fignum_exists(fig.number):
            plt.pause(0.01)
            data = mqtt_connection.client.user_data_get()

            od_values = data.get("od_measure", [])
            
            if len(od_values) > plotted_measurements:
                new_measurements = od_values[plotted_measurements:]
                plotted_measurements += len(new_measurements)
                for measurement_time, measurement_od in new_measurements:
                    if (
                        not isinstance(measurement_od, (int, float))
                        or not math.isfinite(measurement_od)
                    ):
                        continue
                    if first_time_stamp is None:
                        first_time_stamp = measurement_time
                    od_times.append(
                        (measurement_time - first_time_stamp).total_seconds() / 3600.0
                    )
                    od_history.append(measurement_od)

                if od_times:
                    if control_values:
                        update_plot(
                            control_ax,
                            control_line,
                            control_times + [max(od_times[-1], control_times[-1])],
                            control_values + [control_values[-1]],
                        )
                    update_plot(od_ax, od_line, od_times, od_history)

            if not od_values:
                continue

            time_stamp, od = od_values[-1]
            if not isinstance(od, (int, float)) or not math.isfinite(od):
                continue

            if last_time_stamp is None:
                last_time_stamp = time_stamp
                continue

            control_dt_h = (time_stamp - last_time_stamp).total_seconds() / 3600.0
            if control_dt_h <= 0:
                continue

            last_time_stamp = time_stamp
            dilution_rate = pid.update(od, dt=control_dt_h)
            
            control_times.append((time_stamp - first_time_stamp).total_seconds() / 3600.0)
            control_values.append(dilution_rate)
            update_plot(
                control_ax,
                control_line,
                control_times + [max(od_times[-1], control_times[-1])],
                control_values + [dilution_rate],
            )

            response = requests.post(
                f"{api_config['url']}{api_config['path']}",
                json={"dilution_rate_per_h": dilution_rate},
                timeout=10,
            )
            response.raise_for_status()

    except KeyboardInterrupt:
        print("Stopping turbidostat controller")
    finally:
        mqtt_connection.mqtt_stop()


if __name__ == "__main__":
    main()
