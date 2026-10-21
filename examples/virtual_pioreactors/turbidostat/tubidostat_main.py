import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException

from cadet_live.config import create_mqtt_client, get_config
from cadet_live.mqtt import MqttConnection


# --------------------- Reactor Dynamic ---------------------

class TubidostatVirtualPioreacor:

    def __init__(self, vmax, km, y_xs, s_in, x_in, V_init):
        
        self.vmax = vmax
        self.km = km
        self.y_xs = y_xs

        self.s = s_in
        self.s_in = s_in
        self.x = x_in

        self.V = V_init

        self.time = 0.0

        self.dilution_rate = 0.0
        self.lock = Lock()

    def step(self, dt):

        with self.lock:
            D = self.dilution_rate

            s = self.s
            x = self.x
            s_in = self.s_in

            mu = self.vmax * s / (s + self.km)

            dx_dt = mu * x - D * x
            ds_dt = - (mu / self.y_xs) * x + D * s_in - D * s

            self.x += dx_dt * dt
            self.s += ds_dt * dt
            self.time += dt

    def set_dilution_rate(self, dilution_rate: float):
        if not math.isfinite(dilution_rate) or dilution_rate < 0:
            raise ValueError("dilution_rate_per_h must be finite and non-negative")

        with self.lock:
            self.dilution_rate = dilution_rate

reactor = TubidostatVirtualPioreacor(vmax=0.4, km=0.01, y_xs=0.5, s_in=1.0, x_in=0.01, V_init=1)

# -------------------- API ----------------------

app = FastAPI(title="Virtual Tubidaost API")

@app.post("/control")
def control(command: dict[str, float]):
    try:
        dilution_rate = command["dilution_rate_per_h"]
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Missing {error.args[0]}") from error

    try:
        reactor.set_dilution_rate(dilution_rate)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {"status": "ok", "dilution_rate_per_h": dilution_rate}


@app.get("/status")
def status():
    return{"status": "running"}


# ----------------- main -------------------

def main():
    config = get_config(Path(__file__).parent / "tubidostat_config.yml")
    mqtt_connection = MqttConnection(create_mqtt_client(config))
    timestamp_format = config["source"]["mqtt"]["devices"][0]["timestamp_format"]

    api_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "127.0.0.1", "port": 8000},
        daemon=True,
    )
    api_thread.start()

    print("Starting virtual turbidostat")

    simulation_dt = 1 / 36
    wall_clock_dt = 0.01
    simulation_start = datetime.now(timezone.utc)

    try:
        while True:
            reactor.step(simulation_dt)

            time_stamp = (simulation_start + timedelta(hours=reactor.time)).strftime(timestamp_format)

            mqtt_connection.client.publish("virtual_pioreactor/state", json.dumps({ "od": reactor.x, "V": reactor.V, "timestamp": time_stamp,}))

            time.sleep(wall_clock_dt)

    except KeyboardInterrupt:
        print("\nStopping virtual Pioreactor")
    finally:
        mqtt_connection.mqtt_stop()


if __name__ == "__main__":
    main()
