"""Crabtree-Anlage als künstliche MQTT-Quelle.

Simuliert die Anlage (Chang et al. Modell) Schritt für Schritt und
publiziert X und G als verrauschte Messungen auf MQTT.
Empfängt gleichzeitig F_in und F_out vom Control-Loop über MQTT.

Starten (nach mosquitto und vor crabtree_control_loop.py):
    cd CADET-Live
    mamba run -n CADET-Live python examples/crabtree_plant_publisher.py
"""

import json
import logging
import os
import sys
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

# ── Pfade ──────────────────────────────────────────────────────────────────────
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _SRC)

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plant_publisher")

# ══════════════════════════════════════════════════════════════════════════════
# Kinetik-Parameter  (Table 3 – Chang et al.)
# ══════════════════════════════════════════════════════════════════════════════
MU_GF_MAX = 0.39;  K_GF  = 0.5;   K_IGF  = 10.0
K_IEGF    = 10.0;  K_IOGF = 189.0
Y_GF      = 0.11;  Y_EG  = 0.415
MU_GO_MAX = 0.34;  K_GO  = 0.041; K_IGO  = 86.0
K_IEGO    = 10.0;  K_OGO = 3.0;   Y_GO   = 0.58
G_FEED    = 200.0; DO_CONST = 5.0

# ── Simulationsparameter ───────────────────────────────────────────────────────
RNG_SEED   = 42
DT         = 0.1          # Schrittweite [h]
T_END      = 20.0         # Gesamtlaufzeit [h]
DT_REAL_S  = 0.5          # Realzeit pro Simulationsschritt [s]  (Zeitraffer)
MEAS_EVERY = 2            # Alle N Schritte eine Messung publishen

NOISE_X   = 0.02          # Messrauschen Std Biomasse
NOISE_G   = 0.5           # Messrauschen Std Glucose

# Anfangszustand  [X, G, E, V]
X0 = np.array([1.0, 10.0, 0.0, 1.0])

# Topics
TOPIC_BIOMASS  = "crabtree/sim/biomass"
TOPIC_GLUCOSE  = "crabtree/sim/glucose"
TOPIC_FIN      = "crabtree/control/fin"
TOPIC_FOUT     = "crabtree/control/fout"
TOPIC_STATUS   = "crabtree/sim/status"


def _ode_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Euler-Integration eines Schritts des Crabtree-Modells."""
    X_v, G_v, E_v, V_v = x
    Fin, Fout = u[0], u[1]

    G_s = max(G_v, 0.0);  E_s = max(E_v, 0.0);  V_s = max(V_v, 1e-3)

    mu_gf = (MU_GF_MAX
             * G_s / (K_GF + G_s + G_s**2 / K_IGF)
             / (1.0 + E_s / K_IEGF)
             / (1.0 + DO_CONST / K_IOGF))

    mu_go = (MU_GO_MAX
             * G_s / (K_GO + G_s + G_s**2 / K_IGO)
             / (1.0 + E_s / K_IEGO)
             * DO_CONST / (K_OGO + DO_CONST))

    dVdt = Fin - Fout
    dXdt = (mu_gf + mu_go) * X_v - Fin * X_v / V_s
    dGdt = Fin * (G_FEED - G_v) / V_s - (mu_gf / Y_GF + mu_go / Y_GO) * X_v
    dEdt = Y_EG * (mu_gf / Y_GF) * X_v - Fin * E_v / V_s

    x_next = x + dt * np.array([dXdt, dGdt, dEdt, dVdt])
    return np.maximum(x_next, 0.0)


class PlantPublisher:
    def __init__(self, broker_host: str, broker_port: int,
                 username: str = "", password: str = ""):
        self._lock = threading.Lock()
        self._u = np.zeros(2)          # [F_in, F_out] – vom Controller gesetzt

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect
        if username:
            self._client.username_pw_set(username, password)

        self._client.connect(broker_host, broker_port, 60)
        self._client.loop_start()

    # MQTT-Callbacks
    def _on_connect(self, client, userdata, flags, rc, props):
        logger.info("Plant connected to broker: %s", rc)
        client.subscribe(TOPIC_FIN)
        client.subscribe(TOPIC_FOUT)

    def _on_disconnect(self, client, userdata, flags, rc, props):
        logger.info("Plant disconnected: %s", rc)

    def _on_message(self, client, userdata, message):
        """Empfange F_in oder F_out vom Control-Loop."""
        try:
            val = float(json.loads(message.payload.decode()).get("value", 0.0))
        except (json.JSONDecodeError, ValueError, AttributeError):
            return
        with self._lock:
            if message.topic == TOPIC_FIN:
                self._u[0] = val
            elif message.topic == TOPIC_FOUT:
                self._u[1] = val

    # Publish-Hilfsmethode
    def _publish(self, topic: str, value: float, t: float):
        payload = json.dumps({"value": round(value, 6), "t": round(t, 4)})
        self._client.publish(topic, payload)

    # Haupt-Simulationsschleife
    def run(self):
        rng  = np.random.default_rng(RNG_SEED)
        x    = X0.copy()
        n    = int(T_END / DT)

        logger.info("Plant simulation started  (T=%.1fh, dt=%.2fh, %dx Zeitraffer)",
                    T_END, DT, int(1.0 / DT_REAL_S / DT))

        for k in range(n):
            t = k * DT

            # aktuellen Steuervektor thread-safe lesen
            with self._lock:
                u = self._u.copy()

            # Anlage simulieren
            x = _ode_step(x, u, DT)
            # Kleines Prozessrauschen
            x += rng.normal(0.0, [1e-3, 5e-3, 1e-3, 1e-4])
            x  = np.maximum(x, 0.0)

            # Messung publishen
            if (k + 1) % MEAS_EVERY == 0:
                t_meas = (k + 1) * DT
                y_X = max(0.0, x[0] + rng.normal(0.0, NOISE_X))
                y_G = max(0.0, x[1] + rng.normal(0.0, NOISE_G))

                self._publish(TOPIC_BIOMASS, y_X, t_meas)
                self._publish(TOPIC_GLUCOSE, y_G, t_meas)

                logger.info("t=%.2fh  X=%.3f(meas=%.3f)  G=%.3f(meas=%.3f)  "
                            "E=%.3f  V=%.3f  Fin=%.4f  Fout=%.4f",
                            t_meas, x[0], y_X, x[1], y_G, x[2], x[3], u[0], u[1])

            # Status (alle 10 Schritte)
            if k % 10 == 0:
                status = {"t": round(t, 3), "X": round(float(x[0]), 4),
                          "G": round(float(x[1]), 4), "E": round(float(x[2]), 4),
                          "V": round(float(x[3]), 4)}
                self._client.publish(TOPIC_STATUS, json.dumps(status))

            time.sleep(DT_REAL_S)

        # Ende
        self._client.publish(TOPIC_STATUS, json.dumps({"done": True, "t": T_END,
                                                        "E_total": round(float(x[2] * x[3]), 3)}))
        logger.info("Simulation abgeschlossen.  E·V = %.3f g", x[2] * x[3])
        self._client.loop_stop()
        self._client.disconnect()


def main():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crabtree_sim_config.yaml")
    cfg = config.get_config(cfg_path)
    src = cfg["source"]["mqtt"][0]

    pub = PlantPublisher(
        broker_host=src.get("host", "localhost"),
        broker_port=src.get("port", 1883),
        username=src.get("username", ""),
        password=src.get("password", ""),
    )

    # Kurz warten bis Broker-Verbindung steht
    time.sleep(1.0)
    pub.run()


if __name__ == "__main__":
    main()
