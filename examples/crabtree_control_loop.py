"""Crabtree MPC + EnKF Control-Loop über MQTT.

Empfängt X- und G-Messungen von crabtree_plant_publisher.py via MqttBridge,
schätzt den Zustand [X, G, E, V] mit dem EnKF und optimiert F_in/F_out mit
dem MPC zur Maximierung des Ethanolgehalts E·V.

Starten (nach mosquitto und crabtree_plant_publisher.py):
    cd CADET-Live
    mamba run -n CADET-Live python examples/crabtree_control_loop.py
"""

import logging
import os
import sys
import time

import casadi as ca
import numpy as np

# ── Pfade ──────────────────────────────────────────────────────────────────────
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _SRC)

import config
from MqttBridge import MqttBridge
from Model import CasadiModel
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from control.optimalControl import Objective, CasadiOptimalControlProblem, MPCController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crabtree_control_loop")

# ══════════════════════════════════════════════════════════════════════════════
# Kinetik-Parameter  (Table 3 – Chang et al.)
# ══════════════════════════════════════════════════════════════════════════════
MU_GF_MAX = 0.39;  K_GF  = 0.5;   K_IGF  = 10.0
K_IEGF    = 10.0;  K_IOGF = 189.0
Y_GF      = 0.11;  Y_EG  = 0.415
MU_GO_MAX = 0.34;  K_GO  = 0.041; K_IGO  = 86.0
K_IEGO    = 10.0;  K_OGO = 3.0;   Y_GO   = 0.58
G_FEED    = 200.0; DO_CONST = 5.0

# ── Regelparameter ─────────────────────────────────────────────────────────────
DT          = 0.1     # MPC/EnKF-Schrittweite [h]
T_END       = 20.0    # Gesamtlaufzeit [h]
N_STEPS     = int(T_END / DT)
DT_REAL_S   = 0.5     # Realzeit pro Schritt [s] — muss mit Publisher übereinstimmen
V_MAX       = 3.0     # Reaktor-Maximalvolumen [L]
MPC_HORIZON = 3.0

U_MIN = [0.0,  0.0]
U_MAX = [0.15, 0.15]

Q_ETHANOL = 500.0
R_CTRL    =   1.0

# EnKF-Startschätzung  [X, G, E, V]
X0_BELIEF = np.array([0.9, 11.0, 0.0, 1.0])


# ══════════════════════════════════════════════════════════════════════════════
# CasADi ODE (identisch mit crap_tree_unstructured_mpc_enkf.py)
# ══════════════════════════════════════════════════════════════════════════════
def unstructured_ode(x, u):
    X_v, G_v, E_v, V_v = x[0], x[1], x[2], x[3]
    Fin = u[0];  Fout = u[1]
    G_s = ca.fmax(G_v, 0.0);  E_s = ca.fmax(E_v, 0.0);  V_s = ca.fmax(V_v, 1e-3)
    DO  = DO_CONST

    mu_gf = (MU_GF_MAX
             * G_s / (K_GF + G_s + G_s**2 / K_IGF)
             / (1.0 + E_s / K_IEGF)
             / (1.0 + DO  / K_IOGF))
    mu_go = (MU_GO_MAX
             * G_s / (K_GO + G_s + G_s**2 / K_IGO)
             / (1.0 + E_s / K_IEGO)
             * DO / (K_OGO + DO))

    dVdt = Fin - Fout
    dXdt = (mu_gf + mu_go) * X_v - Fin * X_v / V_s
    dGdt = Fin * (G_FEED - G_v) / V_s - (mu_gf / Y_GF + mu_go / Y_GO) * X_v
    dEdt = Y_EG * (mu_gf / Y_GF) * X_v - Fin * E_v / V_s
    return ca.vertcat(dXdt, dGdt, dEdt, dVdt)


# ══════════════════════════════════════════════════════════════════════════════
# Objective: Maximierung E·V
# ══════════════════════════════════════════════════════════════════════════════
class EthanolObjective(Objective):
    def __init__(self, Q_ethanol, R, Q_terminal_scale=10.0):
        self.Q_ethanol        = Q_ethanol
        self.R                = R
        self.Q_terminal_scale = Q_terminal_scale

    def stage_cost(self, state, control):
        E_v = state[2];  V_v = ca.fmax(state[3], 1e-3)
        return (-self.Q_ethanol * E_v * V_v
                + self.R * (control[0]**2 + control[1]**2))

    def terminal_cost(self, state):
        E_v = state[2];  V_v = ca.fmax(state[3], 1e-3)
        return -self.Q_ethanol * self.Q_terminal_scale * E_v * V_v


# ══════════════════════════════════════════════════════════════════════════════
# Hauptprogramm
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # 1. Config laden
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crabtree_sim_config.yaml")
    cfg          = config.get_config(cfg_path)
    client_cfg   = config.config_to_source(cfg)
    topic_map    = config.get_topic_map(cfg)

    # 2. MqttBridge aufbauen und verbinden
    bridge = MqttBridge(client_cfg, topic_map,
                        subscribe_topic="crabtree/#")
    bridge.connect()
    logger.info("Warte auf MQTT-Verbindung …")
    for _ in range(20):
        if bridge.connected:
            break
        time.sleep(0.3)
    else:
        logger.error("Kein MQTT-Broker erreichbar. Abbruch.")
        return
    logger.info("MQTT verbunden.")

    # 3. Modell
    states_sym   = ca.vertcat(*[ca.SX.sym(n) for n in ("X", "G", "E", "V")])
    controls_sym = ca.SX.sym("u", 2)

    model = CasadiModel(
        states=states_sym,
        controls=controls_sym,
        ode=unstructured_ode,
        init_state=X0_BELIEF.copy(),
        process_noise=np.diag([1e-4, 1e-3, 1e-4, 1e-5]),
        dt=DT,
    )

    # 4. EnKF konfigurieren via Bridge-Provider
    #    Alle Messnamen mit state_index aus der Bridge holen, jeden Provider
    #    einzeln an den EnKF übergeben. Reihenfolge nach state_index sortiert.
    state_indices   = {}
    providers_for_enkf = []

    for name in bridge.measurement_names:
        prov = bridge.get_measurement_provider(name)
        var  = prov.get_variable(name)
        if var is not None and var.state_index is not None:
            providers_for_enkf.append(prov)
            state_indices[name] = var.state_index

    ordered_vars = sorted(state_indices, key=lambda n: state_indices[n])
    obs_func     = lambda x: np.array([x[state_indices[n]] for n in ordered_vars])

    enkf = EnKalmanFilter(
        model=model,
        ensemble_size=30,
        initial_covariance=np.diag([0.04, 2.0, 0.01, 0.01]),
        observation_func=obs_func,
        providers=providers_for_enkf,
        random_seed=42,
    )
    logger.info("EnKF initialisiert (obs: %s)", ordered_vars)

    # 5. MPC
    ocp = CasadiOptimalControlProblem(
        model=model,
        objective=EthanolObjective(Q_ETHANOL, R_CTRL),
        time_horizon=MPC_HORIZON,
        u_min=U_MIN,
        u_max=U_MAX,
        ipopt_print_level=0,
        path_constraints=[(3, 0.5, V_MAX)],
    )
    mpc = MPCController(ocp)
    logger.info("MPC initialisiert (Horizont=%.1fh, V_max=%.1fL)", MPC_HORIZON, V_MAX)

    # 6. Warte auf erste Messung vom Publisher
    logger.info("Warte auf erste Messung vom Plant-Publisher …")
    for _ in range(100):
        if all(bridge.get_latest_measurement(n) is not None for n in ordered_vars):
            break
        time.sleep(0.3)
    else:
        logger.warning("Keine Messung empfangen — starte trotzdem.")

    # 7. Regelkreis
    u_last             = np.zeros(2)
    last_meas_time     = -np.inf      # Echtzeit-Sekunden der zuletzt verwendeten Messung

    logger.info("=== Regelkreis gestartet (%d Schritte, dt=%.2fh) ===",
                N_STEPS, DT)

    for k in range(N_STEPS):
        t_sim = (k + 1) * DT          # Simulationszeit [h]

        # ── Ensemble propagieren
        enkf.propagate(t_end=t_sim, u=u_last)

        # ── Neue Messung prüfen (Bridge nutzt Echtzeit-Sekunden)
        meas = {n: bridge.get_latest_measurement(n) for n in ordered_vars}
        new_meas = False

        if all(m is not None for m in meas.values()):
            t_latest = max(float(m[0]) for m in meas.values())

            if t_latest > last_meas_time:
                y = np.array([float(meas[n][1][0]) for n in ordered_vars])
                enkf.update_state(y)
                last_meas_time = t_latest
                new_meas = True

        state = np.maximum(enkf.state.copy(), 0.0)

        # ── MPC lösen
        mpc.solve(state, DT, t_sim)
        u_vec  = mpc.current_control
        u_last = u_vec

        # ── Stellgrößen via Bridge publizieren
        bridge.publish_control("fin",  float(u_vec[0]), t_sim)
        bridge.publish_control("fout", float(u_vec[1]), t_sim)

        logger.info(
            "[%04d] t=%.2fh | X=%.3f G=%.3f E=%.3f V=%.3f | "
            "Fin=%.4f Fout=%.4f | E·V=%.3f g | new_meas=%s",
            k + 1, t_sim,
            state[0], state[1], state[2], state[3],
            u_vec[0], u_vec[1],
            state[2] * state[3],
            new_meas,
        )

        time.sleep(DT_REAL_S)

    # 8. Abschluss
    state = enkf.state.copy()
    logger.info("Regelkreis beendet.")
    logger.info("Letzte Schätzung:  X=%.3f  G=%.3f  E=%.3f  V=%.3f  E·V=%.3f g",
                state[0], state[1], state[2], state[3], state[2] * state[3])
    bridge.disconnect()


if __name__ == "__main__":
    main()
