"""Main control loop: PioReactor -> MqttBridge -> EnKalmanFilter -> PID/MPC -> PioReactor.

PID controls temperature and stirring; MPC controls dosing rate (F_in).

Usage:
    cd examples
    python control_loop.py
"""

import logging
import os
import sys
import time

import numpy as np

# Add src/ to path so all core modules are importable
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _SRC)

import config
from MqttBridge import MqttBridge
from modelLibrary.Casadi.monod_cstr import create_monod_cstr
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from control.PID import PID
from control.optimalControl import (
    TrackingObjective,
    CasadiOptimalControlProblem,
    MPCController,
)
from Provider import ControlProvider
from LivePlot import LivePlot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
DT = 10.0           # Control loop time step [s]
N_ENSEMBLE = 30     # Ensemble size for Kalman filter
MAX_ITERATIONS = 30  # 0 = run forever

# MPC tuning for dosing rate (controls substrate S via F_in and F_out)
MPC_CONFIG = dict(
    time_horizon=100.0,       # 10 steps × DT
    u_min=[0.0, 0.0],         # [F_in_min, F_out_min]  [L/h]
    u_max=[0.2, 0.2],         # [F_in_max, F_out_max]  [L/h]
    Q=10.0,                   # substrate-tracking weight
    R=0.1,                    # control-effort weight
    setpoint=5.0,             # target substrate concentration [g/L]
    state_index=1,            # S is state index 1
)

# PID tuning for temperature and stirring (dosing_rate removed — handled by MPC)
PID_CONFIGS = {
    # Temperature: setpoint = target temperature [°C]
    # Input to PID = measured temperature → output = new target temperature
    "temperature_setpoint": dict(kp=1.0, ki=0.05, kd=0.0, setpoint=30.0, output_limits=(20, 40)),
    # Stirring: setpoint = target RPM, input = measured RPM from sensor
    "stirring_rate": dict(kp=1.0, ki=0.1, kd=0.0, setpoint=500.0, output_limits=(100, 800)),
}


def build_observation_function(provider_names, state_indices_map):
    """Build an observation function that maps full state -> observed measurements.

    Parameters
    ----------
    provider_names : list[str]
        Ordered list of provider names being observed.
    state_indices_map : dict[str, int]
        Maps provider name -> state index it observes.

    Returns
    -------
    callable
        Function x -> y_observed
    """
    indices = [state_indices_map[n] for n in provider_names if n in state_indices_map]

    def obs_func(x):
        return np.array([x[i] for i in indices])

    return obs_func


def main():
    # 1. Load configuration
    cfg = config.get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    client_config = config.config_to_source(cfg)
    topic_map = config.get_topic_map(cfg)

    if topic_map is None:
        logger.error("No topic_map found in config.yaml")
        sys.exit(1)

    # 2. Create MqttBridge
    bridge = MqttBridge(client_config, topic_map)
    bridge.connect()
    logger.info("Waiting for MQTT connection")
    for _ in range(30):
        if bridge.connected:
            break
        time.sleep(0.5)
    else:
        logger.error("Could not connect to MQTT broker")
        bridge.disconnect()
        sys.exit(1)
    logger.info("MQTT connected")

    # 3. Create CasadiModel (Monod CSTR)
    #    Single model shared by EnKF and MPC.  controllable_Fin=True +
    #    controllable_Fout=True exposes both F_in (u[0]) and F_out (u[1])
    #    as CasADi symbolic inputs so the MPC can optimise both and the EnKF
    #    propagation receives the actually applied control vector each step.
    model = create_monod_cstr(
        X0=np.array([0.1, 10.0, 1.0]),  # [Biomass, Substrate, Volume]
        dt=DT,
        controllable_Fin=True,
        controllable_Fout=True,
    )
    logger.info("CasadiModel created (Monod CSTR, 3 states, 2 controls: F_in + F_out)")

    # 4. Create EnKalmanFilter with measurement providers from bridge
    # Iterate all measurement names; collect only those with a state_index.
    # Each per-name provider is passed individually to the EnKF.
    state_indices_map = {}
    providers_for_enkf = []

    for name in bridge.measurement_names:
        prov = bridge.get_measurement_provider(name)
        var = prov.get_variable(name)
        if var is not None and var.state_index is not None:
            providers_for_enkf.append(prov)
            state_indices_map[name] = var.state_index

    has_observations = len(providers_for_enkf) > 0

    if not has_observations:
        logger.warning("No measurement variables with state_index found. "
                       "EnKF will run open-loop (prediction only).")

    obs_func = build_observation_function(
        list(state_indices_map.keys()),
        state_indices_map,
    ) if has_observations else None

    initial_cov = np.diag([0.01, 1.0, 0.001])  # [X, S, V]

    enkf = EnKalmanFilter(
        model=model,
        ensemble_size=N_ENSEMBLE,
        initial_covariance=initial_cov,
        observation_func=obs_func,
        providers=providers_for_enkf,
        random_seed=42,
    )
    logger.info("EnKalmanFilter created with %d providers", len(providers_for_enkf))

    # ------------------------------------------------------------------
    # 5a. Create PID controllers (temperature + stirring)
    # ------------------------------------------------------------------
    configured_controls = {c["name"] for c in topic_map.get("controls", [])}
    pids = {}

    for ctrl_name in configured_controls:
        if ctrl_name in PID_CONFIGS:
            pids[ctrl_name] = PID(**PID_CONFIGS[ctrl_name])
            logger.info("PID controller created for '%s'", ctrl_name)

    # ------------------------------------------------------------------
    # 5b. Create MPC controller (dosing rate via F_in)
    # ------------------------------------------------------------------
    mpc_ctrl = None
    if "dosing_rate" in configured_controls:
        
        mpc_objective = TrackingObjective(
            Q=MPC_CONFIG["Q"],
            R=MPC_CONFIG["R"],
            setpoint=MPC_CONFIG["setpoint"],
            state_index=MPC_CONFIG["state_index"],
        )
        
        mpc_ocp = CasadiOptimalControlProblem(
            model=model,
            objective=mpc_objective,
            time_horizon=MPC_CONFIG["time_horizon"],
            u_min=MPC_CONFIG["u_min"],
            u_max=MPC_CONFIG["u_max"],
            ipopt_print_level=0,
        )
        
        mpc_ctrl = MPCController(mpc_ocp)
        logger.info("MPCController created for 'dosing_rate' (horizon=%.0fs)",
                    MPC_CONFIG["time_horizon"])
    else:
        logger.warning("'dosing_rate' not in config controls — MPC disabled")

    # ControlProvider for the MPC prediction sequence; injected into the
    # controller so the sequence is published automatically after every update().
    # bridge.control_provider still holds only the actually applied u[0].
    mpc_plan_provider = ControlProvider("mpc_plan")
    if mpc_ctrl is not None:
        mpc_ctrl = MPCController(mpc_ocp, provider=mpc_plan_provider, variable_name="dosing_rate")
        logger.info("mpc_plan_provider created for MPC prediction sequence")

    # ------------------------------------------------------------------
    # 6. Wait for initial measurements
    # ------------------------------------------------------------------
    logger.info("Waiting for initial sensor data (up to 30s)...")
    for _ in range(60):
        has_data = any(
            bridge.get_latest_measurement(n) is not None
            for n in state_indices_map
        )
        if has_data:
            break
        time.sleep(0.5)
    else:
        logger.warning("No sensor data received yet. Starting loop anyway.")

    # ------------------------------------------------------------------
    # 7. Main control loop
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 7b. Create live plot
    # ------------------------------------------------------------------
    plot = LivePlot(
        measurement_provider=bridge.measurement_provider,
        control_provider=bridge.control_provider,
        state_names=["X (Biomass)", "S (Substrate)", "V (Volume)"],
    )

    logger.info("=== Starting control loop (dt=%.1fs, %d controls) ===", DT, len(configured_controls))
    iteration = 0
    # Last applied control vector [F_in, F_out] — passed to the EnKF propagation
    # so the ensemble prediction uses the same u that was sent to the plant.
    u_dosing_last: np.ndarray = np.zeros(2)
    # Track the latest measurement time we have already used for an EnKF correction
    # + MPC re-solve.  When a newer measurement arrives we call update(); otherwise
    # we call step() to apply the next value from the pre-computed sequence.
    t_last_meas_used: float = -np.inf

    try:
        while True:
            iteration += 1
            t = enkf.t_current + DT

            # --- Check whether a new MQTT measurement arrived since the last update ---
            new_measurement = False
            if has_observations:
                for vn in state_indices_map:
                    m = bridge.get_latest_measurement(vn)
                    if m is not None and float(m[0]) > t_last_meas_used:
                        new_measurement = True
                        t_last_meas_used = float(m[0])
                        break

            # --- Propagate + Update (EnKF) ---
            u_enkf = u_dosing_last if mpc_ctrl is not None else None
            if new_measurement:
                state = enkf.update_state_with_interpolation(
                    t_end=t, interpolation="nearest", u=u_enkf
                )
            else:
                enkf.propagate(t, u=u_enkf)
                state = enkf.state.copy()

            X_est, S_est, V_est = state[0], state[1], state[2]

            # --- Compute and publish control commands ---
            control_outputs = {}

            # Dosing MPC: re-solve when a new state estimate is available;
            # otherwise apply the next step from the pre-computed sequence.
            if mpc_ctrl is not None:
                if new_measurement:
                    _, _ = mpc_ctrl.solve(state, DT, t)
                    logger.debug("MPC re-solved at t=%.1f (new measurement)", t)
                else:
                    _, _ = mpc_ctrl.step(t)
                    logger.debug("MPC step at t=%.1f (no new measurement)", t)
                u_vec = mpc_ctrl.current_control          # [F_in, F_out]
                u_dosing_last = u_vec
                control_outputs["dosing_rate"]  = float(u_vec[0])
                control_outputs["outlet_rate"]  = float(u_vec[1])
                bridge.publish_control("dosing_rate", control_outputs["dosing_rate"], t)
                bridge.publish_control("outlet_rate", control_outputs["outlet_rate"], t)

            # Temperature PID: controls temperature (if available)
            if "temperature_setpoint" in pids:
                temp_meas = bridge.get_latest_measurement("temperature")
                temp_val = temp_meas[1][0] if temp_meas is not None else pids["temperature_setpoint"].current_setpoint(t)
                _, control_outputs["temperature_setpoint"] = pids["temperature_setpoint"].update(temp_val, DT, t)
                bridge.publish_control("temperature_setpoint", control_outputs["temperature_setpoint"], t)
            
            # Stirring PID: controls stirring based on biomass
            if "stirring_rate" in pids:
                _, control_outputs["stirring_rate"] = pids["stirring_rate"].update(X_est, DT, t)
                bridge.publish_control("stirring_rate", control_outputs["stirring_rate"], t)

            # --- Log ---
            log_msg = f"[{iteration:04d}] t={t:.1f} | X={X_est:.3f} S={S_est:.3f} V={V_est:.3f}"
            for ctrl_name, ctrl_val in control_outputs.items():
                log_msg += f" | {ctrl_name}={ctrl_val:.2f}"
            logger.info(log_msg)

            # --- Update live plot ---
            plot.update(state=state, t=t)

            # --- Check stop condition ---
            if 0 < MAX_ITERATIONS <= iteration:
                logger.info("Max iterations reached (%d). Stopping.", MAX_ITERATIONS)
                break

            time.sleep(DT)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.disconnect()
        plot.close()
        logger.info("Control loop stopped after %d iterations", iteration)


if __name__ == "__main__":
    main()
