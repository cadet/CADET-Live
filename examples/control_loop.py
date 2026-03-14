"""Main control loop: PioReactor -> MqttBridge -> EnKalmanFilter -> PID -> PioReactor.

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
from Provider import MeasurementProvider
from modelLibrary.Casadi.monod_cstr import create_monod_cstr
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from control.PID import PID
from LivePlot import LivePlot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DT = 10.0           # Control loop time step [s]
N_ENSEMBLE = 30     # Ensemble size for Kalman filter
MAX_ITERATIONS = 30  # 0 = run forever

# PID tuning for each control (only used if configured in config.yaml)
PID_CONFIGS = {
    "dosing_rate": dict(kp=0.5, ki=0.01, kd=0.0, setpoint=5.0, output_limits=(0, 10)),
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
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg = config.get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    client_config = config.config_to_source(cfg)
    topic_map = config.get_topic_map(cfg)

    if topic_map is None:
        logger.error("No topic_map found in config.yaml")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Create MqttBridge
    # ------------------------------------------------------------------
    bridge = MqttBridge(client_config, topic_map)
    bridge.connect()
    logger.info("Waiting for MQTT connection...")
    for _ in range(30):
        if bridge.connected:
            break
        time.sleep(0.5)
    else:
        logger.error("Could not connect to MQTT broker")
        bridge.disconnect()
        sys.exit(1)
    logger.info("MQTT connected")

    # ------------------------------------------------------------------
    # 3. Create CasadiModel (Monod CSTR)
    # ------------------------------------------------------------------
    model = create_monod_cstr(
        X0=np.array([0.1, 10.0, 1.0]),  # [Biomass, Substrate, Volume]
        dt=DT,
    )
    logger.info("CasadiModel created (Monod CSTR, 3 states)")

    # ------------------------------------------------------------------
    # 4. Create EnKalmanFilter with measurement providers from bridge
    # ------------------------------------------------------------------
    # The bridge has one MeasurementProvider with all variables (as TimeDependentData).
    # The EnKF needs a provider with only the state-observed variables.
    # We create a filtered provider that shares the same TimeDependentData references,
    # so MQTT writes flow through automatically.
    provider = bridge.measurement_provider
    state_indices_map = {}

    enkf_provider = MeasurementProvider(name="enkf_observed")
    for var_name in provider.variable_names:
        var = provider.get_variable(var_name)
        if var is not None and var.state_index is not None:
            # Share the same TimeDependentData object (not a copy!)
            enkf_provider._data[var_name] = var
            state_indices_map[var_name] = var.state_index

    has_observations = len(enkf_provider.variable_names) > 0
    providers_for_enkf = [enkf_provider] if has_observations else []

    if not has_observations:
        logger.warning("No measurement variables with state_index found. "
                       "EnKF will run open-loop (prediction only).")

    obs_func = build_observation_function(
        enkf_provider.variable_names,
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
    # 5. Create PID controllers (only for configured controls)
    # ------------------------------------------------------------------
    configured_controls = {c["name"] for c in topic_map.get("controls", [])}
    pids = {}
    
    for ctrl_name in configured_controls:
        if ctrl_name in PID_CONFIGS:
            pids[ctrl_name] = PID(**PID_CONFIGS[ctrl_name])
            logger.info("PID controller created for '%s'", ctrl_name)
        else:
            logger.warning("No PID config defined for control '%s'", ctrl_name)

    # ------------------------------------------------------------------
    # 6. Wait for initial measurements
    # ------------------------------------------------------------------
    logger.info("Waiting for initial sensor data (up to 30s)...")
    for _ in range(60):
        has_data = any(
            len(enkf_provider.get_variable(n) or []) > 0
            for n in enkf_provider.variable_names
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

    try:
        while True:
            iteration += 1
            t = enkf.t_current + DT

            # --- Propagate + Update (EnKF) ---
            if has_observations:
                state = enkf.update_state_with_interpolation(
                    t_end=t, interpolation="nearest"
                )
            else:
                enkf.propagate(t)
                state = enkf.state.copy()

            X_est, S_est, V_est = state[0], state[1], state[2]

            # --- Compute and publish control commands ---
            control_outputs = {}
            
            # Dosing PID: controls substrate concentration
            if "dosing_rate" in pids:
                _, control_outputs["dosing_rate"] = pids["dosing_rate"].update(S_est, DT, t)
                bridge.publish_control("dosing_rate", control_outputs["dosing_rate"], t)
            
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
