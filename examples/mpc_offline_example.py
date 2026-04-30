"""Offline MPC example: closed-loop simulation with artificial sensor data.

Simulates a Monod-CSTR bioreactor with:
  - A "true" plant model (with process noise) as the physical system
  - An EnKalmanFilter estimating the state from noisy OD measurements
  - An MPCController driving the dosing rate (F_in) to track a substrate setpoint
  - A PID controller for comparison

No MQTT or hardware required — all sensor readings are generated synthetically.

Usage:
    cd examples
    mamba run -n CADET-Live python mpc_offline_example.py
"""

import os
import sys
import logging

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # interactive window; falls back gracefully on headless systems
import matplotlib.pyplot as plt

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(__file__))

from modelLibrary.Casadi.monod_cstr import create_monod_cstr
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from Provider import MeasurementProvider, ControlProvider
from control.PID import PID
from control.optimalControl import (
    TrackingObjective,
    CasadiOptimalControlProblem,
    MPCController,
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------
RNG_SEED        = 0
DT              = 0.5        # control / EnKF step [h]
T_END           = 40.0       # total simulation time [h]
N_STEPS         = int(T_END / DT)

X0_TRUE         = np.array([0.10, 1.0, 1.0])   # true initial state [X, S, V]  — S below setpoint
X0_INIT_BELIEF  = np.array([0.15, 1.5, 1.0])  # EnKF initial belief

S_SETPOINT      = 4.0        # target substrate concentration [g/L]

# Monod parameters (shared between plant, EnKF model and MPC model)
# mu_max=0.1 h⁻¹: slower growth keeps substrate consumption feasible for u_max=0.2.
# F_out=0.0: fed-batch mode — volume only grows, preventing V→0 singularity
#   that arises when F_in < F_out and V collapses to zero.
MONOD_PARAMS = dict(mu_max=0.1, K_s=0.1, Y_xs=0.5, S_in=10.0, F_out=0.0)

# OD measurement noise std
OD_NOISE_STD = 0.05   # [g/L]

# EnKF measurement interval: how many control steps between OD measurements.
# EnKF_INTERVAL=1  → every step (original behaviour, mpc.update() every time)
# EnKF_INTERVAL=4  → OD every 4 steps, mpc.step() in between
# Mimics PioReactor where OD is read every few minutes but F_in is set every second.
ENKF_INTERVAL   = 4

# MPC horizon (longer horizon suits the slower mu_max=0.1 dynamics)
MPC_HORIZON     = 20.0               # [h]
MPC_U_MIN       = [0.0, 0.0]         # [F_in_min, F_out_min]  [L/h]
MPC_U_MAX       = [0.2, 0.2]         # [F_in_max, F_out_max]  [L/h]

# PID for comparison — gains re-tuned for mu_max=0.1 (slower dynamics)
PID_KP, PID_KI, PID_KD = 0.5, 0.01, 0.05

# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)

# Single model shared by plant simulation, EnKF and MPC.
# controllable_Fin=True + controllable_Fout=True exposes both F_in (u[0]) and
# F_out (u[1]) as CasADi symbolic inputs so the MPC can optimise both and the
# EnKF propagation receives the actually applied control each step.
model = create_monod_cstr(X0=X0_INIT_BELIEF.copy(), dt=DT,
                          controllable_Fin=True, controllable_Fout=True, **MONOD_PARAMS)

# ---------------------------------------------------------------------------
# Build EnKF
# ---------------------------------------------------------------------------
obs_func = lambda x: np.array([x[0]])   # OD ≈ X

measurement_provider = MeasurementProvider("od_sim")
measurement_provider.add_variable(
    "od", noise=np.array([[OD_NOISE_STD ** 2]]), state_index=1
)

enkf = EnKalmanFilter(
    model=model,
    ensemble_size=20,
    initial_covariance=np.diag([0.01, 0.01, 0.01]),
    observation_func=obs_func,
    providers=[measurement_provider],
    random_seed=RNG_SEED,
)

# ---------------------------------------------------------------------------
# Build MPC controller
# ---------------------------------------------------------------------------
objective = TrackingObjective(Q=20.0, R=0.5, setpoint=S_SETPOINT, state_index=1)
ocp = CasadiOptimalControlProblem(
    model=model,
    objective=objective,
    time_horizon=MPC_HORIZON,
    u_min=MPC_U_MIN,
    u_max=MPC_U_MAX,
    ipopt_print_level=0,
    path_constraints=[(2, 0.5, None)],   # V (state[2]) >= 0.5 L throughout horizon
)

# ControlProvider for the MPC prediction sequence
mpc_plan_provider = ControlProvider("mpc_plan")
mpc = MPCController(ocp, provider=mpc_plan_provider, variable_name="dosing_rate")

# ---------------------------------------------------------------------------
# Build PID controller (for comparison)
# ---------------------------------------------------------------------------
pid = PID(kp=PID_KP, ki=PID_KI, kd=PID_KD,
          setpoint=S_SETPOINT, output_limits=(MPC_U_MIN[0], MPC_U_MAX[0]))

# ---------------------------------------------------------------------------
# Storage arrays
# ---------------------------------------------------------------------------
times           = np.zeros(N_STEPS + 1)
x_true          = np.zeros((N_STEPS + 1, 3))
x_est_mpc       = np.zeros((N_STEPS + 1, 3))
x_sim_pid       = np.zeros((N_STEPS + 1, 3))   # PID: pure model simulation
u_mpc_hist      = np.zeros((N_STEPS, 2))       # columns: [F_in, F_out]
u_pid_hist      = np.zeros(N_STEPS)
od_noisy        = np.zeros(N_STEPS)

x_true[0]       = X0_TRUE
x_est_mpc[0]    = X0_INIT_BELIEF
x_sim_pid[0]    = X0_TRUE                       # PID starts from same true state

pid_plant = create_monod_cstr(X0=X0_TRUE.copy(), dt=DT,
                              controllable_Fin=True, controllable_Fout=True, **MONOD_PARAMS)

# Last applied MPC control vector [F_in, F_out] — fed into the EnKF propagation
# so the ensemble prediction uses the same u that was sent to the plant.
u_mpc_last = np.zeros(2)

# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------
print(f"Running offline MPC simulation (T={T_END}h, dt={DT}h, S_ref={S_SETPOINT} g/L, EnKF every {ENKF_INTERVAL} steps) ...")

for k in range(N_STEPS):
    t = k * DT

    # --- Generate artificial OD measurement from true S + noise ---
    od_meas = float(x_true[k, 1]) + rng.normal(0.0, OD_NOISE_STD)
    od_meas = max(0.0, od_meas)   # physical constraint: non-negative
    od_noisy[k] = od_meas

    # Store in provider so EnKF can retrieve it
    measurement_provider.add_measurement("od", t, od_meas)

    # --- EnKF + MPC: only every ENKF_INTERVAL steps ---
    t_next = t + DT
    if k % ENKF_INTERVAL == 0:
        # New measurement available → run EnKF → re-solve OCP
        state_est = enkf.update_state_with_interpolation(t_end=t_next,
                                                          interpolation="nearest",
                                                          u=u_mpc_last)
        state_est = np.maximum(state_est, [0.0, 0.0, 0.0])   # keep V >= 0
        x_est_mpc[k + 1] = state_est
        _, _ = mpc.solve(state_est, DT, t)
    else:
        # No new measurement → apply next step from existing sequence
        x_est_mpc[k + 1] = x_est_mpc[k]   # hold last estimate for logging
        _, _ = mpc.step(t)

    u_vec = mpc.current_control           # [F_in, F_out]
    u_mpc_last = u_vec
    u_mpc_hist[k] = u_vec

    # --- Advance true plant with MPC control + process noise ---
    res = model.integrator(x0=x_true[k], p=u_vec)
    x_next_true = np.array(res['xf']).flatten()
    # Add small process noise to the true plant
    x_next_true += rng.normal(0.0, [1e-3, 1e-2, 1e-4])
    x_next_true = np.maximum(x_next_true, 0.0)
    x_true[k + 1] = x_next_true

    # --- PID control (uses noisy OD directly as S estimate) ---
    _, u_pid = pid.update(od_meas, DT, t)
    u_pid_hist[k] = u_pid
    res_pid = pid_plant.integrator(x0=x_sim_pid[k], p=[u_pid, 0.0])  # F_in=u_pid, F_out=0
    x_pid_next = np.array(res_pid['xf']).flatten()
    x_pid_next += rng.normal(0.0, [1e-3, 1e-2, 1e-4])
    x_pid_next = np.maximum(x_pid_next, 0.0)
    x_sim_pid[k + 1] = x_pid_next

    times[k + 1] = t_next

print("Done.")

# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f"Offline MPC vs PID — Monod CSTR  (S_ref = {S_SETPOINT} g/L)", fontsize=13)

t_plot = times

# --- Substrate S ---
ax = axes[0, 0]
ax.plot(t_plot, x_true[:, 1],      'k-',  lw=2,   label='True S (MPC plant)')
ax.plot(t_plot, x_est_mpc[:, 1],   'b--', lw=1.5, label='EnKF estimate S')
ax.plot(t_plot, x_sim_pid[:, 1],   'r:',  lw=1.5, label='True S (PID plant)')
ax.scatter(times[:-1], od_noisy, s=8, c='grey', alpha=0.5, zorder=3, label='OD measurement')
ax.axhline(S_SETPOINT, color='green', ls=':', lw=1.5, label=f'Setpoint {S_SETPOINT} g/L')
ax.set_ylabel('S [g/L]')
ax.set_title('Substrate concentration')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Biomass X ---
ax = axes[0, 1]
ax.plot(t_plot, x_true[:, 0],     'k-',  lw=2,   label='True X (MPC)')
ax.plot(t_plot, x_est_mpc[:, 0],  'b--', lw=1.5, label='EnKF estimate X')
ax.plot(t_plot, x_sim_pid[:, 0],  'r:',  lw=1.5, label='True X (PID)')
ax.set_ylabel('X [g/L]')
ax.set_title('Biomass concentration')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Volume V ---
ax = axes[1, 0]
ax.plot(t_plot, x_true[:, 2],     'k-',  lw=2,   label='True V (MPC)')
ax.plot(t_plot, x_sim_pid[:, 2],  'r:',  lw=1.5, label='True V (PID)')
ax.set_ylabel('V [L]')
ax.set_title('Volume')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Control inputs ---
ax = axes[1, 1]
ax.step(times[:-1], u_mpc_hist[:, 0], where='post', color='blue',       lw=2,   label='MPC: F_in')
ax.step(times[:-1], u_mpc_hist[:, 1], where='post', color='dodgerblue', lw=1.5, ls='--', label='MPC: F_out')
ax.step(times[:-1], u_pid_hist,        where='post', color='red',        lw=1.5, ls='--', label='PID: F_in')
ax.axhline(MPC_U_MAX[0], color='k', ls=':', lw=1, alpha=0.5)
ax.axhline(MPC_U_MIN[0], color='k', ls=':', lw=1, alpha=0.5)
ax.set_ylabel('Flow rate [L/h]')
ax.set_title('Control inputs (dosing / outlet rate)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Tracking error |S - S_ref| ---
ax = axes[2, 0]
err_mpc = np.abs(x_true[:, 1] - S_SETPOINT)
err_pid = np.abs(x_sim_pid[:, 1] - S_SETPOINT)
ax.plot(t_plot, err_mpc, 'b-',  lw=2,   label=f'MPC  RMSE={np.sqrt(np.mean(err_mpc**2)):.3f}')
ax.plot(t_plot, err_pid, 'r--', lw=1.5, label=f'PID  RMSE={np.sqrt(np.mean(err_pid**2)):.3f}')
ax.set_ylabel('|S - S_ref| [g/L]')
ax.set_title('Substrate tracking error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- MPC prediction sequence at last step ---
ax = axes[2, 1]
t_pred = mpc.time_sequence
ax.step(t_pred, mpc.control_sequence[:, 0], where='post', color='blue',       lw=2,   label='MPC plan: F_in')
ax.step(t_pred, mpc.control_sequence[:, 1], where='post', color='dodgerblue', lw=1.5, ls='--', label='MPC plan: F_out')
ax.axhline(MPC_U_MAX[0], color='k', ls=':', lw=1, alpha=0.5)
ax.axhline(MPC_U_MIN[0], color='k', ls=':', lw=1, alpha=0.5)
ax.set_ylabel('Flow rate [L/h]')
ax.set_title(f'MPC prediction sequence at t={times[-2]:.1f}h')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel('Time [h]')

plt.tight_layout()
plt.savefig("PID_vs_MPC_yeast_grow.png")
plt.show()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Results ===")
print(f"  S setpoint:              {S_SETPOINT:.2f} g/L")
print(f"  MPC final true S:        {x_true[-1, 1]:.3f} g/L")
print(f"  PID final true S:        {x_sim_pid[-1, 1]:.3f} g/L")
print(f"  MPC tracking RMSE:       {np.sqrt(np.mean((x_true[:, 1] - S_SETPOINT)**2)):.4f}")
print(f"  PID tracking RMSE:       {np.sqrt(np.mean((x_sim_pid[:, 1] - S_SETPOINT)**2)):.4f}")
print(f"  MPC total control effort:{np.sum(u_mpc_hist**2):.4f}")
print(f"  PID total control effort:{np.sum(u_pid_hist**2):.4f}")
