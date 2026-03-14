import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'control'))

from Model import CasadiModel
from Provider import DFProvider
from EnKalmanFilter import EnKalmanFilter

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'export_20250921082017')

df_od_readings = pd.read_csv(f"{DATA_PATH}/od_readings/od_readings-Yeast_Grow_4_Days-all_units-20250921102018.csv")
first_10_hours = df_od_readings[df_od_readings['hours_since_experiment_created'] <= 10]

time = first_10_hours['hours_since_experiment_created'].values
od_readings = first_10_hours['od_reading'].values

def create_ode_function(x, u):
    """Create ODE system for yeast fermentation model."""

    X, S, E = x[0], x[1], x[2]

    # Fitted parameters
    mu_S_max = 0.445
    mu_E_max = 1.130
    K_S = 0.077
    K_E = 1.556
    K_I = 0.446
    Y_XS = 0.108
    Y_ES = 1.359
    Y_XE = 2.000

    eps = 1e-8

    # Prevent negative values
    S_pos = ca.fmax(S, eps)
    E_pos = ca.fmax(E, eps)
    X_pos = ca.fmax(X, eps)

    # Kinetics
    mu_S = mu_S_max * S_pos / (K_S + S_pos + eps)
    glc_rep = K_I / (K_I + S_pos + eps)
    mu_E = mu_E_max * E_pos / (K_E + E_pos + eps) * glc_rep

    # ODE System
    dXdt = (mu_S + mu_E) * X_pos
    dSdt = -(1.0 / Y_XS) * mu_S * X_pos
    dEdt = Y_ES * mu_S * X_pos - (1.0 / Y_XE) * mu_E * X_pos

    return ca.vertcat(dXdt, dSdt, dEdt)


X_sym = ca.SX.sym('X')
S_sym = ca.SX.sym('S')
E_sym = ca.SX.sym('E')

x_sym = ca.vertcat(X_sym, S_sym, E_sym)
u_sym = ca.SX.sym('u', 0)

dt = 0.1
T = 10.0
x0 = np.array([0.05, 20.00, 0.0])

casadi_model = CasadiModel(
    states=x_sym,
    controls=u_sym,
    ode=create_ode_function,
    process_noise=np.diag([0.001, 0.01, 0.001]),
    init_state=x0,
    dt=dt,
    T=T
)

time_od_pairs = [(time[i], od_readings[i]) for i in range(len(time))]

df_measurements = pd.DataFrame({'OD': [time_od_pairs]})

od_provider = DFProvider(
    name="OD_Biomass",
    dataframe=df_measurements,
    y_columns=["OD"],
    noise=np.array([[0.02**2]])
)


def observation_func(x_state):
    return np.array([x_state[0]])


enkf = EnKalmanFilter(
    model=casadi_model,
    ensemble_size=5,
    initial_covariance=np.diag([0.05, 1.0, 0.05]),
    observation_func=observation_func,
    providers=[od_provider],
    random_seed=42
)

results = enkf.run_filter(
    t_start=0.0,
    t_end=T,
    use_measurement_times=True,
    interpolation='nearest'
)

enkf_times = results['times']
enkf_states = results['states']

print(f"Filter completed: {len(enkf_times)} time steps")
print(f"Final state: X={enkf_states[-1, 0]:.4f}, S={enkf_states[-1, 1]:.4f}, E={enkf_states[-1, 2]:.4f}")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(time, od_readings, color='red', s=0.5, alpha=0.6, label='OD Measurements')
axes[0].plot(enkf_times, enkf_states[:, 0], 'b-', linewidth=2, label='EnKF Estimate')
axes[0].set_xlabel('Time [h]')
axes[0].set_ylabel('Biomass X [g/L]')
axes[0].set_title('Biomass')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(enkf_times, enkf_states[:, 1], 'g-', linewidth=2)
axes[1].set_xlabel('Time [h]')
axes[1].set_ylabel('Glucose S [g/L]')
axes[1].set_title('Glucose (Unobserved)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(enkf_times, enkf_states[:, 2], color='orange', linewidth=2)
axes[2].set_xlabel('Time [h]')
axes[2].set_ylabel('Ethanol E [g/L]')
axes[2].set_title('Ethanol (Unobserved)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'enkf_yeast_fermentation.png'), dpi=150, bbox_inches='tight')
plt.show()
