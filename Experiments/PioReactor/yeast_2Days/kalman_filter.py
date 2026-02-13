import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import sys
import os

sys.path.insert(0, os.path.join('src', 'control'))

from Model import CasadiModel
from Provider import DFProvider
from EnKalmanFilter import EnKalmanFilter

DATA_PATH = "/Users/berger/fzj/cadet/CADET-Live/Experiments/PioReactor/yeast_2Days/export_20250921082017"

df_od_readings = pd.read_csv(f"{DATA_PATH}/od_readings/od_readings-Yeast_Grow_4_Days-all_units-20250921102018.csv")
first_10_hours = df_od_readings[df_od_readings['hours_since_experiment_created'] <= 10]

time = first_10_hours['hours_since_experiment_created'].values
od_readings = first_10_hours['od_reading'].values

# ---------- CasADi ODE Function ----------
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

# ---------- Define symbolic variables ----------
X = ca.MX.sym('X')   # Biomass [g/L]
S = ca.MX.sym('S')   # Glucose [g/L]
E = ca.MX.sym('E')   # Ethanol [g/L]

x = ca.vertcat(X, S, E)
u = ca.MX.sym('u', 0)  # No external control

# ---------- Create CasADi Model ----------
dt = 0.1  # Time step in hours
T = 10.0  # Total simulation time in hours
x0 = np.array([0.05, 20.00, 0.0])  # Initial state [X, S, E]

casadi_model = CasadiModel(
    states=x,
    controls=u,
    ode=create_ode_function,
    process_noise=np.diag([0.001, 0.01, 0.001]),
    init_state=x0,
    dt=dt,
    T=T
)


time_od_pairs = [(time[i], od_readings[i]) for i in range(len(time))]

df_measurements = pd.DataFrame({
    'OD': [time_od_pairs] 
})

noise_std_OD = 0.02  # OD measurement noise

od_provider = DFProvider(
    name="OD_Biomass",
    DataFrame=df_measurements,
    y_columns=["OD"],  # Specify which columns are measurements
    noise=np.array([[noise_std_OD**2]])
)


# ---------- Define observation function ----------
def observation_func(x_state):
    """Observation function: only measure X (biomass via OD)."""
    return np.array([x_state[0]])

enkf = EnKalmanFilter(
    model=casadi_model,
    ensemble_size=10,
    initial_covariance=np.diag([0.05, 1.0, 0.05]),  # Initial uncertainty
    observation_func=observation_func,
    providers=[od_provider],
    random_seed=42,
    enable_logging=True,
    log_file="enkf_yeast_log.csv"
)

try:
    results = enkf.run_filter(
        t_start=0.0,
        t_end=T,
        use_measurement_times=True,
        interpolation='nearest'
    )
    
    enkf_times = results['times']
    enkf_states = results['states']
    
    print(f"✓ Filter completed: {len(enkf_times)} time steps")
    print(f"  Final state estimate: X={enkf_states[-1, 0]:.4f}, S={enkf_states[-1, 1]:.4f}, E={enkf_states[-1, 2]:.4f}")
    
except Exception as e:
    print(f"Error running filter: {e}")
    import traceback
    traceback.print_exc()
    enkf_states = None
    enkf_times = None

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.scatter(time, od_readings, color='red', s=0.5, alpha=0.6, label='OD Measurements', zorder=3)
if enkf_states is not None:
    ax.plot(enkf_times, enkf_states[:, 0], 'b-', linewidth=2.5, label='EnKF Estimate', zorder=2)
    ax.set_xlabel('Time [h]', fontsize=11)
    ax.set_ylabel('Biomass X [g/L]', fontsize=11)
    ax.set_title('Biomass Estimation with EnKF', fontsize=12, fontweight='bold')
else:
    ax.set_xlabel('Time [h]', fontsize=11)
    ax.set_ylabel('OD Reading', fontsize=11)
    ax.set_title('OD Measurements (Filter failed)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
if enkf_states is not None:
    ax.plot(enkf_times, enkf_states[:, 1], 'g-', linewidth=2.5, label='EnKF Estimate')
    ax.set_xlabel('Time [h]', fontsize=11)
    ax.set_ylabel('Substrate S [g/L]', fontsize=11)
    ax.set_title('Glucose (Unobserved) - EnKF Prediction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Glucose Prediction', fontsize=12, fontweight='bold')

ax = axes[1, 0]
if enkf_states is not None:
    ax.plot(enkf_times, enkf_states[:, 2], 'orange', linewidth=2.5, label='EnKF Estimate')
    ax.set_xlabel('Time [h]', fontsize=11)
    ax.set_ylabel('Ethanol E [g/L]', fontsize=11)
    ax.set_title('Ethanol (Unobserved) - EnKF Prediction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Ethanol Prediction', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('enkf_yeast_fermentation_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: enkf_yeast_fermentation_results.png")
plt.show()
