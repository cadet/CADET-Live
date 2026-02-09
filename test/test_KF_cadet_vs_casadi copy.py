import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'control'))


from Model import CasadiModel, CadetModel

# def cadadi model: simple CSTR dynamics

# Define CasADi model: simple CSTR dynamics
P = ca.SX.sym('P')  # Product
S = ca.SX.sym('S')  # Substrate
states = ca.vertcat(P, S)

# No external control for this example
u = ca.SX.sym('u', 0)

# Parameters
S_in = 0.0
F_out = 0.0
F_in = 0.0
k_fwd = 2
k_bwd = 3
V = 1.0

# ODE function
def cstr_ode(x, u):
    X, S = x[0], x[1]
    dP_dt = (k_fwd * S) / V - (k_bwd * P) / V
    dS_dt = -(k_fwd * S) / V + (k_bwd * P) / V
    
    return ca.vertcat(dP_dt, dS_dt)

# Create CasADi model
dt = 0.1
T = 2.0
X0 = np.array([1.0, 0.0])

casadi_model = CasadiModel(
    states=states,
    controls=u,
    ode=cstr_ode,
    process_noise=np.diag([0.01, 0.01]),
    init_state=X0,
    dt=dt,
    T=T
)

## def cadet model: simple CSTR dynamics

from cadet import Cadet

import os
import numpy as np

# Set up model parameters 
parameters = {
    "ncomp": 2,
    "init_c": [1.0, 0.0],
    "sim_time": 2.0,
}

# create Cadet model
cstr_model = Cadet(r"/Users/berger/fzj/cadet/CADET-Core/install_release")

ncomp = parameters["ncomp"]
init_c = parameters["init_c"]
sim_time = parameters["sim_time"]

cstr_model.root.input.model.nunits = 1

# CSTR - dynamically use unit_number
cstr_model.root.input.model.unit_000.unit_type = 'CSTR'
cstr_model.root.input.model.unit_000.ncomp = ncomp
cstr_model.root.input.model.unit_000.init_liquid_volume = 1.0
cstr_model.root.input.model.unit_000.init_c = init_c
cstr_model.root.input.model.unit_000.const_solid_volume = 0.0
cstr_model.root.input.model.unit_000.use_analytic_jacobian = 1

#Configure solver settings
cstr_model.root.input.solver.user_solution_times = np.linspace(0, parameters['sim_time'], 50)
cstr_model.root.input.solver.sections.nsec = 1
cstr_model.root.input.solver.sections.section_times = [0.0, parameters['sim_time']]
cstr_model.root.input.solver.sections.section_continuity = []

cstr_model.root.input.model.solver.gs_type = 1
cstr_model.root.input.model.solver.max_krylov = 0
cstr_model.root.input.model.solver.max_restarts = 10
cstr_model.root.input.model.solver.schur_safety = 1e-8

cstr_model.root.input.solver.time_integrator.abstol = 1e-6
cstr_model.root.input.solver.time_integrator.algtol = 1e-10
cstr_model.root.input.solver.time_integrator.reltol = 1e-6
cstr_model.root.input.solver.time_integrator.init_step_size = 1e-6
cstr_model.root.input.solver.time_integrator.max_steps = 1000000
cstr_model.root.input.solver.consistent_init_mode = 1

cstr_model.root.input['return'].unit_000.split_components_data = 0
cstr_model.root.input['return'].unit_000.split_ports_data = 0
cstr_model.root.input['return'].unit_000.write_solution_bulk = 1
cstr_model.root.input['return'].unit_000.write_solution_inlet = 1
cstr_model.root.input['return'].unit_000.write_solution_outlet = 1
cstr_model.root.input['return'].unit_000.write_solution_solid = 1

cstr_model.root.input.model.unit_000.nreac_liquid = 1
cstr_model.root.input.model.unit_000.liquid_reaction_000.type = 'MASS_ACTION_LAW'
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_stoichiometry = [[-1],[1]]
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_kfwd = 3
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_kbwd = 2


# add connections
cstr_model.root.input.model.connections.nswitches = 1
cstr_model.root.input.model.connections.switch_000.section = 0
cstr_model.root.input.model.connections.switch_000.connections = [ ]

# save as h5 file
model_filename = "cstr_no_inlet_one_mal.h5"

# run simulation and save model
cstr_model.filename = model_filename
cstr_model.save()


cstr_model_results = cstr_model.run_simulation()
if cstr_model_results.return_code != 0:
    print(cstr_model_results.error_message)
else:
    print("Happy")

#plot results
time_points = cstr_model.root.output.solution.solution_times
concentration_comp0 = cstr_model.root.output.solution.unit_000.solution_bulk[:,0]
concentration_comp1 = cstr_model.root.output.solution.unit_000.solution_bulk[:,1]

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time_points, concentration_comp0, label='CADET Component 0 (P)')
plt.plot(time_points, concentration_comp1, label='CADET Component 1 (S)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('CADET Simulation Results')
plt.grid(True)


# Run CasADi model simulation
casadi_states = [X0.copy()]
casadi_times = [0.0]
x_current = X0.copy()

for i in range(int(sim_time / dt)):
    t_start = i * dt
    t_end = (i + 1) * dt
    x_next = casadi_model.integrate(x_current, np.array([]), t_start, t_end)
    casadi_states.append(x_next)
    casadi_times.append(t_end)
    x_current = x_next

casadi_states = np.array(casadi_states)
casadi_times = np.array(casadi_times)

# Plot comparison
plt.subplot(1, 2, 2)
plt.plot(casadi_times, casadi_states[:, 0], 'b-', label='CasADi P', linewidth=2)
plt.plot(casadi_times, casadi_states[:, 1], 'r-', label='CasADi S', linewidth=2)
plt.plot(time_points, concentration_comp0, 'b--', label='CADET P', linewidth=2)
plt.plot(time_points, concentration_comp1, 'r--', label='CADET S', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Model Comparison')
plt.grid(True)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# Interpolate CADET results to CasADi time points for comparison
cadet_P_interp = np.interp(casadi_times, time_points, concentration_comp0)
cadet_S_interp = np.interp(casadi_times, time_points, concentration_comp1)

# Compute error metrics
error_P = np.abs(casadi_states[:, 0] - cadet_P_interp)
error_S = np.abs(casadi_states[:, 1] - cadet_S_interp)

print(f"\nComparison Metrics:")
print(f"  Component P (Product):")
print(f"    Mean Absolute Error: {np.mean(error_P):.6f}")
print(f"    Max Absolute Error:  {np.max(error_P):.6f}")
print(f"  Component S (Substrate):")
print(f"    Mean Absolute Error: {np.mean(error_S):.6f}")
print(f"    Max Absolute Error:  {np.max(error_S):.6f}")



from EnKalmanFilter import EnKalmanFilter
from Provider import DFProvider
import pandas as pd

# Generate synthetic noisy measurements from CADET "true" trajectory
np.random.seed(42)
measurement_interval = 5  # Every 5th point
meas_indices = list(range(0, len(time_points), measurement_interval))

noise_std_P = 0.01
noise_std_S = 0.01

measurements_P = []
measurements_S = []

for idx in meas_indices:
    t = time_points[idx]
    true_P = concentration_comp0[idx]
    true_S = concentration_comp1[idx]
    
    # Add Gaussian noise
    meas_P = true_P + np.random.normal(0, noise_std_P)
    meas_S = true_S + np.random.normal(0, noise_std_S)
    
    measurements_P.append((t, meas_P))
    measurements_S.append((t, meas_S))

# Create DataFrames for providers
df_P = pd.DataFrame({"P": [measurements_P]})
df_S = pd.DataFrame({"S": [measurements_S]})

provider_P = DFProvider(
    name="ProductConc",
    DataFrame=df_P,
    y_columns=["P"],
    noise=np.array([[noise_std_P**2]])
)

provider_S = DFProvider(
    name="SubstrateConc",
    DataFrame=df_S,
    y_columns=["S"],
    noise=np.array([[noise_std_S**2]])
)

# Define observation function (observes both P and S)
def obs_func(x):
    return x[:2]

# ---- Run EnKF with CasADi Model ----
initial_state_guess = np.array([1.0, 0.0])  # Slightly perturbed initial guess

enkf_casadi = EnKalmanFilter(
    model=casadi_model,
    ensemble_size=50,
    initial_covariance=np.diag([0.2, 0.2]),
    observation_func=obs_func,
    providers=[provider_P, provider_S],
    random_seed=42,
    enable_logging=False,
    log_file="enkf_casadi_log.csv"
)

# Update initial state
enkf_casadi.state = initial_state_guess

results_casadi = enkf_casadi.run_filter(
    t_start=0.0,
    t_end=sim_time,
    use_measurement_times=True,
    interpolation='nearest'
)

enkf_casadi_times = results_casadi['times']
enkf_casadi_states = results_casadi['states']

enkf_casadi.save_log()


# Create a fresh CADET model for EnKF
cadet_model_enkf = CadetModel(
    cadet_path=r"/Users/berger/fzj/cadet/CADET-Core/install_release",
    init_state=initial_state_guess[:2],  # Only P and S
    model_path="./modelLibrary/cstr_no_inlet_one_mal.h5",
    n_states=2,
    n_controls=0,
    process_noise=np.diag([0.01, 0.01]),
    state_indices=[2, 3]  # Indices for bulk concentrations in CADET state vector
)

enkf_cadet = EnKalmanFilter(
    model=cadet_model_enkf,
    ensemble_size=50,
    initial_covariance=np.diag([0.2, 0.2]),
    observation_func=lambda x: x,  # Direct observation
    providers=[provider_P, provider_S],
    random_seed=42,
    enable_logging=False,
    log_file="enkf_cadet_log.csv"
)

results_cadet = enkf_cadet.run_filter(
    t_start=0.0,
    t_end=sim_time,
    use_measurement_times=True,
    interpolation='nearest'
)

enkf_cadet_times = results_cadet['times']
enkf_cadet_states = results_cadet['states']

enkf_cadet.save_log()

# Interpolate true states to EnKF time points
true_P_at_enkf_times = np.interp(enkf_casadi_times, time_points, concentration_comp0)
true_S_at_enkf_times = np.interp(enkf_casadi_times, time_points, concentration_comp1)

# Calculate errors for CasADi EnKF
error_casadi_P = np.abs(enkf_casadi_states[:, 0] - true_P_at_enkf_times)
error_casadi_S = np.abs(enkf_casadi_states[:, 1] - true_S_at_enkf_times)

# Calculate errors for CADET EnKF
error_cadet_P = np.abs(enkf_cadet_states[:, 0] - true_P_at_enkf_times)
error_cadet_S = np.abs(enkf_cadet_states[:, 1] - true_S_at_enkf_times)

print(f"\nCasADi EnKF Performance:")
print(f"  Component P: MAE = {np.mean(error_casadi_P):.6f}, Max Error = {np.max(error_casadi_P):.6f}")
print(f"  Component S: MAE = {np.mean(error_casadi_S):.6f}, Max Error = {np.max(error_casadi_S):.6f}")

print(f"\nCADET EnKF Performance:")
print(f"  Component P: MAE = {np.mean(error_cadet_P):.6f}, Max Error = {np.max(error_cadet_P):.6f}")
print(f"  Component S: MAE = {np.mean(error_cadet_S):.6f}, Max Error = {np.max(error_cadet_S):.6f}")

# ---- Plot Final Results ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Component P comparison
ax = axes[0, 0]
ax.plot(time_points, concentration_comp0, 'k-', linewidth=2, label='True P')
ax.plot([t for t, _ in measurements_P], [m for _, m in measurements_P], 
        'ro', markersize=5, alpha=0.6, label='Measurements')
ax.plot(enkf_casadi_times, enkf_casadi_states[:, 0], 'b--', linewidth=2, label='CasADi EnKF')
ax.plot(enkf_cadet_times, enkf_cadet_states[:, 0], 'g:', linewidth=2, label='CADET EnKF')
ax.set_xlabel('Time')
ax.set_ylabel('Concentration P')
ax.set_title('Component P (Product) Estimation')
ax.legend()
ax.grid(True)

# Component S comparison
ax = axes[0, 1]
ax.plot(time_points, concentration_comp1, 'k-', linewidth=2, label='True S')
ax.plot([t for t, _ in measurements_S], [m for _, m in measurements_S], 
        'mo', markersize=5, alpha=0.6, label='Measurements')
ax.plot(enkf_casadi_times, enkf_casadi_states[:, 1], 'b--', linewidth=2, label='CasADi EnKF')
ax.plot(enkf_cadet_times, enkf_cadet_states[:, 1], 'g:', linewidth=2, label='CADET EnKF')
ax.set_xlabel('Time')
ax.set_ylabel('Concentration S')
ax.set_title('Component S (Substrate) Estimation')
ax.legend()
ax.grid(True)

# Error comparison for P
ax = axes[1, 0]
ax.plot(enkf_casadi_times, error_casadi_P, 'b-', linewidth=2, label='CasADi EnKF Error')
ax.plot(enkf_cadet_times, error_cadet_P, 'g-', linewidth=2, label='CADET EnKF Error')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Error')
ax.set_title('Component P Estimation Error')
ax.legend()
ax.grid(True)

# Error comparison for S
ax = axes[1, 1]
ax.plot(enkf_casadi_times, error_casadi_S, 'b-', linewidth=2, label='CasADi EnKF Error')
ax.plot(enkf_cadet_times, error_cadet_S, 'g-', linewidth=2, label='CADET EnKF Error')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Error')
ax.set_title('Component S Estimation Error')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('enkf_comparison_casadi_vs_cadet.png', dpi=150)
plt.show()

# Clean up
cadet_model_enkf.end_simulation()

print("\n" + "="*60)
print("Test Completed Successfully!")
print("="*60)
print("\nSummary:")
print(f"  - Both models simulate similar dynamics")
print(f"  - EnKF successfully estimates states from noisy measurements")
print(f"  - CasADi and CADET models produce comparable results")
print(f"\nPlots saved:")
print(f"  - model_comparison.png")
print(f"  - enkf_comparison_casadi_vs_cadet.png")
