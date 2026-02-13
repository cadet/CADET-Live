import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

DATA_PATH =  "/Users/berger/fzj/cadet/CADET-Live/Experiments/PioReactor/yeast_2Days/export_20250921082017"

df_od_readings = pd.read_csv(f"{DATA_PATH}/od_readings/od_readings-Yeast_Grow_4_Days-all_units-20250921102018.csv")
first_10_hours = df_od_readings[df_od_readings['hours_since_experiment_created'] <= 10]

X = ca.MX.sym('X')   # Biomass [g/L]
S = ca.MX.sym('S')   # Glucose [g/L]
E = ca.MX.sym('E')   # Ethanol [g/L]

eps = 1e-8

S_pos = ca.fmax(S, eps)
E_pos = ca.fmax(E, eps)
X_pos = ca.fmax(X, eps)

x = ca.vertcat(X, S, E)

mu_S_max = ca.MX.sym('mu_S_max')
mu_E_max = ca.MX.sym('mu_E_max')
K_S      = ca.MX.sym('K_S')
K_E      = ca.MX.sym('K_E')
K_I      = ca.MX.sym('K_I')
Y_XS     = ca.MX.sym('Y_XS')
Y_ES     = ca.MX.sym('Y_ES')
Y_XE     = ca.MX.sym('Y_XE')

p = ca.vertcat(mu_S_max, mu_E_max, K_S, K_E, K_I, Y_XS, Y_ES, Y_XE)

eps = 1e-8

mu_S = mu_S_max * S_pos / (K_S + S_pos + eps)
glc_rep = K_I / (K_I + S_pos + eps)
mu_E = mu_E_max * E_pos / (K_E + E_pos + eps) * glc_rep

dXdt = (mu_S + mu_E) * X_pos
dSdt = -(1.0 / Y_XS) * mu_S * X_pos
dEdt = Y_ES * mu_S * X_pos - (1.0 / Y_XE) * mu_E * X_pos

xdot = ca.vertcat(dXdt, dSdt, dEdt)

t_end = 10
n_steps = 500
dt = t_end / n_steps

dae = {'x': x, 'p': p, 'ode': xdot}

integrator = ca.integrator(
    'integrator',
    'cvodes',
    dae,
    {'tf': dt}
)

params = [0.445, 1.130, 0.077, 1.556, 0.446, 0.108, 1.359, 2.000]
x0 = np.array([0.227, 13.665, 1.945])

t_span = np.linspace(0, t_end, n_steps)

X_sim = np.zeros(n_steps)
S_sim = np.zeros(n_steps)
E_sim = np.zeros(n_steps)

X_sim[0], S_sim[0], E_sim[0] = x0
x_current = x0

for i in range(1, n_steps):
    res = integrator(x0=x_current, p=params)
    x_current = res['xf'].full().flatten()

    x_current = np.maximum(x_current, 0)

    X_sim[i], S_sim[i], E_sim[i] = x_current

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(t_span, X_sim / 8.2, linewidth=2, label='Biomasse (X)')
axes[0].scatter(first_10_hours['hours_since_experiment_created'], first_10_hours['od_reading'], label='OD Reading', color='red', s=0.5)
axes[0].set_ylabel('Biomasse [g/L]')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_span, S_sim, linewidth=2, label='Glucose (S)')
axes[1].set_ylabel('Glucose [g/L]')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_span, E_sim, linewidth=2, label='Ethanol (E)')
axes[2].set_xlabel('Zeit [h]')
axes[2].set_ylabel('Ethanol [g/L]')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
