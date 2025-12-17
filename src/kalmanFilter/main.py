import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# BIOREACTOR
S_in = 0.5
F_out = 1.0

X = ca.SX.sym('X')
S = ca.SX.sym('S')
V = ca.SX.sym('V')
states = ca.vertcat(X, S, V)
n_states = 3

F_in = ca.SX.sym('F_in')
controls = F_in
n_controls = 1

dV_dt = F_in - F_out
dx_dt = (-F_out * X - dV_dt * X) / V
ds_dt = (F_in * S_in - F_out * S - dV_dt * S) / V
f = ca.vertcat(dx_dt, ds_dt, dV_dt)

# MPC Parameter und Setup
dt = 0.05
N_horizont = 20
Q_mpc = np.diag([100.0, 1.0, 1.0])
R_mpc = np.diag([0.01])

F_in_min = 0.0
F_in_max = 5.0
V_min = 1.0
V_max = 5.0

integrator_mpc = ca.integrator(
    "integrator_mpc", "idas",
    {"x": states, "p": controls, "ode": f},
    0.0, dt
)

# Optimierungsvariable
U = ca.SX.sym('U', n_controls, N_horizont)
X0_param = ca.SX.sym('X0_param', n_states)
x_ref_param = ca.SX.sym('x_ref_param', n_states)

cost = 0
x_k = X0_param
g = []
lbg = []
ubg = []

for k in range(N_horizont):
    # Zustandskosten + Stellgrößenkosten
    cost += (x_k - x_ref_param).T @ Q_mpc @ (x_k - x_ref_param)
    cost += U[:, k].T @ R_mpc @ U[:, k]
    
    # Propagation
    res = integrator_mpc(x0=x_k, p=U[:, k])
    x_k = res['xf']
    
    # Volumen-Constraint
    g.append(x_k[2])
    lbg.append(V_min)
    ubg.append(V_max)

# Terminalkosten
cost += 10.0 * (x_k - x_ref_param).T @ Q_mpc @ (x_k - x_ref_param)

nlp = {
    'x': ca.reshape(U, -1, 1),
    'f': cost,
    'g': ca.vertcat(*g) if g else ca.SX.zeros(0, 1),
    'p': ca.vertcat(X0_param, x_ref_param)
}

opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.max_iter': 200,
    'ipopt.tol': 1e-6
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

lbx = [F_in_min] * N_horizont
ubx = [F_in_max] * N_horizont

# EnKF SETUP
N_ens = 10
x0 = np.array([3.0, 2.0, 5.0])
P0 = np.diag([0.01, 0.5, 0.01])
X_ens = np.random.multivariate_normal(x0, P0, N_ens)

R_meas = np.array([[0.01]])
Q_noise = np.diag([0.001, 0.001, 0.001])

# KOMBINIERTE SIMULATION
T_sim = 50.0
time = np.arange(0, T_sim, dt)

x_true = x0.copy()
S_ref = 1.0
X_ref = 1.0

X_true_hist = [x_true.copy()]
X_est_hist = [np.mean(X_ens, axis=0)]
U_hist = []
S_ref_hist = []

# Warm-Start
U_last = np.ones(N_horizont) * 2.5

for t in time:
    # Sollwert ändern
    if t > 10:
        S_ref = 0.5
    S_ref_hist.append(S_ref)

    # 1. MPC mit Warm-Start
    x_est = np.mean(X_ens, axis=0)
    x_ref_val = np.array([X_ref, S_ref, 5.0])

    p_val = np.concatenate([x_est, x_ref_val])

    # Warm-Start: Verschiebe letzte Lösung
    x0_warm = np.concatenate([U_last[1:], [U_last[-1]]])

    try:
        sol = solver(
            x0=x0_warm,
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg,
            p=p_val
        )

        u_opt = np.array(sol['x']).flatten()
        U_last = u_opt  # Speichere für nächsten Schritt
        u_apply = u_opt[0]
    except:
        print(f"MPC failed at t={t}, using last control")
        u_apply = U_last[0]

    U_hist.append(u_apply)

    # 2. Wahres System propagieren
    res = integrator_mpc(x0=x_true, p=u_apply)
    x_true = np.array(res['xf']).flatten()
    X_true_hist.append(x_true.copy())

    # 3. EnKF Propagation
    for i in range(N_ens):
        res = integrator_mpc(x0=X_ens[i, :], p=u_apply)
        x_pred = np.array(res['xf']).flatten()
        w = np.random.multivariate_normal(np.zeros(3), Q_noise)
        X_ens[i, :] = x_pred + w

    # 4. Messung
    y_meas = x_true[0] + np.random.normal(0, np.sqrt(R_meas[0, 0]))
    Y_ens = X_ens[:, 0:1] + np.random.normal(0, np.sqrt(R_meas[0, 0]), (N_ens, 1))

    # 5. EnKF Update
    x_mean = np.mean(X_ens, axis=0, keepdims=True)
    y_mean = np.mean(Y_ens, axis=0, keepdims=True)

    X_dev = X_ens - x_mean
    Y_dev = Y_ens - y_mean

    P_xy = (X_dev.T @ Y_dev) / (N_ens - 1)
    P_yy = (Y_dev.T @ Y_dev) / (N_ens - 1) + R_meas

    K = P_xy @ np.linalg.inv(P_yy)

    for i in range(N_ens):
        X_ens[i, :] = X_ens[i, :] + (K @ (y_meas - Y_ens[i])).flatten()

    X_est_hist.append(np.mean(X_ens, axis=0))

# Arrays konvertieren
X_true_hist = np.array(X_true_hist)
X_est_hist = np.array(X_est_hist)
U_hist = np.array(U_hist)

# PLOT
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(time, X_true_hist[:-1, 0], 'b-', linewidth=2, label='X true')
axes[0].plot(time, X_est_hist[:-1, 0], 'r--', linewidth=2, label='X estimated')
axes[0].set_ylabel('X [g/L]')
axes[0].legend()
axes[0].grid()
axes[0].set_title('MPC + Ensemble Kalman Filter')

axes[1].plot(time, X_true_hist[:-1, 1], 'b-', linewidth=2, label='S true')
axes[1].plot(time, X_est_hist[:-1, 1], 'r--', linewidth=2, label='S estimated')
axes[1].plot(time, S_ref_hist, 'g:', linewidth=2, label='S_ref')
axes[1].set_ylabel('S [g/L]')
axes[1].legend()
axes[1].grid()

axes[2].plot(time, X_true_hist[:-1, 2], 'b-', linewidth=2, label='V true')
axes[2].plot(time, X_est_hist[:-1, 2], 'r--', linewidth=2, label='V estimated')
axes[2].axhline(V_min, color='k', linestyle='--', alpha=0.5, label='V limits')
axes[2].axhline(V_max, color='k', linestyle='--', alpha=0.5)
axes[2].set_ylabel('V [m³]')
axes[2].legend()
axes[2].grid()

axes[3].step(time, U_hist, 'g-', where='post', linewidth=2, label='F_in (MPC)')
axes[3].axhline(F_in_min, color='k', linestyle='--', alpha=0.5, label='F_in limits')
axes[3].axhline(F_in_max, color='k', linestyle='--', alpha=0.5)
axes[3].set_ylabel('F_in [m³/s]')
axes[3].set_xlabel('Zeit [s]')
axes[3].legend()
axes[3].grid()

plt.tight_layout()
plt.show()

print("\nHappy!")