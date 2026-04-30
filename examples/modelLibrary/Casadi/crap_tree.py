import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

X0 = 0.1    # Biomasse         [g/L]
S0 = 10.0   # Glucose          [g/L]
E0 = 0.1    # Ethanol          [g/L]
V0 = 1.0    # Volumen          [L]

F_in  = 0.01  # Zulaufrate    [L/h]
F_out = 0.01  # Ablaufrate    [L/h]
S_in  = 200.0 # Glucose im Zulauf [g/L]

mu_s_max = 0.7   # [1/h]
mu_e_max = 0.3   # [1/h]

ks = 10   #[g/L]
ke = 1   #[g/L]

ki = 0.5

k1 = 0.05

y_xs = 0.3   # Ausbeute Biomasse/Glucose   [g/g]
y_xe = 0.3   # Ausbeute Biomasse/Ethanol   [g/g]

X = ca.SX.sym("X")  # Biomasse
S = ca.SX.sym("S")  # Glucose
E = ca.SX.sym("E")  # Ethanol
V = ca.SX.sym("V")  # Volumen

states = ca.vertcat(X, S, E, V)

mu_s = mu_s_max * (S / (ks + S))  * ki / (ki + E)       # Wachstum auf Glucose
mu_e = mu_e_max * (E / (ke + E))                        # Wachstum auf Ethanol

dVdt = F_in - F_out
dXdt = (mu_s + mu_e) * X - F_out * X / V - dVdt * X / V 
dSdt = -(mu_s / y_xs) * X - k1 * S - F_out * S / V - dVdt * S / V + F_in * S_in / V  
dEdt =  k1 * S - (mu_e / y_xe) * X - F_out * E / V - dVdt * E / V

rhs = ca.vertcat(dXdt, dSdt, dEdt, dVdt)

t_sym = ca.SX.sym("t")
dt    = 0.05   #[h]
T_end = 20.0   #[h]

ode = {"x": states, "t": t_sym, "ode": rhs}
integrator = ca.integrator("integrator", "cvodes", ode, 0, dt)

t_span  = np.arange(0, T_end + dt, dt)
n_steps = len(t_span) - 1

traj = np.zeros((n_steps + 1, 4))
traj[0] = [X0, S0, E0, V0]

x_k = traj[0]
for k in range(n_steps):
    result = integrator(x0=x_k)
    x_k    = np.array(result["xf"]).flatten()
    traj[k + 1] = x_k

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].plot(t_span, traj[:, 0], label="Biomasse X")
axes[0].set_ylabel("X [g/L]")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_span, traj[:, 1], label="Glucose S", color="tab:orange")
axes[1].plot(t_span, traj[:, 2], label="Ethanol E",  color="tab:green")
axes[1].set_ylabel("Konzentration [g/L]")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_span, traj[:, 3], label="Volumen V", color="tab:red")
axes[2].set_ylabel("V [L]")
axes[2].set_xlabel("Zeit [h]")
axes[2].legend()
axes[2].grid(True)

fig.suptitle("Crabtree-CSTR – Batch-Simulation")
plt.tight_layout()
plt.show()

