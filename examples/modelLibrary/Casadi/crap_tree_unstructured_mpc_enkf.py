"""Unstrukturiertes Crabtree-Modell nach Chang et al. mit MPC + EnKF.

Modell:    Chang, Liu, Henson –
           'Nonlinear model predictive control of fed-batch fermentations
            using dynamic flux balance models'

Gleichungen (7) und (8) aus dem Paper, Table 3 Parameter.

Regelziel:  Maximierung des Ethanolgehalts  (E·V = Gesamtmenge im Reaktor)
Stellgrößen: F_in (Zulaufpumpe), F_out (Ablaufpumpe)
Nebenbedingung: V ≤ 3 L
Beobachter: EnKF aus verrauschter X- und G-Messung (OD + Glucosemeter)

Usage:
    cd CADET-Live
    mamba run -n CADET-Live python examples/modelLibrary/Casadi/crap_tree_unstructured_mpc_enkf.py
"""

import os
import sys
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ── Pfade ──────────────────────────────────────────────────────────────────────
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

from Model import CasadiModel
from Provider import MeasurementProvider
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from control.optimalControl import Objective, CasadiOptimalControlProblem, MPCController

# ══════════════════════════════════════════════════════════════════════════════
# Kinetik-Parameter  (Table 3 – Chang et al.)
# ══════════════════════════════════════════════════════════════════════════════
# Fermentative Wachstumsrate  μ_gf
MU_GF_MAX = 0.39     # h⁻¹
K_GF      = 0.5      # g/L  – Sättigungskonstante Glucose
K_IGF     = 10.0     # g/L  – Substrathemmung Glucose
K_IEGF    = 10.0     # g/L  – Ethanolhemmung
K_IOGF    = 189.0    # %    – DO-Hemmungskonstante
Y_GF      = 0.11     # g Biomasse / g Glucose
Y_EG      = 0.415    # g Ethanol   / g Glucose

# Oxidative Wachstumsrate  μ_go
MU_GO_MAX = 0.34     # h⁻¹
K_GO      = 0.041    # g/L
K_IGO     = 86.0     # g/L
K_IEGO    = 10.0     # g/L
K_OGO     = 3.0      # %  – Sättigungskonstante DO
Y_GO      = 0.58     # g Biomasse / g Glucose

G_FEED   = 200.0     # Glucose-Feedkonzentration      [g/L]
DO_CONST =  5.0      # Gelöster Sauerstoff (konstant) [% Sättigung]
                     # Niedriger DO fördert fermentative Ethanolproduktion

# ══════════════════════════════════════════════════════════════════════════════
# Simulationsparameter
# ══════════════════════════════════════════════════════════════════════════════
RNG_SEED = 42
DT       = 0.1    # [h]
T_END    = 20.0   # [h]
N_STEPS  = int(T_END / DT)

# Anfangszustand Anlage  [X, G, E, V]
X0_TRUE   = np.array([1.0, 10.0, 0.0, 1.0])

# EnKF-Startschätzung (leicht daneben)
X0_BELIEF = np.array([0.9, 11.0, 0.0, 1.0])

# Messrauschen (Std)
NOISE_X = 0.02    # Biomasse [g/L]
NOISE_G = 0.5     # Glucose  [g/L]

# EnKF-Intervall
ENKF_INTERVAL = 2

V_MAX = 3.0   # Reaktor-Maximalvolumen [L]

# ── MPC ─────────────────────────────────────────────────────────────────────────
MPC_HORIZON = 3.0
# u[0] = F_in  [L/h],  u[1] = F_out [L/h]
U_MIN = [0.0,  0.0]
U_MAX = [0.15, 0.15]

Q_ETHANOL  = 500.0   # Gewicht auf E·V (maximieren)
R_CTRL     =   1.0   # Bestrafung Stellaufwand

# ══════════════════════════════════════════════════════════════════════════════
# ODE  –  unstrukturiertes Crabtree-Modell (Gl. 7 + 8, fed-batch)
# ══════════════════════════════════════════════════════════════════════════════
def unstructured_ode(x, u):
    """
    Zustände: [X, G, E, V]
    Eingänge: [F_in, F_out]

    dV/dt = F_in - F_out
    dX/dt = (μ_gf + μ_go)·X  -  F_in·X/V
    dG/dt = F_in·(G_feed - G)/V  -  (μ_gf/Y_gf + μ_go/Y_go)·X
    dE/dt = Y_eg·(μ_gf/Y_gf)·X  -  F_in·E/V
    """
    X_v, G_v, E_v, V_v = x[0], x[1], x[2], x[3]
    Fin = u[0]
    Fout = u[1]

    G_s = ca.fmax(G_v, 0.0)
    E_s = ca.fmax(E_v, 0.0)
    V_s = ca.fmax(V_v, 1e-3)
    DO  = DO_CONST

    # μ_gf  (Gl. 8, fermentativ)
    mu_gf = (MU_GF_MAX 
             * G_s / (K_GF + G_s + G_s**2 / K_IGF)
             * 1.0  / (1.0 + E_s / K_IEGF)
             * 1.0  / (1.0 + DO  / K_IOGF))

    # μ_go  (Gl. 8, oxidativ)
    mu_go = (MU_GO_MAX
             * G_s / (K_GO + G_s + G_s**2 / K_IGO)
             * 1.0  / (1.0 + E_s / K_IEGO)
             * DO   / (K_OGO + DO))

    dVdt = Fin - Fout
    dXdt = (mu_gf + mu_go) * X_v - Fin * X_v / V_s
    dGdt = Fin * (G_FEED - G_v) / V_s - (mu_gf / Y_GF + mu_go / Y_GO) * X_v
    dEdt = Y_EG * (mu_gf / Y_GF) * X_v - Fin * E_v / V_s

    return ca.vertcat(dXdt, dGdt, dEdt, dVdt)


# ══════════════════════════════════════════════════════════════════════════════
# Objective: Maximierung Ethanolgehalt  E·V  (Gesamtmenge im Reaktor)
# ══════════════════════════════════════════════════════════════════════════════
class EthanolObjective(Objective):
    """Minimiert  -Q_E·E·V + R·F_in²  ⟺  maximiert Ethanolmenge E·V.

    Terminalkost erhält großes Gewicht auf E·V, damit der Regler
    auch am Horizont-Ende auf hohen Ethanolgehalt optimiert.
    """

    def __init__(self, Q_ethanol, R, Q_terminal_scale=5.0):
        self.Q_ethanol       = Q_ethanol
        self.R               = R
        self.Q_terminal_scale = Q_terminal_scale

    def stage_cost(self, state, control):
        E_v = state[2]
        V_v = ca.fmax(state[3], 1e-3)
        return (-self.Q_ethanol * E_v * V_v
                + self.R * (control[0]**2 + control[1]**2))

    def terminal_cost(self, state):
        E_v = state[2]
        V_v = ca.fmax(state[3], 1e-3)
        return -self.Q_ethanol * self.Q_terminal_scale * E_v * V_v


# ══════════════════════════════════════════════════════════════════════════════
# Modell aufbauen  (Anlage + EnKF-Modell)
# ══════════════════════════════════════════════════════════════════════════════
states_sym   = ca.vertcat(*[ca.SX.sym(n) for n in ("X", "G", "E", "V")])
controls_sym = ca.SX.sym("u", 2)   # [F_in, F_out]

model = CasadiModel(
    states=states_sym,
    controls=controls_sym,
    ode=unstructured_ode,
    init_state=X0_BELIEF.copy(), #TODO auch setter funktion
    process_noise=np.diag([1e-4, 1e-3, 1e-4, 1e-5]),
    dt=DT,
)

# Messprovider  (Biomasse X + Glucose G)
prov_X = MeasurementProvider("od_sensor")
prov_X.add_variable("X", noise=np.array([[NOISE_X**2]]), state_index=0)

prov_G = MeasurementProvider("glucose_sensor")
prov_G.add_variable("G", noise=np.array([[NOISE_G**2]]), state_index=1)
#TODO: MX varibalen in CasADi


obs_func = lambda x: np.array([x[0], x[1]])   # h(x) = [X, G]

# EnKF aufbauen
enkf = EnKalmanFilter(
    model=model,
    ensemble_size=10,
    initial_covariance=np.diag([0.04, 2.0, 0.01, 0.01]),
    observation_func=obs_func,
    providers=[prov_X, prov_G],
    random_seed=RNG_SEED, #TODO Process noise und Measurement noise
)

# MPC aufbauen
objective = EthanolObjective(
    Q_ethanol=Q_ETHANOL,
    R=R_CTRL,
    Q_terminal_scale=10.0,
)

ocp = CasadiOptimalControlProblem(
    model=model,
    objective=objective,
    time_horizon=MPC_HORIZON,
    u_min=U_MIN,
    u_max=U_MAX,
    ipopt_print_level=0,
    path_constraints=[(3, 0.5, V_MAX)],   # 0.5 L ≤ V ≤ V_MAX
)

mpc = MPCController(ocp)


#------------------------------------------------------------------

plant = CasadiModel(
    states=states_sym,
    controls=controls_sym,
    ode=unstructured_ode,
    init_state=X0_TRUE.copy(), 
    process_noise=np.diag([1e-4, 1e-3, 1e-4, 1e-5]), #TODO
    dt=DT,
)

# Closed-loop Simulation
rng = np.random.default_rng(RNG_SEED)

times  = np.arange(N_STEPS + 1) * DT
x_true = np.zeros((N_STEPS + 1, 4))
x_est  = np.zeros((N_STEPS + 1, 4))
u_hist = np.zeros((N_STEPS, 2))
meas_X = []
meas_G = []
meas_t = []

x_true[0] = X0_TRUE
x_est[0]  = X0_BELIEF

print(f"Chang et al. – Unstrukturiertes Modell | MPC + EnKF  (T={T_END}h, dt={DT}h)")
print(f"Ziel: Maximierung E·V  |  DO={DO_CONST}%  |  V_max={V_MAX} L")
print(f"Messintervall: alle {ENKF_INTERVAL} Schritte  |  Ensemble: 30\n")

for k in range(N_STEPS):
    t = times[k]

    # MPC lösen 
    mpc.solve(x_est[k], DT, t)
    u_vec     = mpc.current_control
    u_hist[k] = u_vec

    # ── Anlage vorwärts integrieren
    res    = plant.integrator(x0=x_true[k], p=u_vec)
    x_next = np.maximum(np.array(res["xf"]).flatten(), 0.0)
    x_next += rng.normal(0.0, [1e-3, 5e-3, 1e-3, 1e-4]) #TODO
    x_next  = np.maximum(x_next, 0.0)
    x_true[k + 1] = x_next

    # ── Künstliche Messungen + EnKF-Update
    if (k + 1) % ENKF_INTERVAL == 0:
        t_meas = times[k + 1]
        y_X = max(0.0, float(x_true[k + 1, 0]) + rng.normal(0.0, NOISE_X))
        y_G = max(0.0, float(x_true[k + 1, 1]) + rng.normal(0.0, NOISE_G))

        prov_X.add_measurement("X", t_meas, y_X)
        prov_G.add_measurement("G", t_meas, y_G)

        meas_X.append(y_X);  meas_G.append(y_G);  meas_t.append(t_meas)

        x_upd = enkf.update_state_with_interpolation(
            t_end=t_meas, interpolation="nearest", u=u_vec
        )
    else:
        enkf.propagate(t_end=times[k + 1], u=u_vec)
        x_upd = enkf.state.copy()

    x_est[k + 1] = np.maximum(x_upd, 0.0)

# Auswertung
E_total_true = x_true[:, 2] * x_true[:, 3]   # E·V  [g]
E_total_est  = x_est[:,  2] * x_est[:,  3]

print("\nSimulation abgeschlossen.\n")
print(f"Endwerte (Anlage):   X={x_true[-1,0]:.3f}  G={x_true[-1,1]:.3f}  "
      f"E={x_true[-1,2]:.3f}  V={x_true[-1,3]:.3f}")
print(f"Endwerte (EnKF):     X={x_est[-1,0]:.3f}  G={x_est[-1,1]:.3f}  "
      f"E={x_est[-1,2]:.3f}  V={x_est[-1,3]:.3f}")
print(f"Gesamtethanol E·V (Anlage): {E_total_true[-1]:.3f} g")
print(f"Gesamtethanol E·V (EnKF):   {E_total_est[-1]:.3f} g")

# Plot
meas_t = np.array(meas_t);  meas_X = np.array(meas_X);  meas_G = np.array(meas_G)

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    "Chang et al. – Unstrukturiertes Modell | MPC + EnKF (fed-batch)\n"
    f"Ziel: Maximierung Ethanolgehalt E·V  (DO={DO_CONST}%)",
    fontsize=12,
)

# ── Gesamtethanol E·V ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(times, E_total_true, "g-",  lw=2,   label="E·V Anlage [g]")
ax.plot(times, E_total_est,  "g--", lw=1.5, label="E·V EnKF")
ax.set_ylabel("Gesamt-Ethanol E·V [g]");  ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# ── Ethanol- und Biomassekonzentration ─────────────────────────────────────────
ax = axes[1]
ax.plot(times, x_true[:, 2], "g-",  lw=2,   label="E Anlage [g/L]")
ax.plot(times, x_est[:,  2], "g--", lw=1.5, label="E EnKF")
ax.plot(times, x_true[:, 0], "k-",  lw=2,   label="X Anlage [g/L]")
ax.plot(times, x_est[:,  0], "k--", lw=1.5, label="X EnKF")
ax.set_ylabel("Konzentration [g/L]");  ax.legend(fontsize=8, ncol=2);  ax.grid(True, alpha=0.3)

# ── Glucose G und Volumen V ────────────────────────────────────────────────────
ax = axes[2];  ax2 = ax.twinx()
ax.plot(times,  x_true[:, 1], color="saddlebrown", lw=2,   label="G Anlage [g/L]")
ax.plot(times,  x_est[:,  1], color="orange",      lw=1.5, ls="--", label="G EnKF")
ax.scatter(meas_t, meas_G, s=10, c="orange", alpha=0.5, zorder=4, label="G Messung")
ax2.plot(times, x_true[:, 3], "r-",  lw=2,   label="V Anlage [L]")
ax2.plot(times, x_est[:,  3], "r--", lw=1.5, label="V EnKF")
ax2.axhline(V_MAX, color="pink", ls=":", lw=1, label=f"V_max={V_MAX}")
ax.set_ylabel("G [g/L]", color="saddlebrown");  ax2.set_ylabel("V [L]", color="red")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")
ax.grid(True, alpha=0.3)

# ── Stellgrößen F_in und F_out ─────────────────────────────────────────────────
ax = axes[3]
ax.step(times[:-1], u_hist[:, 0], where="post", color="blue",       lw=2,   label="F_in [L/h]")
ax.step(times[:-1], u_hist[:, 1], where="post", color="dodgerblue", lw=1.5, ls="--", label="F_out [L/h]")
ax.axhline(U_MAX[0], color="k", ls=":", lw=0.8, alpha=0.4, label=f"F_max={U_MAX[0]}")
ax.set_ylabel("Durchfluss [L/h]");  ax.set_xlabel("Zeit [h]")
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("crap_tree_unstructured_mpc_enkf.png", dpi=120)
plt.show()
