"""MPC-Regelung des Crabtree-Bioreaktors.

Regelziel : Wachstumsrate  dX/dt = const. (Setpoint: XDOT_REF)
Stellgrößen: F_in  (Zulauf, u[0])
             F_out (Ablauf, u[1])  → reguliert zusätzlich das Volumen

Kein MQTT/Hardware nötig – reine Offline-Simulation.

Usage:
    cd CADET-Live
    mamba run -n CADET-Live python examples/modelLibrary/Casadi/crap_tree_mpc.py
"""

import os
import sys
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ── Pfad zu src/ ─────────────────────────────────────────────────────────────
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

from Model import CasadiModel
from control.optimalControl import Objective, CasadiOptimalControlProblem, MPCController

# ══════════════════════════════════════════════════════════════════════════════
# Parameter  (identisch mit crap_tree.py)
# ══════════════════════════════════════════════════════════════════════════════
MU_S_MAX = 0.7    # max. Wachstumsrate auf Glucose   [1/h]
MU_E_MAX = 0.3    # max. Wachstumsrate auf Ethanol   [1/h]
KS       = 10.0   # Monod-Halbsättigung Glucose      [g/L]
KE       = 1.0    # Monod-Halbsättigung Ethanol      [g/L]
KI       = 0.5    # Ethanolhemmung auf µ_s           [g/L]
K1       = 0.05   # Spontane Ethanolproduktion       [1/h]
Y_XS     = 0.3    # Ausbeute Biomasse/Glucose        [g/g]
Y_XE     = 0.3    # Ausbeute Biomasse/Ethanol        [g/g]
S_IN     = 200.0  # Glucose im Zulauf                [g/L]

# ── Simulations-Parameter ─────────────────────────────────────────────────────
DT    = 0.1    # MPC-Schrittweite   [h]
T_END = 20.0   # Simulationszeit    [h]
X0    = np.array([0.1, 10.0, 0.1, 1.0, 0.0])  # [X, S, E, V, e_int]  — wie crap_tree.py + Integralzustand

# ── Regelziele ────────────────────────────────────────────────────────────────
XDOT_REF = 0.05   # Soll-Wachstumsrate    dX/dt  [g/L/h]
V_REF    = 1.0    # Soll-Volumen                  [L]

# ── MPC-Einstellungen ─────────────────────────────────────────────────────────
MPC_HORIZON = 3.0   # Prädiktionshorizont  [h]
U_MIN = [0.0, 0.0]  # [F_in_min, F_out_min] [L/h]
U_MAX = [0.3, 0.3]  # [F_in_max, F_out_max] [L/h]

Q_RATE = 100.0   # Gewicht Wachstumsrate-Fehler   (dXdt - ref)^2
Q_VOL  =  10.0   # Gewicht Volumen-Fehler         (V - V_ref)^2
Q_INT  =  50.0   # Gewicht Integralfehler         e_int^2
R_CTRL =   0.5   # Penalisierung Stellaufwand     ||u||^2

# ══════════════════════════════════════════════════════════════════════════════
# ODE  (Crabtree-CSTR mit F_in und F_out als Eingänge)
# ══════════════════════════════════════════════════════════════════════════════
def crabtree_ode(x, u):
    """Massenbilanz Crabtree-CSTR – identisch mit crap_tree.py.

    Zustände: x = [X, S, E, V]
    Eingänge: u = [F_in, F_out]

    mu_s mit Ethanolhemmung (ki/(ki+E)).
    Spontane Ethanolproduktion via k1*S.
    """
    X_v, S_v, E_v, V_v = x[0], x[1], x[2], x[3]
    Fin  = u[0]
    Fout = u[1]

    S_s = ca.fmax(S_v, 0.0)
    E_s = ca.fmax(E_v, 0.0)
    V_s = ca.fmax(V_v, 1e-3)

    mu_s = MU_S_MAX * (S_s / (KS + S_s)) * KI / (KI + E_s)   # Ethanolhemmung
    mu_e = MU_E_MAX * (E_s / (KE + E_s))

    dVdt     = Fin - Fout
    dXdt     = (mu_s + mu_e) * X_v - Fin * X_v / V_s
    dSdt     = -(mu_s / Y_XS) * X_v - K1 * S_s + Fin * (S_IN - S_v) / V_s
    dEdt     = K1 * S_s - (mu_e / Y_XE) * X_v - Fin * E_v / V_s
    de_intdt = dXdt - XDOT_REF   # Integralfehler: akkumuliert (dX/dt - ref)

    return ca.vertcat(dXdt, dSdt, dEdt, dVdt, de_intdt)

# ══════════════════════════════════════════════════════════════════════════════
# Custom Objective:  dX/dt-Tracking  +  Volumen-Regelung
# ══════════════════════════════════════════════════════════════════════════════
class GrowthRateObjective(Objective):
    """Hält dX/dt auf konstantem Setpoint und regelt das Volumen.

    Stufenkosten:
        J_k = Q_rate * (dX/dt - xdot_ref)^2
            + Q_vol  * (V   - v_ref)^2
            + R      * (F_in^2 + F_out^2)
    """

    def __init__(self, Q_rate: float, Q_vol: float, Q_int: float, R: float,
                 xdot_ref: float, v_ref: float):
        self.Q_rate   = Q_rate
        self.Q_vol    = Q_vol
        self.Q_int    = Q_int
        self.R        = R
        self.xdot_ref = xdot_ref
        self.v_ref    = v_ref

    def _dXdt_sym(self, state: ca.SX, control: ca.SX) -> ca.SX:
        """Symbolischer Ausdruck für dX/dt (identisch mit crap_tree.py)."""
        X_v, S_v, E_v, V_v = state[0], state[1], state[2], state[3]
        Fin = control[0]

        S_s = ca.fmax(S_v, 0.0)
        E_s = ca.fmax(E_v, 0.0)
        V_s = ca.fmax(V_v, 1e-3)

        mu_s = MU_S_MAX * (S_s / (KS + S_s)) * KI / (KI + E_s)
        mu_e = MU_E_MAX * (E_s / (KE + E_s))

        return (mu_s + mu_e) * X_v - Fin * X_v / V_s

    def stage_cost(self, state: ca.SX, control: ca.SX) -> ca.SX:
        err_rate = self._dXdt_sym(state, control) - self.xdot_ref
        err_vol  = state[3] - self.v_ref
        e_int    = state[4]
        return (self.Q_rate * err_rate**2
                + self.Q_vol  * err_vol**2
                + self.Q_int  * e_int**2
                + self.R      * ca.dot(control, control))

    def terminal_cost(self, state: ca.SX) -> ca.SX:
        u_zero   = ca.SX.zeros(2)
        err_rate = self._dXdt_sym(state, u_zero) - self.xdot_ref
        err_vol  = state[3] - self.v_ref
        e_int    = state[4]
        return self.Q_rate * err_rate**2 + self.Q_vol * err_vol**2 + self.Q_int * e_int**2

# ══════════════════════════════════════════════════════════════════════════════
# Modell und MPC aufbauen
# ══════════════════════════════════════════════════════════════════════════════
states_sym   = ca.vertcat(*[ca.SX.sym(n) for n in ("X", "S", "E", "V", "e_int")])
controls_sym = ca.SX.sym("u", 2)

model = CasadiModel(
    states=states_sym,
    controls=controls_sym,
    ode=crabtree_ode,
    init_state=X0.copy(),
    process_noise=np.diag([1e-5, 1e-4, 1e-5, 1e-6, 0.0]),
    dt=DT,
)

objective = GrowthRateObjective(
    Q_rate=Q_RATE, Q_vol=Q_VOL, Q_int=Q_INT, R=R_CTRL,
    xdot_ref=XDOT_REF, v_ref=V_REF,
)

ocp = CasadiOptimalControlProblem(
    model=model,
    objective=objective,
    time_horizon=MPC_HORIZON,
    u_min=U_MIN,
    u_max=U_MAX,
    ipopt_print_level=0,
    path_constraints=[(3, 0.05, None)],   # V >= 0.05 L (Sicherheitsgrenze)
)

mpc = MPCController(ocp)

# ══════════════════════════════════════════════════════════════════════════════
# Geschlossener Regelkreis (closed-loop Simulation)
# ══════════════════════════════════════════════════════════════════════════════
def compute_xdot(x, u):
    """Numerische dX/dt-Berechnung (für Auswertung)."""
    X_v, S_v, E_v, V_v = x[0], x[1], x[2], x[3]
    Fin = u[0]
    S_s = max(S_v, 0.0)
    E_s = max(E_v, 0.0)
    V_s = max(V_v, 1e-3)
    mu_s = MU_S_MAX * (S_s / (KS + S_s)) * KI / (KI + E_s)
    mu_e = MU_E_MAX * (E_s / (KE + E_s))
    return (mu_s + mu_e) * X_v - Fin * X_v / V_s

N_STEPS  = int(T_END / DT)
times    = np.arange(N_STEPS + 1) * DT
traj     = np.zeros((N_STEPS + 1, 5))
u_hist   = np.zeros((N_STEPS, 2))
xdot_arr = np.zeros(N_STEPS + 1)

traj[0]     = X0
xdot_arr[0] = compute_xdot(X0, [0.0, 0.0])

print(f"Starte Crabtree-MPC  (T={T_END} h,  dt={DT} h,  dX/dt_ref={XDOT_REF} g/L/h)")
print(f"Initiale Wachstumsrate: {xdot_arr[0]:.4f} g/L/h\n")

for k in range(N_STEPS):
    x_k = traj[k]

    mpc.solve(x_k, DT, times[k])
    u_vec = mpc.current_control       # [F_in, F_out]
    u_hist[k] = u_vec

    # Anlage einen Schritt vorwärts integrieren
    res    = model.integrator(x0=x_k, p=u_vec)
    x_next = np.maximum(np.array(res["xf"]).flatten(), 0.0)

    traj[k + 1]     = x_next
    xdot_arr[k + 1] = compute_xdot(x_next, u_vec)

print("Simulation abgeschlossen.\n")
rmse = float(np.sqrt(np.mean((xdot_arr - XDOT_REF) ** 2)))
print(f"{'Endwerte':}")
print(f"  X      = {traj[-1, 0]:.3f} g/L")
print(f"  S      = {traj[-1, 1]:.3f} g/L")
print(f"  E      = {traj[-1, 2]:.3f} g/L")
print(f"  V      = {traj[-1, 3]:.3f} L")
print(f"  e_int  = {traj[-1, 4]:.5f} g/L")
print(f"  dX/dt  = {xdot_arr[-1]:.4f} g/L/h  (Setpoint: {XDOT_REF})")
print(f"  RMSE dX/dt = {rmse:.5f} g/L/h")

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
fig.suptitle(f"Crabtree MPC  —  dX/dt = const. = {XDOT_REF} g/L/h", fontsize=13)

ax = axes[0]
ax.plot(times, xdot_arr, "k-", lw=2, label="dX/dt (gemessen)")
ax.axhline(XDOT_REF, color="red", ls="--", lw=1.5, label=f"Setpoint {XDOT_REF} g/L/h")
ax.set_ylabel("dX/dt [g/L/h]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(times, traj[:, 0], label="X Biomasse",  color="tab:blue")
ax.plot(times, traj[:, 1], label="S Glucose",   color="tab:orange")
ax.plot(times, traj[:, 2], label="E Ethanol",   color="tab:green")
ax.set_ylabel("Konzentration [g/L]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(times, traj[:, 3], color="tab:red", lw=2, label="Volumen V")
ax.axhline(V_REF, color="grey", ls=":", lw=1, label=f"V_ref = {V_REF} L")
ax.set_ylabel("V [L]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(times, traj[:, 4], color="purple", lw=2, label="Integralfehler e_int")
ax.axhline(0, color="grey", ls=":", lw=1)
ax.set_ylabel("e_int [g/L]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[4]
ax.step(times[:-1], u_hist[:, 0], where="post", color="blue",       lw=2,   label="F_in")
ax.step(times[:-1], u_hist[:, 1], where="post", color="dodgerblue", lw=1.5, ls="--", label="F_out")
ax.axhline(U_MAX[0], color="k", ls=":", lw=1, alpha=0.4)
ax.set_ylabel("Durchfluss [L/h]")
ax.set_xlabel("Zeit [h]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("crap_tree_mpc.png", dpi=120)
plt.show()
