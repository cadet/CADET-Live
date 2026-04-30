"""Crabtree-MPC mit Ensemble-Kalman-Filter und künstlichen Messungen.

Regelziel   : dX/dt = const. (XDOT_REF)
Stellgrößen : F_in (u[0]),  F_out (u[1])
Beobachter  : EnKF schätzt [X, S, E, V] aus verrauschter X- und S-Messung
              (simuliert z.B. OD-Sensor + Glucosemeter).

Kein MQTT/Hardware nötig – alle Messungen werden synthetisch erzeugt.

Usage:
    cd CADET-Live
    mamba run -n CADET-Live python examples/modelLibrary/Casadi/crap_tree_mpc_enkf.py
"""

import os
import sys
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ── Pfade ─────────────────────────────────────────────────────────────────────
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

from Model import CasadiModel
from Provider import MeasurementProvider
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from control.optimalControl import Objective, CasadiOptimalControlProblem, MPCController

# ══════════════════════════════════════════════════════════════════════════════
# Kinetik-Parameter  (identisch mit crap_tree.py)
# ══════════════════════════════════════════════════════════════════════════════
MU_S_MAX = 0.7
MU_E_MAX = 0.3
KS       = 10.0
KE       = 1.0
KI       = 0.5    # Ethanolhemmung auf µ_s
K1       = 0.05   # Spontane Ethanolproduktion [1/h]
Y_XS     = 0.3
Y_XE     = 0.3
S_IN     = 200.0

# ══════════════════════════════════════════════════════════════════════════════
# Simulationsparameter
# ══════════════════════════════════════════════════════════════════════════════
RNG_SEED = 42
DT       = 0.1     # MPC / EnKF-Schrittweite [h]
T_END    = 20.0    # Gesamtzeit              [h]
N_STEPS  = int(T_END / DT)

# Wahre Anfangswerte (Anlage)
X0_TRUE  = np.array([0.10, 10.0, 0.05, 1.0, 0.0])   # [X, S, E, V, e_int]
# EnKF-Startschätzung (leicht daneben)
X0_BELIEF = np.array([0.15, 9.0, 0.05, 1.0, 0.0])

# Messrauschen (Std)
NOISE_X = 0.01   # Biomasse (OD)  [g/L]
NOISE_S = 0.2    # Glucose        [g/L]

# EnKF-Intervall: alle ENKF_INTERVAL Schritte wird gemessen + Update
ENKF_INTERVAL = 2

# ── Regelziele ─────────────────────────────────────────────────────────────────
XDOT_REF_INIT      = 0.1   # Startwert Wachstumsrate [g/L/h]
XDOT_REF_INCREMENT = 0.1   # Schrittweite alle 10 h  [g/L/h]
XDOT_STEP_INTERVAL = 10.0  # Intervall pro Stufe     [h]
V_REF              = 1.0   # Soll-Volumen            [L]


def xdot_ref_profile(t: float) -> float:
    """Stufenförmiges Sollwertprofil: alle 10 h wird XDOT_REF um 0.1 erhöht."""
    step = int(t / XDOT_STEP_INTERVAL)
    return XDOT_REF_INIT + step * XDOT_REF_INCREMENT


# ── MPC ────────────────────────────────────────────────────────────────────────
MPC_HORIZON = 3.0
# u[0]=F_in, u[1]=F_out, u[2]=xdot_ref (symbolischer Sollwert-Parameter)
U_MIN = [0.0, 0.0, XDOT_REF_INIT]
U_MAX = [0.3, 0.3, XDOT_REF_INIT]

Q_RATE = 100.0
Q_VOL  =  10.0
Q_INT  =  50.0
R_CTRL =   0.5

# ══════════════════════════════════════════════════════════════════════════════
# ODE  (Crabtree-CSTR, F_in und F_out als Eingänge)
# ══════════════════════════════════════════════════════════════════════════════
def crabtree_ode(x, u):
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
    de_intdt = dXdt - u[2]   # u[2] = xdot_ref (Stufenprofil, zur Laufzeit gesetzt)

    return ca.vertcat(dXdt, dSdt, dEdt, dVdt, de_intdt)

# Custom Objective: dX/dt-Tracking + Volumen
class GrowthRateObjective(Objective):
    def __init__(self, Q_rate, Q_vol, Q_int, R, xdot_ref, v_ref):
        self.Q_rate   = Q_rate
        self.Q_vol    = Q_vol
        self.Q_int    = Q_int
        self.R        = R
        self.xdot_ref = xdot_ref
        self.v_ref    = v_ref

    def _dXdt_sym(self, state, control):
        X_v, S_v, E_v, V_v = state[0], state[1], state[2], state[3]
        Fin = control[0]
        S_s = ca.fmax(S_v, 0.0)
        E_s = ca.fmax(E_v, 0.0)
        V_s = ca.fmax(V_v, 1e-3)
        mu_s = MU_S_MAX * (S_s / (KS + S_s)) * KI / (KI + E_s)
        mu_e = MU_E_MAX * (E_s / (KE + E_s))
        return (mu_s + mu_e) * X_v - Fin * X_v / V_s

    def stage_cost(self, state, control):
        xdot_ref = control[2]          # symbolischer Sollwert aus u[2]
        err_rate = self._dXdt_sym(state, control) - xdot_ref
        err_vol  = state[3] - self.v_ref
        e_int    = state[4]
        ctrl_eff = control[0]**2 + control[1]**2  # nur F_in/F_out bestrafen
        return (self.Q_rate * err_rate**2
                + self.Q_vol  * err_vol**2
                + self.Q_int  * e_int**2
                + self.R      * ctrl_eff)

    def terminal_cost(self, state):
        # Kein Control-Vektor am Horizont-Ende → nur Volumen + Integralfehler
        err_vol = state[3] - self.v_ref
        e_int   = state[4]
        return self.Q_vol * err_vol**2 + self.Q_int * e_int**2

# ══════════════════════════════════════════════════════════════════════════════
# Modell aufbauen  (wird von Anlage, EnKF und MPC geteilt)
# ══════════════════════════════════════════════════════════════════════════════
states_sym   = ca.vertcat(*[ca.SX.sym(n) for n in ("X", "S", "E", "V", "e_int")])
controls_sym = ca.SX.sym("u", 3)   # [F_in, F_out, xdot_ref]

model = CasadiModel(
    states=states_sym,
    controls=controls_sym,
    ode=crabtree_ode,
    init_state=X0_BELIEF.copy(),
    process_noise=np.diag([1e-5, 1e-4, 1e-5, 1e-6, 0.0]),
    dt=DT,
)

# ── Anlage (zweite Instanz mit echten Anfangswerten) ──────────────────────────
plant = CasadiModel(
    states=states_sym,
    controls=controls_sym,
    ode=crabtree_ode,
    init_state=X0_TRUE.copy(),
    process_noise=np.diag([1e-5, 1e-4, 1e-5, 1e-6, 0.0]),
    dt=DT,
)

# ══════════════════════════════════════════════════════════════════════════════
# Messprovider  (X = Biomasse via OD,  S = Glucose)
# ══════════════════════════════════════════════════════════════════════════════
prov_X = MeasurementProvider("od_sensor")
prov_X.add_variable("X", noise=np.array([[NOISE_X**2]]), state_index=0)

prov_S = MeasurementProvider("glucose_sensor")
prov_S.add_variable("S", noise=np.array([[NOISE_S**2]]), state_index=1)

# Beobachtungsfunktion: h(x) = [X, S]
obs_func = lambda x: np.array([x[0], x[1]])

# ══════════════════════════════════════════════════════════════════════════════
# EnKF aufbauen
# ══════════════════════════════════════════════════════════════════════════════
enkf = EnKalmanFilter(
    model=model,
    ensemble_size=30,
    initial_covariance=np.diag([0.02, 1.0, 0.01, 0.01, 1e-8]),
    observation_func=obs_func,
    providers=[prov_X, prov_S],
    random_seed=RNG_SEED,
)

# ══════════════════════════════════════════════════════════════════════════════
# MPC aufbauen
# ══════════════════════════════════════════════════════════════════════════════
objective = GrowthRateObjective(
    Q_rate=Q_RATE, Q_vol=Q_VOL, Q_int=Q_INT, R=R_CTRL,
    xdot_ref=XDOT_REF_INIT, v_ref=V_REF,
)

ocp = CasadiOptimalControlProblem(
    model=model,
    objective=objective,
    time_horizon=MPC_HORIZON,
    u_min=U_MIN,
    u_max=U_MAX,
    ipopt_print_level=0,
    path_constraints=[(3, 0.05, None)],   # V >= 0.05 L
)

mpc = MPCController(ocp)

# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktion: numerisches dX/dt
# ══════════════════════════════════════════════════════════════════════════════
def compute_xdot(x, u):
    X_v, S_v, E_v, V_v = x[0], x[1], x[2], x[3]
    Fin = u[0]
    S_s = max(S_v, 0.0);  E_s = max(E_v, 0.0);  V_s = max(V_v, 1e-3)
    mu_s = MU_S_MAX * (S_s / (KS + S_s)) * KI / (KI + E_s)
    mu_e = MU_E_MAX * (E_s / (KE + E_s))
    return (mu_s + mu_e) * X_v - Fin * X_v / V_s

# ══════════════════════════════════════════════════════════════════════════════
# Geschlossener Regelkreis (closed-loop)
# ══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(RNG_SEED)

times     = np.arange(N_STEPS + 1) * DT
x_true    = np.zeros((N_STEPS + 1, 5))
x_est     = np.zeros((N_STEPS + 1, 5))
u_hist    = np.zeros((N_STEPS, 2))
xdot_true = np.zeros(N_STEPS + 1)
xdot_est  = np.zeros(N_STEPS + 1)
meas_X    = []
meas_S    = []
meas_t    = []

x_true[0]    = X0_TRUE
x_est[0]     = X0_BELIEF
xdot_true[0] = compute_xdot(X0_TRUE,  [0.0, 0.0])
xdot_est[0]  = compute_xdot(X0_BELIEF,[0.0, 0.0])

u_last = np.array([0.0, 0.0, XDOT_REF_INIT])

print(f"Crabtree MPC + EnKF  (T={T_END}h, dt={DT}h)")
print(f"Sollwertprofil: alle {XDOT_STEP_INTERVAL}h + {XDOT_REF_INCREMENT} g/L/h")
print(f"Messintervall: alle {ENKF_INTERVAL} Schritte  |  Ensemble: 30\n")

_xdot_ref_prev = XDOT_REF_INIT  # Vergleichswert für Stufenerkennung

for k in range(N_STEPS):
    t = times[k]
    xdot_ref_now = xdot_ref_profile(t)

    # ── Sollwert-Stufe: NLP-Schranken und Integral aktualisieren ────────────
    if xdot_ref_now != _xdot_ref_prev:
        N_hor = ocp._N
        lbx_arr = np.array(ocp._lbx, dtype=float).reshape(N_hor, 3)
        ubx_arr = np.array(ocp._ubx, dtype=float).reshape(N_hor, 3)
        lbx_arr[:, 2] = xdot_ref_now
        ubx_arr[:, 2] = xdot_ref_now
        ocp._lbx = lbx_arr.flatten().tolist()
        ocp._ubx = ubx_arr.flatten().tolist()
        # Integral-Reset bei Sollwertsprung (Anti-Windup)
        x_est[k, 4]  = 0.0
        x_true[k, 4] = 0.0
        _xdot_ref_prev = xdot_ref_now
        print(f"  t={t:.1f}h: Sollwertsprung → dX/dt_ref = {xdot_ref_now:.2f} g/L/h")

    # ── MPC lösen (immer mit aktueller Schätzung) ──────────────────────────
    mpc.solve(x_est[k], DT, t)
    u_vec = mpc.current_control
    u_hist[k] = u_vec[:2]   # nur F_in und F_out speichern

    # ── Anlage vorwärts integrieren ────────────────────────────────────────
    res    = plant.integrator(x0=x_true[k], p=u_vec)
    x_next = np.maximum(np.array(res["xf"]).flatten(), 0.0)
    # Kleines Prozessrauschen auf die Anlage
    x_next += rng.normal(0.0, [1e-3, 5e-3, 1e-3, 1e-4, 0.0])
    x_next  = np.maximum(x_next, 0.0)
    x_true[k + 1] = x_next

    # ── Künstliche Messungen erzeugen (alle ENKF_INTERVAL Schritte) ────────
    if (k + 1) % ENKF_INTERVAL == 0:
        t_meas = times[k + 1]
        y_X = float(x_true[k + 1, 0]) + rng.normal(0.0, NOISE_X)
        y_S = float(x_true[k + 1, 1]) + rng.normal(0.0, NOISE_S)
        y_X = max(0.0, y_X)
        y_S = max(0.0, y_S)

        prov_X.add_measurement("X", t_meas, y_X)
        prov_S.add_measurement("S", t_meas, y_S)

        meas_X.append(y_X)
        meas_S.append(y_S)
        meas_t.append(t_meas)

        # EnKF: propagieren + korrigieren
        x_upd = enkf.update_state_with_interpolation(
            t_end=times[k + 1], interpolation="nearest", u=u_vec
        )
    else:
        # Zwischen Messungen: nur propagieren
        enkf.propagate(t_end=times[k + 1], u=u_vec)
        x_upd = enkf.state.copy()

    x_est[k + 1]     = np.maximum(x_upd, 0.0)
    xdot_true[k + 1] = compute_xdot(x_true[k + 1], u_vec)
    xdot_est[k + 1]  = compute_xdot(x_est[k + 1],  u_vec)

xdot_ref_arr = np.array([xdot_ref_profile(t) for t in times])
print("Simulation abgeschlossen.\n")
rmse_true = float(np.sqrt(np.mean((xdot_true - xdot_ref_arr)**2)))
rmse_est  = float(np.sqrt(np.mean((xdot_est  - xdot_ref_arr)**2)))
print(f"Endwerte (Anlage):   X={x_true[-1,0]:.3f}  S={x_true[-1,1]:.3f}  "
      f"E={x_true[-1,2]:.3f}  V={x_true[-1,3]:.3f} g/L")
print(f"Endwerte (EnKF):     X={x_est[-1,0]:.3f}  S={x_est[-1,1]:.3f}  "
      f"E={x_est[-1,2]:.3f}  V={x_est[-1,3]:.3f} g/L")
print(f"RMSE dX/dt (Anlage): {rmse_true:.5f} g/L/h")
print(f"RMSE dX/dt (EnKF):   {rmse_est:.5f} g/L/h")

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
meas_t = np.array(meas_t)
meas_X = np.array(meas_X)
meas_S = np.array(meas_S)

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle("Crabtree MPC + EnKF  —  dX/dt Sollwert-Stufen (+0.1 g/L/h alle 10h)", fontsize=13)

# ── Wachstumsrate ────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(times, xdot_true, "k-",  lw=2,   label="dX/dt Anlage")
ax.plot(times, xdot_est,  "b--", lw=1.5, label="dX/dt EnKF-Schätzung")
ax.step(times, xdot_ref_arr, where="post", color="red", ls=":", lw=1.5, label="Sollwert")
ax.set_ylabel("dX/dt [g/L/h]")
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# ── Biomasse X und Glucose S ──────────────────────────────────────────────────
ax = axes[1]
ax.plot(times, x_true[:, 0], "k-",  lw=2,   label="X Anlage")
ax.plot(times, x_est[:,  0], "b--", lw=1.5, label="X EnKF")
ax.scatter(meas_t, meas_X, s=12, c="blue", alpha=0.5, zorder=4, label="X Messung")
ax.plot(times, x_true[:, 1], color="saddlebrown", lw=2,   label="S Anlage")
ax.plot(times, x_est[:,  1], color="orange",      lw=1.5, ls="--", label="S EnKF")
ax.scatter(meas_t, meas_S, s=12, c="orange", alpha=0.5, zorder=4, label="S Messung")
ax.set_ylabel("Konzentration [g/L]")
ax.legend(fontsize=8, ncol=2);  ax.grid(True, alpha=0.3)

# ── Ethanol E und Volumen V ───────────────────────────────────────────────────
ax = axes[2]
ax2 = ax.twinx()
ax.plot(times,  x_true[:, 2], "g-",  lw=2,   label="E Anlage")
ax.plot(times,  x_est[:,  2], "g--", lw=1.5, label="E EnKF")
ax2.plot(times, x_true[:, 3], "r-",  lw=2,   label="V Anlage")
ax2.plot(times, x_est[:,  3], "r--", lw=1.5, label="V EnKF")
ax2.axhline(V_REF, color="pink", ls=":", lw=1)
ax.set_ylabel("E [g/L]",   color="green")
ax2.set_ylabel("V [L]",    color="red")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Stellgrößen ───────────────────────────────────────────────────────────────
ax = axes[3]
ax.step(times[:-1], u_hist[:, 0], where="post", color="blue",       lw=2,   label="F_in")
ax.step(times[:-1], u_hist[:, 1], where="post", color="dodgerblue", lw=1.5, ls="--", label="F_out")
ax.axhline(U_MAX[0], color="k", ls=":", lw=0.8, alpha=0.4)
ax.set_ylabel("Durchfluss [L/h]")
ax.set_xlabel("Zeit [h]")
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("crap_tree_mpc_enkf.png", dpi=120)

plt.show()
