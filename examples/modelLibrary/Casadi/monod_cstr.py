"""Bioprozess-Modelle für CasADi-basierte Simulation.

Definiert ODE-Modelle (z.B. Monod-Kinetik CSTR), die mit der
CasadiModel-Klasse aus control/Model.py verwendet werden können.
"""

import casadi as ca
import numpy as np

from Model import CasadiModel


def create_monod_cstr(
    X0=None,
    dt=0.1,
    mu_max=0.4,
    K_s=0.1,
    Y_xs=0.5,
    S_in=10.0,
    F_in=0.05,
    F_out=0.05,
    process_noise=None,
):
    """Create a Monod-kinetics CSTR model for yeast growth.

    States:
        X - Biomass concentration [g/L]
        S - Substrate concentration [g/L]
        V - Volume [L]

    Parameters
    ----------
    X0 : array-like, optional
        Initial state [X, S, V]. Default: [0.1, 10.0, 1.0]
    dt : float
        Integration time step [s or h, depending on parameter units].
    mu_max : float
        Maximum specific growth rate.
    K_s : float
        Monod half-saturation constant.
    Y_xs : float
        Yield coefficient (biomass / substrate).
    S_in : float
        Substrate concentration in feed.
    F_in : float
        Inlet flow rate.
    F_out : float
        Outlet flow rate.
    process_noise : np.ndarray, optional
        3x3 process noise covariance. Default: small diagonal.

    Returns
    -------
    CasadiModel
        Ready-to-use model instance.
    """
    if X0 is None:
        X0 = np.array([0.1, 10.0, 1.0])

    # Symbolic states
    X = ca.SX.sym("X")  # Biomass
    S = ca.SX.sym("S")  # Substrate
    V = ca.SX.sym("V")  # Volume
    states = ca.vertcat(X, S, V)

    # No external control input (PID controls flow rates externally)
    u = ca.SX.sym("u", 0)

    def cstr_ode(x, u):
        X_val, S_val, V_val = x[0], x[1], x[2]

        # Monod growth rate
        mu = mu_max * S_val / (K_s + S_val)

        # Mass balances
        dX_dt = mu * X_val - (F_out / V_val) * X_val
        dS_dt = -(mu / Y_xs) * X_val + (F_in / V_val) * (S_in - S_val)
        dV_dt = F_in - F_out

        return ca.vertcat(dX_dt, dS_dt, dV_dt)

    if process_noise is None:
        process_noise = np.diag([1e-4, 1e-4, 1e-6])

    model = CasadiModel(
        states=states,
        controls=u,
        ode=cstr_ode,
        init_state=X0,
        process_noise=process_noise,
        dt=dt,
    )

    return model
