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
    controllable_Fin=False,
    controllable_Fout=False,
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
        Nominal inlet flow rate. Used as a fixed parameter when
        ``controllable_Fin=False``, ignored when ``controllable_Fin=True``.
    F_out : float
        Outlet flow rate. Used as a fixed parameter when
        ``controllable_Fout=False``, ignored when ``controllable_Fout=True``.
    process_noise : np.ndarray, optional
        3x3 process noise covariance. Default: small diagonal.
    controllable_Fin : bool
        If True, F_in is exposed as symbolic control input u[0] so an MPC
        can optimise it. If False (default), F_in is a fixed parameter
        baked into the ODE (backward-compatible, used by the EnKF model).
    controllable_Fout : bool
        If True (and ``controllable_Fin=True``), F_out is exposed as symbolic
        control input u[1].  Ignored when ``controllable_Fin=False``.

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

    if controllable_Fin:
        if controllable_Fout:
            # Both F_in (u[0]) and F_out (u[1]) are MPC decision variables
            u = ca.SX.sym("u", 2)

            def cstr_ode(x, u):
                X_val, S_val, V_val = x[0], x[1], x[2]
                F_in_ctrl  = u[0]
                F_out_ctrl = u[1]

                mu = mu_max * S_val / (K_s + S_val)

                dX_dt = mu * X_val - (F_out_ctrl / V_val) * X_val
                dS_dt = -(mu / Y_xs) * X_val + (F_in_ctrl / V_val) * (S_in - S_val)
                dV_dt = F_in_ctrl - F_out_ctrl

                return ca.vertcat(dX_dt, dS_dt, dV_dt)

        else:
            # Only F_in is controllable; F_out is fixed from the closure
            u = ca.SX.sym("u", 1)

            def cstr_ode(x, u):
                X_val, S_val, V_val = x[0], x[1], x[2]
                F_in_ctrl = u[0]
                V_MIN = 0.01
                V_safe = (V_val + V_MIN + ca.sqrt((V_val - V_MIN)**2 + 1e-10)) / 2
                S_safe = (S_val + ca.sqrt(S_val**2 + 1e-10)) / 2
                mu = mu_max * S_safe / (K_s + S_safe)

                dX_dt = mu * X_val - (F_out / V_safe) * X_val
                dS_dt = -(mu / Y_xs) * X_val + (F_in_ctrl / V_safe) * (S_in - S_val)
                dV_dt = F_in_ctrl - F_out

                return ca.vertcat(dX_dt, dS_dt, dV_dt)

    else:
        # No external control input — F_in baked in (default, used by EnKF)
        u = ca.SX.sym("u", 0)

        def cstr_ode(x, u):
            X_val, S_val, V_val = x[0], x[1], x[2]
            V_MIN = 0.01
            V_safe = (V_val + V_MIN + ca.sqrt((V_val - V_MIN)**2 + 1e-10)) / 2
            S_safe = (S_val + ca.sqrt(S_val**2 + 1e-10)) / 2

            # Monod growth rate
            mu = mu_max * S_safe / (K_s + S_safe)

            # Mass balances
            dX_dt = mu * X_val - (F_out / V_safe) * X_val
            dS_dt = -(mu / Y_xs) * X_val + (F_in / V_safe) * (S_in - S_val)
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
