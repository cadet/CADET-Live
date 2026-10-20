"""Monod-kinetics CSTR example using the current CasADi model API."""

import casadi as ca
import numpy as np
from cadet_live import CasadiModel


def create_monod_cstr(
    X0: np.ndarray | list[float] | None = None,
    mu_max: float = 0.4,
    K_s: float = 0.1,
    Y_xs: float = 0.5,
    S_in: float = 10.0,
    F_in: float = 0.05,
    F_out: float = 0.05,
) -> CasadiModel:
    """Create a Monod-kinetics CSTR model for yeast growth.

    States:
        X - Biomass concentration [g/L]
        S - Substrate concentration [g/L]
        V - Volume [L]

    Parameters
    ----------
    X0 : array-like, optional
        Initial state [X, S, V]. Default: [0.1, 10.0, 1.0]
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

    def cstr_ode(x: ca.SX) -> ca.SX:
        """Return the biomass, substrate, and volume derivatives."""
        X_val, S_val, V_val = x[0], x[1], x[2]

        # Monod growth rate
        mu = mu_max * S_val / (K_s + S_val)

        # Mass balances
        dX_dt = mu * X_val - (F_out / V_val) * X_val
        dS_dt = -(mu / Y_xs) * X_val + (F_in / V_val) * (S_in - S_val)
        dV_dt = F_in - F_out

        return ca.vertcat(dX_dt, dS_dt, dV_dt)

    model = CasadiModel(states=states, ode=cstr_ode)
    model.set_state(np.asarray(X0, dtype=float))

    return model
