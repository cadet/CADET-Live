import logging

import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from typing import Optional

from Model import CasadiModel
from Provider import ControlProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Objective ABC + concrete implementations
# ---------------------------------------------------------------------------

class Objective(ABC):
    """Abstract base class for MPC objective functions.

    Subclasses define the running cost per time step and an optional
    terminal cost at the end of the prediction horizon.
    """

    @abstractmethod
    def stage_cost(self, state: ca.SX, control: ca.SX) -> ca.SX:
        """Running cost at a single time step.

        Parameters
        ----------
        state : ca.SX
            Symbolic state vector at step k.
        control : ca.SX
            Symbolic control vector at step k.

        Returns
        -------
        ca.SX
            Scalar stage cost expression.
        """
        pass

    def terminal_cost(self, state: ca.SX) -> ca.SX:
        """Cost at the final predicted state. Default: zero."""
        return ca.SX(0)


class TrackingObjective(Objective):
    """Quadratic setpoint-tracking objective.

    Minimises:
        sum_k  Q * (x[state_index] - setpoint)^2 + R * ||u||^2
        + Q * (x_N[state_index] - setpoint)^2   (terminal)

    Parameters
    ----------
    Q : float
        State-tracking weight.
    R : float
        Control-effort weight.
    setpoint : float
        Desired value for the tracked state.
    state_index : int
        Index of the state variable to track (default 1 = substrate S).
    """

    def __init__(self, Q: float, R: float, setpoint: float, state_index: int = 1):
        self.Q = Q
        self.R = R
        self.setpoint = setpoint
        self.state_index = state_index

    def stage_cost(self, state: ca.SX, control: ca.SX) -> ca.SX:
        error = state[self.state_index] - self.setpoint
        return self.Q * error ** 2 + self.R * ca.dot(control, control)

    def terminal_cost(self, state: ca.SX) -> ca.SX:
        error = state[self.state_index] - self.setpoint
        return self.Q * error ** 2


class BiomassMaxObjective(Objective):
    """Biomass-maximising objective.

    Minimises:  -X + R * ||u||^2  (i.e. maximises biomass, penalises effort)

    Parameters
    ----------
    R : float
        Control-effort penalty weight.
    biomass_index : int
        Index of the biomass state (default 0 = X).
    """

    def __init__(self, R: float = 0.01, biomass_index: int = 0):
        self.R = R
        self.biomass_index = biomass_index

    def stage_cost(self, state: ca.SX, control: ca.SX) -> ca.SX:
        return -state[self.biomass_index] + self.R * ca.dot(control, control)


# ---------------------------------------------------------------------------
# OptimalControlProblem ABC + CasADi implementation
# ---------------------------------------------------------------------------

class OptimalControlProblem(ABC):
    """Abstract base class for optimal control problems."""

    def __init__(
        self,
        model: CasadiModel,
        objective: Objective,
        time_horizon: float,
        u_min: float = 0.0,
        u_max: float = 10.0,
    ):
        self.model = model
        self.objective = objective
        self.time_horizon = time_horizon
        self.u_min = u_min
        self.u_max = u_max

    @abstractmethod
    def solve(self, x0: np.ndarray) -> np.ndarray:
        """Solve the OCP for a given initial state.

        Parameters
        ----------
        x0 : np.ndarray
            Current state estimate (e.g. from EnKF).

        Returns
        -------
        np.ndarray
            Optimal control sequence, shape (n_controls * N_horizon,).
        """
        pass


class CasadiOptimalControlProblem(OptimalControlProblem):
    """Single-shooting OCP solved with IPOPT via CasADi.

    The NLP and solver are compiled **once** in ``__init__``; only the
    initial-state parameter ``p`` changes on each call to ``solve()``.
    Warm-starting is applied automatically by shifting the previous solution.

    Parameters
    ----------
    model : CasadiModel
        The dynamical model.  Must expose a ``.integrator`` property.
    objective : Objective
        Stage and terminal cost definitions.
    time_horizon : float
        Prediction horizon length (same units as model.dt).
    u_min, u_max : float
        Box constraints on all control inputs.
    ipopt_print_level : int
        IPOPT verbosity (0 = silent).
    path_constraints : list of (int, float|None, float|None), optional
        Path constraints on states at every rollout step, as a list of
        ``(state_index, lower_bound, upper_bound)`` tuples.  Use ``None`` for
        one-sided constraints.  Example: ``[(2, 0.5, None)]`` keeps the
        third state (e.g. volume) >= 0.5 throughout the horizon.
    """

    def __init__(
        self,
        model: CasadiModel,
        objective: Objective,
        time_horizon: float,
        u_min: float = 0.0,
        u_max: float = 10.0,
        ipopt_print_level: int = 0,
        path_constraints: list = None,
    ):
        super().__init__(model, objective, time_horizon, u_min, u_max)

        self._N = int(time_horizon / model.dt)
        self._n_states = model.nStates
        self._n_controls = model.nControls
        self._prev_solution = np.zeros(self._n_controls * self._N)
        self._path_constraints = path_constraints or []

        # Build NLP solver once
        self._solver, self._lbx, self._ubx, self._lbg, self._ubg = self._build_solver(ipopt_print_level)

    def _build_solver(self, ipopt_print_level: int):
        """Compile the symbolic NLP and return the solver + bounds."""
        N = self._N
        n_controls = self._n_controls

        # Decision variables: controls over the horizon
        U = ca.SX.sym('U', n_controls, N)
        # Parameter: initial state (updated on each solve call)
        X0_sym = ca.SX.sym('X0', self._n_states)

        # Single-shooting rollout
        cost = ca.SX(0)
        current_state = X0_sym
        g_exprs = []
        lbg = []
        ubg = []

        for k in range(N):
            res = self.model.integrator(x0=current_state, p=U[:, k])
            current_state = res['xf']
            cost += self.objective.stage_cost(current_state, U[:, k])
            # Optional path constraints on states (e.g. V >= V_min)
            for (idx, lb, ub) in self._path_constraints:
                g_exprs.append(current_state[idx])
                lbg.append(-1e20 if lb is None else float(lb))
                ubg.append( 1e20 if ub is None else float(ub))

        cost += self.objective.terminal_cost(current_state)

        g_sym = ca.vertcat(*g_exprs) if g_exprs else ca.SX([])
        nlp = {
            'x': ca.reshape(U, -1, 1),
            'f': cost,
            'g': g_sym,
            'p': X0_sym,
        }

        opts = {
            'ipopt.print_level': ipopt_print_level,
            'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.hessian_approximation': 'limited-memory',
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        # Support scalar or per-control array bounds.
        # Decision variables are laid out as [u0_step0, u1_step0, u0_step1, ...]
        # so tile the per-control bounds across the horizon.
        u_min_arr = np.atleast_1d(np.asarray(self.u_min, dtype=float))
        u_max_arr = np.atleast_1d(np.asarray(self.u_max, dtype=float))
        lbx = np.tile(u_min_arr, N).tolist()
        ubx = np.tile(u_max_arr, N).tolist()

        return solver, lbx, ubx, lbg, ubg

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """Solve the OCP for the current state (warm-started).

        Parameters
        ----------
        x0 : np.ndarray
            Current full state vector.

        Returns
        -------
        np.ndarray
            Flattened optimal control sequence of length n_controls * N.
        """
        result = self._solver(
            x0=self._prev_solution,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=self._lbg,
            ubg=self._ubg,
            p=x0,
        )
        solution = np.array(result['x']).flatten()

        # Shift solution for warm-start on next call
        self._prev_solution = np.roll(solution, -self._n_controls)

        return solution

    def solve_multiple_shooting(self):
        pass  # TODO: implement multiple-shooting transcription


# MPCController
class MPCController:
    """Receding-horizon MPC with the same interface as :class:`PID`.

    Wraps a :class:`CasadiOptimalControlProblem` and applies only the
    **first** control input at each step (apply-first-control principle).

    After every call to ``update()`` the full predicted control sequence and
    the corresponding time grid are accessible via the read-only properties
    ``control_sequence`` and ``time_sequence``.

    The ``update()`` method is drop-in compatible with ``PID.update()``:
    both return ``(time: float, control: float)``.

    Parameters
    ----------
    ocp : CasadiOptimalControlProblem
        The compiled optimal control problem.
    provider : ControlProvider, optional
        If given, the full predicted control sequence is written into this
        provider after every ``update()`` and ``step()`` call.  The
        variable is created automatically if it does not yet exist.
    variable_name : str, optional
        Name of the variable inside ``provider`` to write to.
        Required when ``provider`` is set.
    """

    def __init__(
        self,
        ocp: CasadiOptimalControlProblem,
        provider: Optional[ControlProvider] = None,
        variable_name: Optional[str] = None,
    ):
        self._ocp = ocp
        self._n_controls = ocp._n_controls
        self._N = ocp._N
        self._dt = ocp.model.dt
        # Shape: (N, n_controls) — stores the latest optimal sequence
        self._control_sequence: np.ndarray = np.zeros((self._N, self._n_controls))
        # Absolute time stamps for each step in the sequence
        self._time_sequence: np.ndarray = np.zeros(self._N)
        # Pointer into the current sequence; reset to 0 on each update()
        self._seq_idx: int = 0
        # Last state passed to update() — used for fallback re-solve if needed
        self._last_state: np.ndarray = np.zeros(ocp._n_states)

        # Optional ControlProvider — receives the sequence after every solve
        if provider is not None and variable_name is None:
            raise ValueError("variable_name must be set when provider is given")
        self._provider = provider
        self._variable_name = variable_name
        if provider is not None and variable_name not in provider.variable_names:
            provider.add_variable(variable_name)

    @property
    def control_sequence(self) -> np.ndarray:
        """Latest optimal control sequence, shape ``(N, n_controls)``.

        Row k contains the optimal control input for prediction step k,
        starting from the time at which ``update()`` was last called.
        Updated after every call to ``update()``.
        """
        return self._control_sequence.copy()

    @property
    def time_sequence(self) -> np.ndarray:
        """Absolute time stamps for each step of ``control_sequence``.

        ``time_sequence[k] = t + (k+1) * dt`` where ``t`` is the time
        passed to the last ``update()`` call.
        """
        return self._time_sequence.copy()

    @property
    def current_control(self) -> np.ndarray:
        """Full control vector of the most recently applied step, shape ``(n_controls,)``.

        Returns the control that was last sent to the plant (either via
        ``update()`` or ``step()``).  Use this to get all control inputs
        when the model has more than one (e.g. ``[F_in, F_out]``).
        """
        idx = max(0, self._seq_idx - 1)
        return self._control_sequence[idx, :].copy()

    def solve(
        self, state: np.ndarray, dt: float, t: float = None
    ) -> tuple:
        """Re-solve the OCP for a new state and reset the sequence pointer.

        Call this whenever a fresh state estimate is available (e.g. from
        the EnKF).  Between calls, use ``step()`` to retrieve the next
        pre-computed control input without re-solving.

        Parameters
        ----------
        state : np.ndarray
            Current full state vector (e.g. from EnKF).
        dt : float
            Time step — kept for interface compatibility with PID, not used
            internally (the horizon is fixed in the OCP).
        t : float, optional
            Current time for logging and for building ``time_sequence``.

        Returns
        -------
        tuple[float, float]
            ``(time, u_first)`` — identical signature to ``PID.update()``.
        """
        time_val = 0.0 if t is None else t
        self._last_state = np.asarray(state, dtype=float)

        logger.debug("update() called at t=%.4f, state=%s", time_val, self._last_state)

        u_flat = self._ocp.solve(self._last_state)

        # Reshape flat solution to (N, n_controls) and reset pointer
        self._control_sequence = u_flat.reshape(self._N, self._n_controls)
        self._time_sequence = time_val + (np.arange(1, self._N + 1) * self._dt)
        self._seq_idx = 0

        u_first = float(self._control_sequence[0, 0])
        self._seq_idx = 1

        self._publish_sequence()
        logger.debug(
            "OCP solved at t=%.4f: u[0]=%.4f, horizon=%d steps",
            time_val, u_first, self._N,
        )
        return (time_val, u_first)

    def step(self, t: float = None) -> tuple:
        """Return the next pre-computed control input without re-solving.

        Use this between ``update()`` calls to apply the remaining steps of
        the current optimal sequence.  If the sequence is exhausted, the
        last control value is frozen and a warning is logged (Option A).

        Parameters
        ----------
        t : float, optional
            Current time for logging.

        Returns
        -------
        tuple[float, float]
            ``(time, u)`` — same signature as ``update()``.
        """
        time_val = 0.0 if t is None else t

        if self._seq_idx < self._N:
            u = float(self._control_sequence[self._seq_idx, 0])
            logger.debug(
                "step() at t=%.4f: applying sequence[%d]=%.4f",
                time_val, self._seq_idx, u,
            )
            self._seq_idx += 1
        else:
            u = float(self._control_sequence[-1, 0])
            logger.warning(
                "step() at t=%.4f: sequence exhausted (N=%d), freezing last u=%.4f. "
                "Call update() with a new state to re-solve.",
                time_val, self._N, u,
            )

        self._publish_sequence()
        return (time_val, u)

    def _publish_sequence(self) -> None:
        """Write the current control sequence into the ControlProvider (if set)."""
        if self._provider is None:
            return
        self._provider.replace_variable(
            self._variable_name,
            times=self._time_sequence.copy(),
            values=self._control_sequence.copy(),
        )
        logger.debug(
            "Published sequence to provider '%s' variable '%s'",
            self._provider.name, self._variable_name,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simple 2-state damped spring: dx/dt = Ax + Bu
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])

    states_sym = ca.SX.sym('x', 2)
    control_sym = ca.SX.sym('u', 1)

    def ode_func(x, u):
        return ca.mtimes(A, x) + ca.mtimes(B, u)

    model = CasadiModel(
        states=states_sym,
        controls=control_sym,
        ode=ode_func,
        init_state=[1.0, 0.0],
        dt=0.1,
        T=10.0,
    )

    objective = TrackingObjective(Q=10.0, R=1.0, setpoint=0.0, state_index=0)

    ocp = CasadiOptimalControlProblem(
        model=model,
        objective=objective,
        time_horizon=10.0,
        u_min=-10.0,
        u_max=10.0,
        ipopt_print_level=0,
    )

    mpc = MPCController(ocp)

    # Closed-loop simulation
    N_sim = int(ocp.time_horizon / model.dt)
    x = np.array([1.0, 0.0])
    x_history = [x.copy()]
    u_history = []
    time_sim = [0.0]

    for k in range(N_sim):
        t_k = k * model.dt
        _, u_k = mpc.solve(x, model.dt, t_k)
        u_history.append(u_k)
        res = model.integrator(x0=x, p=[u_k])
        x = np.array(res['xf']).flatten()
        x_history.append(x.copy())
        time_sim.append((k + 1) * model.dt)

    x_history = np.array(x_history)
    u_history = np.array(u_history)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(time_sim, x_history[:, 0], label='x0 (position)')
    axes[0].plot(time_sim, x_history[:, 1], label='x1 (velocity)')
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('States')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].step(time_sim[:-1], u_history, where='post', color='green')
    axes[1].set_ylabel('Control u')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n=== MPC Results ===")
    print(f"Final state: {x_history[-1]}")
    print(f"Total control effort: {np.sum(u_history**2):.4f}")

