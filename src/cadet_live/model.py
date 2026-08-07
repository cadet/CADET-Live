from abc import ABC, abstractmethod
from collections.abc import Callable

import casadi as ca
import numpy as np
from cadet import Cadet


class Model(ABC):
    """Abstract base class for stateful dynamical-system models.

    Implementations advance their current state one time step at a time and keep
    track of their own model time. Models without control inputs inherit
    ``nControls == 0``.
    """

    @abstractmethod
    def integrate_timestep(
        self,
        t_end: float | None = None,
        controls: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the model state and return the resulting state."""
        raise NotImplementedError

    @abstractmethod
    def update_state(self, x0: np.ndarray, t_start: float) -> None:
        """Reset the model state and its current time."""
        raise NotImplementedError

    @property
    @abstractmethod
    def nStates(self) -> int:
        """Number of state variables."""
        raise NotImplementedError

    @property
    def nControls(self) -> int:
        """Number of control inputs."""
        return 0


class CadetModel(Model):
    """CADET-based dynamical-system model."""

    def __init__(
        self,
        cadet_path: str,
        init_state: np.ndarray,
        model_path: str,
        n_states: int,
        n_controls: int = 0,
        state_indices: list[int] | None = None,
    ) -> None:
        """Initialize a CADET model.

        Parameters
        ----------
        cadet_path : str
            Path to the CADET installation.
        init_state : np.ndarray
            Initial state vector.
        model_path : str
            Path to the CADET model file.
        n_states : int
            Number of state variables to track.
        n_controls : int, optional
            Number of control inputs.
            State process-noise covariance matrix.
        state_indices : list[int], optional
            State indices to extract from the CADET result.
        """
        self._nStates = n_states
        self._nControls = n_controls
        self._state_indices = (
            state_indices if state_indices is not None else list(range(n_states))
        )

        self._state = np.array(init_state, dtype=float)
        self.model = Cadet(install_path=cadet_path, use_dll=True)
        self.model.filename = model_path
        self.model.load_from_file()
        self.model.save()
        self.model.initialize_simulation()
        self.t_curr = 0.0

    def update_state(self, x0: np.ndarray, t_start: float) -> None:
        """Update the state in the CADET model."""
        try:
            x0_full = np.zeros(7)
            x0_full[self._state_indices] = x0
            x0_full[4] = 1.0
            self.model.update_state(x0_full, t_start, len(x0_full))
            self._state = np.array(x0, dtype=float)
            self.t_curr = t_start
        except Exception as error:
            print(f"CADET state update error: {error}")

    def integrate_timestep(
        self,
        t_end: float | None = None,
        controls: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate the CADET model to an explicit end time."""
        if t_end is None:
            raise ValueError("CadetModel requires an explicit t_end.")
        if controls is not None:
            raise NotImplementedError(
                "Passing controls directly is not supported by CadetModel."
            )

        try:
            result = self.model.perform_simulation_step(t_end)
            if result[0].return_code != 0:
                raise RuntimeError(
                    f"CADET simulation step failed: {result[0].error_message}"
                )

            full_state = self.model.cadet_runner.res.last_state_y()
            state = np.array([full_state[index] for index in self._state_indices])
            self._state = state
            self.t_curr = t_end
            return state.copy()
        except Exception as error:
            print(f"CADET integration error: {error}")
            return self._state.copy()

    def end_simulation(self) -> object:
        """End the CADET simulation."""
        return self.model.end_simulation()

    @property
    def state(self) -> np.ndarray:
        """Current state."""
        return self._state.copy()

    @property
    def nStates(self) -> int:
        """Number of state variables."""
        return self._nStates

    @property
    def nControls(self) -> int:
        """Number of control inputs."""
        return self._nControls


class CasadiModel(Model):
    """Dynamical-system model using a CasADi integrator."""

    _SUPPORTED_INTEGRATORS = {"idas", "cvodes"}

    def __init__(
        self,
        states: ca.SX,
        ode: Callable[..., ca.SX],
        init_state: np.ndarray,
        controls: ca.SX | None = None,
        dt: float = 0.1,
        simulation_time: float = 1.0,
        integrator_type: str = "cvodes",
    ) -> None:
        """Initialize a CasADi model.

        Parameters
        ----------
        states : ca.SX
            Symbolic state vector.
        ode : callable
            Function ``f(x)`` if ``controls`` is ``None`` and ``f(x, u)``
            otherwise.
        init_state : np.ndarray
            Initial state vector.
        controls : ca.SX, optional
            Symbolic input vector. An empty symbolic vector is permitted.
            State process-noise covariance matrix.
        dt : float
            Default integration time step.
        simulation_time : float
            Absolute end time used by :meth:`run_integration`.
        integrator_type : str
            One of ``"idas"`` and ``"cvodes"``.
        """
        if integrator_type not in self._SUPPORTED_INTEGRATORS:
            supported = ", ".join(sorted(self._SUPPORTED_INTEGRATORS))
            raise ValueError(
                f"Unsupported integrator_type {integrator_type!r}; expected {supported}."
            )

        self._states_sym = states
        self._controls_sym = controls
        self._ode = ode
        self._integrator_type = integrator_type
        self._nStates = int(states.numel())
        self._nControls = 0 if controls is None else int(controls.numel())

        self._dt = self._validate_positive_time(dt, "dt")
        self._T = self._validate_nonnegative_time(simulation_time, "simulation_time")
        self.t_curr = 0.0

        self._init_state = self._as_vector(init_state, self._nStates, "init_state")
        self._state = self._init_state.copy()
        self._controls = np.zeros(self._nControls, dtype=float)
        self._trajectory: list[tuple[float, np.ndarray]] = [ (self.t_curr, self._state.copy())]

        self._integrator_func = self._create_integrator(self._dt)

    @staticmethod
    def _validate_positive_time(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite value greater than zero.")
        return value

    @staticmethod
    def _validate_nonnegative_time(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value.")
        return value

    @staticmethod
    def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != size:
            raise ValueError(f"{name} must contain {size} values, got {vector.size}.")
        return vector.copy()

    def _create_integrator(self, integration_time: float) -> ca.Function:
        """Create an integrator for one interval of ``integration_time``."""
        if self._controls_sym is None:
            ode_expression = self._ode(self._states_sym)
            dae = {"x": self._states_sym, "ode": ode_expression}
        else:
            ode_expression = self._ode(self._states_sym, self._controls_sym)
            dae = {
                "x": self._states_sym,
                "p": self._controls_sym,
                "ode": ode_expression,
            }

        options = {
            "abstol": 1e-8,
            "reltol": 1e-8,
            "max_num_steps": 10000,
        }
        return ca.integrator(
            "casadi_integrator",
            self._integrator_type,
            dae,
            0.0,
            integration_time,
            options,
        )

    def _get_next_timepoint(self, stop_time: float | None = None) -> float:
        """Return the next model time without advancing the model state."""
        next_time = self.t_curr + self._dt
        if stop_time is None:
            return next_time

        stop_time = self._validate_nonnegative_time(stop_time, "stop_time")
        if stop_time < self.t_curr and not np.isclose(stop_time, self.t_curr):
            raise ValueError("stop_time must not be earlier than the current model time.")
        return min(next_time, stop_time)

    def _get_controls(self, controls: np.ndarray | None) -> np.ndarray:
        """Validate supplied controls and remember the latest values."""
        if controls is None:
            return self._controls.copy()

        control_vector = self._as_vector(controls, self._nControls, "controls")
        self._controls = control_vector
        return control_vector.copy()

    def update_state(self, x0: np.ndarray, t_start: float) -> None:
        """Reset the internal state, current time, and trajectory recorder."""
        self._state = self._as_vector(x0, self._nStates, "x0")
        self.t_curr = self._validate_nonnegative_time(t_start, "t_start")
        self._trajectory = [(self.t_curr, self._state.copy())]

    def integrate_timestep(
        self,
        t_end: float | None = None,
        controls: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance by one step or integrate to an explicit end time."""
        if t_end is None:
            t_end = self._get_next_timepoint()
        else:
            t_end = self._validate_nonnegative_time(t_end, "t_end")

        integration_time = t_end - self.t_curr
        if integration_time <= 0.0:
            raise ValueError("t_end must be later than the current model time.")

        control_vector = self._get_controls(controls)
        integrator = self._integrator_func
        if not np.isclose(integration_time, self._dt, rtol=1e-12, atol=1e-15):
            integrator = self._create_integrator(integration_time)

        arguments = {"x0": self._state}
        if self._controls_sym is not None:
            arguments["p"] = control_vector
        result = integrator(**arguments)

        self._state = np.asarray(result["xf"], dtype=float).reshape(-1)
        self.t_curr = t_end
        self._trajectory.append((self.t_curr, self._state.copy()))
        return self._state.copy()

    def run_integration(self, controls: np.ndarray | None = None) -> np.ndarray:
        """Integrate from the current time to the configured end time."""
        while self.t_curr < self._T:
            next_time = self._get_next_timepoint(stop_time=self._T)
            self.integrate_timestep(t_end=next_time, controls=controls)
        return self.state

    def plot_trajectory(self) -> None:
        """Plot every state recorded during integration."""
        import matplotlib.pyplot as plt

        states = np.vstack([state for _, state in self._trajectory])
        plt.plot(self.solution_times, states)
        plt.show()

    @property
    def state(self) -> np.ndarray:
        """Current state."""
        return self._state.copy()

    @property
    def solution_times(self) -> np.ndarray:
        """Recorded model times."""
        return np.array([time for time, _ in self._trajectory], dtype=float)

    @property
    def trajectory(self) -> tuple[tuple[float, np.ndarray], ...]:
        """Recorded time-state pairs."""
        return tuple((time, state.copy()) for time, state in self._trajectory)

    @property
    def dt(self) -> float:
        """Default integration time step."""
        return self._dt

    @property
    def T(self) -> float:
        """Configured absolute end time."""
        return self._T

    @property
    def init_state(self) -> np.ndarray:
        """Initial state."""
        return self._init_state.copy()

    @property
    def nStates(self) -> int:
        """Number of state variables."""
        return self._nStates

    @property
    def nControls(self) -> int:
        """Number of control inputs."""
        return self._nControls

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = self._validate_positive_time(value, "dt")
        self._integrator_func = self._create_integrator(self._dt)

    @T.setter
    def T(self, value: float) -> None:
        self._T = self._validate_nonnegative_time(value, "T")
