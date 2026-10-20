import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
from cadet import Cadet


@dataclass
class SimulationResult:
    """Time points, states, and controls returned by a simulation."""

    time: np.ndarray
    state: np.ndarray
    controls: np.ndarray | None = None


class Model(ABC):
    """Base interface for models with a current state and time."""

    @abstractmethod
    def set_state(self, state: np.ndarray, time: float = 0.0) -> None:
        """Set the current state and model time."""
        raise NotImplementedError

    @abstractmethod
    def simulate_timestep( self, t_end: float, controls: np.ndarray | None = None ) -> np.ndarray:
        """Advance the model state and return the resulting state."""
        raise NotImplementedError

    def run_simulation(
        self, times: np.ndarray, controls: np.ndarray | None = None) -> SimulationResult:
        """Advance through strictly increasing times starting at the current time."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError( "times must be a nonempty, one-dimensional array of finite values.")

        if times[0] != self.t_curr:
            raise ValueError( f"The first time point must equal the current time {self.t_curr}.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing.")

        states = [self.state]
        for t in times[1:]:
            state = self.simulate_timestep(t_end=t, controls=controls)
            states.append(state)

        return SimulationResult( time=np.asarray(times), state=np.asarray(states), controls=controls)

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Return a copy of the current state."""
        raise NotImplementedError


class CadetModel(Model):
    """Stateful CADET model using the CADET C API."""

    def __init__(self, model_path: str, cadet_path: str | None = None) -> None:
        """Load a single-unit CADET model from a file."""
        install_path = cadet_path or os.getenv("CADET_INSTALL_PATH") or sys.prefix

        self.model = Cadet(install_path=install_path, use_dll=True)
        self.model.filename = model_path

        self.model.load_from_file()
        self.model.save()

        self.t_curr = 0.0

        if self.model.root.input.model.nunits != 1:
            raise NotImplementedError("CadetModel currently supports state updates only for single-unit models.")

    def initialise_simulation(self) -> None:
        """Initialize the CADET simulator."""
        result = self.model.initialize_simulation()

        if result.return_code != 0:
            raise RuntimeError(result.error_message)

    def set_state(self, state: np.ndarray, time: float = 0.0) -> None:
        """Update the simulator state and current model time."""
        result = self.model.update_state(0, state, time) 
                # unit_type is always zero

        if result.return_code != 0:
            raise RuntimeError(result.error_message)

        self._state = np.asarray(state).copy()
        self.t_curr = time

    def simulate_timestep(self, t_end: float, controls: np.ndarray | None = None) -> np.ndarray:
        """Advance the simulator to the requested absolute time."""
        if controls is not None:
            raise NotImplementedError("Passing controls directly is not supported by CadetModel.")
        
        result = self.model.perform_simulation_step(t_end)

        if result[0].return_code != 0:
            raise RuntimeError(result[0].error_message)

        full_state = self.model.cadet_runner.res.last_state_y()

        self._state = full_state.copy()
        self.t_curr = result[1]

        return self._state.copy()

    def end_simulation(self) -> object:
        """End the CADET simulation."""
        return self.model.end_simulation()

    @property
    def state(self) -> np.ndarray:
        """Current state."""
        return self._state.copy()


class CasadiModel(Model):
    """Stateful ODE model integrated with CasADi."""

    def __init__(self, states: ca.SX, ode: Callable[..., ca.SX], controls: ca.SX | None = None) -> None:
        """Store symbolic states, dynamics, and optional controls."""
        self._states_sym = states
        self._controls_sym = controls
        self._ode = ode

        self._state = np.zeros(states.numel())
        self.t_curr = 0.0

    def set_state(self, state: np.ndarray, time: float = 0.0) -> None:
        """Set the current state and model time."""
        state = np.asarray(state, dtype=float).reshape(-1)
        
        if state.size != self._states_sym.numel():
            raise ValueError(f"state must contain {self._states_sym.numel()} values.")
        if not np.isfinite(time):
            raise ValueError("time must be finite.")
        
        self._state = state.copy()
        self.t_curr = time

    def create_simulator(self, dt: float) -> ca.Function:
        """Create an integrator for one time interval."""
        
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

        self._integrator_func = ca.integrator("casadi_integrator", "idas", dae, 0.0, dt)
        

    def simulate_timestep(self, t_end: float, controls: np.ndarray | None = None) -> np.ndarray:
        """Advance the ODE to the requested absolute time."""
        dt = t_end - self.t_curr
        if not np.isfinite(t_end) or dt <= 0:
            raise ValueError("t_end must be finite and later than the current time.")
        integrator = self.create_simulator(dt)

        arguments = {"x0": self._state}

        if self._controls_sym is not None:
            arguments["p"] = controls

        results = integrator(**arguments)

        self._state = np.asarray(results["xf"]).reshape(-1)
        self.t_curr = t_end

        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state."""
        return self._state.copy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example = {
        "model_path": (
            Path(__file__).resolve().parents[3]
            / "examples/models/CADET/only_cstr_no_reac.h5"
        ),
    }

    x = ca.SX.sym("x")
    model = CasadiModel(states=x, ode=lambda state: -state)
    model.set_state(np.array([1.0]))
    result = model.run_simulation(np.linspace(0.0, 3.0, 31))

    plt.plot(result.time, result.state[:, 0], label="CasADi state")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.show()

    model = CadetModel(str(example["model_path"]))
    model.initialise_simulation()
    try:
        initial_state = model.model.cadet_runner.res.last_state_y().copy()
        model.set_state(initial_state)
        result = model.run_simulation(np.linspace(0.0, 1.0, 21))
    finally:
        model.end_simulation()

    plt.plot(result.time, result.state[:, 0], label="CADET state 0")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.show()
