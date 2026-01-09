import numpy as np
import casadi as ca


class CasadiModel:
    """
    A generic dynamical system model using CasADi's IDAS/SUNDIALS solver
    for ODE integration.
    """

    def __init__(self,
                 states: ca.SX,
                 controls: ca.SX,
                 ode: callable,
                 init_state: np.ndarray,
                 dt: float,
                 T: float,
                 integrator_type: str = "idas"):
        """
        Parameters
        ----------
        states : ca.SX
            Symbolic state vector.
        controls : ca.SX
            Symbolic input vector.
        ode : callable
            Function f(x,u) → dx/dt.
        init_state : np.ndarray
            Initial state vector.
        dt : float
            Sampling time.
        T : float
            Total simulation horizon.
        integrator_type : str
            One of ["idas", "cvodes"].
        """

        self._states = states
        self._controls = controls
        self._ode = ode
        self._init_state = np.array(init_state, dtype=float)
        self._dt = float(dt)
        self._T = float(T)

        # Dimensions
        self._nStates = states.size1()
        self._nControls = controls.size1()

        # Build CasADi integrator
        self.integrator = self._create_integrator(integrator_type)


    def _create_integrator(self, integrator_type: str):

        f = self._ode(self._states, self._controls)

        dae = {
            'x': self._states,
            'p': self._controls,
            'ode': f
        }

        opts = {
            'tf': self._dt,
            'abstol': 1e-8,
            'reltol': 1e-8,
        }

        return ca.integrator(
            'idas_integrator',
            integrator_type,
            dae,
            opts
        )


    @property
    def dt(self):
        return self._dt

    @property
    def T(self):
        return self._T

    @property
    def init_state(self):
        return self._init_state

    @property
    def nStates(self):
        return self._nStates

    @property
    def nControls(self):
        return self._nControls
    
    @property
    def states(self):
        return self._states
    
    @property
    def controls(self):
        return self._controls
    
    @states.setter
    def states(self, value: ca.SX):
        self._states = value
        self._nStates = value.size1()

    @controls.setter
    def controls(self, value: ca.SX):
        self._controls = value
        self._nControls = value.size1()
    
    @init_state.setter
    def init_state(self, value: np.ndarray):
        self._init_state = np.array(value, dtype=float)

    @dt.setter
    def dt(self, value: float):
        self._dt = float(value)

    @T.setter
    def T(self, value: float):
        self._T = float(value)
