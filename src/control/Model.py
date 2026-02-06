import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from cadet import Cadet
#things to do:
# TODO: think about what is an experiment and what is a model 

class Model(ABC):
    """Abstract base class for dynamical system models.
    
    All model implementations must provide:
    - integrate(x0, u, t_start, t_end): Integrate state from t_start to t_end
    - nStates: Number of state variables
    - nControls: Number of control inputs
    """

    @abstractmethod
    def integrate(self, 
                  x0: np.ndarray, 
                  u: np.ndarray,
                  t_start: float, 
                  t_end: float) -> np.ndarray:
        """Integrate the model from t_start to t_end.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector.
        u : np.ndarray
            Control input vector.
        t_start : float
            Start time of integration.
        t_end : float
            End time of integration.
            
        Returns
        -------
        np.ndarray
            State vector at t_end.
        """
        pass

    @property
    @abstractmethod
    def nStates(self) -> int:
        """Number of state variables."""
        pass

    @property
    @abstractmethod
    def nControls(self) -> int:
        """Number of control inputs."""
        pass


class CadetModel(Model):
    """CADET-based dynamical system model."""

    def __init__(self, 
                 cadet_path: str,
                 init_state: np.ndarray, #TODO delete this and get this from cadet
                 model_path: str,
                 n_states: int,
                 n_controls: int = 0,
                 process_noise: np.ndarray = None,
                 state_indices: list = None): #TODO make experiment handle this
        """
        Parameters
        ----------
        cadet_path : str
            Path to CADET installation.
        model_path : str
            Path to the CADET model file (.h5).
        n_states : int
            Number of state variables to track.
        n_controls : int
            Number of control inputs.
        state_indices : list, optional
            Indices of the states to extract from CADET result.
            If None, uses range(n_states).
        """
        self._nStates = n_states
        self._nControls = n_controls
        self._state_indices = state_indices if state_indices is not None else list(range(n_states))
        if process_noise is not None:
            self._process_noise = process_noise
        else:
            self._process_noise = np.diag([0.000]*self._nStates)

        self._state = np.array(init_state, dtype=float)
        self.model = Cadet(install_path=cadet_path, use_dll=True)
        self.model.filename = model_path
        self.model.load_from_file()
        self.model.save()
        self.model.initialize_simulation()
        self.t_curr = 0.0

    def integrate(self, 
                  x0: np.ndarray, 
                  u: np.ndarray,
                  t_start: float,
                  t_end: float) -> np.ndarray:
        """Integrate the CADET model from t_start to t_end."""
        try:
            # Update state in CADET model
            self.model.update_state(x0, t_start, len(x0))
            self.t_curr = t_start
            
            # Perform simulation step
            ret = self.model.perform_simulation_step(t_end)
            if ret[0].return_code != 0:
                raise RuntimeError(f"CADET simulation step failed: {ret[0].error_message}")

            # Extract state from result
            res = self.model.cadet_runner.res
            full_state = res.last_state_y()
            
            # Extract relevant state components
            state = np.array([full_state[i] for i in self._state_indices])
            self.t_curr = t_end
            
            return state
        
        except Exception as e:
            print(f"CADET integration error: {e}")
            return x0.copy()  # Return unchanged state on error
    
    def end_simulation(self):
        """End the CADET simulation."""
        return self.model.end_simulation()
    
    @property
    def state(self) -> np.ndarray:
        if self._state is None:
            print("Warning: State has not been initialized.")
            #TODO Cadet-Core initialize state in initialization funktion
        return self._state
        

    @property
    def nStates(self) -> int:
        return self._nStates

    @property
    def nControls(self) -> int:
        return self._nControls



class CasadiModel(Model):
    """
    A generic dynamical system model using CasADi's IDAS/SUNDIALS solver
    for ODE integration.
    """

    def __init__(self,
                 states: ca.SX,
                 controls: ca.SX,
                 ode: callable,
                 init_state: np.ndarray,
                 process_noise: np.ndarray = None,
                 dt: float = 0.1,
                 T: float = 1.0,
                 integrator_type: str = "cvodes"):
        """
        Parameters
        ----------
        states : ca.SX
            Symbolic state vector.
        controls : ca.SX
            Symbolic input vector (can be empty: ca.SX.sym('u', 0)).
        ode : callable
            Function f(x, u) → dx/dt.
        init_state : np.ndarray
            Initial state vector.
        dt : float
            Default sampling time for integration.
        T : float
            Total simulation horizon.
        integrator_type : str
            One of ["idas", "cvodes"].
        """
        self._states_sym = states
        self._controls_sym = controls
        self._ode = ode
        self._state = np.array(init_state, dtype=float)
        self._dt = float(dt)
        self._T = float(T)

        # Dimensions
        self._nStates = states.size1()
        self._nControls = controls.size1()
        self._integrator_type = integrator_type

        if process_noise is not None:
            self._process_noise = process_noise
        else:
            self._process_noise = np.diag([0.0]*self._nStates)



        # Build CasADi integrator function
        self._integrator_func = self._create_integrator(integrator_type)

    def _create_integrator(self, integrator_type: str):
        """Create a CasADi integrator for the ODE system with default time step."""
        f = self._ode(self._states_sym, self._controls_sym)

        dae = {
            'x': self._states_sym,
            'p': self._controls_sym,
            'ode': f
        }

        opts = {
            'tf': self._dt,  # Default integration time
            'abstol': 1e-8,
            'reltol': 1e-8,
            'max_num_steps': 10000,
        }

        return ca.integrator(
            'casadi_integrator',
            integrator_type,
            dae,
            opts
        )

    def integrate(self,
                  x0: np.ndarray,
                  u: np.ndarray,
                  t_start: float,
                  t_end: float) -> np.ndarray:
        """
        Integrate the CasADi model from t_start to t_end.
        For autonomous ODEs (time-invariant), uses relative time (t_end - t_start).
        """
        x_current = np.array(x0, dtype=float)
        
        # Handle empty control vector
        if self._nControls == 0:
            u_param = np.array([])
        else:
            u_param = np.array(u, dtype=float).flatten()
        
        # For autonomous systems, integrate for duration (t_end - t_start)
        # Create integrator with correct time span
        duration = t_end - t_start
        
        if abs(duration - self._dt) > 1e-10:
            # Need different integration time - create temporary integrator
            temp_integrator = self._create_integrator_with_tf(duration)
            result = temp_integrator(x0=x_current, p=u_param)
        else:
            result = self._integrator_func(x0=x_current, p=u_param)
        
        x_next = np.array(result['xf']).flatten()
        
        return x_next
    
    def _create_integrator_with_tf(self, tf: float):
        """Create integrator with specific final time (for variable step sizes)."""
        f = self._ode(self._states_sym, self._controls_sym)
        
        dae = {
            'x': self._states_sym,
            'p': self._controls_sym,
            'ode': f
        }
        
        opts = {
            'tf': tf,
            'abstol': 1e-8,
            'reltol': 1e-8,
            'max_num_steps': 10000,
        }
        
        return ca.integrator(
            'temp_integrator',
            self._integrator_type,
            dae,
            opts
        )
    
    @property
    def state(self) -> np.ndarray:
        return self._state
    
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def T(self) -> float:
        return self._T

    @property
    def init_state(self) -> np.ndarray:
        return self._init_state

    @property
    def nStates(self) -> int:
        return self._nStates

    @property
    def nControls(self) -> int:
        return self._nControls
    
    @property
    def states_sym(self) -> ca.SX:
        """Symbolic state vector."""
        return self._states_sym
    
    @property
    def controls_sym(self) -> ca.SX:
        """Symbolic control vector."""
        return self._controls_sym
    
    @property
    def ode(self) -> callable:
        """ODE function f(x, u)."""
        return self._ode
    
    @init_state.setter
    def init_state(self, value: np.ndarray):
        self._init_state = np.array(value, dtype=float)

    @dt.setter
    def dt(self, value: float):
        self._dt = float(value)
        # Rebuild integrator with new dt
        self._integrator_func = self._create_integrator(self._integrator_type)

    @T.setter
    def T(self, value: float):
        self._T = float(value)
