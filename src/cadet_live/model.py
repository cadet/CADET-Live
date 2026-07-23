import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from cadet import Cadet

# TODO: think about what is an experiment and what is a model
# TODO: change from indicies to cadet variable and state dict
# TODO: add optinal indices to update state
class Model(ABC):
    @abstractmethod
    def integrate(self,
                  t_end: float) -> np.ndarray:
        """Integrate the model from t_start to t_end."""
        pass

    def update_state(self, x0: np.ndarray, t_start: float):
        """Optional method to update the model's internal state before integration."""
        pass

    @property
    @abstractmethod
    def nStates(self) -> int:
        """Number of state variables."""
        pass

    @property
    def nControls(self) -> int:
        """Optional Number of control inputs."""
        pass


class CadetModel(Model):

    def __init__(self, 
                 cadet_path: str, #TODO be flexible also allow the direct model object 
                 init_state: np.ndarray,
                 model_path: str,
                 n_states: int,
                 n_controls: int = 0,
                 process_noise: np.ndarray = None,
                 state_indices: list = None):
        
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
    
    def update_state(self, x0: np.ndarray, t_start: float):
        try:
            X0_full = np.zeros(7)
            X0_full[self._state_indices] = x0
            X0_full[4] = 1.0 # Assuming index 4 is a required state (e.g., volume)
            self.model.update_state(X0_full, t_start, len(X0_full))
            self.t_curr = t_start
        
        except Exception as e:
            print(f"CADET state update error: {e}")

    def integrate(self,
                  t_end: float) -> np.ndarray:
        """Integrate the CADET model from t_start to t_end."""
        try:

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
    def __init__(self,
                 states: ca.SX,
                 controls: ca.SX,
                 ode: callable,
                 init_state: np.ndarray,
                 process_noise: np.ndarray = None,
                 dt: float = 0.1,
                 T: float = 1.0,
                 integrator_type: str = "cvodes"):
        
        self._states_sym = states
        self._controls_sym = controls
        self._ode = ode
        self._init_state = np.array(init_state, dtype=float)
        self._state = np.array(init_state, dtype=float)
        self._dt = float(dt)
        self._T = float(T)
        self.t_curr = 0.0

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

        f = self._ode(self._states_sym, self._controls_sym)

        dae = {
            'x': self._states_sym,
            'p': self._controls_sym,
            'ode': f
        }

        opts = {
            'abstol': 1e-8,
            'reltol': 1e-8,
            'max_num_steps': 10000,
        }

        return ca.integrator(
            'casadi_integrator',
            integrator_type,
            dae,
            0, self._dt,
            opts
        )

    def update_state(self, x0: np.ndarray, t_start: float):
        self._state = np.array(x0, dtype=float)
        self.t_curr = t_start

    def integrate(self, t_end: float) -> np.ndarray:
        # Handle empty control vector
        if self._nControls == 0:
            u_param = np.array([])
        else:
            u_param = np.zeros(self._nControls)

        duration = t_end - self.t_curr

        if abs(duration - self._dt) > 1e-10:
            temp_integrator = self._create_integrator_with_tf(duration)
            result = temp_integrator(x0=self._state, p=u_param)
        else:
            result = self._integrator_func(x0=self._state, p=u_param)

        x_next = np.array(result['xf']).flatten()
        self.t_curr = t_end

        return x_next
    
    def _create_integrator_with_tf(self, tf: float):
        f = self._ode(self._states_sym, self._controls_sym)
        
        dae = {
            'x': self._states_sym,
            'p': self._controls_sym,
            'ode': f
        }
        
        opts = {
            'abstol': 1e-8,
            'reltol': 1e-8,
            'max_num_steps': 10000,
        }

        return ca.integrator(
            'temp_integrator',
            self._integrator_type,
            dae,
            0, tf,
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

    @dt.setter
    def dt(self, value: float):
        self._dt = float(value)
        # Rebuild integrator with new dt
        self._integrator_func = self._create_integrator(self._integrator_type)

    @T.setter
    def T(self, value: float):
        self._T = float(value)
