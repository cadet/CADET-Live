import numpy as np
import pandas as pd
from typing import Callable, Optional, Union, List, Dict


from Provider import MeasurementProvider, DFProvider, Mesuremtents
from Model import Model


class EnKalmanFilter:
    """
    Ensemble Kalman Filter for state estimation.

    """

    def __init__(self,
                 model: Model,
                 ensemble_size: int,
                 initial_covariance: np.ndarray,
                 observation_func: Optional[Callable] = None,
                 providers: Optional[List[MeasurementProvider]] = None,
                 random_seed: Optional[int] = None):#TODO delete this
        """
        Parameters
        ----------
        model : Model
            Dynamical system model (CadetModel or CasadiModel).
        ensemble_size : int
            Number of ensemble members.
        initial_covariance : np.ndarray
            Initial state covariance matrix (nStates x nStates).
        observation_func : callable, optional
            Nonlinear observation function h(x) -> y.
        providers : List[MeasurementProvider], optional
            List of measurement providers for multiple measurement sources.
        random_seed : int, optional
            Random seed for reproducibility.
        """
        self.model = model
        self.N_ens = ensemble_size
        self.cov = np.atleast_2d(initial_covariance)
        
        # Set up random number generator
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize state
        self.state = model.state
        
        # Handle multiple providers
        self._providers: Dict[str, MeasurementProvider] = {}
        self._provider_obs_indices: Dict[str, List[int]] = {}  # Which state indices each provider observes
        
        if providers is not None:
            for prov in providers:
                self.add_provider(prov)
            
        # Build measurement noise from providers if not explicitly given
        self.measurement_noise = self._build_measurement_noise_from_providers()
        
        # Set up observation model
        if observation_func is not None:
            self._obs_func = observation_func
        else:
            # Default: observe all states
            self._obs_func = lambda x: x
        
        # Determine measurement dimension
        test_obs = self._obs_func(self.state)
        self._nMeas = len(np.atleast_1d(test_obs))
        
        # Initialize ensemble around initial state
        self.X_ens = self.rng.multivariate_normal(
            self.state, 
            self.cov, 
            self.N_ens
        )
        
        # Track current time
        self.t_current = 0.0
        
        # Control input (default: zero)
        self._control = np.zeros(model.nControls)
    
    
    def add_provider(self, 
                     provider: MeasurementProvider,
                     observed_state_indices: Optional[List[int]] = None):
        """
        Add a measurement provider.
        
        Parameters
        ----------
        provider : MeasurementProvider
            The measurement provider to add.
        observed_state_indices : List[int], optional
            Which state indices this provider observes.
            If None, assumes sequential indices based on provider order.
        """
        self._providers[provider.name] = provider
        if observed_state_indices is not None:
            self._provider_obs_indices[provider.name] = observed_state_indices
    
    def remove_provider(self, name: str):
        """Remove a measurement provider by name."""
        if name in self._providers:
            del self._providers[name]
            if name in self._provider_obs_indices:
                del self._provider_obs_indices[name]
    
    def get_provider(self, name: str) -> Optional[MeasurementProvider]:
        """Get a measurement provider by name."""
        return self._providers.get(name)
    
    @property
    def providers(self) -> Dict[str, MeasurementProvider]:
        """Get all registered measurement providers."""
        return self._providers
    
    @property
    def provider_names(self) -> List[str]:
        """Get names of all registered providers."""
        return list(self._providers.keys())
    
    def _build_measurement_noise_from_providers(self) -> np.ndarray:
        """Build combined measurement noise matrix from all providers."""
        if not self._providers:
            return np.array([[0.01]])  # Default fallback
        
        # Collect noise values from all providers
        noise_values = []
        for name, prov in self._providers.items():
            noise = np.atleast_2d(prov.noise)
            # Flatten diagonal if it's a matrix
            if noise.shape[0] == noise.shape[1]:
                noise_values.extend(np.diag(noise))
            else:
                noise_values.extend(noise.flatten())
        
        # Build diagonal noise matrix
        return np.diag(noise_values) if noise_values else np.array([[0.01]])
        
    def get_all_measurements(self, timepoint: float) -> np.ndarray:
        """
        Get combined measurements from all providers at a timepoint.
        
        Parameters
        ----------
        timepoint : float
            Time at which to retrieve measurements.
            
        Returns
        -------
        np.ndarray
            Combined measurement vector from all providers.
        """
        measurements = []
        for name, prov in self._providers.items():
            try:
                meas = prov.getMeasurement(timepoint)
                measurements.extend(np.atleast_1d(meas).flatten())
            except ValueError:
                # No measurement available at this time from this provider
                pass
        
        return np.array(measurements) if measurements else None
    
    def get_measurements(self, 
                                       timepoint: float, 
                                       method: str = 'nearest') -> np.ndarray:
        """
        Get measurements with interpolation for missing timepoints.
        
        Parameters
        ----------
        timepoint : float
            Time at which to retrieve measurements.
        method : str
            Interpolation method: 'nearest', 'linear', or 'zero' (use zero if missing).
            
        Returns
        -------
        np.ndarray
            Combined measurement vector.
        """
        measurements = []
        for name, prov in self._providers.items():
            meas = self._get_measurement(prov, timepoint, method)
            if meas is not None:
                measurements.extend(np.atleast_1d(meas).flatten())
        
        return np.array(measurements) if measurements else None
    
    def _get_measurement(self, 
                                       provider: MeasurementProvider,
                                       timepoint: float,
                                       method: str) -> Optional[np.ndarray]:
        """Get measurement from single provider with interpolation."""
        try:
            return provider.getMeasurement(timepoint)
        except ValueError:
            # Timepoint not available, try interpolation
            available_times = provider.times
            if len(available_times) == 0:
                return None
            
            if method == 'nearest':
                idx = np.argmin(np.abs(available_times - timepoint))
                nearest_time = available_times[idx]
                return provider.getMeasurement(nearest_time)
            elif method == 'linear':
                pass #TODO implement linear interpolation between closest time points
            
            elif method == 'zero':
                return None
            
            return None
    
    def get_next_measurement_time(self):
       pass #TODO implement method to get next measurement time after a given time, optionally for a specific provider
    
    def get_all_measurement_times(self) -> np.ndarray:
        """Get sorted array of all unique measurement times from all providers."""
        all_times = set()
        for prov in self._providers.values():
            all_times.update(prov.times)
        return np.array(sorted(all_times))

    def set_control(self, u: np.ndarray):
        """Set the control input for the next propagation step."""
        self._control = np.array(u, dtype=float).flatten()

    def propagate(self, 
                  t_start: float, 
                  t_end: float,
                  u: Optional[np.ndarray] = None):
        """
        Propagate the ensemble from t_start to t_end.
        
        Parameters
        ----------
        t_start : float
            Start time.
        t_end : float
            End time.
        u : np.ndarray, optional
            Control input. If None, uses previously set control.
        """
        if u is not None:
            self._control = np.array(u, dtype=float).flatten()
        
        for i in range(self.N_ens):
            x_prev = self.X_ens[i, :]
            
            # Integrate using model
            x_next = self.model.integrate(
                x0=x_prev,
                u=self._control,
                t_start=t_start,
                t_end=t_end
            )
            
            # Add process noise
            w = self.rng.multivariate_normal(
                np.zeros(self.model.nStates), 
                self.model._process_noise
            )
            
            self.X_ens[i, :] = x_next + w
        
        # Update current state estimate
        self.state = np.mean(self.X_ens, axis=0)
        self.t_current = t_end

    def update(self,
               measurement: Union[np.ndarray, float],
               timepoint: Optional[float] = None):
        """
        Update the ensemble based on a measurement.
        
        Parameters
        ----------
        measurement : np.ndarray or float
            Measurement vector or scalar.
        timepoint : float, optional
            Time of measurement (for logging purposes).
        """
        y_meas = np.atleast_1d(measurement).flatten()
        
        # Generate ensemble of predicted observations
        Y_ens = np.zeros((self.N_ens, self._nMeas))
        for i in range(self.N_ens):
            y_pred = self._obs_func(self.X_ens[i, :])
            v = self.rng.multivariate_normal(
                np.zeros(self._nMeas), 
                self.measurement_noise
            )
            Y_ens[i, :] = np.atleast_1d(y_pred).flatten() + v
        
        # Calculate ensemble means
        x_mean = np.mean(self.X_ens, axis=0)
        y_mean = np.mean(Y_ens, axis=0)
        
        # Calculate deviations from mean
        X_dev = self.X_ens - x_mean
        Y_dev = Y_ens - y_mean
        
        # Cross-covariance and measurement covariance
        P_xy = (X_dev.T @ Y_dev) / (self.N_ens - 1)
        P_yy = (Y_dev.T @ Y_dev) / (self.N_ens - 1) + self.measurement_noise
        
        # Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)
        
        # Update each ensemble member
        for i in range(self.N_ens):
            innovation = y_meas - Y_ens[i, :]
            self.X_ens[i, :] += (K @ innovation).flatten()
        
        # Update state estimate and covariance
        self.state = np.mean(self.X_ens, axis=0)
        self.cov = np.cov(self.X_ens.T) + 1e-9 * np.eye(self.model.nStates)

    def update_from_provider(self, timepoint: float):
        """
        Update using measurement from the configured provider(s).
        
        Parameters
        ----------
        timepoint : float
            Time at which to retrieve measurement.
        """
        if not self._providers:
            raise ValueError("No measurement providers configured.")
        
        y_meas = self.get_all_measurements(timepoint)
        if y_meas is None or len(y_meas) == 0:
            raise ValueError(f"No measurements available at time {timepoint}.")
        
        self.update(y_meas, timepoint)
    
    def update_from_providers_interpolated(self, 
                                            timepoint: float,
                                            method: str = 'nearest'):
        """
        Update using interpolated measurements from all providers.
        
        Parameters
        ----------
        timepoint : float
            Time at which to retrieve measurement.
        method : str
            Interpolation method: 'nearest', 'linear', or 'zero'.
        """
        if not self._providers:
            raise ValueError("No measurement providers configured.")
        
        y_meas = self.get_measurements(timepoint, method)
        if y_meas is None or len(y_meas) == 0:
            raise ValueError(f"No measurements available near time {timepoint}.")
        
        self.update(y_meas, timepoint)
    
    def step_with_providers(self,
                            t_start: float,
                            t_end: float,
                            u: Optional[np.ndarray] = None,
                            interpolation: str = 'nearest') -> np.ndarray:
        """
        Perform EnKF step using measurements from configured providers.
        
        Parameters
        ----------
        t_start : float
            Start time.
        t_end : float
            End time (measurement time).
        u : np.ndarray, optional
            Control input during propagation.
        interpolation : str
            Interpolation method for measurements.
            
        Returns
        -------
        np.ndarray
            Updated state estimate.
        """
        self.propagate(t_start, t_end, u)
        self.update_from_providers_interpolated(t_end, interpolation)
        return self.state.copy()
    
    def run_filter(self,
                   t_start: float = 0.0,
                   t_end: Optional[float] = None,
                   dt: Optional[float] = None,
                   use_measurement_times: bool = True,
                   interpolation: str = 'nearest') -> Dict[str, np.ndarray]:
        """
        Run the EnKF over a time range using configured providers.
        
        Parameters
        ----------
        t_start : float
            Start time.
        t_end : float, optional
            End time. If None, uses last measurement time.
        dt : float, optional
            Time step for propagation. If None, uses measurement times.
        use_measurement_times : bool
            If True, step at measurement times. If False, use fixed dt.
        interpolation : str
            Interpolation method for measurements.
            
        Returns
        -------
        Dict with 'times', 'states', 'covariances' arrays.
        """
        if not self._providers:
            raise ValueError("No measurement providers configured.")
        
        # Determine time grid
        all_meas_times = self.get_all_measurement_times()
        
        if t_end is None:
            t_end = all_meas_times[-1] if len(all_meas_times) > 0 else t_start
        
        if use_measurement_times:
            # Use measurement times within range
            time_grid = all_meas_times[(all_meas_times > t_start) & (all_meas_times <= t_end)]
        else:
            if dt is None:
                dt = self.model.dt if hasattr(self.model, 'dt') else 0.1
            time_grid = np.arange(t_start + dt, t_end + dt/2, dt)
        
        # Run filter
        times = [t_start]
        states = [self.state.copy()]
        covariances = [self.cov.copy()]
        
        t_current = t_start
        for t_next in time_grid:
            state_est = self.step_with_providers(
                t_start=t_current,
                t_end=t_next,
                interpolation=interpolation
            )
            times.append(t_next)
            states.append(state_est.copy())
            covariances.append(self.cov.copy())
            t_current = t_next
        
        return {
            'times': np.array(times),
            'states': np.array(states),
            'covariances': np.array(covariances)
        }

    def step(self,
             t_start: float,
             t_end: float,
             measurement: Union[np.ndarray, float],
             u: Optional[np.ndarray] = None):
        """
        Perform a complete EnKF step: propagate and update.
        
        Parameters
        ----------
        t_start : float
            Start time.
        t_end : float
            End time (measurement time).
        measurement : np.ndarray or float
            Measurement at t_end.
        u : np.ndarray, optional
            Control input during propagation.
            
        Returns
        -------
        np.ndarray
            Updated state estimate.
        """
        self.propagate(t_start, t_end, u)
        self.update(measurement, t_end)
        return self.state.copy()


    @property
    def nMeas(self) -> int:
        """Number of measurement variables."""
        return self._nMeas

    @property
    def nEnsembles(self) -> int:
        """Number of ensemble members."""
        return self.N_ens

    def get_state(self, idx: Optional[int] = None) -> np.ndarray:
        """
        Get current state estimate.
        
        Parameters
        ----------
        idx : int, optional
            Index of specific state. If None, returns full state.
        """
        if idx is None:
            return self.state.copy()
        return self.state[idx]

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance estimate."""
        return self.cov.copy()
    
    def get_ensemble(self) -> np.ndarray:
        """Get current ensemble matrix (N_ens x nStates)."""
        return self.X_ens.copy()


if __name__ == "__main__":
    import casadi as ca
    import matplotlib.pyplot as plt
    from Model import CasadiModel, CadetModel

    print("="*60)
    print("Example 1: EnKF with CasADi Model and Multiple Providers")
    print("="*60)

    # Define CasADi model: simple CSTR dynamics
    X = ca.SX.sym('X')  # Biomass
    S = ca.SX.sym('S')  # Substrate
    V = ca.SX.sym('V')  # Volume
    states = ca.vertcat(X, S, V)
    
    # No external control for this example
    u = ca.SX.sym('u', 0)
    
    # Parameters
    S_in = 0.5
    F_out = 1.0
    F_in = 1.0
    
    # ODE function
    def cstr_ode(x, u):
        X, S, V = x[0], x[1], x[2]
        dV_dt = F_in - F_out
        dX_dt = (-F_out * X - dV_dt * X) / V
        dS_dt = (F_in * S_in - F_out * S - dV_dt * S) / V
        return ca.vertcat(dX_dt, dS_dt, dV_dt)
    
    # Create CasADi model
    dt = 0.1
    T = 1.0
    X0 = np.array([1.0, 0.5, 2.0])
    
    casadi_model = CasadiModel(
        states=states,
        controls=u,
        ode=cstr_ode,
        init_state=X0,
        dt=dt,
        T=T
    )
    
    # Generate synthetic measurements for MULTIPLE sensors
    np.random.seed(42)
    time_points = []
    true_states = [X0.copy()]
    
    # Separate measurement lists for different "sensors"
    biomass_measurements = []  # Sensor 1: measures X (biomass)
    substrate_measurements = []  # Sensor 2: measures S (substrate)
    
    x_current = X0.copy()
    for t_idx in range(int(T / dt)):
        t = t_idx * dt
        time_points.append(t)
        
        # Simulate true system
        x_next = casadi_model.integrate(x_current, np.array([]), t, t + dt)
        true_states.append(x_next.copy())
        
        # Create noisy measurements from different sensors
        # Sensor 1: Biomass (e.g., OD sensor)
        biomass_meas = x_next[0] + np.random.normal(0, 0.05)
        biomass_measurements.append((t + dt, biomass_meas))
        
        # Sensor 2: Substrate (e.g., concentration sensor) - measured less frequently
        if t_idx % 2 == 0:  # Every other time step
            substrate_meas = x_next[1] + np.random.normal(0, 0.03)
            substrate_measurements.append((t + dt, substrate_meas))
        
        x_current = x_next
    
    # Create DataFrames for providers
    df_biomass = pd.DataFrame({
        "X": [biomass_measurements]
    })
    
    df_substrate = pd.DataFrame({
        "S": [substrate_measurements]
    })
    
    # Create MeasurementProviders
    biomass_provider = DFProvider(
        name="BiomassOD",
        DataFrame=df_biomass,
        y_columns=["X"],
        noise=np.array([[0.05**2]])  # Variance
    )
    
    substrate_provider = DFProvider(
        name="SubstrateConc",
        DataFrame=df_substrate,
        y_columns=["S"],
        noise=np.array([[0.03**2]])  # Variance
    )
    
    print(f"Biomass provider times: {biomass_provider.times}")
    print(f"Substrate provider times: {substrate_provider.times}")

    def func(x):
        return np.asarray(x)[:2]
    
    # Create EnKF with multiple providers
    # Observation matrix: [1, 0, 0] for X, [0, 1, 0] for S -> combined [1,0,0; 0,1,0]
    enkf = EnKalmanFilter(
        model=casadi_model,
        initial_state=X0,
        ensemble_size=50,
        initial_covariance=np.diag([0.1, 0.1, 0.1]),
        observation_func=func,
        providers=[biomass_provider, substrate_provider],
        random_seed=42
    )
    
    print(f"\nRegistered providers: {enkf.provider_names}")
    print(f"All measurement times: {enkf.get_all_measurement_times()}")
    
    # Run EnKF using the run_filter method
    print("\n--- Running EnKF with multiple providers ---")
    results = enkf.run_filter(
        t_start=0.0,
        t_end=T,
        use_measurement_times=True,
        interpolation='nearest'
    )
    
    print(f"Filter ran for {len(results['times'])} time steps")
    
    # Also demonstrate manual stepping
    print("\n--- Manual stepping example ---")
    enkf2 = EnKalmanFilter(
        model=casadi_model,
        initial_state=X0,
        ensemble_size=50,
        initial_covariance=np.diag([0.1, 0.1, 0.1]),
        observation_func=func,
        providers=[biomass_provider, substrate_provider],
        random_seed=42
    )
    
    # Step manually with measurement retrieval
    estimated_states = [X0.copy()]
    meas_times = enkf2.get_all_measurement_times()
    t_prev = 0.0
    
    for t in meas_times:
        if t <= 0:
            continue
        
        # Get measurements (with interpolation for missing values)
        meas = enkf2.get_measurements(t, method='nearest')
        print(f"Time {t:.2f}: Measurements = {meas}")
        
        # Step
        state_est = enkf2.step(
            t_start=t_prev,
            t_end=t,
            measurement=meas
        )
        estimated_states.append(state_est.copy())
        t_prev = t
    
    # Plot results
    estimated_arr = np.array(estimated_states)
    true_arr = np.array(true_states)
    time_arr = np.array([0] + list(meas_times))[:len(estimated_arr)]
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([0] + [t for t, _ in biomass_measurements], 
             [X0[0]] + [m for _, m in biomass_measurements], 
             'ro', markersize=4, label='Biomass Meas')
    plt.plot(np.linspace(0, T, len(true_arr)), true_arr[:, 0], 'b-', label='True X')
    plt.plot(time_arr, estimated_arr[:, 0], 'g--', label='Estimated X')
    plt.xlabel('Time')
    plt.ylabel('Biomass X')
    plt.legend()
    plt.title('Biomass Estimation')
    
    plt.subplot(1, 3, 2)
    plt.plot([t for t, _ in substrate_measurements], 
             [m for _, m in substrate_measurements], 
             'mo', markersize=4, label='Substrate Meas')
    plt.plot(np.linspace(0, T, len(true_arr)), true_arr[:, 1], 'b-', label='True S')
    plt.plot(time_arr, estimated_arr[:, 1], 'g--', label='Estimated S')
    plt.xlabel('Time')
    plt.ylabel('Substrate S')
    plt.legend()
    plt.title('Substrate Estimation')
    
    plt.subplot(1, 3, 3)
    plt.plot(np.linspace(0, T, len(true_arr)), true_arr[:, 2], 'b-', label='True V')
    plt.plot(time_arr, estimated_arr[:, 2], 'g--', label='Estimated V')
    plt.xlabel('Time')
    plt.ylabel('Volume V')
    plt.legend()
    plt.title('Volume Estimation (unobserved)')
    
    plt.tight_layout()
    plt.savefig('enkf_multi_provider_results.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("Multi-Provider EnKF example completed successfully!")
    print("="*60)
