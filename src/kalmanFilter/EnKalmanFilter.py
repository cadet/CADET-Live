import numpy as np
"""
?Questions: 
- Sollten die Messungen ahnung vom Model haben? | Ja
ToDos:
- [ ] Measurements soll warend eines Prozesses hinzugefuegzt werden koennen.
- [ ] Messungen mussen erst transformiert werden um in den Zustandsraum zu passen. 
"""


class Measurement:
    
    def __init__(self, 
                 state_name: str,
                 times: np.array,
                 observed_states: np.array,
                 noise: np.ndarray = np.array([[0.0]])):
        
        self.state_name = state_name
        
        if (len(times) != len(observed_states)):
            raise ValueError("Times and observed states must have the same length.")
        
        self.time_stamp = times
        self.observed_states = observed_states
        self.noise = noise

    @property
    def nMeasurements(self)-> int:
        return len(self.observed_states)
    
    @property
    def measurementTimes(self) -> np.array:
        return self.time_stamp
    
    @property
    def nObservedStates(self) -> int:
        return self.observed_states.shape[1]
    
    def printMeasurements(self):
        for t, obs in zip(self.time_stamp, self.observed_states):
            print(f"Time: {t}, Observed State: {obs}")

class EnKalmanFilter:

    def __init__(self, 
                  integrator: object,
                  measuremnts: Measurement,
                  ensemble_size: int,
                  process_noise: np.ndarray,
                  initial_state: np.ndarray,
                  initial_covariance: np.ndarray):
        
        self.integrator = integrator
        self.measurements = measuremnts
        self.N_ens = ensemble_size
        self.noise = process_noise
        self.state = initial_state
        self.cov = initial_covariance
        self.meas_index = 0

        self.X_ens = np.random.multivariate_normal(self.state, self.cov, self.N_ens)

    def propagate(self):
        """Propagate the ensemble using the integrator."""
        for i in range(self.N_ens):
            # get inital sate for each ensemble member
            x_prev = self.X_ens[i, :]
            # integrate the state
            sol = self.integrator(x0=x_prev)
            w = np.random.multivariate_normal(np.zeros(len(x_prev)), self.noise)

            self.X_ens[i] = sol + w
    
    def update(self):
        """Update the ensemble based on the measurement."""
        
        y_meas = self.measurements.observed_states[self.meas_index]
        Y_ens = np.array([y_meas + np.random.normal(0,self.measurements.noise)[0] for _ in range(self.N_ens)])


        # Calculate mean
        x_mean = np.mean(self.X_ens, axis=0, keepdims=True)
        y_mean = np.mean(Y_ens, axis=0, keepdims=True)

        # Calculate deviations
        X_dev = self.X_ens - x_mean
        Y_dev = Y_ens - y_mean

        # Calculate Kalman gain
        P_xy = (X_dev.T @ Y_dev) / (self.N_ens - 1)
        P_yy = (Y_dev.T @ Y_dev) / (self.N_ens - 1) + self.measurements.noise

        K = P_xy @ np.linalg.inv(P_yy)

        # Update ensemble
        for i in range(self.N_ens):
            self.X_ens[i] += (K @ (y_meas - Y_ens[i])).flatten()
        
        # Update state and covariance
        self.state = np.mean(self.X_ens, axis=0)
        self.cov = np.cov(self.X_ens.T)

        self.meas_index += 1


if __name__ == "__main__":
    
    import casadi as ca

    # Model Definition
    S_in = 0.5
    F_out = 1.0
    F_in = 1.0

    X = ca.SX.sym('X')
    S = ca.SX.sym('S')
    V = ca.SX.sym('V')
    states = ca.vertcat(X, S, V)
    n_states = 3

    dV_dt = F_in - F_out
    dx_dt = (-F_out * X - dV_dt * X) / V
    ds_dt = (F_in * S_in - F_out * S - dV_dt * S) / V
    f = ca.vertcat(dx_dt, ds_dt, dV_dt)

    T = 1.0
    dt = 0.1
    X0 = np.array([1.0, 0.5, 2.0])

    integrato = ca.integrator(
    "integrator", "idas",
    {"x": states,  "ode": f},
    0.0, dt
    )
    
    X_sim = [1.0]
    S_sim = [0.5]
    V_sim = [2.0]

    # create atifical measurements
    np.random.seed(42)
    time_points = [ ]
    measurements = [ ]
    for t in range(int(T//dt)):
        time_points.append(t*dt)
        
        res = integrato(x0=np.array([X_sim[-1], S_sim[-1], V_sim[-1]]))
        
        X_sim.append(res['xf'][0].full().flatten()[0])
        S_sim.append(res['xf'][1].full().flatten()[0])
        V_sim.append(res['xf'][2].full().flatten()[0])

        measurements.append(res['xf'][0].full().flatten()[0] + np.random.normal(0, 0.1))
            
    meas = Measurement("X", np.array(time_points), np.array(measurements), np.array([[0.1]]))

    enkf = EnKalmanFilter(
        integrator=lambda x0: integrato(x0=x0)['xf'].full().flatten(),
        measuremnts=meas,
        ensemble_size=50,
        process_noise=np.diag([0.01, 0.01, 0.01]),
        initial_state=np.array([1.0, 0.5, 2.0]),
        initial_covariance=np.diag([0.1, 0.1, 0.1])
    )

    for t in range(len(time_points)):
        enkf.propagate()
        enkf.update()
        print(f"Time: {time_points[t]}, Estimated State: {enkf.state}")
