import numpy as np
import pandas as pd
import casadi as ca

from Mesurement import MeasurementProvider, DFProvider
from EnKalmanFilter import EnKalmanFilter
from Controller import Controller

# Purpose: 
# - connect measurment and state / model information to an experiment
# - connect state and controller

class Experiment:
    
    def __init__(self,
                data: MeasurementProvider,
                model: object,
                state_estimator: EnKalmanFilter,
                controller: Controller,
                verbose: bool = False
                ):
        
        self.data = data
        self.model = model
        self.state_estimator = state_estimator
        self.controller = controller

        self.verbose = verbose
        self.true_states = []
        self.estimated_states = []
        self.measurements = []
        self.covariances = []

    def run(self, time_points: list):
        
        for t in range(len(time_points)):
            
            self.state_estimator.propagate()
            self.state_estimator.update(t*dt, self.data)
            
            if self.verbose:
                self.covariances.append(self.state_estimator.cov.copy())
                self.estimated_states.append(self.state_estimator.state.copy())

    @property
    def trueStates(self) -> np.ndarray:
        return np.array(self.true_states)
    
    @property
    def estimatedStates(self) -> np.ndarray:
        return np.array(self.estimated_states)
    
    def estimatedState(self, index: int) -> np.ndarray:
        return np.array([state[index] for state in self.estimated_states])
    
    @property
    def Measurements(self) -> np.ndarray:
        return np.array(self.measurements)
    
    @property
    def Covariances(self) -> np.ndarray:
        return np.array(self.covariances)

if __name__ == "__main__":


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

    T = 4.0
    dt = 0.05

    integrator = ca.integrator(
    "integrator", "idas",
    {"x": states,  "ode": f},
    0.0, dt
    )
    
    X_sim = [10.0]
    S_sim = [0.5]
    V_sim = [2.0]

    # create atifical measurements
    np.random.seed(42)
    time_points = [ ]
    measurements = [ ]

    for t in range(int(T//dt)):
        
        time_points.append(t*dt)
        res = integrator(x0=np.array([X_sim[-1], S_sim[-1], V_sim[-1]]))
        
        X_sim.append(res['xf'][0].full().flatten()[0])
        S_sim.append(res['xf'][1].full().flatten()[0])
        V_sim.append(res['xf'][2].full().flatten()[0])

        measurements.append(res['xf'][0].full().flatten()[0] + np.random.normal(0, 0.1))
            

    enkf = EnKalmanFilter(
        integrator=lambda x0: integrator(x0=x0)['xf'].full().flatten(),
        ensemble_size=50,
        process_noise=np.diag([0.01, 0.01, 0.01]) * dt, # dt is important here
        initial_state=np.array([10.0, 0.5, 2.0]),
        initial_covariance=np.diag([0.1, 0.1, 0.1])
    )

    df = pd.DataFrame({
        "t": time_points,
        "X": measurements
    })

    meas = DFProvider(
        df,
        x_column="t",
        y_columns=["X"],
        noise=np.array([[0.1]])
    )
    
    exp = Experiment(
        data=meas,
        model=integrator,
        state_estimator=enkf,
        controller=None,
        verbose=True)
    
    exp.run(time_points)

    #plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot True State, Measurements, and Estimated State
    axs[0].plot(time_points, X_sim[1:], label='True State X', color='blue')
    axs[0].plot(time_points, S_sim[1:], label='True State S', color='orange')
    axs[0].plot(time_points,  V_sim[1:], label='True State V', color='green')
    axs[0].scatter(time_points, measurements, label='Measurements', color='red', s=10)
    axs[0].plot(time_points, exp.estimatedState(index=0), label='Estimated State X', color='blue', linestyle='--')
    axs[0].plot(time_points, exp.estimatedState(index=1), label='Estimated State S', color='orange', linestyle='--')
    axs[0].plot(time_points, exp.estimatedState(index=2), label='Estimated State V', color='green', linestyle='--')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('State X')
    axs[0].legend()
    axs[0].set_title('Ensemble Kalman Filter State Estimation')

    # Plot Covariance
    axs[1].plot(time_points, exp.Covariances[:, 0, 0], label='Covariance of State X', color='purple')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Covariance')
    axs[1].legend()
    axs[1].set_title('Ensemble Kalman Filter State Covariance')

    plt.tight_layout()
    plt.show()
