import numpy as np
import matplotlib.pyplot as plt

from cadet import Cadet
from scipy.integrate import solve_ivp

def propagate(model, X_ens, t_start, t_end, noise, random_seed=None):
    rng = np.random.default_rng(random_seed)
    N_ens, nStates = X_ens.shape

    for i in range(N_ens):
        x_prev = np.mean(X_ens, axis=0)
        #set state in cadet model
        nStates = x_prev.shape[0]
        if i != 0:
            model.update_state(x_prev, t_start, nStates)

        try:
            #perform_simulation_step may return (return_info, t_reached)
            ret = model.perform_simulation_step(t_end)
            if ret[0].return_code != 0:
                raise Exception("Cadet simulation step failed")

            res = model.cadet_runner.res
            state = res.last_state_y()
            w = rng.multivariate_normal(np.zeros(nStates), noise)

            X_ens[i, :] = np.asarray(state[2:4]).reshape(nStates,) + w
            continue

        except Exception:
            print("something went wring")

        # simple propagation: identity dynamics + process noise
        if np.isscalar(noise):
            cov = noise * np.eye(nStates)
        else:
            cov = np.asarray(noise)

        w = rng.multivariate_normal(np.zeros(nStates), cov)
        X_ens[i, :] = x_prev + w
    return X_ens

def update(measure: tuple, noise, X_ens, H=None, random_seed=None):
    rng = np.random.default_rng(random_seed)

    # support both (time, y) and scalar y
    if isinstance(measure, tuple) or isinstance(measure, list):
        y_meas = measure[1]
    else:
        y_meas = measure

    N_ens, nStates = X_ens.shape

    if H is None:
        H = lambda x: x

    Y_ens = np.array([])
    for i in range(X_ens.shape[1]):
        x_ens = X_ens[:, i]
        y_ens = np.array([H(x_ens[i]) + rng.normal(0.0, noise) for i in range(N_ens)])
        Y_ens = np.column_stack((Y_ens, y_ens)) if Y_ens.size else y_ens.reshape(N_ens,1)

    # Calculate means
    x_mean = np.mean(X_ens, axis=0)
    y_mean = np.mean(Y_ens, axis=0)

    # Deviations
    X_dev = (X_ens - x_mean)
    Y_dev = (Y_ens - y_mean)

    # Cross covariance and measurement covariance
    P_xy = (X_dev.T @ Y_dev) / (N_ens - 1)           # shape (nStates, 1)
    P_yy = (Y_dev.T @ Y_dev) / (N_ens - 1) + noise   # shape (1,1)

    # Kalman gain
    K = P_xy @ np.linalg.inv(P_yy)

    # Update ensemble
    for i in range(N_ens):
        innovation = (y_meas - Y_ens[i])
        X_ens[i, :] = X_ens[i, :] + (K @ innovation)

    state = np.mean(X_ens, axis=0)
    cov = np.cov(X_ens.T) + 1e-9 * np.eye(nStates)

    return state, cov, X_ens


def test():

    cadet_root = "/Users/berger/fzj/cadet/CADET-Core/install_release"

    cadet_model = Cadet(install_path=cadet_root, use_dll=True)

    cadet_model.filename = "/Users/berger/fzj/cadet/CADET-Live/src/modelLibrary/only_cstr_no_reac.h5"
    cadet_model.load_from_file()
    cadet_model.save()

    return_info = cadet_model.initialize_simulation()

    print(return_info)

    t_target = 10.0
    return_info, t_reached = cadet_model.perform_simulation_step(t_target)

    # Check that step was successful
    assert return_info.return_code == 0
    assert return_info.error_message == ""
    assert isinstance(t_reached, float)
    assert t_reached <= t_target
    assert t_reached > 0.0

    print(return_info)
    print(t_reached)


if __name__ == "__main__":

    np.random.seed(42)
    cadet_root = "/Users/berger/fzj/cadet/CADET-Core/install_release"

    ## Model / Process Definition
    def reaction(t, y):
        dydt = np.zeros(2)
        dydt[0] = -3 * y[0] + 2 * y[1]
        dydt[1] = 3 * y[0] - 2 * y[1]
        return dydt

    t_span = (0.0, 2)
    t_eval = np.linspace(0, 2, 40)
    y0 = np.array([1.0, 0.0])

    sol = solve_ivp(reaction, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)

    #plt.plot(sol.t, sol.y[0], label='y0 (true)')
    #plt.plot(sol.t, sol.y[1], label='y1 (true)')
    #plt.show()

    # transfrom solution into measurements
    m_noise = 0.01
    measurements = [(sol.t[i], [sol.y[0, i] + m_noise * np.random.randn(),sol.y[1, i] + m_noise * np.random.randn()]) for i in range(len(sol.t))]


    # set up Cadet model
    model = Cadet(install_path=cadet_root, use_dll=True)
    model.filename = "./modelLibrary/cstr_no_inlet_one_mal.h5"
    model.load_from_file()
    model.save()
    model.initialize_simulation()

    N_ens = 50
    nStates = 2
    dt = 0.5
    time_points = np.linspace(0, 2, 100)

    p_noise = 0.01 * np.eye(nStates)
    X_ens = np.vstack([np.array([1.0, 0.0]) + 0.01 * np.random.randn(nStates) for _ in range(N_ens)])

    states = [np.array([1.0, 0.0])]
    for idx, t in enumerate(time_points):
        if idx == 0:
            continue
        # propagate ensemble to next time
        X_ens = propagate(model, X_ens, t, t + dt, p_noise)

        # get matching measurement from the secondary simulation (use closest index)
        if idx < len(measurements):
            meas_time, meas_state = measurements[idx]
        else:
            # fallback: use last available measurement timestamp/state
            meas_time, meas_state = measurements[-1]

        state, cov, X_ens = update((meas_time, meas_state), m_noise, X_ens)

        print(f"Time: {meas_time:.1f}, Estimated State: {state}")
        states.append(state.copy())

    return_code = model.end_simulation()

    states_arr = np.vstack(states)

    plt.figure(figsize=(10, 5))
    # Estimated states (EnKF)
    plt.plot(time_points, states_arr[:, 0], '-o', label='Estimated state[0]', markersize=4)
    plt.plot(time_points, states_arr[:, 1], '-s', label='Estimated state[1]', markersize=4)

    # Measurements
    plt.scatter(sol.t, [m[1][0] for m in measurements], color='blue', label='Measurements y0', s=20, alpha=0.6)
    plt.scatter(sol.t, [m[1][1] for m in measurements], color='orange', label='Measurements y1', s=20, alpha=0.6)

    # True states
    plt.plot(sol.t, sol.y[0], color='blue', linestyle='--', label='True State y0')
    plt.plot(sol.t, sol.y[1], color='orange', linestyle='--', label='True State y1')

    plt.xlabel('time')
    plt.ylabel('state')
    plt.title('EnKF estimated states, measurements and true states')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("enkf_cadet_results.png")
    print("Happy")
