cadet_root = "/Users/berger/fzj/cadet/CADET-Core/install_release"

from cadet import Cadet
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'control'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test_stepped_vs_full_simulation(plot_results: bool = False, tolerance: float = 1e-5):
    """
    Test that stepped simulation results match full simulation results.
    
    Verifies that state points from stepped simulation are consistent with 
    the full simulation solution within the specified tolerance.
    
    Parameters
    ----------
    plot_results : bool, optional
        Whether to plot the results for visualization (default: False).
    tolerance : float, optional
        Maximum allowed difference between stepped and full simulation results (default: 1e-5).
    """
    # ---- Run full simulation ----
    model_full = Cadet(install_path=cadet_root, use_dll=True)
    model_full.filename = "./modelLibrary/cstr_no_inlet_one_mal.h5"
    model_full.load_from_file()
    model_full.save()

    return_info_full = model_full.run_simulation()
    assert return_info_full.return_code == 0, "Full simulation failed"

    # Store results from full simulation
    solution_full = model_full.root.output.solution.unit_000.solution_bulk.copy()
    solution_times_full = model_full.root.output.solution.solution_times.copy()

    # ---- Run stepped simulation ----
    model_step = Cadet(install_path=cadet_root, use_dll=True)
    model_step.filename = "./modelLibrary/cstr_no_inlet_one_mal.h5"
    model_step.load_from_file()
    model_step.save()

    return_info = model_step.initialize_simulation()
    assert return_info.return_code == 0, "Stepped simulation initialization failed"

    # Step through simulation in larger chunks
    total_time = 2.0
    n_steps = 5
    step_size = total_time / n_steps

    solutions_step = []
    times_step = []

    for i in range(n_steps):
        t_target = min((i + 1) * step_size, total_time)
        times_step.append(t_target)

        return_info, t_reached = model_step.perform_simulation_step(t_target)
        assert return_info.return_code == 0, f"Stepped simulation step {i} failed"

        res = model_step.cadet_runner.res
        state = res.last_state_y()
        solutions_step.append(state[2:4])

    # Clean up stepped simulation
    model_step.end_simulation()

    # ---- Verify that stepped solution points exist in full solution ----
    for t_step, sol_step in zip(times_step, solutions_step):
        # Find the closest time point in the full solution
        idx = np.argmin(np.abs(solution_times_full - t_step))
        sol_full_at_t = solution_full[idx, 0:2]

        # Check that the difference is within tolerance
        diff = np.abs(sol_step - sol_full_at_t)
        max_diff = np.max(diff)

        assert max_diff < tolerance, (
            f"At time {t_step}: stepped solution {sol_step} differs from "
            f"full solution {sol_full_at_t} by {max_diff} (tolerance: {tolerance})"
        )

    # ---- Optional plotting ----
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(solution_times_full, solution_full[:, 0], label='Concentration C_A (full simulation)', linewidth=2)
        plt.plot(solution_times_full, solution_full[:, 1], label='Concentration C_B (full simulation)', linewidth=2)
        plt.scatter(times_step, [sol[0] for sol in solutions_step], 
                   color='red', label='C_A (stepped simulation)', zorder=5, s=100)
        plt.scatter(times_step, [sol[1] for sol in solutions_step], 
                   color='green', label='C_B (stepped simulation)', zorder=5, s=100)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('CSTR Concentrations Over Time - Stepped vs Full Simulation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def test_update_state_simulation(plot_results: bool = False, tolerance: float = 1e-5):
    """
    Test updating state during simulation and comparing with full simulation.
    
    Simulates iterative state updates with noise and verifies consistency
    between stepped simulation and full simulation at each step.
    
    Parameters
    ----------
    plot_results : bool, optional
        Whether to plot the results for visualization (default: False).
    tolerance : float, optional
        Maximum allowed difference between stepped and full simulation results (default: 1e-5).
    """
    # ---- Initialize models ----
    model_full = Cadet(install_path=cadet_root, use_dll=True)
    model_full.filename = "./modelLibrary/cstr_no_inlet_one_mal_full.h5"
    model_full.load_from_file()
    model_full.save()

    model_step = Cadet(install_path=cadet_root, use_dll=True)
    model_step.filename = "./modelLibrary/cstr_no_inlet_one_mal_step.h5"
    model_step.load_from_file()
    model_step.save()

    return_info = model_step.initialize_simulation()
    assert return_info.return_code == 0, "Stepped simulation initialization failed"

    # ---- Simulation parameters ----
    total_time = 10.0
    n_steps = 5
    step_size = 2.0
    init_c = np.array([1.0, 0.0])

    # ---- Data collection ----
    solutions_step = [init_c]
    solutions_full = [init_c]
    times_step = [0]
    
    # Set initial conditions for both models
    model_full.root.input.model.unit_000.init_c = init_c
    model_full.save()
    model_step.root.input.model.unit_000.init_c = init_c
    model_step.save()
    # ---- Iterative simulation with state updates ----

    for i in range(n_steps):
        t_target = min((i + 1) * step_size, total_time)
        times_step.append(t_target)
        
        if i != 0:

            return_info = model_step.update_state(state_step, times_step[i], len(state_step))
            if return_info.return_code != 0:
                print(return_info.error_message)
                
        
        return_info, t_reached = model_step.perform_simulation_step(t_target)
        if return_info.return_code != 0:
            print(return_info.error_message)
            

        res_step = model_step.cadet_runner.res
        state_step = res_step.last_state_y().copy()
        sol_step_bulk = state_step[2:4].copy()
        solutions_step.append(sol_step_bulk)

        # Run full simulation from current initial conditions to end time
        return_info = model_full.run_simulation()
        if return_info.return_code != 0:
            print(return_info.error_message)
        
        sol_full_bulk = model_full.root.output.solution.unit_000.solution_bulk[-1, 0:2]
        solutions_full.append(sol_full_bulk)

        # Verify consistency within tolerance
        diff = np.abs(sol_step_bulk - sol_full_bulk)
        max_diff = np.max(diff)
        
        assert max_diff < tolerance, (
            f"At step {i} (t={t_target}): stepped solution {sol_step_bulk} differs from "
            f"full solution {sol_full_bulk} by {max_diff} (tolerance: {tolerance})"
        )

        # Update state with small random perturbation (measurement noise)
        updated_c = sol_full_bulk + np.random.randn(2) * 0.05
        state_step[2:4] = updated_c

        if return_info.return_code != 0:
            print(return_info.error_message)
        model_full.root.input.model.unit_000.init_c = updated_c
        model_full.save()

    # Clean up stepped simulation
    model_step.end_simulation()

    # ---- Optional plotting of endpoint trajectories ----
    if plot_results:
        solutions_step = np.array(solutions_step)
        solutions_full = np.array(solutions_full)

        plt.figure(figsize=(10, 6))
        plt.plot(times_step, solutions_step[:, 0], 'o-', label='Stepped Sim - C_A',
                    linewidth=2, markersize=8, color='red')
        plt.plot(times_step, solutions_step[:, 1], 's-', label='Stepped Sim - C_B',
                    linewidth=2, markersize=8, color='blue')
        plt.plot(times_step, solutions_full[:, 0], 'o--', label='Full Sim - C_A',
                    linewidth=2, markersize=8, color='orange', alpha=0.7)
        plt.plot(times_step, solutions_full[:, 1], 's--', label='Full Sim - C_B',
                    linewidth=2, markersize=8, color='violet', alpha=0.7)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def test_enkf_with_cadet_model(plot_results: bool = False, tolerance: float = 0.2):
    """
    Test Ensemble Kalman Filter with CADET model.
    
    Creates a CADET model, generates synthetic measurements with noise,
    runs the EnKF to estimate states, and validates that estimates are
    within tolerance of true states.
    
    Parameters
    ----------
    plot_results : bool, optional
        Whether to plot the results for visualization (default: False).
    tolerance : float, optional
        Maximum allowed mean absolute error between estimated and true states (default: 0.2).
    """
    from Model import CadetModel
    from EnKalmanFilter import EnKalmanFilter
    from Provider import DFProvider
    
    # ---- Set up CADET model for "true" system ----
    model_true = Cadet(install_path=cadet_root, use_dll=True)
    model_true.filename = "./modelLibrary/cstr_no_inlet_one_mal.h5"
    model_true.load_from_file()
    model_true.save()
    
    # Run full simulation to get "true" trajectory
    return_info = model_true.run_simulation()
    assert return_info.return_code == 0, "True simulation failed"
    
    solution_true = model_true.root.output.solution.unit_000.solution_bulk.copy()
    solution_times = model_true.root.output.solution.solution_times.copy()
    
    # ---- Generate synthetic measurements with noise ----
    # Sample measurements at fewer time points
    measurement_interval = 5  # Every 5th point
    meas_indices = range(0, len(solution_times), measurement_interval)
    
    m_noise_std = 0.3
    np.random.seed(42)
    
    # Create measurements for both states (C_A and C_B)
    measurements_A = []
    measurements_B = []
    
    for idx in meas_indices:
        t = solution_times[idx]
        true_A = solution_true[idx, 0]
        true_B = solution_true[idx, 1]
        
        # Add measurement noise
        meas_A = true_A + np.random.normal(0, m_noise_std)
        meas_B = true_B + np.random.normal(0, m_noise_std)
        
        measurements_A.append((t, meas_A))
        measurements_B.append((t, meas_B))
    
    # Create DataFrames for providers
    df_A = pd.DataFrame({"C_A": [measurements_A]})
    df_B = pd.DataFrame({"C_B": [measurements_B]})
    
    provider_A = DFProvider(
        name="ConcentrationA",
        dataframe=df_A,
        y_columns=["C_A"],
        noise=np.array([[m_noise_std**2]])
    )

    provider_B = DFProvider(
        name="ConcentrationB",
        dataframe=df_B,
        y_columns=["C_B"],
        noise=np.array([[m_noise_std**2]])
    )
    
    # ---- Initialize EnKF ----
    initial_state = np.array([1.0, 0.0])  # Initial guess
    
    # ---- Set up CADET model for EnKF ----
    cadet_model_enkf = CadetModel(
        cadet_path=cadet_root,
        init_state=initial_state,
        model_path="./modelLibrary/cstr_no_inlet_one_mal.h5",
        n_states=2,
        n_controls=0,
        process_noise = np.diag([0.01,0.01]),
        state_indices=[2, 3]  # Extract bulk concentrations from CADET state
    )
    

    
    enkf = EnKalmanFilter(
        model=cadet_model_enkf,
        ensemble_size=50,
        initial_covariance=np.diag([0.5, 0.5]),
        providers=[provider_A, provider_B],
        random_seed=42
    )
    
    results = enkf.run_filter(
        t_start=0.0,
        t_end=solution_times[-1],
        use_measurement_times=True,
        interpolation='nearest'
    )
    
    enkf_times = results['times']
    enkf_states = results['states']
    enkf_covs = results['covariances']
    
    
    true_states_interp = np.zeros((len(enkf_times), 2))
    for i, t in enumerate(enkf_times):
        idx = np.argmin(np.abs(solution_times - t))
        true_states_interp[i] = solution_true[idx, 0:2]
    
    # Calculate errors
    errors = np.abs(enkf_states - true_states_interp)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    
    # Check tolerance
    assert mean_error < tolerance, (
        f"Mean error {mean_error:.4f} exceeds tolerance {tolerance}"
    )
    
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Concentration A
        ax = axes[0, 0]
        ax.plot(solution_times, solution_true[:, 0], 'b-', label='True C_A', linewidth=2)
        ax.plot(enkf_times, enkf_states[:, 0], 'g--', label='EnKF Estimate', linewidth=2)
        ax.scatter([t for t, _ in measurements_A], [m for _, m in measurements_A],
                  color='red', label='Measurements', s=50, zorder=5)
        
        std_A = np.sqrt([enkf_covs[i][0, 0] for i in range(len(enkf_times))])
        ax.fill_between(enkf_times, 
                        enkf_states[:, 0] - std_A,
                        enkf_states[:, 0] + std_A,
                        alpha=0.3, color='green', label='±1 sigma')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration C_A')
        ax.set_title('Concentration A Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Concentration B
        ax = axes[0, 1]
        ax.plot(solution_times, solution_true[:, 1], 'b-', label='True C_B', linewidth=2)
        ax.plot(enkf_times, enkf_states[:, 1], 'g--', label='EnKF Estimate', linewidth=2)
        ax.scatter([t for t, _ in measurements_B], [m for _, m in measurements_B],
                  color='red', label='Measurements', s=50, zorder=5)
        
        # Add uncertainty bands
        std_B = np.sqrt([enkf_covs[i][1, 1] for i in range(len(enkf_times))])
        ax.fill_between(enkf_times,
                        enkf_states[:, 1] - std_B,
                        enkf_states[:, 1] + std_B,
                        alpha=0.3, color='green', label='±1 sigma')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration C_B')
        ax.set_title('Concentration B Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Estimation errors
        ax = axes[1, 0]
        ax.plot(enkf_times, errors[:, 0], 'o-', label='Error C_A', markersize=4)
        ax.plot(enkf_times, errors[:, 1], 's-', label='Error C_B', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Estimation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 4: Covariance trace
        ax = axes[1, 1]
        trace_cov = [np.trace(enkf_covs[i]) for i in range(len(enkf_times))]
        ax.plot(enkf_times, trace_cov, 'o-', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Trace of Covariance Matrix')
        ax.set_title('Estimation Uncertainty (Covariance Trace)')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('test_enkf_cadet_results.png', dpi=150)
        plt.show()
    
    # Clean up
    cadet_model_enkf.end_simulation()


def test_update_reset_trajectory(plot_results: bool = False, tolerance: float = 1e-5):
    """
    Test that resetting to initial state produces identical trajectories.
    
    Simulates forward, then resets to the initial state using update_state,
    and verifies that the second trajectory matches the first one exactly.
    This ensures that the update_state function correctly resets the simulation.
    
    Parameters
    ----------
    plot_results : bool, optional
        Whether to plot the results for visualization (default: False).
    tolerance : float, optional
        Maximum allowed difference between the two trajectories (default: 1e-5).
    """
    # ---- Initialize model ----
    model = Cadet(install_path=cadet_root, use_dll=True)
    model.filename = "./modelLibrary/cstr_no_inlet_one_mal.h5"
    model.load_from_file()
    model.save()
    
    return_info = model.initialize_simulation() 
    assert return_info.return_code == 0, "Simulation initialization failed"
    
    # Store initial state

    initial_time = 0.0
    t_init = 0.01
    unitId = 0

    return_info, t_reached = model.perform_simulation_step(t_init)
    assert return_info.return_code == 0, "Initial simulation step failed"
    
    res = model.cadet_runner.res

    initial_state = res.last_state_y().copy()
    initial_state = initial_state[2:4]  # Set bulk concentrations to initial values
    return_info = model.update_bulk_state(unitId,initial_time, initial_state, len(initial_state))

    total_time = 2.0
    n_steps = 5
    step_size = total_time / n_steps
    
    trajectory_1 = []
    times_1 = []
    
    for i in range(n_steps):

        t_target = (i + 1) * step_size
        times_1.append(t_target)
        
        return_info, t_reached = model.perform_simulation_step(t_target)
        assert return_info.return_code == 0, f"First trajectory step {i} failed"
        
        res = model.cadet_runner.res
        state = res.last_state_y()
        trajectory_1.append(state[2:4].copy())  # Store bulk concentrations
    
    # ---- Reset to initial state using update_state ----
    return_info = model.update_bulk_state(unitId,initial_time, initial_state, len(initial_state))
    assert return_info.return_code == 0, "State reset failed"
    
    # ---- Second trajectory: simulate forward again from reset state ----
    trajectory_2 = []
    times_2 = []
    
    for i in range(n_steps):
        t_target = (i + 1) * step_size
        times_2.append(t_target)
        
        return_info, t_reached = model.perform_simulation_step(t_target)
        assert return_info.return_code == 0, f"Second trajectory step {i} failed"
        
        res = model.cadet_runner.res
        state = res.last_state_y()
        trajectory_2.append(state[2:4].copy())  # Store bulk concentrations
    
    # Clean up
    model.end_simulation()
    
    # ---- Verify that both trajectories are identical ----
    trajectory_1 = np.array(trajectory_1)
    trajectory_2 = np.array(trajectory_2)
    
    for i, (t, sol1, sol2) in enumerate(zip(times_1, trajectory_1, trajectory_2)):
        diff = np.abs(sol1 - sol2)
        max_diff = np.max(diff)
        
        assert max_diff < tolerance, (
             f"At time {t} (step {i}): first trajectory {sol1} differs from "
             f"second trajectory {sol2} by {max_diff} (tolerance: {tolerance})"
          )
    
    print(f"\n--- Test Passed ---")
    print(f"Both trajectories are identical within tolerance {tolerance}")
    print(f"Max difference across all steps: {np.max(np.abs(trajectory_1 - trajectory_2)):.2e}")
    
    # ---- Optional plotting ----
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(times_1, trajectory_1[:, 0], 'o-', label='First trajectory - C_A',
                linewidth=2, markersize=8, color='blue')
        plt.plot(times_1, trajectory_1[:, 1], 's-', label='First trajectory - C_B',
                linewidth=2, markersize=8, color='red')
        plt.plot(times_2, trajectory_2[:, 0], 'x--', label='Second trajectory - C_A',
                linewidth=2, markersize=10, color='cyan', alpha=0.7)
        plt.plot(times_2, trajectory_2[:, 1], 'd--', label='Second trajectory - C_B',
                linewidth=2, markersize=10, color='orange', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('CSTR Concentrations - Trajectory Reset Test')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    # Run tests with optional plotting
    np.random.seed(42)
    
    test_stepped_vs_full_simulation(plot_results=True, tolerance=1e-2)
    
    test_update_state_simulation(plot_results=True, tolerance=1e-2)
    
    test_enkf_with_cadet_model(plot_results=True, tolerance=0.2)
    
    test_update_reset_trajectory(plot_results=True, tolerance=1e-5)