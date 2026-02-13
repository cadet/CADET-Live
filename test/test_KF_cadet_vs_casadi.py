"""Test: CasADi vs CADET model comparison and EnKF with both backends."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'control'))

import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
from cadet import Cadet
from Model import CasadiModel, CadetModel
from EnKalmanFilter import EnKalmanFilter
from Provider import DFProvider

CADET_ROOT = "/Users/berger/fzj/cadet/CADET-Core/install_release"
MODEL_FILE = "./modelLibrary/cstr_no_inlet_one_mal.h5"

# Shared CSTR parameters
INIT_C = np.array([1.0, 0.0])  # [P, S]
SIM_TIME = 2.0
DT = 0.1

# Reaction rates (from CasADi perspective)
# k1: S -> P production,  k2: P -> S consumption
# CADET stoichiometry [[-1],[1]] means forward = P->S, so:
#   CADET mal_kfwd = k2 = 3,  CADET mal_kbwd = k1 = 2
K1 = 2.0
K2 = 3.0
V_CONST = 1.0


def create_casadi_model():
    """Create CasADi CSTR model matching the CADET model (2 states: P, S)."""
    P = ca.SX.sym('P')
    S = ca.SX.sym('S')
    states = ca.vertcat(P, S)
    u = ca.SX.sym('u', 0)

    def ode(x, u):
        dp = (K1 * x[1] - K2 * x[0]) / V_CONST
        ds = (-K1 * x[1] + K2 * x[0]) / V_CONST
        return ca.vertcat(dp, ds)

    return CasadiModel(
        states=states,
        controls=u,
        ode=ode,
        init_state=INIT_C.copy(),
        dt=DT,
        T=SIM_TIME
    )


def run_cadet_reference():
    """Run full CADET simulation and return reference trajectory."""
    model = Cadet(install_path=CADET_ROOT, use_dll=True)
    model.filename = MODEL_FILE
    model.load_from_file()
    model.save()

    ret = model.run_simulation()
    assert ret.return_code == 0, "CADET reference simulation failed"

    times = model.root.output.solution.solution_times
    bulk = model.root.output.solution.unit_000.solution_bulk
    return times, bulk


def generate_measurements(times, true_P, true_S, noise_std=0.1, interval=5, seed=42):
    """Generate synthetic noisy measurements from true trajectory."""
    np.random.seed(seed)
    indices = list(range(0, len(times), interval))

    meas_P = [(times[i], true_P[i] + np.random.normal(0, noise_std)) for i in indices]
    meas_S = [(times[i], true_S[i] + np.random.normal(0, noise_std)) for i in indices]

    provider_P = DFProvider(
        name="P",
        dataframe=pd.DataFrame({"P": [meas_P]}),
        y_columns=["P"],
        noise=np.array([[noise_std**2]])
    )
    provider_S = DFProvider(
        name="S",
        dataframe=pd.DataFrame({"S": [meas_S]}),
        y_columns=["S"],
        noise=np.array([[noise_std**2]])
    )
    return provider_P, provider_S, meas_P, meas_S


def test_model_comparison(plot_results=False, tolerance=5e-3):
    """Verify CasADi and CADET produce matching CSTR trajectories."""
    cadet_times, cadet_bulk = run_cadet_reference()
    casadi_model = create_casadi_model()

    # Simulate CasADi step by step
    casadi_states = [INIT_C.copy()]
    casadi_times = [0.0]
    x = INIT_C.copy()

    for i in range(int(SIM_TIME / DT)):
        t = i * DT
        casadi_model.update_state(x, t)
        x = casadi_model.integrate(t + DT)
        casadi_states.append(x.copy())
        casadi_times.append(t + DT)

    casadi_states = np.array(casadi_states)
    casadi_times = np.array(casadi_times)

    # Interpolate CADET to CasADi time points and compare
    cadet_P_interp = np.interp(casadi_times, cadet_times, cadet_bulk[:, 0])
    cadet_S_interp = np.interp(casadi_times, cadet_times, cadet_bulk[:, 1])

    err_P = np.abs(casadi_states[:, 0] - cadet_P_interp)
    err_S = np.abs(casadi_states[:, 1] - cadet_S_interp)

    print(f"Model Comparison:")
    print(f"  P: MAE={np.mean(err_P):.6f}, Max={np.max(err_P):.6f}")
    print(f"  S: MAE={np.mean(err_S):.6f}, Max={np.max(err_S):.6f}")

    assert np.max(err_P) < tolerance, (
        f"P max error {np.max(err_P):.6f} exceeds tolerance {tolerance}"
    )
    assert np.max(err_S) < tolerance, (
        f"S max error {np.max(err_S):.6f} exceeds tolerance {tolerance}"
    )

    if plot_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(cadet_times, cadet_bulk[:, 0], 'b-', label='CADET P', linewidth=2)
        axes[0].plot(casadi_times, casadi_states[:, 0], 'r--', label='CasADi P', linewidth=2)
        axes[0].plot(cadet_times, cadet_bulk[:, 1], 'b:', label='CADET S', linewidth=2)
        axes[0].plot(casadi_times, casadi_states[:, 1], 'r-.', label='CasADi S', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Concentration')
        axes[0].set_title('Model Comparison')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(casadi_times, err_P, 'b-', label='Error P')
        axes[1].plot(casadi_times, err_S, 'r-', label='Error S')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title('CasADi vs CADET Error')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150)
        plt.show()


def test_enkf_casadi_vs_cadet(plot_results=False, tolerance=0.3):
    """Run EnKF with CasADi and CADET backends, verify both estimate well."""
    cadet_times, cadet_bulk = run_cadet_reference()
    true_P = cadet_bulk[:, 0]
    true_S = cadet_bulk[:, 1]

    # Generate measurements
    noise_std = 0.1
    provider_P, provider_S, meas_P, meas_S = generate_measurements(
        cadet_times, true_P, true_S, noise_std=noise_std
    )

    # EnKF with CasADi
    casadi_model = create_casadi_model()
    enkf_casadi = EnKalmanFilter(
        model=casadi_model,
        ensemble_size=50,
        initial_covariance=np.diag([0.2, 0.2]),
        providers=[provider_P, provider_S],
        random_seed=42
    )
    results_casadi = enkf_casadi.run_filter(
        t_start=0.0, t_end=SIM_TIME,
        use_measurement_times=True, interpolation='nearest'
    )

    # EnKF with CADET
    cadet_model = CadetModel(
        cadet_path=CADET_ROOT,
        init_state=INIT_C.copy(),
        model_path=MODEL_FILE,
        n_states=2,
        state_indices=[2, 3]
    )
    enkf_cadet = EnKalmanFilter(
        model=cadet_model,
        ensemble_size=50,
        initial_covariance=np.diag([0.2, 0.2]),
        providers=[provider_P, provider_S],
        random_seed=42
    )
    results_cadet = enkf_cadet.run_filter(
        t_start=0.0, t_end=SIM_TIME,
        use_measurement_times=True, interpolation='nearest'
    )

    # Calculate errors
    enkf_times_casadi = results_casadi['times']
    enkf_times_cadet = results_cadet['times']

    true_P_interp = np.interp(enkf_times_casadi, cadet_times, true_P)
    true_S_interp = np.interp(enkf_times_casadi, cadet_times, true_S)

    err_casadi_P = np.abs(results_casadi['states'][:, 0] - true_P_interp)
    err_casadi_S = np.abs(results_casadi['states'][:, 1] - true_S_interp)
    err_cadet_P = np.abs(results_cadet['states'][:, 0] - true_P_interp)
    err_cadet_S = np.abs(results_cadet['states'][:, 1] - true_S_interp)

    print(f"\nCasADi EnKF: P MAE={np.mean(err_casadi_P):.4f}, S MAE={np.mean(err_casadi_S):.4f}")
    print(f"CADET EnKF:  P MAE={np.mean(err_cadet_P):.4f}, S MAE={np.mean(err_cadet_S):.4f}")

    assert np.mean(err_casadi_P) < tolerance, (
        f"CasADi P MAE {np.mean(err_casadi_P):.4f} exceeds tolerance {tolerance}"
    )
    assert np.mean(err_cadet_P) < tolerance, (
        f"CADET P MAE {np.mean(err_cadet_P):.4f} exceeds tolerance {tolerance}"
    )

    cadet_model.end_simulation()

    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(cadet_times, true_P, 'k-', linewidth=2, label='True P')
        ax.plot([t for t, _ in meas_P], [m for _, m in meas_P],
                'ro', markersize=5, alpha=0.6, label='Measurements')
        ax.plot(enkf_times_casadi, results_casadi['states'][:, 0], 'b--', linewidth=2, label='CasADi EnKF')
        ax.plot(enkf_times_cadet, results_cadet['states'][:, 0], 'g:', linewidth=2, label='CADET EnKF')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration P')
        ax.set_title('Component P Estimation')
        ax.legend()
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(cadet_times, true_S, 'k-', linewidth=2, label='True S')
        ax.plot([t for t, _ in meas_S], [m for _, m in meas_S],
                'mo', markersize=5, alpha=0.6, label='Measurements')
        ax.plot(enkf_times_casadi, results_casadi['states'][:, 1], 'b--', linewidth=2, label='CasADi EnKF')
        ax.plot(enkf_times_cadet, results_cadet['states'][:, 1], 'g:', linewidth=2, label='CADET EnKF')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration S')
        ax.set_title('Component S Estimation')
        ax.legend()
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(enkf_times_casadi, err_casadi_P, 'b-', linewidth=2, label='CasADi Error')
        ax.plot(enkf_times_cadet, err_cadet_P, 'g-', linewidth=2, label='CADET Error')
        ax.set_xlabel('Time')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Component P Error')
        ax.legend()
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(enkf_times_casadi, err_casadi_S, 'b-', linewidth=2, label='CasADi Error')
        ax.plot(enkf_times_cadet, err_cadet_S, 'g-', linewidth=2, label='CADET Error')
        ax.set_xlabel('Time')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Component S Error')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('enkf_comparison_casadi_vs_cadet.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    test_model_comparison(plot_results=True)
    test_enkf_casadi_vs_cadet(plot_results=True)
