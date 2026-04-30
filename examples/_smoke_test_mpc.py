"""Headless smoke-test for the offline MPC example (no matplotlib window)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')

from modelLibrary.Casadi.monod_cstr import create_monod_cstr
from stateEsimator.EnKalmanFilter import EnKalmanFilter
from Provider import MeasurementProvider, ControlProvider
from control.PID import PID
from control.optimalControl import (
    TrackingObjective, CasadiOptimalControlProblem, MPCController,
)

RNG_SEED = 0; DT = 1.0; T_END = 20.0; N_STEPS = int(T_END / DT)
S_SETPOINT = 4.0
MONOD_PARAMS = dict(mu_max=0.1, K_s=0.1, Y_xs=0.5, S_in=10.0, F_out=0.0)

rng = np.random.default_rng(RNG_SEED)

plant     = create_monod_cstr(X0=[0.10, 1., 1.], dt=DT, controllable_Fin=True, controllable_Fout=True, **MONOD_PARAMS)
model     = create_monod_cstr(X0=[0.15, 1.5, 1.], dt=DT, controllable_Fin=True, controllable_Fout=True, **MONOD_PARAMS)

mp = MeasurementProvider('od_sim')
mp.add_variable('od', noise=np.array([[0.05**2]]), state_index=1)

enkf = EnKalmanFilter(
    model=model, ensemble_size=50,
    initial_covariance=np.diag([0.01, 2., 0.01]),
    observation_func=lambda x: np.array([x[1]]),
    providers=[mp], random_seed=RNG_SEED,
)

ocp = CasadiOptimalControlProblem(
    model=model,
    objective=TrackingObjective(Q=20., R=0.5, setpoint=S_SETPOINT, state_index=1),
    time_horizon=20., u_min=[0., 0.], u_max=[0.2, 0.2], ipopt_print_level=0,
)
mpc = MPCController(ocp)

pid_plant = create_monod_cstr(X0=[0.10, 1., 1.], dt=DT, controllable_Fin=True, controllable_Fout=True, **MONOD_PARAMS)
pid = PID(kp=0.05, ki=0.01, kd=0.0, setpoint=S_SETPOINT, output_limits=(0., 0.2))

mpc_plan_provider = ControlProvider('mpc_plan')
mpc = MPCController(ocp, provider=mpc_plan_provider, variable_name='dosing_rate')

x_true = np.array([0.10, 1., 1.])
x_pid  = np.array([0.10, 1., 1.])
s_hist_mpc = [x_true[1]]
s_hist_pid = [x_pid[1]]
u_mpc_last = np.zeros(2)   # [F_in, F_out]

for k in range(N_STEPS):
    t = k * DT
    od = max(0.0, float(x_true[1]) + rng.normal(0, 0.05))
    mp.add_measurement('od', t, od)

    est = enkf.update_state_with_interpolation(t_end=t + DT, interpolation='nearest',
                                               u=u_mpc_last)
    _, _ = mpc.solve(est, DT, t)
    u_vec = mpc.current_control          # [F_in, F_out]
    u_mpc_last = u_vec

    res = plant.integrator(x0=x_true, p=u_vec)
    x_true = np.clip(np.array(res['xf']).flatten() + rng.normal(0, [1e-3, 1e-2, 1e-4]), 0, None)
    s_hist_mpc.append(x_true[1])

    _, u_pid = pid.update(od, DT, t)
    res_pid = pid_plant.integrator(x0=x_pid, p=[u_pid])
    x_pid = np.clip(np.array(res_pid['xf']).flatten() + rng.normal(0, [1e-3, 1e-2, 1e-4]), 0, None)
    s_hist_pid.append(x_pid[1])

rmse_mpc = float(np.sqrt(np.mean((np.array(s_hist_mpc) - S_SETPOINT)**2)))
rmse_pid = float(np.sqrt(np.mean((np.array(s_hist_pid) - S_SETPOINT)**2)))

print('Simulation OK')
print(f'Final S (MPC): {x_true[1]:.3f} g/L  (setpoint={S_SETPOINT})')
print(f'Final S (PID): {x_pid[1]:.3f} g/L')
print(f'RMSE MPC: {rmse_mpc:.4f}')
print(f'RMSE PID: {rmse_pid:.4f}')

# Verify MPC plan provider has entries by checking internal TimeDependentData
tdd = mpc_plan_provider._data.get('dosing_rate')
assert tdd is not None, "mpc_plan_provider should have 'dosing_rate' variable"
assert len(tdd._data) > 0, "mpc_plan_provider should have data entries"
last_u = float(np.atleast_1d(tdd._data[-1][1])[0])
print(f'MPC plan provider: OK  ({len(tdd._data)} entries, last dosing_rate={last_u:.4f})')

print('\n=== Smoke test PASSED ===')
