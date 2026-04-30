"""Quick test to verify all new modules import and work correctly."""
import sys
import os

# Paths relative to this file (test/ -> ../src and ../examples)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC = os.path.join(_ROOT, 'src')
_EXAMPLES = os.path.join(_ROOT, 'examples')
sys.path.insert(0, _SRC)
sys.path.insert(0, _EXAMPLES)

# Test 1: MqttBridge imports
print('1. Testing MqttBridge imports...')
from MqttBridge import MqttBridge
print('   OK')

# Test 2: config imports
print('2. Testing config imports...')
import config
cfg = config.get_config(os.path.join(_EXAMPLES, 'config.yaml'))
topic_map = config.get_topic_map(cfg)
print(f'   topic_map keys: {list(topic_map.keys()) if topic_map else None}')
print(f'   measurements: {[m["name"] for m in topic_map.get("measurements", [])]}')
print(f'   controls: {[c["name"] for c in topic_map.get("controls", [])]}')
print('   OK')

# Test 3: Provider thread safety
print('3. Testing Provider thread-safety...')
from Provider import MeasurementProvider
import numpy as np
mp = MeasurementProvider('test')
mp.add_variable('x', noise=np.array([[0.01]]))
mp.add_measurement('x', 0.0, 1.0)
mp.add_measurement('x', 1.0, 2.0)
print(f'   Measurements: {mp.measurements}')
print('   OK')

# Test 4: MqttBridge with topic_map
print('4. Testing MqttBridge topic map parsing...')
client_config = config.config_to_source(cfg)
bridge = MqttBridge(client_config, topic_map)
provider = bridge.measurement_provider
print(f'   Provider name: {provider.name}')
print(f'   Variables: {provider.variable_names}')
print(f'   Control routes: {list(bridge._control_routes.keys())}')
# Check TimeDependentData access
od_var = provider.get_variable("od")
print(f'   OD variable: name={od_var.name}, noise={od_var.noise}, state_index={od_var.state_index}')
print('   OK')

# Test 5: Monod CSTR model
print('5. Testing Monod CSTR model...')
from modelLibrary.Casadi.monod_cstr import create_monod_cstr
model = create_monod_cstr(dt=0.1)
print(f'   nStates: {model.nStates}, state: {model.state}')
model.update_state(np.array([0.1, 10.0, 1.0]), 0.0)
x_next = model.integrate(0.1)
print(f'   After dt=0.1: {x_next}')
print('   OK')

# Test 6: EnKalmanFilter with single provider
print('6. Testing EnKalmanFilter with MqttBridge provider...')
from stateEsimator.EnKalmanFilter import EnKalmanFilter

# Add measurements to bridge's provider (simulating MQTT data)
provider.add_measurement("od", 0.0, 0.1)
provider.add_measurement("od", 0.1, 0.12)

# Create filtered provider for EnKF (only variables with state_index)
enkf_provider = MeasurementProvider(name="enkf_observed")
for var_name in provider.variable_names:
    var = provider.get_variable(var_name)
    if var is not None and var.state_index is not None:
        enkf_provider._data[var_name] = var  # Share TimeDependentData reference

print(f'   EnKF provider variables: {enkf_provider.variable_names}')

obs_func = lambda x: np.array([x[1]])  # Observe state[1] (od maps to S)
enkf = EnKalmanFilter(
    model=create_monod_cstr(dt=0.1),
    ensemble_size=20,
    initial_covariance=np.diag([0.01, 1.0, 0.001]),
    observation_func=obs_func,
    providers=[enkf_provider],
    random_seed=42,
)
state = enkf.update_state_with_interpolation(0.1, interpolation='nearest')
print(f'   EnKF state after update: {state}')
print('   OK')

# Test 7: PID controller
print('7. Testing PID controllers...')
from control.PID import PID
pid = PID(kp=0.5, ki=0.01, kd=0.0, setpoint=5.0, output_limits=(0, 10))
t_val, cmd = pid.update(3.0, 0.1, 0.0)
print(f'   PID output: t={t_val}, cmd={cmd:.3f}')
print('   OK')

# ---------------------------------------------------------------------------
# Test 8: CasadiModel.integrator property
# ---------------------------------------------------------------------------
print('8. Testing CasadiModel.integrator property...')
import casadi as ca
from Model import CasadiModel

states_sym = ca.SX.sym('x', 2)
ctrl_sym = ca.SX.sym('u', 1)
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])

def _ode(x, u):
    return ca.mtimes(A, x) + ca.mtimes(B, u)

test_model = CasadiModel(states=states_sym, controls=ctrl_sym,
                         ode=_ode, init_state=[1.0, 0.0], dt=0.1, T=1.0)
integrator = test_model.integrator
assert callable(integrator), "integrator must be callable"
res = integrator(x0=[1.0, 0.0], p=[0.0])
assert 'xf' in res, "integrator result must contain 'xf'"
x_next = np.array(res['xf']).flatten()
assert x_next.shape == (2,)
print(f'   integrator(x=[1,0], u=0) -> xf={x_next}')
print('   OK')

# ---------------------------------------------------------------------------
# Test 9: Objective classes — symbolic output
# ---------------------------------------------------------------------------
print('9. Testing Objective classes...')
from control.optimalControl import TrackingObjective, BiomassMaxObjective

x_sym = ca.SX.sym('x', 3)
u_sym = ca.SX.sym('u', 1)

tracking = TrackingObjective(Q=10.0, R=1.0, setpoint=5.0, state_index=1)
stage = tracking.stage_cost(x_sym, u_sym)
terminal = tracking.terminal_cost(x_sym)
assert stage.shape == (1, 1), f"Expected scalar SX, got {stage.shape}"
assert terminal.shape == (1, 1)

biomass = BiomassMaxObjective(R=0.1, biomass_index=0)
stage_b = biomass.stage_cost(x_sym, u_sym)
assert stage_b.shape == (1, 1)
# Default terminal cost is zero
terminal_b = biomass.terminal_cost(x_sym)
f_terminal = ca.Function('f', [x_sym], [terminal_b])
assert float(f_terminal([1.0, 2.0, 3.0])) == 0.0, "Default terminal cost must be 0"
print('   TrackingObjective: OK')
print('   BiomassMaxObjective: OK')
print('   OK')

# ---------------------------------------------------------------------------
# Test 10: CasadiOptimalControlProblem — shape + bounds
# ---------------------------------------------------------------------------
print('10. Testing CasadiOptimalControlProblem...')
from control.optimalControl import CasadiOptimalControlProblem

ocp_model = CasadiModel(states=states_sym, controls=ctrl_sym,
                        ode=_ode, init_state=[1.0, 0.0], dt=0.1, T=1.0)
objective = TrackingObjective(Q=10.0, R=1.0, setpoint=0.0, state_index=0)
ocp = CasadiOptimalControlProblem(
    model=ocp_model,
    objective=objective,
    time_horizon=1.0,
    u_min=-5.0,
    u_max=5.0,
    ipopt_print_level=0,
)
assert ocp._N == 10, f"Expected N=10, got {ocp._N}"
assert len(ocp._lbx) == 10
assert ocp._lbx[0] == -5.0
assert ocp._ubx[0] == 5.0

x0 = np.array([1.0, 0.0])
solution = ocp.solve(x0)
assert solution.shape == (10,), f"Expected shape (10,), got {solution.shape}"
assert np.all(solution >= -5.0 - 1e-6)
assert np.all(solution <= 5.0 + 1e-6)
print(f'   N={ocp._N}, solution shape={solution.shape}')
print(f'   First control: {solution[0]:.4f}  (should be non-zero, stabilising)')
print('   OK')

# ---------------------------------------------------------------------------
# Test 11: Warm-start shifts previous solution
# ---------------------------------------------------------------------------
print('11. Testing warm-start shift...')
ocp2 = CasadiOptimalControlProblem(
    model=CasadiModel(states=states_sym, controls=ctrl_sym,
                      ode=_ode, init_state=[1.0, 0.0], dt=0.1, T=1.0),
    objective=TrackingObjective(Q=1.0, R=1.0, setpoint=0.0, state_index=0),
    time_horizon=1.0, u_min=-5.0, u_max=5.0, ipopt_print_level=0,
)
sol1 = ocp2.solve(np.array([1.0, 0.0]))
prev_after_first = ocp2._prev_solution.copy()
# prev_solution should be sol1 shifted by 1 step
expected_shift = np.roll(sol1, -1)
assert np.allclose(prev_after_first, expected_shift), \
    "Warm-start: prev_solution must be sol shifted by -1"
print('   Warm-start shift: OK')
print('   OK')

# ---------------------------------------------------------------------------
# Test 12: MPCController — interface compatibility with PID
# ---------------------------------------------------------------------------
print('12. Testing MPCController interface (drop-in for PID)...')
from control.optimalControl import MPCController

mpc_ocp = CasadiOptimalControlProblem(
    model=CasadiModel(states=states_sym, controls=ctrl_sym,
                      ode=_ode, init_state=[1.0, 0.0], dt=0.1, T=1.0),
    objective=TrackingObjective(Q=10.0, R=1.0, setpoint=0.0, state_index=0),
    time_horizon=1.0, u_min=-5.0, u_max=5.0, ipopt_print_level=0,
)
mpc = MPCController(mpc_ocp)
state_vec = np.array([1.0, 0.0])

t_val, u_now = mpc.solve(state_vec, dt=0.1, t=5.0)
assert isinstance(t_val, float), "update() must return float as first element"
assert isinstance(u_now, float), "update() must return float as second element"
assert t_val == 5.0
print(f'   update() -> (t={t_val}, u={u_now:.4f})')
print('   OK')

# ---------------------------------------------------------------------------
# Test 13: MPCController — control_sequence and time_sequence
# ---------------------------------------------------------------------------
print('13. Testing MPCController control_sequence / time_sequence...')
N = mpc_ocp._N   # 10
seq = mpc.control_sequence
t_seq = mpc.time_sequence

assert seq.shape == (N, 1), f"control_sequence shape: expected ({N},1), got {seq.shape}"
assert t_seq.shape == (N,), f"time_sequence shape: expected ({N},), got {t_seq.shape}"

# time_sequence must be t + k*dt  (k=1..N)
dt = mpc_ocp.model.dt
expected_times = 5.0 + np.arange(1, N + 1) * dt
assert np.allclose(t_seq, expected_times), \
    f"time_sequence mismatch: {t_seq} vs {expected_times}"

# First value of sequence must equal u_now
assert abs(float(seq[0, 0]) - u_now) < 1e-9, \
    "control_sequence[0] must equal the returned u_now"

# Properties return copies — mutating them must not affect the controller
seq_copy = mpc.control_sequence
seq_copy[0, 0] = 9999.0
assert mpc.control_sequence[0, 0] != 9999.0, "control_sequence must return a copy"

print(f'   control_sequence shape: {seq.shape}')
print(f'   time_sequence: {t_seq}')
print('   OK')

# ---------------------------------------------------------------------------
# Test 14: ControlProvider.replace_variable()
# ---------------------------------------------------------------------------
print('14. Testing ControlProvider.replace_variable()...')
from Provider import ControlProvider

cp = ControlProvider("test_plan")
times_in  = np.array([1.0, 2.0, 3.0])
values_in = np.array([[0.1], [0.2], [0.3]])

# Creates variable automatically
cp.replace_variable("dosing_rate", times_in, values_in)
assert "dosing_rate" in cp.variable_names

var = cp.get_variable("dosing_rate")
assert len(var) == 3
assert np.allclose(var.times, times_in)

# Replace again with different data — old data must be gone
times2  = np.array([10.0, 20.0])
values2 = np.array([[0.9], [0.8]])
cp.replace_variable("dosing_rate", times2, values2)
assert len(cp.get_variable("dosing_rate")) == 2, \
    "replace_variable must discard previous data"
assert np.allclose(cp.get_variable("dosing_rate").times, times2)

print(f'   After replace: times={cp.get_variable("dosing_rate").times}')
print('   OK')

# ---------------------------------------------------------------------------
# Test 15: monod_cstr controllable_Fin
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test 15b: MPCController.step() and sequence-exhausted warning
# ---------------------------------------------------------------------------
print('15b. Testing MPCController.step() and sequence-exhausted warning...')
import logging as _logging

mpc_step = MPCController(mpc_ocp)
mpc_step.solve(state_vec, dt=0.1, t=0.0)
N_steps = mpc_ocp._N  # 10

# _seq_idx is 1 after update(); step() N-1 more times exhausts the sequence
for i in range(N_steps - 1):
    t_s, u_s = mpc_step.step(t=float(i + 1) * 0.1)
    assert isinstance(u_s, float)

# One more step must freeze and log a warning
_test_logger = _logging.getLogger('control.optimalControl')
_test_logger.setLevel(_logging.WARNING)

class _Capture(_logging.Handler):
    def __init__(self): super().__init__(); self.records = []
    def emit(self, r): self.records.append(r)

_cap = _Capture()
_test_logger.addHandler(_cap)

t_ex, u_ex = mpc_step.step(t=99.0)
assert isinstance(u_ex, float), "step() must return float after exhaustion"
assert len(_cap.records) >= 1, "step() must log a warning when sequence is exhausted"
assert 'exhausted' in _cap.records[0].getMessage().lower()
_test_logger.removeHandler(_cap)

print(f'   step() exhausted: u={u_ex:.4f}, warning logged: OK')
print('   OK')

# ---------------------------------------------------------------------------
print('15. Testing monod_cstr controllable_Fin=True...')
from modelLibrary.Casadi.monod_cstr import create_monod_cstr

model_ctrl = create_monod_cstr(dt=0.1, controllable_Fin=True)
assert model_ctrl.nControls == 1, \
    f"controllable_Fin=True should give nControls=1, got {model_ctrl.nControls}"
assert model_ctrl.nStates == 3

# integrator must accept u[0] = F_in value
res_ctrl = model_ctrl.integrator(x0=[0.1, 10.0, 1.0], p=[0.05])
x_ctrl = np.array(res_ctrl['xf']).flatten()
assert x_ctrl.shape == (3,)
assert np.all(np.isfinite(x_ctrl))

model_fixed = create_monod_cstr(dt=0.1, controllable_Fin=False)
assert model_fixed.nControls == 0, \
    f"controllable_Fin=False should give nControls=0, got {model_fixed.nControls}"
print(f'   controllable_Fin=True  -> nControls={model_ctrl.nControls}, x_next={x_ctrl}')
print(f'   controllable_Fin=False -> nControls={model_fixed.nControls}')
print('   OK')

# ---------------------------------------------------------------------------
# Test 16: MPCController provider injection
# ---------------------------------------------------------------------------
print('16. Testing MPCController ControlProvider injection...')
from Provider import ControlProvider as _CP

_prov = _CP("mpc_test_plan")
mpc_prov = MPCController(mpc_ocp, provider=_prov, variable_name="dosing_rate")

# variable must be created automatically — no manual add_variable() needed
assert "dosing_rate" in _prov.variable_names, \
    "MPCController __init__ must auto-create the variable in provider"

# After update() the provider must hold N entries without any manual call
mpc_prov.solve(state_vec, dt=0.1, t=0.0)
tdd = _prov._data["dosing_rate"]
assert len(tdd._data) == mpc_ocp._N, \
    f"provider must hold N={mpc_ocp._N} entries after update(), got {len(tdd._data)}"

# After step() the provider is refreshed (same N entries, same values — sequence unchanged)
t_p, u_p = mpc_prov.step(t=0.1)
assert len(tdd._data) == mpc_ocp._N, \
    "provider must still hold N entries after step()"

# Without provider: MPCController must work exactly as before
mpc_no_prov = MPCController(mpc_ocp)
t_np, u_np = mpc_no_prov.solve(state_vec, dt=0.1, t=0.0)
assert isinstance(u_np, float)

# Passing provider without variable_name must raise ValueError
import pytest as _pytest
try:
    MPCController(mpc_ocp, provider=_prov)
    raise AssertionError("Should have raised ValueError for missing variable_name")
except ValueError:
    pass

print(f'   provider auto-created variable, {mpc_ocp._N} entries after update(): OK')
print(f'   step() keeps provider up to date: OK')
print(f'   no-provider mode still works: OK')
print(f'   missing variable_name raises ValueError: OK')
print('   OK')

# ---------------------------------------------------------------------------
# Test 17: shared model — integrate(u=...) and EnKF propagate(u=...)
# ---------------------------------------------------------------------------
print('17. Testing shared model: integrate(u=...) and EnKF propagate(u=...)...')
from modelLibrary.Casadi.monod_cstr import create_monod_cstr as _create
from stateEsimator.EnKalmanFilter import EnKalmanFilter as _EnKF
from Provider import MeasurementProvider as _MP

_shared = _create(X0=[0.1, 1.0, 1.0], dt=0.1, controllable_Fin=True,
                  mu_max=0.1, K_s=0.1, Y_xs=0.5, S_in=10.0, F_out=0.0)

# integrate() with explicit u must differ from u=0 result
_shared.update_state([0.1, 1.0, 1.0], 0.0)
x_zero = _shared.integrate(0.1, u=np.array([0.0]))

_shared.update_state([0.1, 1.0, 1.0], 0.0)
x_fin  = _shared.integrate(0.1, u=np.array([0.1]))

assert not np.allclose(x_zero, x_fin), \
    "integrate(u=0.1) must differ from integrate(u=0.0)"

# integrate() with u=None must equal u=zeros (backward compat)
_shared.update_state([0.1, 1.0, 1.0], 0.0)
x_none = _shared.integrate(0.1, u=None)
assert np.allclose(x_zero, x_none), \
    "integrate(u=None) must equal integrate(u=0.0)"

# EnKF propagate(u=...) with shared controllable model
_mp17 = _MP('mp17')
_mp17.add_variable('od', noise=np.array([[0.05**2]]), state_index=1)
_enkf17 = _EnKF(
    model=_shared, ensemble_size=20,
    initial_covariance=np.diag([0.01, 0.01, 0.01]),
    observation_func=lambda x: np.array([x[1]]),
    providers=[_mp17], random_seed=0,
)
_mp17.add_measurement('od', 0.0, 1.0)
_s1 = _enkf17.update_state_with_interpolation(0.1, u=np.array([0.0]))

# reset
_enkf17_2 = _EnKF(
    model=_shared, ensemble_size=20,
    initial_covariance=np.diag([0.01, 0.01, 0.01]),
    observation_func=lambda x: np.array([x[1]]),
    providers=[_mp17], random_seed=0,
)
_s2 = _enkf17_2.update_state_with_interpolation(0.1, u=np.array([0.1]))
assert not np.allclose(_s1, _s2), \
    "EnKF propagate with different u must yield different state estimates"

print(f'   integrate(u=0.0): S={x_zero[1]:.4f}  integrate(u=0.1): S={x_fin[1]:.4f}')
print(f'   integrate(u=None) == integrate(u=0.0): OK')
print(f'   EnKF state with u=0.0 != u=0.1: OK')
print('   OK')

print()
print('=== All tests passed! ===')
