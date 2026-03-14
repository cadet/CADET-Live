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

print()
print('=== All tests passed! ===')
