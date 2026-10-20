from types import SimpleNamespace
from unittest.mock import Mock

import casadi as ca
import numpy as np
import pytest
from cadet_live.model import CadetModel, CasadiModel
from cadet_live.model import model as model_module


def test_casadi_autonomous_system_matches_analytic_solution() -> None:
    states = ca.SX.sym("x", 2)
    model = CasadiModel(states, lambda x: ca.vertcat(-x[0], 2 * x[0] - x[1]))
    model.set_state(np.array([2.0, 1.0]), time=1.0)

    first = model.simulate_timestep(1.4)
    first[0] = 100.0
    second = model.simulate_timestep(2.0)

    np.testing.assert_allclose(second, [2 * np.exp(-1), 5 * np.exp(-1)], rtol=1e-5)
    assert model.t_curr == 2.0
    assert model.state[0] != 100.0


def test_casadi_control_changes_between_unequal_steps() -> None:
    states = ca.SX.sym("x")
    controls = ca.SX.sym("u")
    model = CasadiModel(states, lambda x, u: u - x, controls=controls)
    model.set_state(np.array([2.0]))

    first = model.simulate_timestep(0.5, controls=np.array([3.0]))
    second = model.simulate_timestep(1.25, controls=np.array([-1.0]))

    expected_first = 3.0 - np.exp(-0.5)
    expected_second = -1.0 + (expected_first + 1.0) * np.exp(-0.75)
    np.testing.assert_allclose(first, [expected_first], rtol=1e-5)
    np.testing.assert_allclose(second, [expected_second], rtol=1e-5)
    assert model.t_curr == 1.25


def test_casadi_run_simulation_records_nonuniform_trajectory() -> None:
    states = ca.SX.sym("x")
    controls = ca.SX.sym("u")
    model = CasadiModel(states, lambda x, u: u, controls=controls)
    model.set_state(np.array([1.0]), time=2.0)
    times = np.array([2.0, 2.25, 3.0])
    control_values = np.array([2.0])

    result = model.run_simulation(times, controls=control_values)

    np.testing.assert_array_equal(result.time, times)
    np.testing.assert_allclose(result.state[:, 0], [1.0, 1.5, 3.0], rtol=1e-5)
    np.testing.assert_array_equal(result.controls, control_values)
    np.testing.assert_allclose(model.state, [3.0], rtol=1e-5)


def test_casadi_initial_state_is_copied_and_example_runs() -> None:
    from examples.modelLibrary.Casadi.monod_cstr import create_monod_cstr

    initial_state = np.array([0.2, 5.0, 1.0])
    model = create_monod_cstr(X0=initial_state)
    initial_state[0] = 100.0

    np.testing.assert_array_equal(model.state, [0.2, 5.0, 1.0])
    result = model.run_simulation(np.array([0.0, 0.1, 0.2]))
    assert result.state.shape == (3, 3)
    assert result.state[-1, 0] > result.state[0, 0]


def test_cadet_single_unit_lifecycle_and_state_snapshots(monkeypatch) -> None:
    cadet = Mock()
    cadet.root.input.model.nunits = 1
    cadet.initialize_simulation.return_value = SimpleNamespace(return_code=0)
    cadet.update_state.return_value = SimpleNamespace(return_code=0)
    cadet.perform_simulation_step.return_value = (SimpleNamespace(return_code=0), 1.45)
    cadet.cadet_runner.res.last_state_y.return_value = np.array([4.0, 5.0])
    monkeypatch.setattr(model_module, "Cadet", Mock(return_value=cadet))

    model = CadetModel(model_path="model.h5", cadet_path="/cadet")
    model.initialise_simulation()
    initial_state = np.array([1.0, 2.0])
    model.set_state(initial_state, time=0.5)
    stepped_state = model.simulate_timestep(1.5)
    stepped_state[0] = 99.0
    model_state = model.state
    model_state[1] = 99.0

    model_module.Cadet.assert_called_once_with(install_path="/cadet", use_dll=True)
    cadet.load_from_file.assert_called_once_with()
    cadet.save.assert_not_called()
    cadet.update_state.assert_called_once_with(0, initial_state, 0.5)
    cadet.perform_simulation_step.assert_called_once_with(1.5)
    np.testing.assert_array_equal(model.state, [4.0, 5.0])
    assert model.t_curr == 1.45
    model.end_simulation()
    cadet.end_simulation.assert_called_once_with()



def test_cadet_failed_operations_preserve_state_and_report_errors(monkeypatch) -> None:
    cadet = Mock()
    cadet.root.input.model.nunits = 1
    cadet.initialize_simulation.return_value = SimpleNamespace(
        return_code=1, error_message="initialization failed"
    )
    cadet.update_state.return_value = SimpleNamespace(return_code=0)
    cadet.perform_simulation_step.return_value = (
        SimpleNamespace(return_code=2, error_message="step failed"),
        1.0,
    )
    monkeypatch.setattr(model_module, "Cadet", Mock(return_value=cadet))
    model = CadetModel(model_path="model.h5", cadet_path="/cadet")

    with pytest.raises(RuntimeError, match="initialization failed"):
        model.initialise_simulation()
    model.set_state(np.array([1.0, 2.0]), time=0.25)
    cadet.update_state.return_value = SimpleNamespace(
        return_code=3, error_message="state update failed"
    )
    with pytest.raises(RuntimeError, match="state update failed"):
        model.set_state(np.array([3.0, 4.0]), time=0.5)
    with pytest.raises(RuntimeError, match="step failed"):
        model.simulate_timestep(1.0)

    np.testing.assert_array_equal(model.state, [1.0, 2.0])
    assert model.t_curr == 0.25
