import os
import shutil
from pathlib import Path

import casadi as ca
import numpy as np
import pytest
from cadet import Cadet
from cadet_live.model import CadetModel, CasadiModel
from packaging.version import Version
from scipy.integrate import solve_ivp


def test_casadi_monod_cstr_matches_scipy_with_changing_feed() -> None:
    states = ca.SX.sym("state", 3)
    feed = ca.SX.sym("feed")
    biomass, substrate, volume = states[0], states[1], states[2]
    growth = 0.5 * substrate / (0.3 + substrate)
    ode = ca.Function(
        "monod_cstr",
        [states, feed],
        [
            ca.vertcat(
                growth * biomass - feed * biomass / volume,
                -growth * biomass / 0.6 + feed * (8.0 - substrate) / volume,
                feed - 0.05,
            )
        ],
    )
    model = CasadiModel(states, ode, controls=feed)
    initial_state = np.array([0.2, 4.0, 1.0])
    model.set_state(initial_state)
    times = np.array([0.0, 0.35, 0.9, 1.5, 2.0])
    feed_rates = [0.08, 0.02, 0.1, 0.04]
    expected = initial_state.copy()

    for start, end, rate in zip(times[:-1], times[1:], feed_rates):

        def reference_ode(_time, state):
            biomass_value, substrate_value, volume_value = state
            growth_value = 0.5 * substrate_value / (0.3 + substrate_value)
            return [
                growth_value * biomass_value - rate * biomass_value / volume_value,
                -growth_value * biomass_value / 0.6
                + rate * (8.0 - substrate_value) / volume_value,
                rate - 0.05,
            ]

        expected = solve_ivp(
            reference_ode, (start, end), expected, rtol=1e-11, atol=1e-12
        ).y[:, -1]
        actual = model.simulate_timestep(end, controls=np.array([rate]))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-6)
        assert model.t_curr == end

    np.testing.assert_allclose(model.state, expected, rtol=2e-4, atol=1e-6)


@pytest.mark.slow
def test_cadet_stepped_adsorption_matches_full_cadet_run(tmp_path) -> None:
    cadet_path = os.getenv("CADET_INSTALL_PATH")
    if cadet_path is None:
        pytest.skip(
            "Set CADET_INSTALL_PATH to a CADET-Core installation with C API >= 1.1.0a2"
        )

    reference = Cadet(install_path=cadet_path, use_dll=True)
    if reference.cadet_runner._cadet_capi_version < Version("1.1.0a2"):
        pytest.skip("CADET-Core C API >= 1.1.0a2 is required for stepped simulation")

    source_path = (
        Path(__file__).resolve().parents[1]
        / "examples/models/CADET/only_cstr_no_reac.h5"
    )
    reference_path = tmp_path / "reference.h5"
    stepped_path = tmp_path / "stepped.h5"
    shutil.copyfile(source_path, reference_path)
    reference.filename = str(reference_path)
    reference.load_from_file()
    unit = reference.root.input.model.unit_000
    unit.init_c = np.array([1.0])
    unit.init_q = np.array([0.0])
    unit.adsorption.lin_ka = np.array([1.0])
    unit.adsorption.lin_kd = np.array([0.2])
    times = np.array([0.0, 0.25, 0.5, 1.0])
    reference.root.input.solver.user_solution_times = times
    reference.save()
    shutil.copyfile(reference_path, stepped_path)

    reference_result = reference.run_simulation()
    assert reference_result.return_code == 0, reference_result.error_message
    expected_bulk = reference.root.output.solution.unit_000.solution_bulk[:, 0]
    expected_solid = reference.root.output.solution.unit_000.solution_solid[:, 0]
    assert expected_bulk[-1] < expected_bulk[0]
    assert expected_solid[-1] > expected_solid[0]

    model = CadetModel(model_path=str(stepped_path), cadet_path=cadet_path)
    model.initialise_simulation()
    try:
        initial_state = model.model.cadet_runner.res.last_state_y().copy()
        model.set_state(initial_state)
        result = model.run_simulation(times)
    finally:
        model.end_simulation()

    np.testing.assert_array_equal(result.time, times)
    assert result.state.shape == (len(times), len(initial_state))
    assert model.t_curr == pytest.approx(times[-1])
    for expected in (expected_bulk, expected_solid):
        maximum_errors = np.max(np.abs(result.state - expected[:, None]), axis=0)
        assert np.min(maximum_errors) < 1e-3
