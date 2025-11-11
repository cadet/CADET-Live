from cadet import Cadet

import os
import numpy as np

# Set up model parameters 
parameters = {
    "ncomp": 2,
    "init_c": [1.0, 0.0],
    "sim_time": 10.0,
    "kfwd": 0.5,
    "kbwd": 0.1,
    "stoichiometric_matrix": [[-1], [1]],
    "Q": 1.0,
}

# create Cadet model
cstr_model = Cadet(r"C:\Users\berger\CADET-Corev5\out\install\aRELEASE\bin\cadet-cli.exe")

# Set up inlet
cstr_model.root.input.model.unit_000.unit_type           = 'INLET'
cstr_model.root.input.model.unit_000.inlet_type          = 'PIECEWISE_CUBIC_POLY'
cstr_model.root.input.model.unit_000.ncomp               = parameters["ncomp"]


# CSTR - dynamically use unit_number
cstr_model.root.input.model.unit_001.unit_type = 'CSTR'
cstr_model.root.input.model.unit_001.ncomp = parameters["ncomp"]
cstr_model.root.input.model.unit_001.init_liquid_volume = 1.0
cstr_model.root.input.model.unit_001.init_c = parameters["init_c"]
cstr_model.root.input.model.unit_001.init_q = parameters["init_c"]
cstr_model.root.input.model.unit_001.nbound = parameters["ncomp"] * [1.0]
cstr_model.root.input.model.unit_001.const_solid_volume = 1.0
cstr_model.root.input.model.unit_001.use_analytic_jacobian = 1

#Configure solver settings
cstr_model.root.input.solver.user_solution_times = np.linspace(0, parameters["sim_time"], 1000)
cstr_model.root.input.solver.sections.nsec = 1
cstr_model.root.input.solver.sections.section_times = [0.0, parameters["sim_time"]]
cstr_model.root.input.solver.sections.section_continuity = []

cstr_model.root.input.model.solver.gs_type = 1
cstr_model.root.input.model.solver.max_krylov = 0
cstr_model.root.input.model.solver.max_restarts = 10
cstr_model.root.input.model.solver.schur_safety = 1e-8

cstr_model.root.input.solver.time_integrator.abstol = 1e-6
cstr_model.root.input.solver.time_integrator.algtol = 1e-10
cstr_model.root.input.solver.time_integrator.reltol = 1e-6
cstr_model.root.input.solver.time_integrator.init_step_size = 1e-6
cstr_model.root.input.solver.time_integrator.max_steps = 1000000
cstr_model.root.input.solver.consistent_init_mode = 1

cstr_model.root.input['return'].split_components_data = 0
cstr_model.root.input['return'].split_ports_data = 0
cstr_model.root.input['return'].write_solution_bulk = 1
cstr_model.root.input['return'].write_solution_inlet = 1
cstr_model.root.input['return'].write_solution_outlet = 1
cstr_model.root.input['return'].write_solution_solid = 1

cstr_model.root.input.model.unit_001.adsorption_model = 'LINEAR'
cstr_model.root.input.model.unit_001.adsorption.is_kinetic  = True
cstr_model.root.input.model.unit_001.adsorption.lin_ka = [0,0,0]
cstr_model.root.input.model.unit_001.adsorption.lin_kd = [0,0,0]

cstr_model.root.input.model.unit_001.reaction_model = "MASS_ACTION_LAW"
cstr_model.root.input.model.unit_001.reaction_bulk.mal_kfwd_bulk = [parameters["kfwd"]]
cstr_model.root.input.model.unit_001.reaction_bulk.mal_kbwd_bulk = [parameters["kbwd"]]
cstr_model.root.input.model.unit_001.reaction_bulk.mal_stoichiometry_bulk = parameters["stoichiometric_matrix"]


cstr_model.root.input.model.unit_002.unit_type = 'OUTLET'
cstr_model.root.input.model.unit_002.ncomp = parameters["ncomp"]

# add connections
cstr_model.root.input.model.connections.nswitches = 1
cstr_model.root.input.model.connections.switch_000.section = 0
cstr_model.root.input.model.connections.switch_000.connections = [
    0, 1, -1, -1, parameters["Q"], # unit_000, unit_001, all components, all components, Q/ m^3/s
    1, 2, -1, -1, parameters["Q"], #unit_001 unit_002, all components, all components, Q/ m^3/s
]

cstr_model.root.input.model.unit_000.sec_000.const_coeff = [0.0, 0.0] # mol / m^3
cstr_model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,0.0]
cstr_model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,0.0]
cstr_model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,0.0]

# save as h5 file
current_script_name = os.path.splitext(os.path.basename(__file__))[0]
model_filename = os.path.join(os.path.dirname(__file__), f'{current_script_name}.h5')


# run simulation and save model
cstr_model.filename = model_filename
cstr_model.save()
cstr_model_results = cstr_model.run_simulation()
if cstr_model_results.return_code != 0:
    print(cstr_model_results.error_message)
else:   
    print("Happy")
