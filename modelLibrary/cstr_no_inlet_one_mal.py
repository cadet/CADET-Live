from cadet import Cadet

import os
import numpy as np

# Set up model parameters 
parameters = {
    "ncomp": 2,
    "init_c": [1.0, 0.0],
    "sim_time": 2.0,
}

# create Cadet model
cstr_model = Cadet(r"/Users/berger/fzj/cadet/CADET-Core/install_release")

ncomp = parameters["ncomp"]
init_c = parameters["init_c"]
sim_time = parameters["sim_time"]

cstr_model.root.input.model.nunits = 1

# CSTR - dynamically use unit_number
cstr_model.root.input.model.unit_000.unit_type = 'CSTR'
cstr_model.root.input.model.unit_000.ncomp = ncomp
cstr_model.root.input.model.unit_000.init_liquid_volume = 1.0
cstr_model.root.input.model.unit_000.init_c = init_c
cstr_model.root.input.model.unit_000.const_solid_volume = 1.0
cstr_model.root.input.model.unit_000.use_analytic_jacobian = 1

#Configure solver settings
cstr_model.root.input.solver.user_solution_times = np.linspace(0, parameters['sim_time'], 50)
cstr_model.root.input.solver.sections.nsec = 1
cstr_model.root.input.solver.sections.section_times = [0.0, parameters['sim_time']]
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

cstr_model.root.input['return'].unit_000.split_components_data = 0
cstr_model.root.input['return'].unit_000.split_ports_data = 0
cstr_model.root.input['return'].unit_000.write_solution_bulk = 1
cstr_model.root.input['return'].unit_000.write_solution_inlet = 1
cstr_model.root.input['return'].unit_000.write_solution_outlet = 1
cstr_model.root.input['return'].unit_000.write_solution_solid = 1

cstr_model.root.input.model.unit_000.nreac_liquid = 1
cstr_model.root.input.model.unit_000.liquid_reaction_000.type = 'MASS_ACTION_LAW'
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_stoichiometry = [[-1],[1]]
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_kfwd = 3
cstr_model.root.input.model.unit_000.liquid_reaction_000.mal_kbwd = 2


# add connections
cstr_model.root.input.model.connections.nswitches = 1
cstr_model.root.input.model.connections.switch_000.section = 0
cstr_model.root.input.model.connections.switch_000.connections = [ ]

# save as h5 file
model_filename = "cstr_no_inlet_one_mal.h5"


# run simulation and save model
cstr_model.filename = model_filename
cstr_model.save()
cstr_model_results = cstr_model.run_simulation()
if cstr_model_results.return_code != 0:
    print(cstr_model_results.error_message)
else:
    print("Happy")

#plot results
time_points = cstr_model.root.output.solution.solution_times
concentration_comp0 = cstr_model.root.output.solution.unit_000.solution_bulk[:,0]
concentration_comp1 = cstr_model.root.output.solution.unit_000.solution_bulk[:,1]

import matplotlib.pyplot as plt
plt.plot(time_points, concentration_comp0, label='Component 0')
plt.plot(time_points, concentration_comp1, label='Component 1')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()