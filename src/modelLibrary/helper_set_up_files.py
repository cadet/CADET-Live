from pyexpat import model
from addict import Dict
import numpy as np

def cstr_setup(model, unit_number, parameters):

    ncomp = parameters["ncomp"]
    init_c = parameters["init_c"]
    sim_time = parameters["sim_time"]

    model.root.input.model.nunits = 1

    # CSTR - dynamically use unit_number
    unit_name = f"unit_{unit_number}"
    unit = getattr(model.root.input.model, unit_name)
    
    unit.unit_type = 'CSTR'
    unit.ncomp = ncomp
    unit.init_liquid_volume = 1.0
    unit.init_c = init_c
    unit.init_q = init_c
    unit.nbound = ncomp * [1.0]
    unit.const_solid_volume = 1.0
    unit.use_analytic_jacobian = 1

    #Configure solver settings
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    unit_return = getattr(model.root.input['return'], unit_name)
    unit_return.write_solution_bulk = 1
    unit_return.write_solution_inlet = 1
    unit_return.write_solution_outlet = 1
    unit_return.write_solution_solid = 1

    unit.adsorption_model = 'LINEAR'
    unit.adsorption.is_kinetic  = True
    unit.adsorption.lin_ka = [0,0,0]
    unit.adsorption.lin_kd = [0,0,0]

    return model


def setup_connections(model, parameters):
    
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [ ]

    return model


def mal_setup(model, unit_number, parameters):
    
    #Configure the reaction system
    unit_name = f"unit_{unit_number}"
    unit = getattr(model.root.input.model, unit_name)
    
    unit.reaction_model = "MASS_ACTION_LAW"
    unit.reaction_bulk.mal_kfwd_bulk = [parameters["kfwd"]]
    unit.reaction_bulk.mal_kbwd_bulk = [parameters["kbwd"]]

        # Stoichiometry matrix 2D array [components][reaction]
    unit.reaction_bulk.mal_stoichiometry_bulk = parameters["stoichiometric_matrix"]

    return model

def inlet_setup(model, unit_number, parameters):

    unit_name = f"unit_{unit_number}"
    unit = getattr(model.root.input.model, unit_name)

    unit.unit_type           = 'INLET'
    unit.inlet_type          = 'PIECEWISE_CUBIC_POLY'
    unit.ncomp               = parameters["ncomp"]

    return model

def outlet_setup(model, unit_number, parameters):

    unit_name = f"unit_{unit_number}"
    unit = getattr(model.root.input.model, unit_name)
    unit.unit_type = 'OUTLET'
    unit.ncomp = parameters["ncomp"]

    return model