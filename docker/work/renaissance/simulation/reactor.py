import numpy as np
import matplotlib as plt
import pandas as pd
import time
from pytfa.optim.constraints import *

import time

def simulate_bioreactor(parameter_set, 
                          steady_state_sample, 
                          kinetic_model, 
                          concentrations, 
                          reactor,
                          reactor_volume,
                          max_time,
                          total_time,
                          steps):
    """
    Run the simulation for a given parameter set on a given kinetic model
    """

    # Load tfa sample and kinetic parameters into kinetic model
    kinetic_model.parameters = parameter_set

    # Fetch equilibrium constants
    # load_equilibrium_constants(steady_state_sample, tmodel, kinetic_model,
    #                         concentration_scaling=CONCENTRATION_SCALING,
    #                         in_place=True)
     
    reactor = reset_reactor(reactor, parameter_set, steady_state_sample, concentrations, reactor_volume)

    start = time.time()
    if hasattr(reactor, 'solver'):
        delattr(reactor, 'solver')

        # Function to stop integration
    def rootfn(t, y, g, user_data):
        t_0 = user_data['time_0']
        t_max = user_data['max_time']
#         print(t)
        curr_t = time.time()
        if (curr_t - t_0) >= t_max:
            g[0] = 0
            # print('Did not converge in time')
        else:
            g[0] = 1


    start = time.time()
    user_data = {'time_0': start,
                 'max_time': max_time} #2 minutes in seconds

    # Solve the ODEs
    sol_ode_wildtype = reactor.solve_ode(np.linspace(0, total_time, steps),
        solver_type='cvode',
        rtol=1e-9,
        atol=1e-9,
        max_steps=1e9,
        rootfn=rootfn,
        nr_rootfns=1,
        user_data=user_data)
    end = time.time()
    print("Compelted in {:.2f} seconds".format(end - start))

    final_biomass = sol_ode_wildtype.concentrations.iloc[-1]['biomass_strain_1'] * 0.28e-12 / 0.05
    final_anthranilate = sol_ode_wildtype.concentrations.iloc[-1]['anth_e'] * 1e-9 * 136.13
    final_glucose = sol_ode_wildtype.concentrations.iloc[-1]['glc_D_e'] * 1e-9 * 180.156
    print("Final biomass is : {}, final anthranilate is : {}, final glucose is : {}".format(final_biomass,
                                                                                                final_anthranilate,
                                                                                                final_glucose))
    return sol_ode_wildtype, final_biomass, final_anthranilate, final_glucose


def reset_reactor(reactor, parameter_set, steady_state_sample, concentrations, reactor_volume):
    """
    Function to reset the reactor and load the concentrations and parameters before each simulation
    """
    # Parameterize the rector and initialize with incolum and medium data
    reactor.parametrize(parameter_set, 'strain_1')
    reactor.initialize(concentrations, 'strain_1')
    reactor.initial_conditions['biomass_strain_1'] = 0.037 * 0.05 / 0.28e-12

    for met_ in reactor.medium.keys():
        LC_id = 'LC_' + met_
        LC_id = LC_id.replace('_L', '-L')
        LC_id = LC_id.replace('_D', '-D')
        reactor.initial_conditions[met_] = np.exp(steady_state_sample.loc[LC_id]) * 1e9

    # Volume parameters for the reactor
    reactor.models.strain_1.parameters.strain_1_volume_e.value = reactor_volume
    reactor.models.strain_1.parameters.strain_1_cell_volume_e.value = 1.0  # 1.0 #(mum**3)
    reactor.models.strain_1.parameters.strain_1_cell_volume_c.value = 1.0  # 1.0 #(mum**3)
    reactor.models.strain_1.parameters.strain_1_cell_volume_p.value = 1.0  # 1.0 #(mum**3)
    reactor.models.strain_1.parameters.strain_1_volume_c.value = 0.9 * 1.0  # (mum**3)
    reactor.models.strain_1.parameters.strain_1_volume_p.value = 0.1 * 1.0  # (mum**3)

    return reactor
