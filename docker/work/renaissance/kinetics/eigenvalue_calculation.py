from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.core import *
from skimpy.mechanisms import *
from scipy.linalg import eigvals as eigenvalues
from sympy import Symbol
from skimpy.core.parameters import ParameterValues
import numpy as np

def calc_eigenvalues(kinetic_model, param_val, flux_series, concentration_series, k_eq):
    
    param_val = ParameterValues(param_val, kinetic_model)
    kinetic_model.parameters = k_eq
    kinetic_model.parameters = param_val
    parameter_sample = {v.symbol: v.value for k, v in kinetic_model.parameters.items()}

    # Set all vmax/flux parameters to 1.
    # TODO Generalize into Flux and Saturation parameters
    for this_reaction in kinetic_model.reactions.values():
        vmax_param = this_reaction.parameters.vmax_forward
        parameter_sample[vmax_param.symbol] = 1

    kinetic_model.flux_parameter_function(
        kinetic_model,
        parameter_sample,
        concentration_series,
        flux_series
    )
    for c in concentration_series.index:
        if c in kinetic_model.parameters:
            c_sym = kinetic_model.parameters[c].symbol
            parameter_sample[c_sym] = concentration_series[c]
    this_jacobian = kinetic_model.jacobian_fun(flux_series[kinetic_model.reactions],
                                        concentration_series[kinetic_model.reactants], parameter_sample)

    this_real_eigenvalues = sorted(np.real(eigenvalues(this_jacobian.todense())))

    this_param_sample = ParameterValues(parameter_sample, kinetic_model)
    return this_param_sample, this_real_eigenvalues

def parameter_sampling(kinetic_model, param_val, flux_series, concentration_series):
    
    param_val = ParameterValues(param_val, kinetic_model)
    kinetic_model.parameters = param_val
    parameter_sample = {v.symbol: v.value for k, v in kinetic_model.parameters.items()}

    # Set all vmax/flux parameters to 1.
    for this_reaction in kinetic_model.reactions.values():
        vmax_param = this_reaction.parameters.vmax_forward
        parameter_sample[vmax_param.symbol] = 1

    kinetic_model.flux_parameter_function(
        kinetic_model,
        parameter_sample,
        concentration_series,
        flux_series
    )

    for c in concentration_series.index:
        if c in kinetic_model.parameters:
            c_sym = kinetic_model.parameters[c].symbol
            parameter_sample[c_sym] = concentration_series[c]

    this_param_sample = ParameterValues(parameter_sample, kinetic_model)

    return this_param_sample
