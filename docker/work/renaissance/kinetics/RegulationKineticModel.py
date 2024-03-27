import pandas as pd
import numpy as np
import yaml
import time 
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.core import *
from skimpy.mechanisms import *
from scipy.linalg import eigvals as eigenvalues
from sympy import Symbol
from skimpy.core.parameters import ParameterValues
from skimpy.io.regulation import load_enzyme_regulation
from skimpy.core.reactor import Reactor
from skimpy.utils.general import make_subclasses_dict


class RegulationKineticModel():
    """
    This class represents a kinetic model that has the capability of 
    loading regulation, omics data, calculate eigenvalue and sampling parameters
    """
    def __init__(self, NCPU, CONCENTRATION_SCALING, TIME_SCALING, DENSITY, GDW_GWW_RATIO, N_SAMPLES):
        self.NCPU = NCPU
        self.CONCENTRATION_SCALING = CONCENTRATION_SCALING # 1 mol to 1 mmol
        self.TIME_SCALING = TIME_SCALING
        # Parameters of the E. Coli cell
        self.DENSITY = DENSITY
        self.GDW_GWW_RATIO = GDW_GWW_RATIO
        self.N_SAMPLES = N_SAMPLES
        self.jacobian_set = []
        self.parameter_sample_set = []

    def load_model(self, path_to_kmodel, path_to_regulation=None):
        """
        Load a regulation model from specified paths and add regulation data to the kinetic model.

        Parameters:
        path_to_kmodel (str): Path to the kinetic model file.
        path_to_regulation (str): Path to the regulation file.
        """
        self.kmodel = load_yaml_model(path_to_kmodel)
        if path_to_regulation:
            # Add regulation data to kinetic model
            df = pd.read_csv(path_to_regulation)
            df_regulations_all = df[df['reaction_id'].isin(list(self.kmodel.reactions.keys()))]
            df_regulations_all = df_regulations_all[df_regulations_all['regulator'].isin(list(self.kmodel.reactants.keys()))]
            self.kmodel = load_enzyme_regulation(self.kmodel, df_regulations_all)
            print("Regulation data added to the kinetic model.")
        self.kmodel.prepare()
        self.kmodel.compile_jacobian(sim_type=QSSA, ncpu=self.NCPU)
        print("Model loaded.")

    def load_generated_kinetic_parameters(self):
        """
        Return the kinetic parameters and the regulation parameters from the model.

        Returns:
            k_names (list): A list of parameter names.
            k_regulation (list): A list of regulation parameter names.
        """
        k_names = []
        k_regulation = []

        for k, v in self.kmodel.parameters.items():
            if k.startswith("km_") or "activator" in k or "inhibitor" in k or "activation" in k or "inhibition" in k:
                if k not in k_names:
                    k_names.append(k)
                    
                if "activator" in k or "inhibitor" in k or "activation" in k or "inhibition" in k:
                    k_regulation.append(k)

        print(f"The number of k for generation is {len(k_names)}, \nand the number of regulation parameters is {len(k_regulation)}.")
        self.generated_k_names = k_names
        self.regulation_k_names = k_regulation

    def load_steady_state_sample(self, path_to_tmodel, path_to_steady_state_samples, ss_idx):
        """
        Load a steady state sample and thermodynamic models into the model.
        
        Parameters:
        - path_to_tmodel (str): Path to the JSON model file.
        - path_to_steady_state_samples (str): Path to the steady state samples file.
        - ss_idx (int): Index of the steady state samples.
        """
        self.tmodel = load_json_model(path_to_tmodel)

        self.samples = pd.read_csv(path_to_steady_state_samples, header=0, index_col=0).iloc[ss_idx, 0:]

        flux_dict = load_fluxes(self.samples, self.tmodel, self.kmodel,
                                    density=self.DENSITY,
                                    ratio_gdw_gww=self.GDW_GWW_RATIO,
                                    concentration_scaling=self.CONCENTRATION_SCALING,
                                    time_scaling=self.TIME_SCALING)

        conc_dict = load_concentrations(self.samples, self.tmodel, self.kmodel,
                                        concentration_scaling=self.CONCENTRATION_SCALING)

        fluxes = pd.Series([flux_dict[reaction.name] for reaction in self.kmodel.reactions.values()])
        concentrations = pd.Series([conc_dict[variable] for variable in self.kmodel.variables.keys()])

        # Fetch equilibrium constants
        k_eq = load_equilibrium_constants(self.samples, self.tmodel, self.kmodel,
                                        concentration_scaling=self.CONCENTRATION_SCALING,
                                        in_place=True)

        self.flux_series = fluxes
        self.conc_series = concentrations
        self.conc_dict = conc_dict
        self.flux_dict = flux_dict
        self.k_eq = k_eq
        self.model_parameters = self.kmodel.parameters


    def parameter_sampling(self, n_samples=1):

        sampling_parameters = SimpleParameterSampler.Parameters(n_samples=n_samples)
        sampler = SimpleParameterSampler(sampling_parameters)
        sampler._compile_sampling_functions(self.kmodel, self.conc_dict)

    def calc_max_eigenvalues(self):
        """
        Calculate eigenvalues, set vmax/flux parameters to 1, and store the eigenvalues.
        """
        #print('Calculating eigenvalues.....')
        #self.parameter_set = self.parameter_set.drop('Unnamed: 0', axis=1)
        store_eigen = []
        param_population=[]
        for j in range(len(self.parameter_set.index)):
            param_val = self.parameter_set.loc[j]*self.CONCENTRATION_SCALING  ####
            param_val = ParameterValues(param_val,self.kmodel)

            # Load parameters from multiple sources
            self.kmodel.parameters = self.k_eq
            self.kmodel.parameters = param_val
            self.parameter_sample= {v.symbol: v.value for k,v in self.kmodel.parameters.items()}
   
            for this_reaction in self.kmodel.reactions.values():
                vmax_param = this_reaction.parameters.vmax_forward
                self.parameter_sample[vmax_param.symbol] = 1
            
            model_parameters = self.kmodel.parameters
            for k, v in self.conc_dict.items():
                if str(k) in model_parameters:
                    self.parameter_sample[Symbol(k)] = v

            self.kmodel.flux_parameter_function(
                self.kmodel, self.parameter_sample, self.conc_dict, self.flux_dict
            )
            for c in self.conc_series.index:
                if c in self.model_parameters:
                    c_sym = self.kmodel.parameters[c].symbol
                    self.parameter_sample[c_sym]=self.conc_series[c]
            self.parameter_sample_set.append(self.parameter_sample)

            # Calculate eigenvalues
            this_jacobian = self.kmodel.jacobian_fun(self.flux_series,
                                    self.conc_series,self.parameter_sample)
            self.jacobian_set.append(this_jacobian)
            this_real_eigenvalues = np.max(np.real(eigenvalues(this_jacobian.todense())))

            store_eigen.append(this_real_eigenvalues)
            this_param_sample = ParameterValues(self.parameter_sample, self.kmodel)
            param_population.append(this_param_sample)

            del(this_param_sample)
            del(param_val)
            del(this_jacobian)
            del(this_real_eigenvalues)
        return store_eigen, param_population

    def prepare_parameters(self, parameters, parameter_names, GAN = True):

        if GAN: parameters = np.exp(parameters)
        self.parameter_set = pd.DataFrame(parameters, columns = parameter_names)
        
    def calculate_eigen_new_parameter_set(self, new_param_set):
        self.parameter_set = new_param_set
        store_eigen = []
        param_population=[]
        for j in range(len(self.parameter_set.index)):
            param_val = self.parameter_set.loc[j]*self.CONCENTRATION_SCALING  ####
            param_val = ParameterValues(param_val, self.kmodel)

            self.kmodel.parameters = self.k_eq
            self.kmodel.parameters = param_val
            self.parameter_sample = {v.symbol: v.value for k, v in self.kmodel.parameters.items()}

            # Set all vmax/flux parameters to 1.
            # TODO Generalize into Flux and Saturation parameters
            for this_reaction in self.kmodel.reactions.values():
                vmax_param = this_reaction.parameters.vmax_forward
                self.parameter_sample[vmax_param.symbol] = 1

            model_parameters = self.kmodel.parameters
            for k, v in self.conc_dict.items():
                if str(k) in model_parameters:
                    self.parameter_sample[Symbol(k)] = v


            self.kmodel.flux_parameter_function(
                self.kmodel,
                self.parameter_sample,
                self.conc_dict, 
                self.flux_dict
            )
            for c in self.conc_series.index:
                if c in self.kmodel.parameters:
                    c_sym = self.kmodel.parameters[c].symbol
                    self.parameter_sample[c_sym] = self.conc_series[c]
            this_jacobian = self.kmodel.jacobian_fun(self.flux_series,
                                                self.conc_series, self.parameter_sample)

            this_real_eigenvalues = sorted(np.real(eigenvalues(this_jacobian.todense())))
            store_eigen.append(this_real_eigenvalues)
            this_param_sample = ParameterValues(self.parameter_sample, self.kmodel)
            param_population.append(this_param_sample)
            
        return param_population, store_eigen
    def get_boundary_subclasses(self):
        return make_subclasses_dict(BoundaryCondition)

    def prepare_param_population(self, gen_params, k_names):
        parameter_set = pd.DataFrame(np.exp(gen_params))
        parameter_set.columns = k_names

        param_population = []
        for j in range(len(parameter_set.index)):
            param_val = parameter_set.loc[j] * self.CONCENTRATION_SCALING
            param_sample, eigen = self.calc_eigenvalues(param_val)
            param_population.append(param_sample)

        param_population = ParameterValuePopulation(param_population, kmodel=self.kmodel)

        return param_population

    def calc_eigenvalues(self, param_val):
        
        param_val = ParameterValues(param_val, self.kmodel)
        self.kmodel.parameters = self.k_eq
        self.kmodel.parameters = param_val
        parameter_sample = {v.symbol: v.value for k, v in self.kmodel.parameters.items()}

        # Set all vmax/flux parameters to 1.
        # TODO Generalize into Flux and Saturation parameters
        for this_reaction in self.kmodel.reactions.values():
            vmax_param = this_reaction.parameters.vmax_forward
            parameter_sample[vmax_param.symbol] = 1


        self.kmodel.flux_parameter_function(
            self.kmodel,
            parameter_sample,
            self.conc_series,
            self.flux_series
        )
        for c in self.conc_series.index:
            if c in self.kmodel.parameters:
                c_sym = self.kmodel.parameters[c].symbol
                parameter_sample[c_sym] = self.conc_series[c]
        this_jacobian = self.kmodel.jacobian_fun(self.flux_series[self.kmodel.reactions],
                                            self.conc_series[self.kmodel.reactants], parameter_sample)

        this_real_eigenvalues = sorted(np.real(eigenvalues(this_jacobian.todense())))

        this_param_sample = ParameterValues(parameter_sample, self.kmodel)
        return this_param_sample, this_real_eigenvalues


    # def prepare_bioreactor(self, path):

    #     with open(path,'r') as fid:
    #             the_dict = yaml.full_load(fid)

    #     # Load scaling
    #     concentration_scaling = float(the_dict['scaling']['concentration'])
    #     density = float(the_dict['scaling']['density'])
    #     gDW_gWW = float(the_dict['scaling']['gDW_gWW'])
    #     time_scaling = float(the_dict['scaling']['time'])
    #     flux_scaling =  1e-3 * (gDW_gWW * density) \
    #                     * concentration_scaling \
    #                     / time_scaling

    #     biomass_scaling = {k:flux_scaling/float(v) for k,v in
    #                     the_dict['biomass_scaling'].items()}

    #     biomass_reactions = dict()
    #     for name, biomass_rxn in the_dict['biomass'].items():
    #         biomass_reactions[name] = self.kmodel.reactions[biomass_rxn]

    #     extracellular_compartment = the_dict['extracellular_compartment']
    #     self.reactor_volume = float(the_dict['reactor_volume'])
    
    #     self.reactor = Reactor([self.kmodel], biomass_reactions, biomass_scaling,
    #                     extracellular_compartment=extracellular_compartment)

    #     for model in self.reactor.models.values():
    #         model.compartments[extracellular_compartment].parameters.volume.value \
    #             = self.reactor_volume

    #     # Boundary Conditions
    #     for the_bc_dict in the_dict['boundary_conditions'].values():
    #         TheBoundaryCondition = self.get_boundary_subclasses()[the_bc_dict.pop('class')]
    #         reactant = self.reactor.variables[the_bc_dict.pop('reactant')]
    #         the_bc = TheBoundaryCondition(reactant, **the_bc_dict)
    #         self.reactor.add_boundary_condition(the_bc)

    #     # Init the reactor initial conditions in correct order
    #     self.reactor.initial_conditions = TabDict([(x,0.0) for x in self.reactor.variables])
    #     # Add medium:
    #     for met, conc in the_dict['initial_medium'].items():
    #         self.reactor.initial_conditions[met]  = float(conc) * concentration_scaling

    #     # Add scaling fator to the rectaor
    #     self.reactor.concentration_scaling = concentration_scaling
    #     self.reactor.flux_scaling = flux_scaling

    # def simulate_bioreactor(self, parameter_set, 
    #                         max_time=60,
    #                         total_time=60,
    #                         steps=1000):
    #     """
    #     Run the simulation for a given parameter set on a given kinetic model
    #     """

    #     # Load tfa sample and kinetic parameters into kinetic model
    #     self.kmodel.parameters = parameter_set

    #     # Fetch equilibrium constants
    #     # load_equilibrium_constants(steady_state_sample, tmodel, self.kmodel,
    #     #                         concentration_scaling=CONCENTRATION_SCALING,
    #     #                         in_place=True)
        
    #     self.reset_reactor(self.reactor, parameter_set)

    #     start = time.time()
    #     print("Starting Simulation")
    #     if hasattr(self.reactor, 'solver'):
    #         delattr(self.reactor, 'solver')

    #         # Function to stop integration
    #     def rootfn(t, y, g, user_data):
    #         t_0 = user_data['time_0']
    #         t_max = user_data['max_time']
    # #         print(t)
    #         curr_t = time.time()
    #         if (curr_t - t_0) >= t_max:
    #             g[0] = 0
    #             print('Did not converge in time')
    #         else:
    #             g[0] = 1


    #     start = time.time()
    #     user_data = {'time_0': start,
    #                 'max_time': max_time} #2 minutes in seconds

    #     # Solve the ODEs
    #     sol_ode_wildtype = self.reactor.solve_ode(np.linspace(0, total_time, steps),
    #         solver_type='cvode',
    #         rtol=1e-9,
    #         atol=1e-9,
    #         max_steps=1e9,
    #         rootfn=rootfn,
    #         nr_rootfns=1,
    #         user_data=user_data)
    #     end = time.time()
    #     print("Compelted in {:.2f} seconds\n-----------------".format(end - start))

    #     final_biomass = sol_ode_wildtype.concentrations.iloc[-1]['biomass_strain_1'] * 0.28e-12 / 0.05
    #     final_anthranilate = sol_ode_wildtype.concentrations.iloc[-1]['anth_e'] * 1e-6 * 136.13
    #     final_glucose = sol_ode_wildtype.concentrations.iloc[-1]['glc_D_e'] * 1e-6 * 180.156
    #     print("Final biomass is : {}, final anthranilate is : {}, final glucose is : {}".format(final_biomass,
    #                                                                                                 final_anthranilate,
    #                                                                                                 final_glucose))
    #     return sol_ode_wildtype, final_biomass, final_anthranilate, final_glucose


    # def reset_reactor(self, parameter_set):
    #     """
    #     Function to reset the reactor and load the concentrations and parameters before each simulation
    #     """
    #     # Parameterize the rector and initialize with incolum and medium data
    #     self.reactor.parametrize(parameter_set, 'strain_1')
    #     self.reactor.initialize(self.conc_series, 'strain_1')
    #     self.reactor.initial_conditions['biomass_strain_1'] = 0.037 * 0.05 / 0.28e-12

    #     for met_ in self.reactor.medium.keys():
    #         LC_id = 'LC_' + met_
    #         LC_id = LC_id.replace('_L', '-L')
    #         LC_id = LC_id.replace('_D', '-D')
    #         self.reactor.initial_conditions[met_] = np.exp(self.samples.loc[LC_id]) * 1e9

    #     # Volume parameters for the reactor
    #     self.reactor.models.strain_1.parameters.strain_1_volume_e.value = self.reactor_volume
    #     self.reactor.models.strain_1.parameters.strain_1_cell_volume_e.value = 1.0  # 1.0 #(mum**3)
    #     self.reactor.models.strain_1.parameters.strain_1_cell_volume_c.value = 1.0  # 1.0 #(mum**3)
    #     self.reactor.models.strain_1.parameters.strain_1_cell_volume_p.value = 1.0  # 1.0 #(mum**3)
    #     self.reactor.models.strain_1.parameters.strain_1_volume_c.value = 0.9 * 1.0  # (mum**3)
    #     self.reactor.models.strain_1.parameters.strain_1_volume_p.value = 0.1 * 1.0  # (mum**3)
