import inspect
import os

import numpy as np
import pandas as pd
import yaml
from scipy import constants
from scipy.integrate import solve_ivp

from pygmo_fwork.config import Config
from pygmo_fwork.pygmol.equations import Equations
from pygmo_fwork.pygmol.exceptions import InitialSolutionError, ModelSolutionError, ModelInitError
from pygmo_fwork.pygmol.model_parameters import ModelParameters


# noinspection DuplicatedCode
class Model(object):
    """A global model solving the equations defined in the equations.Equations class
    """
    mandatory_chemistry_methods = (
        'get_species_name',
        'get_species_charge',
        'get_species_mass',
        'get_species_lj_sigma',
        'get_species_stick_coef',

        'get_reactions_arrh_a',
        'get_reactions_arrh_b',
        'get_reactions_arrh_c',
        'get_reactions_el_en_loss',
        'get_reactions_elastic',
        'get_reactions_stoich_coefs_electron',
        'get_reactions_stoich_coefs_arbitrary',
        'get_reactions_id',

        'get_return_matrix',
        'get_stoichiomatrix',

        'num_species',
        'num_reactions'
    )

    def __init__(self, chemistry, model_params):
        """Initialiser of the GlobalModel class. The model solves equations for density balance for Ns species
        and electron temperature balance. A Chemistry class instance needs to be passed as chemistry, defining
        the kinetic scheme for the model. The chemistry class needs to implement some mandatory methods.
        The model equations support reactions with arbitrary specie 'M', which is not explicitly solved for (it's
        density is simple sum of all species densities) and also electron 'e', which is also not explicitly solved
        for, since it's density follows from neutrality. All the other species (densities) are solved for.
        The chemistry passed needs to be self-consistent and needs to contain electron and at least one positive ion.
        No chemistry consistency checks are done here, apart from ensuring the chemistry parameter implements all
        the mandatory methods listed in the class attribute. Check the plaschem.chemistry.Chemistry class or
        pygmofork.pygmol.equations.Equations class for the docs for all the mandatory methods!

        :param chemistry: (Chemistry instance) - this parameter is forwarded to the Equations class, for documentation
                          of the Chemistry class, see the Equations.__init__ docstring.
        :param model_params: (dict or ModelParameters) - if dict is passed, it gets turned to a ModelParameters
                             instance. For the structure of the dict, see the ModelParameters.__init__ docstring and
                             the Equations.__init__ class docstrings (since the model_params get forwarded to the
                             Equations class)
        """
        for mandatory_chemistry_method in self.mandatory_chemistry_methods:
            if not hasattr(chemistry, mandatory_chemistry_method) or \
                    not inspect.ismethod(getattr(chemistry, mandatory_chemistry_method)):
                raise ModelInitError('The passed chemistry argument does not implement all the mandatory methods!')
        self.chemistry = chemistry

        # validate model parameters passed:
        if not isinstance(model_params, ModelParameters):
            model_params = ModelParameters(model_params)
        self.model_params = model_params

        self.equations = None  # placeholder for the Equations instance
        self._build_equations()  # populate the self.equations attribute

        self.sol_raw = None  # placeholder for whatever the selected solver returns
        self.t = None  # placeholder for an array of timesteps
        self.sol_primary = None  # array of vectors y = (n, rho) for all timesteps as returned by the solver in [m-3]
        self.sol_secondary = None  # pd.DataFrame of all densities (incl. 'e'), time, Te etc... in [m-3, s, eV]

    # ****************************************** HELPER METHODS  ***************************************************** #

    def _build_equations(self):
        """This method populates self.equations with an Equations instance.
        :return: None
        """
        self.equations = Equations(self.chemistry, self.model_params)

    def _build_y0(self, el_temp=1.0, ionisation_degree=1.e-15, neg_ion_fraction=1.e-15, non_feed_sp_fraction=1.e-15):
        """A method to create a self-consistent initial values of the solution. This is pretty much only useful if
        running the PyGMol as a stand-alone. If running via the GlobalModel wrapper, the initial guess will be
        defaulted there and passed into self.run.
        :param el_temp: Electron temperature in [eV]
        :param ionisation_degree: ratio between electron density and total density (not strictly ionisation degree)
        :param neg_ion_fraction: ratio between density of a negative ion and total density
        :param non_feed_sp_fraction: ratio between density of of a non-feed gas species and the total density
        :return: (sp.array) of initial solution vector (n0_1, n0_2, ..., n0_N, rho0)
        """
        gas_temp = self.model_params['Tg']  # (initial) gas temperature
        pressure = self.model_params['p']  # initial gas pressure
        feeds_dict = self.model_params['feeds']  # dict of feeds in [sccm] with sp names as keys

        charges = np.array(self.chemistry.get_species_charge())  # charges of all heavy species in elementary charges
        feeds = np.array(
            [feeds_dict[sp] if sp in feeds_dict else 0. for sp in self.chemistry.get_species_name()]
        )  # feeds in [sccm]

        n_tot = pressure / (constants.k * gas_temp)  # total initial density in [m-3]
        n_e = n_tot * ionisation_degree
        n_neg_ions_tot = n_tot * -sum(charges[charges < 0]) * neg_ion_fraction  # total density of negative ions
        n_neg_tot = n_neg_ions_tot + n_e  # total density of negative species (including electron)
        n_pos_tot = n_neg_tot  # preserving neutrality!

        n0 = np.zeros(len(charges))
        n0[charges > 0] = n_pos_tot / sum(charges[charges > 0])  # initial densities of positive ions
        if (charges < 0).any():
            n0[charges < 0] = -n_neg_ions_tot / sum(charges[charges < 0])  # initial densities of negative ions
        n0[(charges == 0) & (
                feeds == 0)] = n_tot * non_feed_sp_fraction  # initial densities of non-feed neutral species

        # rest of the densities is divided between feed gas species in proportion to their feed flows, if defined:
        if feeds.sum() > 0:
            n_residual = n_tot - n0.sum()
            n0[feeds > 0] = feeds[feeds > 0] / feeds.sum() * n_residual
        else:  # if no feed flows defined, distribute the remaining density between all the neutral species...
            n_residual = n_tot - n0[charges != 0].sum()
            n0[charges == 0] = n_residual / len(n0[charges == 0])

        # add the electron energy density:
        rho0 = 3 / 2 * n_e * el_temp

        return np.r_[n0, rho0]

    def _y0_from_init_params(self, init_el_temp, init_n):
        """A method to create the initial solution vector y0 out of the initial electron temperature and initial
        densities for all species in the chemistry
        :param init_el_temp: (float) in [eV]
        :param init_n: (dict) with species names as keys and values in [m-3]. Also None is allowed, in which case the
                       y0 is built by self._build_y0(el_temp=init_el_temp) instead...
        :return: (np.array) initial solution vector [n_i0] + [rho0]
        """
        if init_n is not None:
            n_series = pd.Series(init_n)
            if list(sorted(n_series.index)) != list(sorted(list(self.chemistry.get_species_name()))):
                raise InitialSolutionError('The init_n dict needs to contain all the species names as keys!')
            n_series = n_series[list(self.chemistry.get_species_name())]  # reorder
            n_array = np.array(n_series)
            n_e = sum(n_array * np.array(self.chemistry.get_species_charge()))
            rho = 3 / 2 * n_e * init_el_temp
            y0 = np.r_[n_array, rho]
        else:
            y0 = self._build_y0(el_temp=init_el_temp)
        return y0

    def _validate_y0(self, y0):
        """A method to assert consistency of the initial solution vector.
        :return: None
        """
        # check for the correct dimension:
        if len(y0) != self.dimension():
            raise InitialSolutionError('Incorrect dimension of the initial values vector!')
        # check that the charge density of the heavy species is positive (need something to allow electrons in)
        n0 = y0[:-1]
        charges = np.array(self.chemistry.get_species_charge())  # charges of all heavy species in elementary charges
        if sum(n0 * charges) <= 0:
            raise InitialSolutionError('Heavy species total charge density needs to be positive to allow for '
                                       'a positive value of electron density!')

    def _solve(self, y0=None, method='BDF', jacobian=False):
        """Method to solve the model with a chosen solver (hardcoded here). Takes the initial value of the solution
        vector. The solution vector is [n_1, ..., n_N, rho], where rho is the electron energy density in [eV/m3]
        and all the densities are in [m-3]. This method runs the solver and stores it's raw solution as well as primary
        solution (see self._parse_raw_sol) and secondary solution (see self._post_process).

        :param y0: (np.array) initial value of the solution vector. If not passed, it's built internally. The y0
                   is a vector of initial densities for all species (the same order as in chemistry.get_species()) with
                   the last value being rho0 (initial electron energy density - 3/2 * Te_0 * [e])
        :param method: (str) this is the method forwarded to the solve_ivp solver. See the solve_ivp documentation
                       for accepted method values.
        :param jacobian: (bool) if the jacobian function is to be used in the solve_ivp.
        :return: None
        """
        # WARNING: new Equations instance NEEDS to be created before each run, to reflect possible changes in chemistry
        self._build_equations()

        if y0 is None:
            y0 = self._build_y0()
        self._validate_y0(y0)

        t_end = self.model_params['t_end']

        func, jac = self.equations.get_objective_functions(jacobian=jacobian)
        try:
            self.sol_raw = solve_ivp(func, (0, t_end), y0, method=method, t_eval=None, jac=jac)
        except ValueError as ve:
            raise ModelSolutionError('solve_ivp raised ValueError: {}'.format(str(ve)))
        self._parse_raw_sol()  # create the primary solution np matrix
        self._post_process()  # create the secondary solution dataframe

    def _parse_raw_sol(self):
        """Method to parse the raw solution returned by the solver and create a primary solution (self.sol_primary),
        which is just a np.array of solution vectors in time. Also saves the array of timesteps to self.t

        :return: None
        """
        self.t = self.sol_raw.t
        self.sol_primary = self.sol_raw.y.T

    def _post_process(self):
        """Method to postprocess the primary solution and save the secondary solution in self.sol_secondary.
        The secondary solution is a DataFrame of results in time with columns of densities of all chemistry heavy
        species (m-3), 'e' (m-3), 'Te' (eV), 'Tg' (K), 'p' (Pa), 't' (s)

        :return: None
        """
        self.sol_secondary = pd.DataFrame(
            self.sol_primary[:, :-1], columns=list(self.chemistry.get_species_name())
        )
        ne = [self.equations.get_electron_density(y) for y in self.sol_primary]
        temp_e = np.array([self.equations.get_electron_temperature(y) for y in self.sol_primary])
        self.sol_secondary['e'] = pd.Series(ne, index=self.sol_secondary.index)
        self.sol_secondary['Te'] = pd.Series(temp_e, index=self.sol_secondary.index)
        self.sol_secondary['Tg'] = pd.Series(self.model_params['Tg'], index=self.sol_secondary.index)
        self.sol_secondary['p'] = pd.Series(self.diagnose('total_pressure')['value'], index=self.sol_secondary.index)
        self.sol_secondary['t'] = pd.Series(self.t, index=self.sol_secondary.index)
        # rearrange:
        self.sol_secondary = self.sol_secondary[['t'] + list(self.sol_secondary.columns[:-1])]

    def diagnose(self, quantity, sol_primary=None, totals=False):
        """Method to diagnose any of the equations' partial results for the whole time evolution of a finished
        solution. Either can be run after self.solve in this instance, or primary solution can be fed, without the
        need to run the self.solve first. Creates a dataframe with the quantities in time. The partial result
        represented by the quantity str may be either scalar, or vector of results for all species or for all
        reactions. Each returned dataframe has a 't' column (in seconds) and either a 'value' column for scalar
        quantities or multiple columns labeled by sp.get_name() for species quantities or r.id for reactions
        quantities. Optionally it also might contain a column labelled 'total' with a sum of all columns for each time.

        :param quantity: (str) for partial results from the Equations class. eg. quantity='total_pressure' will
                         diagnose the Equations.get_total_pressure(y) for y for each time in the timesteps.
        :param sol_primary: (np.array) optional already calculated primary solution.
        :param totals: (bool) if to include the total sum column
        :return: (pd.DataFrame) as described in the docstring.
        """
        if sol_primary is None:
            sol_primary = self.sol_primary
        diagnostics = pd.DataFrame([getattr(self.equations, 'get_{}'.format(quantity))(y).copy() for y in sol_primary])
        if diagnostics.shape[1] == self.equations.num_species:  # a value for each species
            diagnostics.columns = list(self.chemistry.get_species_name())
        elif diagnostics.shape[1] == self.equations.num_reactions:  # a value for each reaction
            diagnostics.columns = list(self.chemistry.get_reactions_id())
        elif diagnostics.shape[1] == self.equations.num_species + 1:  # a value for each component of the y vector
            diagnostics.columns = list(self.chemistry.get_species_name()) + ['rho']
        else:  # one value for the whole system
            diagnostics.columns = ['value']
        # add the totals column:
        if totals:
            diagnostics['total'] = diagnostics.sum(axis=1)
        # add the time column:
        diagnostics['t'] = pd.Series(self.t, index=diagnostics.index)
        diagnostics = diagnostics[['t'] + list(diagnostics.columns[:-1])]
        return pd.DataFrame(diagnostics)

    def dimension(self):
        """Returns the dimension of the model (or number of linearly independent outputs)
        :return: (float)
        """
        return self.chemistry.num_species() + 1

    # **************************** METHODS REQUIRED BY THE HIGHER-LEVEL WRAPPER ************************************** #

    def run(self, init_el_temp=1.0, init_n=None):
        """Method to run the model with passed (or automatically built) initial densities and return the final solution.
        Raises the ModelSolutionError if the solver did not finish successfully!
        :param init_el_temp: (float) Initial electron temperature in [eV]. Defaults to 1.0
        :param init_n: (dict) Initial densities. Needs to contain as keys species names for all the active species
                       in the self.chemistry. All values in [SI]. If not passed, these are built internally.
        :return: (pd.Series) of the final values indexed by species names and parameters names...
        """
        y0 = self._y0_from_init_params(init_el_temp, init_n)
        self._solve(y0=y0)
        if not self.success():
            raise ModelSolutionError(self.sol_raw.message)

    def success(self):
        """Method to get a success of the solution. True, if solver finished successfully, False if not.
        :return: (bool)
        """
        return bool(self.sol_raw.success)

    def get_solution(self):
        """Returns a pd.DataFrame holding the time-dependent solution of the model. The DataFrame has columns
        of 't', species names, 'p', 'Te', 'Tg', 'e' etc. Index of rows are ascending integers - arbitrary.
        :return: (pd.DataFrame)
        """
        if self.sol_primary is None:
            raise ModelSolutionError('Rates can only be extracted after the model is solved!')
        return self.sol_secondary.copy()

    def get_solution_final(self):
        """Returns the solution at the final time. It's a pd.Series with index of 't', species names, 'Te', 'Tg', 'p',
        'e' etc...
        :return: (pd.Series)
        """
        return self.get_solution().iloc[-1]

    def get_rates(self):
        """Returns a pd.DataFrame holding the time-dependent rates for all the reactions in the chemistry.
        The DataFrame has columns of 't' and reactions ids. The rows index are arbitrary ascending integers.
        :return: (pd.DataFrame)
        """
        if self.sol_primary is None:
            raise ModelSolutionError('Rates can only be extracted after the model is solved!')
        rates = self.diagnose('reaction_rates')
        return rates

    def get_rates_final(self):
        """Returns the rates at the final time. It's a pd.Series with index of reaction.id for all the reactions in the
        chemistry.
        :return: (pd.Series)
        """
        return self.get_rates().iloc[-1].drop('t')

    def get_wall_fluxes(self):
        """Returns a pd.DataFrame holding the time-dependent wall fluxes for all the species in the chemistry.
        The DataFrame has columns of 't' and species names. The rows index are arbitrary ascending integers.
        :return: (pd.DataFrame)
        """
        if self.sol_primary is None:
            raise ModelSolutionError('Rates can only be extracted after the model is solved!')
        wall_fluxes = self.diagnose('wall_fluxes')
        return wall_fluxes

    def get_wall_fluxes_final(self):
        """Returns the wall fluxes at the final time. It's a pd.Series with index of species names for all the species
        in the chemistry.
        :return: (pd.Series)
        """
        return self.get_wall_fluxes().iloc[-1].drop('t')

    def num_species(self):
        return self.chemistry.num_species()

    def num_reactions(self):
        return self.chemistry.num_reactions()

    def dump_logs(self, run_results_dir):
        """Saves/copies all the relevant inputs/outputs of the model/solver. Only used for logs specific to this
        particular model (backend), solution, rates and chemistry/model parameters are logged from the higher level
        wrapper. This method only logs the solve_ivp solver output!
        :param run_results_dir: (str) path where to save the log file
        :return: None
        """
        sol_raw_attributes = \
            {attr: getattr(self.sol_raw, attr) for attr in ['nfev', 'njev', 'nlu', 'status', 'message', 'success']}
        with open(os.path.join(run_results_dir, Config.default_log_name('solver_log')), 'w') as stream:
            yaml.dump(sol_raw_attributes, stream=stream)
