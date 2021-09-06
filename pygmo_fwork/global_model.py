
import inspect
import os
import shutil
import yaml
import time

import numpy as np
import pandas as pd
from scipy import constants

from pygmo_fwork.pygmol.model import Model as PyGMol
from pygmo_fwork.results import ResultsParser
from pygmo_fwork.config import Config
from pygmo_fwork.exceptions import \
    GlobalModelInitialGuessError, GlobalModelAttributeError, BackendNoRunError, \
    BackendRunUnsuccessfulError, GlobalModelLogError


class GlobalModel(object):
    """Higher-level wrapper around different backends - namely PyGMol and PyGKin. While both lower-level python modules
    carry on all the model solving, this class implements logging the results, comparing the solutions,
    pretty-printing the comparisons etc... This wrapper is also called from the OptiChem package. In general, the
    lower-level backend modules should not be used on their own from outside the PyGMoFork package, instead they should
    always be called via this class.
    """
    mandatory_backend_methods = (
        'run',
        'success',
        'get_solution',
        'get_solution_final',
        'get_rates',
        'get_rates_final',
        'get_wall_fluxes',
        'get_wall_fluxes_final',
        'num_species',
        'num_reactions',
        'dump_logs'
    )

    def __init__(self, chemistry, model_parameters, backend='PyGMol'):
        """Instantiation of the GlobalModel object. The chemistry and model_parameters arguments need to have some
        particular structure. Check docstrings for the backend Model objects for further information.
        The backend attribute is validated here, while the chemistry and model_parameter objects are
        validated during the backend model instantiation.
        The chemistry instance can be modified in place, with the model reflecting these modifications in subsequent
        runs.
        The model_parameters object becomes "frozen" after instantiation and the only way to change the model
        parameters is to call self.reload_model_parameters passing the new updated model_parameters!
        :param chemistry: (plaschem.chemistry.Chemistry or similar) check the backend model docs
        :param model_parameters: (plaschem.model_parameters.ModelParameters or dict or similar) check the backend docs
        :param backend: (str) identifier of the backend model.
        """
        self.backend = backend
        self.model = None
        if self.backend.lower() not in Config.accepted_backends:
            raise GlobalModelAttributeError('Unsupported backend identifier: {}'.format(backend))
        if self.backend.lower() == 'pygmol':
            self.model = PyGMol(chemistry=chemistry, model_params=model_parameters)
        elif self.backend.lower() == 'pygkin':
            raise NotImplementedError  # TODO: Implement PyGKin!
        else:
            raise GlobalModelAttributeError('This can never happen! Fix the Config.accepted_backends class attribute!')
        # sanity check: does the backend model implement all the mandatory methods?
        for method in self.mandatory_backend_methods:
            if not hasattr(self.model, method) or not inspect.ismethod(getattr(self.model, method)):
                # this should never happen!
                raise GlobalModelAttributeError('Some mandatory methods missing from the backend model class!')

        self.last_run_success = None  # this is a flag showing if the last backend model run was successful

    def reload_model_parameters(self, model_parameters):
        """Method to load some new values of model parameters. Basically re-instantiates the class with the new
        model_parameters attribute:
        :param model_parameters:
        :return: None
        """
        self.__init__(chemistry=self.model.chemistry, model_parameters=model_parameters, backend=self.backend)

    # noinspection DuplicatedCode
    def generate_initial_n(self, ionisation_degree=1.e-15, neg_ion_fraction=1.e-15, non_feed_sp_fraction=1.e-15):
        """Generates a self_consistent initial densities for all heavy species. Those will add up to the pressure
        defined in model_parameters and will ensure that the charge density of all heavy species is positive,
        so there is a space for electrons.
        :param ionisation_degree: ratio between electron density and total density (not strictly ionisation degree)
        :param neg_ion_fraction: ratio between density of a negative ion and total density
        :param non_feed_sp_fraction: ratio between density of of a non-feed gas species and the total density
        :return: (dict) of sp_name: initial_sp_n in [m-3]
        """
        gas_temp = self.model.model_params['Tg']  # (initial) gas temperature
        pressure = self.model.model_params['p']  # initial gas pressure
        feeds_dict = self.model.model_params['feeds']  # dict of feeds in [sccm] with sp names as keys

        species_names = self.model.chemistry.get_species_name()  # names for all the species in chemistry
        charges = np.array(self.model.chemistry.get_species_charge())  # charges of all heavy sp in [e]
        feeds = np.array(
            [feeds_dict[sp] if sp in feeds_dict else 0. for sp in species_names]
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

        # convert explicitly to floats, so it can be nicely yamled...
        n0 = [float(n) for n in n0]
        init_n = dict(zip(species_names, n0))
        self.validate_initial_n(init_n)
        return init_n

    def validate_initial_n(self, init_n):
        """Validates the passed initial_n dict. Checks if all the species are in keys and also that the total heavy
        species charge density is positive. Does not check the total pressure against the value in model parameters!!
        If the init_n dict is in any way inconsistent, raises the GlobalModelInitialGuessError
        :param init_n: (dict) sp_name: init_sp_n in [m-3]
        :return: None
        """
        species_names = np.array(self.model.chemistry.get_species_name())
        if not set(init_n) == set(species_names):
            raise GlobalModelInitialGuessError('The passed initial densities dict is inconsistent with chemistry spcs"')
        species_charges = np.array(self.model.chemistry.get_species_charge())
        species_densities = np.array([init_n[sp_name] for sp_name in species_names])
        total_charge_density = sum(species_charges*species_densities)
        if not total_charge_density > 0:
            raise GlobalModelInitialGuessError('The total heavy species charge density needs to be positive!')

    def run(self, run_id=None, init_el_temp=1.0, init_n=None, log_results=True, log_to=None, verbose=True,
            overwrite=False, compress_logs=False):
        """Method to run the backend model. See the backend model run method documentation for more info.
        :param run_id: (str) identifier of this current run. Only needed if log_results == True
        :param init_el_temp: (float) initial electron temperature. Defaults to 1.0
        :param init_n: (dict) initial densities in [m-3] for all the heavy species in the chemistry (keyed by the
                       species names). If not supplied, a self-consistent dict id auto-generated.
        :param log_results: (bool) if True, all the relevant results and logs will be saved/copied to the
                            log_to/run_id directory. Defaults to True
        :param log_to: (str) path to the model results folder. Only needed if log_results is True
        :param verbose: (bool) if True, various run info is printed to the stdout as well as the results if run
                        successful or a warning if the run not successful.
        :param overwrite: (bool) if False and if logging and the log_to/run_id folder exists, throws error. If True,
                          the original results are deleted and overwritten by new ones. Only relevant if log_results
                          is True.
        :param compress_logs: (bool) if True, only compresses solution, rates and wall_fluxes will be logged
                              Only in action if log_results is True.
        :return: None
        """

        # preset this, so I can catch whatever error the backend model raises and still be able to assess if the
        # run was successful or not:
        self.last_run_success = False

        # some validators and some defaulting:
        if log_results:
            if not isinstance(run_id, str):  # if logging, run_id needs to be provided
                raise GlobalModelAttributeError('If logging the run results, (str) run_id needs to be specified')
            if log_to is None:
                log_to = Config().get_results_dir(self.backend)  # config-default results directory
            if not os.path.isdir(log_to):
                raise GlobalModelLogError('Passed "log_to" path is not a valid directory!')

        if init_el_temp <= 0:
            raise GlobalModelInitialGuessError('Initial electron temperature needs to be positive!')
        if init_n is None:
            init_n = self.generate_initial_n()
        else:
            self.validate_initial_n(init_n)

        if verbose:
            print()
            print(
                '{:*^100}'.format('{} Running {} - Run ID: {} '.format(time.strftime('%H:%M:%S'), self.backend, run_id))
            )

        start_time = time.time()
        self.model.run(init_el_temp=init_el_temp, init_n=init_n)
        run_time = time.time() - start_time

        self.last_run_success = self.model.success()

        if verbose:
            if self.success():
                convergency = self.check_convergency(self.get_solution(), self.model.model_params, verbose=verbose)
                if (~convergency.values).any():
                    print('{:.^100}'.format(''))
                with pd.option_context('display.float_format', '{:.1E}'.format):
                    print(pd.DataFrame(self.get_solution_final()).T)
                print('Run Time: {} sec'.format(round(run_time, 2)))
            else:
                print('WARNING: Solution Unsuccessful!')
            print('{:*^100}'.format(''))
            print()

        if log_results and self.success():
            init_params = {'init_el_temp': init_el_temp, 'init_n': init_n}
            self.log_results(
                run_id=run_id, results_dir=log_to, init_params=init_params, overwrite=overwrite, compress=compress_logs
            )

    def success(self):
        """Returns if the last backend model run was successful or not. Raises a GlobalModelBackendError if called
        before the backend model was run...
        :return: (bool)
        """
        if self.last_run_success is None:
            raise BackendNoRunError('Backend model has not been run!')
        return self.last_run_success

    def assert_success(self):
        """A helper method asserting a successful backend model run. Will rise the relevant exception if not successful.
        :return: None
        """
        if not self.success():
            raise BackendRunUnsuccessfulError('Last run unsuccessful - no solution!')

    def log_results(self, run_id, results_dir, init_params, overwrite=False, compress=False):
        """Logs/copies all the appropriate results into the results_dir/run_id folder. Does not check for the folder
        existence, permissions etc. This should be done on the level of calling methods (it is implemented in
        self.run). Raises corresponding errors if the run was unsuccessful or not run at all...
        :param run_id: (str) identifier of the run being logged.
        :param results_dir: (str) path to a results directory where the run results will be saved under the run_id
                            subdirectory
        :param init_params: (dict) dictionary of whatever initial parameters passed to run, to be logged.
        :param overwrite: (bool) if True and the run_id subdir already exists, it will be removed and logged into again.
        :param compress: (bool) if True, only compressed solution, rates and wall_fluxes will be logged.
        :return: None
        """
        self.assert_success()

        if not os.path.isdir(results_dir):
            raise GlobalModelLogError('Invalid or non existing results_dir!')
        run_results_dir = os.path.join(results_dir, run_id)
        if os.path.isdir(run_results_dir):
            if overwrite:
                shutil.rmtree(run_results_dir)
            else:
                raise GlobalModelLogError('The {} results subdirectory already exists!'.format(run_id))
        os.mkdir(run_results_dir)

        # log the solution, rates and wall fluxes:
        sol = self.get_solution()
        rates = self.get_rates()
        wall_fluxes = self.get_wall_fluxes()
        if compress:
            sol = ResultsParser.compress_solution(sol)
            rates = ResultsParser.compress_solution(rates)
            wall_fluxes = ResultsParser.compress_solution(wall_fluxes)
        sol.to_csv(os.path.join(run_results_dir, Config.default_log_name('solution', run_id=run_id)))
        rates.to_csv(os.path.join(run_results_dir, Config.default_log_name('rates', run_id=run_id)))
        wall_fluxes.to_csv(os.path.join(run_results_dir, Config.default_log_name('wall_fluxes', run_id=run_id)))

        # log the model parameters and chemistry attributes:
        self.model.chemistry.dump_chemistry_attributes(
            os.path.join(run_results_dir, Config.default_log_name('chemistry_attributes', run_id=run_id))
        )
        model_params = dict(self.model.model_params)
        with open(os.path.join(
                run_results_dir, Config.default_log_name('model_attributes', run_id=run_id)), 'w') as stream:
            yaml.dump(
                {'model_params': model_params, 'initial_params': init_params},
                stream=stream
            )

        # save the solver log (for PyGMol) or copy all the inputs and outputs (for GlobalKin)
        self.model.dump_logs(run_results_dir=run_results_dir)

    @staticmethod
    def check_convergency(solution, model_params, eps=0.1, timewindow=0.1, verbose=False):
        """Method to check the computed backend model results and check if the results converged in time. Also
        checks if the prescribed t_end has been reached by the calculation. The method is coded as static one,
        so it can be reused in other classes, such as the one handling loading and comparing different past logged
        runs.
        :param solution (pd.DataFrame) what gets returned by self.get_solution()
        :param model_params (dict-like) with the same structure as self.model_params
        :param eps (float) maximum relative difference of values inside the last t_end*timewindow time window
                    for which the model is considered converged. Between 0. - 1.
        :param timewindow (float) Needs to be between 0. - 1. Defines the time window inside which to check for
                           convergence as (t_end - t_end*time_frame, t_end)
        :param verbose (bool) if True, messages about not converging and not reaching prescribed endpoint are
                        printed to stdout.
        :return: (pd.Series) of bool values with te same index as the solution's columns.
        """
        # some validation first:
        messages = []
        timesteps = solution['t']
        t_end = timesteps.iloc[-1]

        # check if the prescribed end time has been reached
        t_end_0 = model_params['t_end']
        if t_end < t_end_0 and not np.isclose(t_end, t_end_0, rtol=0.05):
            messages.append('Terminated at t={} (t_end={} not reached!)'.format(t_end, t_end_0))

        # check convergency:
        delta_t = t_end*timewindow

        values_end = solution.iloc[-1]  # end values for all the model outputs
        t_start = t_end - delta_t  # start of the convergency check time window
        # start values for all the model outputs need to be interpolated from the results!
        values_start = pd.Series(
            [np.interp(t_start, solution['t'], solution[a]) for a in solution.columns], index=solution.columns
        )
        rel_diff = (values_end - values_start) / (values_end + values_start) * 2

        outputs_to_ignore = {'t', }  # I dont care for convergency of these parameters
        rel_diff = rel_diff.drop(outputs_to_ignore)
        convergency = abs(rel_diff) <= eps

        messages.extend([
            'The {} value changed by {}% over the last {}% of total time!'.format(
                output, round(rel_diff[output]*100, 2), timewindow * 100
            ) for output in rel_diff[~convergency].index
        ])

        # printout the convergency warning, if any received and if in verbose mode:
        if verbose and len(messages):
            print('CONVERGENCY WARNING:')
            print('\n'.join(messages))

        return convergency

    # ***************************** METHODS SIMPLE FORWARDED FROM THE BACKEND MODEL CLASS **************************** #

    def get_solution(self):
        """Check the backend model method documentation!
        :return: (pd.DataFrame) copy of the backend model solution
        """
        self.assert_success()
        return self.model.get_solution()

    def get_solution_final(self):
        """Check the backend model method documentation!
        :return: (pd.Series) copy of the backend model final solution
        """
        self.assert_success()
        return self.model.get_solution_final()

    def get_rates(self):
        """Check the backend model method documentation!
        :return: (pd.DataFrame) copy of the backend model rates
        """
        self.assert_success()
        return self.model.get_rates()

    def get_rates_final(self):
        """Check the backend model method documentation!
        :return: (pd.Series) copy of the backend model final rates
        """
        self.assert_success()
        return self.model.get_rates_final()

    def get_wall_fluxes(self):
        """Check the backend model method documentation!
        :return: (pd.Series) copy of the backend model particle wall fluxes in time.
        """
        self.assert_success()
        return self.model.get_wall_fluxes()

    def get_wall_fluxes_final(self):
        """Check the backend model method documentation!
        :return: (pd.Series) copy of the backend model final particle wall fluxes.
        """
        self.assert_success()
        return self.model.get_wall_fluxes_final()

    def num_species(self):
        """Check the backend model method documentation!
        :return: (int) number of species in the model.
        """
        return self.model.num_species()

    def num_reactions(self):
        """Check the backend model method documentation!
        :return: (int) number of reactions in the model.
        """
        return self.model.num_reactions()
