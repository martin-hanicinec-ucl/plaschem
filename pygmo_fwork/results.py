
import os
import yaml
import warnings

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pygmo_fwork.config import Config
from pygmo_fwork.pygmol.model_parameters import TypeCheck
from pygmo_fwork.exceptions import ResultsAttributeError, ResultsNotFoundError


class ResultsParser(object):
    """
    This class handles parsing previously logged results from pygmofwork.GlobalKin wrapper runs. Methods are
    implemented to load in solutions, rates and other logs from runs identified by GlobalKin wrapper run_id and
    backend model name or explicit results directory.
    """
    unit_conversion_factors = {
        'm-3/s': 1,
        'm-3.s-1': 1,
        'm-3s-1': 1,
        'cm-3/s': 1e-6,
        'cm-3.s-1': 1e-6,
        'cm-3s-1': 1e-6,
    }

    def __init__(self, nominal_backend=None, nominal_results_dir=None):
        """Instantiates the ResultsParser class. If called without arguments, the nominal backend defaults to 'pygmol'.
        Only one of the arguments might be passed, passing both will result in a conflict and exception being raised.
        :param nominal_backend: (str) identifier of the nominal backend model. Must be from Config.accepted_backends
        :param nominal_results_dir: (str) nominal path to the results directory (filled with sub-directories with
                                    run_id names)
        """
        if nominal_backend is None and nominal_results_dir is None:
            nominal_backend = 'pygmol'  # default
        self.nominal_results_dir = self._fetch_results_dir(nominal_backend, nominal_results_dir)  # this is what matters

    # ****************************************** Helper Methods ****************************************************** #

    def _fetch_results_dir(self, backend=None, results_dir=None):
        """Helper method to get a path to the results directory. That is the parent directory to a run_id directory.
        At most one of the arguments needs to be passed. If results_dir is passed, it is simply returned, if backend
        is passed, results_dir from Config gets returned. If none is passed, nominal results dir gets returned
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory
        :return: (str) path to the results directory
        """
        if backend is None and results_dir is not None:
            return results_dir
        elif backend is not None and results_dir is None:
            return Config().get_results_dir(backend)
        elif backend is None and results_dir is None:
            return self.nominal_results_dir
        else:
            raise ResultsAttributeError('Invalid combination of attributes!')

    def _fetch_run_dir(self, run_id, backend=None, results_dir=None):
        """The same as _fetch_results_dir defined above, but will return run directory, as a child directory to the
        results directory with the run_id name. At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory
        :return: (str) absolute path to a run_id directory
        """
        results_dir = self._fetch_results_dir(backend=backend, results_dir=results_dir)
        return os.path.join(results_dir, run_id)

    def _fetch_model_attributes(self, run_id, backend=None, results_dir=None, attributes_dump_name=None):
        """This method will return the model attributes dictionary loaded from the .yaml logged file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory
        :param attributes_dump_name: (str) optional custom name of the model attributes yaml dump file. if not passed,
                                     defaults to the default log name.
        :return: (dict) model attributes
        """
        run_dir = self._fetch_run_dir(run_id, backend, results_dir)
        if attributes_dump_name is None:
            attributes_dump_name = Config.default_log_name('model_attributes', run_id=run_id)
        model_attributes_path = os.path.join(run_dir, attributes_dump_name)
        if not os.path.isfile(model_attributes_path):
            raise ResultsNotFoundError('Model attributes not found in "{}"!'.format(model_attributes_path))
        with open(model_attributes_path) as stream:
            model_attributes = yaml.load(stream=stream, Loader=yaml.FullLoader)
        return model_attributes

    @staticmethod
    def _parse_run_identifiers(run):
        """Helper method to return **kwargs for the main solution fetching method from an overloaded "run" parameter.
        run parameter might be one of the following:
            'run_id',
            ('run_id', 'backend'),
            ('run_id', 'results_dir'),
            ('run_id', 'solution_file_name'),
            ('run_id', 'backend', 'solution_file_name'),
            ('run_id', 'results_dir', 'solution_file_name')
        Only str (for run_id) or tuples of strings are supported, with all parameters not supplied in tuples being
        defaulted or using the nominal ones.
        The method decides itself which parameter in the run tuple is which. If it cannot decide, raises
        ResultsAttributeError.
        :param run: (str or tuple of str) - see above
        :return: (dict) with keys, values describing **kwargs for self.fetch_solution method.
        """
        run_identifiers = {'run_id': None, 'backend': None, 'results_dir': None, 'file_name': None}
        if not TypeCheck.is_collection(run):
            run = [run, ]  # only run_id
        else:
            run = list(run)  # if iterable passed, make it a list explicitly
        run_identifiers['run_id'] = run.pop(0)  # pop the first element, which needs to be run_id
        if not len(run):
            pass
        elif len(run) == 1:  # only one more element is in the run iterable:
            if run[0].lower() in Config.accepted_backends:  # it's backend
                run_identifiers['backend'] = run[0]
            elif os.path.isdir(run[0]):  # its results_dir
                run_identifiers['results_dir'] = run[0]
            else:  # it needs to be solution_file_name
                run_identifiers['file_name'] = run[0]
        elif len(run) == 2:  # two more parameters are in the run iterable
            # the last one needs to be solution_file_name, since backend and res_dir are mutually exclusive
            run_identifiers['file_name'] = run[-1]
            if run[0].lower() in Config.accepted_backends:  # the second one is backend
                run_identifiers['backend'] = run[0]
            else:  # the second one is results_dir
                run_identifiers['results_dir'] = run[0]
        else:
            raise ResultsAttributeError('Invalid run identifiers!')

        return run_identifiers

    # **************************** Methods fetching logs from results directories ************************************ #

    def fetch_solution(self, run_id, backend=None, results_dir=None, file_name=None):
        """This method will return the solution DataFrame loaded from the .csv logged file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        In the most cases, self.fetch_solution will only be used with one attribute of run_id, with results
        being stored in the config-default backend-specific results_directory and with backend being the nominal
        backend specified in the ResultsParser instantiation. For other backends, backend might be explicitly passed.
        For results outside the normal pygmofwork workspace directory, the results_dir might be explicitly passed
        (mutually exclusive with backend argument.)
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param file_name: (str) optional custom name of the model solution csv dump file. if not passed,
                          defaults to the default log name.
        :return: (pd.DataFrame) solution loaded from the .csv log. See GlobalModel.get_solution for explanation of the
                 solution DataFrame structure.
        """
        run_dir = self._fetch_run_dir(run_id, backend, results_dir)
        if file_name is None:
            file_name = Config.default_log_name('solution', run_id=run_id)  # get default file name
        sol_path = os.path.join(run_dir, file_name)
        if not os.path.isfile(sol_path):
            raise ResultsNotFoundError('Solution not found in "{}"!'.format(sol_path))
        solution = pd.read_csv(sol_path, index_col=0)  # read the solution from the csv log file
        return solution

    def fetch_rates(self, run_id, backend=None, results_dir=None, file_name=None):
        """This method will return the rates DataFrame loaded from the .csv logged file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        In the most cases, self.fetch_rates will only be used with one attribute of run_id, with results
        being stored in the config-default backend-specific results_directory and with backend being the nominal
        backend specified in the ResultsParser instantiation. For other backends, backend might be explicitly passed.
        For results outside the normal pygmofwork workspace directory, the results_dir might be explicitly passed
        (mutually exclusive with backend argument.)
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param file_name: (str) optional custom name of the model rates csv dump file. if not passed,
                          defaults to the default log name.
        :return: (pd.DataFrame) solution loaded from the .csv log. See GlobalModel.get_rates for explanation of the
                 rates DataFrame structure.
        """
        run_dir = self._fetch_run_dir(run_id, backend, results_dir)
        if file_name is None:
            file_name = Config.default_log_name('rates', run_id=run_id)  # default file name
        rates_path = os.path.join(run_dir, file_name)
        if not os.path.isfile(rates_path):
            raise ResultsNotFoundError('Rates not found in "{}"!'.format(rates_path))
        rates = pd.read_csv(rates_path, index_col=0)  # read the csv into a DataFrame
        # change the columns for reaction ids from str to int...
        columns = np.array(rates.columns, dtype=object)
        columns[1:] = np.array(columns[1:], dtype=int)
        rates.columns = columns
        return rates

    def fetch_wall_fluxes(self, run_id, backend=None, results_dir=None, file_name=None):
        """This method will return the wall fluxes DataFrame loaded from the .csv logged file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        In the most cases, self.fetch_wall_fluxes will only be used with one attribute of run_id, with results
        being stored in the config-default backend-specific results_directory and with backend being the nominal
        backend specified in the ResultsParser instantiation. For other backends, backend might be explicitly passed.
        For results outside the normal pygmofwork workspace directory, the results_dir might be explicitly passed
        (mutually exclusive with backend argument.)
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param file_name: (str) optional custom name of the model wall_fluxes csv dump file. if not passed,
                          defaults to the default log name.
        :return: (pd.DataFrame) solution loaded from the .csv log. See GlobalModel.get_wall_fluxes for explanation of
                 the wall_fluxes DataFrame structure.
        """
        run_dir = self._fetch_run_dir(run_id, backend, results_dir)
        if file_name is None:
            file_name = Config.default_log_name('wall_fluxes', run_id=run_id)  # default file name
        wall_fluxes_path = os.path.join(run_dir, file_name)
        if not os.path.isfile(wall_fluxes_path):
            raise ResultsNotFoundError('Wall fluxes not found in "{}"!'.format(wall_fluxes_path))
        wall_fluxes = pd.read_csv(wall_fluxes_path, index_col=0)  # read the csv into a DataFrame
        return wall_fluxes

    @staticmethod
    def get_results_frames(results_df, times):
        """Interpolate model outputs for certain times passed in the times array.
        :param results_df: (pd.DataFrame) results_df log. Might be what gets returned by self.fetch_solution or
                           self.fetch_rates or any other DataFrame with the monotonic time 't' column
        :param times: (array) of float times inside the solution time range (solution['t'])
        :return: (pd.DataFrame) the same structure as the solution, but only with rows corresponding to times in times
                 (linearly interpolated from original solution rows, extrapolation is forbidden - raises
                 AssertionError)
        """
        # coherence check:
        for t in times:
            assert results_df['t'].iloc[0] <= t <= results_df['t'].iloc[-1], \
                'time={} is outside the results_df range'.format(t)

        frames = pd.DataFrame(columns=results_df.columns)
        frames.loc[:, 't'] = times
        ignore_columns = {'t'}
        for col in results_df.columns:
            if col not in ignore_columns:
                vals_at_times = np.interp(times, results_df['t'], results_df[col])
                frames.loc[:, col] = vals_at_times
        return frames

    def fetch_model_params(self, run_id, backend=None, results_dir=None, attributes_dump_name=None):
        """This method will return the model parameters dict from the logged .yaml file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        In the most cases, self.fetch_rates will only be used with one attribute of run_id, with results
        being stored in the config-default backend-specific results_directory and with backend being the nominal
        backend specified in the ResultsParser instantiation. For other backends, backend might be explicitly passed.
        For results outside the normal pygmofwork workspace directory, the results_dir might be explicitly passed
        (mutually exclusive with backend argument.)
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param attributes_dump_name: (str) optional custom name of the model attributes yaml dump file. if not passed,
                                     defaults to the default log name.
        :return: (dict) loaded from the .yaml log. See ModelParameters class for the dict structure.
        """
        model_attributes = self._fetch_model_attributes(run_id, backend, results_dir, attributes_dump_name)
        return model_attributes['model_params']

    def fetch_initial_params(self, run_id, backend=None, results_dir=None, attributes_dump_name=None):
        """This method will return the initial parameters dict from the logged .yaml file. Returns from
        the run_id run directory (see self._fetch_run_dir defined above). At most one of backend/results_dir might
        be passed. If none is, nominal results directory is used.
        In the most cases, self.fetch_rates will only be used with one attribute of run_id, with results
        being stored in the config-default backend-specific results_directory and with backend being the nominal
        backend specified in the ResultsParser instantiation. For other backends, backend might be explicitly passed.
        For results outside the normal pygmofwork workspace directory, the results_dir might be explicitly passed
        (mutually exclusive with backend argument.)
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param attributes_dump_name: (str) optional custom name of the model attributes yaml dump file. if not passed,
                                     defaults to the default log name.
        :return: (dict) loaded from the .yaml log. See GlobalModel class' run method for the dict structure.
        """
        model_attributes = self._fetch_model_attributes(run_id, backend, results_dir, attributes_dump_name)
        return model_attributes['initial_params']

    # ****************** Methods showing value-sorted contributions to time derivatives of species ******************* #

    @staticmethod
    def _assert_consistency(sp, chemistry, solution_frame=None, rates_frame=None, wall_fluxes_frame=None):
        """Method to assert that passed chemistry instance and solution, rates and wall_fluxes log dataframes are
        consistent with each other.
        :param sp: (RP, str or int) either RP instance, or RP id or name.
        :param chemistry: (Chemistry instance) - See Chemistry doc
        :param solution_frame: (pd.Series) - One row of self.fetch_solution (may or may not have the 't' value)
        :param rates_frame: (pd.Series) - One row of self.fetch_rates (may or may not have the 't' value)
        :param wall_fluxes_frame: (pd.Series) - One row of self.fetch_wall_fluxes (may or may not have the 't' value)
        :return: None
        """
        # species needs to be in chemistry:
        chemistry.get_rp(sp)  # will raise an error if the species not found in the chemistry.
        # all the species from chemistry need to be in the solution index
        if solution_frame is not None:
            assert set(chemistry.get_species_name()) == set(solution_frame.index) - {'t', 'e', 'Te', 'Tg', 'p'}, \
                'Inconsistency between the passed chemistry and solution!'
        # all the reaction ids from chemistry need to be identical to the rates index
        if rates_frame is not None:
            assert set(chemistry.get_reactions().index) == set(rates_frame.index) - {'t'}, \
                'Inconsistency between the passed chemistry and rates!'
        # all the species names from chemistry need to be identical to index of wall fluxes series:
        if wall_fluxes_frame is not None:
            assert set(chemistry.get_species_name()) == set(wall_fluxes_frame.index) - {'t'}, \
                'Inconsistency between the passed chemistry and wall fluxes!'

    @staticmethod
    def get_volumetric_rates(sp, chemistry, rates_frame, annotate=False):
        """A method to get production and consumption rates for all the volumetric reactions producing or consuming the
        species "sp". Positive values are volume production processes and negative values are volume consumption
        processes. The returned Series is sorted by descending values.
        :param sp: (str, int or RP instance) either RP instance, or RP id or name.
        :param chemistry: (Chemistry instance) - See Chemistry doc
        :param rates_frame: (pd.Series) - One row of self.fetch_rates (may or may not have the 't' value)
        :param annotate: (bool) - If False (default), then return Series has indices of Reaction ids for all the
                         reactions either producing or consuming "sp". If True, then the ids are user-readable
                         reaction strings. Annotate=True is meant for only printing the values.
        :return: (pd.Series) All volumetric process rates for processes producing or consuming the species "sp".
                 Indexed by either species ids, or user-readable reaction strings (if annotate==True)
        """
        ResultsParser._assert_consistency(sp=sp, chemistry=chemistry, rates_frame=rates_frame)
        sp_id = chemistry.get_rp(sp).id
        stoichiomatrix = chemistry.get_stoichiomatrix(disabled=False, special=True, method='net')
        # I'm certain that 'M' is neither consumed nor produced, so vol rates for M will be empty.
        stoichiovector = stoichiomatrix.loc[stoichiomatrix[sp_id] != 0, sp_id]

        # stoichiovector has index of reactions ids which produce or consume the sp and values are stoich coefs
        # (-ive for consumption!)
        volumetric_rates = stoichiovector * rates_frame.loc[stoichiovector.index]
        # sort values:
        volumetric_rates = volumetric_rates.sort_values(ascending=False)
        # annotate => replace the reaction ids with reaction strings:
        if annotate:
            volumetric_rates.index = [
                '{: <10}{}'.format('R{}'.format(r_id), chemistry.get_reaction(r_id))
                for r_id in volumetric_rates.index
            ]
        return volumetric_rates

    @staticmethod
    def get_surface_rates(sp, chemistry, model_params, wall_fluxes_frame, annotate=False):
        """A method to get production and consumption rates for all the surface conversions producing or consuming the
        species "sp". Positive values are surface production processes and negative values are surface consumption
        processes. The returned Series is sorted by descending values.
        :param sp: (str, int or RP instance) either RP instance, or RP id or name.
        :param chemistry: (Chemistry instance) - See Chemistry doc
        :param model_params: (dict or ModelParameters instance) - See ModelParameters doc
        :param wall_fluxes_frame: (pd.Series) - One row of self.fetch_wall_fluxes (may or may not have the 't' value)
        :param annotate: (bool) - If False (default), then return Series has indices of RP ids for all the species
                         which are being converted to "sp". If True, then the ids are user-readable surface reactions.
                         Annotate=True is meant for only printing the values.
        :return: (pd.Series) All surface conversion process rates for processes producing or consuming the species "sp".
                 Indexed by either species ids, or user-readable wall conversion strings (if annotate==True)
        """
        ResultsParser._assert_consistency(sp=sp, chemistry=chemistry, wall_fluxes_frame=wall_fluxes_frame)

        # surface rates do not make sense for special species, which do not get converted!
        if chemistry.get_rp(sp).is_special():
            warnings.warn('Surface rates not defined for the species {}!'.format(sp))
            # returns empty series, the same as any species which is not produced or consumed by any surface processes
            return pd.Series(dtype=np.float64)

        sp_id = chemistry.get_rp(sp).id
        ret_matrix = chemistry.get_return_matrix(disabled=False, special=False)
        area = 2*np.pi*model_params['radius'] * (model_params['radius'] + model_params['length'])  # in [m2]
        volume = np.pi*model_params['radius']**2 * model_params['length']  # in [m3]

        # wall fluxes frame needs to have species ids as index, not names:
        wall_fluxes_frame = wall_fluxes_frame.copy()
        if 't' in wall_fluxes_frame.index:
            wall_fluxes_frame = wall_fluxes_frame.drop('t')  # drop the time stamp from the Series
        wall_fluxes_frame.index = [chemistry.get_rp(sp).id for sp in wall_fluxes_frame.index]

        # some sanity checks:
        assert list(wall_fluxes_frame.index) == list(ret_matrix.index) == list(ret_matrix.columns), \
            'Inconsistent return matrix and wall fluxes frame!'

        # build the rates matrix:
        sticking_rates = wall_fluxes_frame[ret_matrix.index] * area/volume
        return_rates = -ret_matrix.multiply(sticking_rates, axis='columns')
        surface_rates_matrix = \
            pd.DataFrame(np.diag(sticking_rates), columns=sticking_rates.index, index=sticking_rates.index) + \
            return_rates

        # select just non-zero elements from one row belonging to the species sp:
        surface_rates = surface_rates_matrix.loc[sp_id]
        surface_rates = surface_rates[surface_rates != 0]

        # annotate:
        if annotate:
            new_index = []
            sp_names = chemistry.get_species_name()
            for i in surface_rates.index:
                if i == sp_id:
                    r_string = '{} + wall > wall'.format(sp)
                else:
                    r_string = '{} + wall > {} + wall'.format(sp_names[i], sp)
                new_index.append('{: <10}{}'.format('S{}'.format(i), r_string))
            surface_rates.index = new_index

        return surface_rates

    @staticmethod
    def print_production_processes(sp, chemistry, model_params, rates_frame, wall_fluxes_frame, n=None, unit='m-3/s'):
        """A method to print all the most prominent production processes together with production rates. Includes
        volumetric production processes as well as surface conversions.
        :param sp: (str, int or RP instance) either RP instance, or RP id or name.
        :param chemistry: (Chemistry instance) - See Chemistry doc
        :param model_params: (dict or ModelParameters instance) - See ModelParameters doc
        :param rates_frame: (pd.Series) - One row of self.fetch_rates (may or may not have the 't' value)
        :param wall_fluxes_frame: (pd.Series) - One row of self.fetch_wall_fluxes (may or may not have the 't' value)
        :param n: (int) - How many most prominent processes to print out. If default (None), prints all.
        :param unit: (str) - Which unit of production rates to print in - from {'cm-3/s', 'm-3/s'}, default is 'm-3/s'
        :return: None
        """
        # build the production/consumption rates in volume and on surfaces
        volumetric_rates = \
            ResultsParser.get_volumetric_rates(sp=sp, chemistry=chemistry, rates_frame=rates_frame, annotate=True)
        surface_rates = \
            ResultsParser.get_surface_rates(sp=sp, chemistry=chemistry, model_params=model_params,
                                            wall_fluxes_frame=wall_fluxes_frame, annotate=True)
        all_rates = volumetric_rates.append(surface_rates, verify_integrity=True)
        production_rates = all_rates[all_rates > 0]
        # capping
        if n is None:
            n = len(production_rates)
        # print the stuff:
        print('\nProcesses and rates of production for {} [{}]:'.format(sp, unit))
        series_to_print = \
            production_rates.sort_values(ascending=False).iloc[:n] * ResultsParser.unit_conversion_factors[unit]
        print(series_to_print.to_string())
        if n < len(production_rates):
            print('{: ^50}'.format('...'))
        print()

    @staticmethod
    def print_consumption_processes(sp, chemistry, model_params, rates_frame, wall_fluxes_frame, n=None, unit='m-3/s'):
        """A method to print all the most prominent consumption processes together with consumption rates. Includes
        volumetric consumption processes as well as surface conversions.
        :param sp: (str, int or RP instance) either RP instance, or RP id or name.
        :param chemistry: (Chemistry instance) - See Chemistry doc
        :param model_params: (dict or ModelParameters instance) - See ModelParameters doc
        :param rates_frame: (pd.Series) - One row of self.fetch_rates (may or may not have the 't' value)
        :param wall_fluxes_frame: (pd.Series) - One row of self.fetch_wall_fluxes (may or may not have the 't' value)
        :param n: (int) - How many most prominent processes to print out. If default (None), prints all.
        :param unit: (str) - Which unit of consumption rates to print in - from {'cm-3/s', 'm-3/s'}, default is 'm-3/s'
        :return: None
        """
        # build the production/consumption rates in volume and on surfaces
        volumetric_rates = \
            ResultsParser.get_volumetric_rates(sp=sp, chemistry=chemistry, rates_frame=rates_frame, annotate=True)
        surface_rates = \
            ResultsParser.get_surface_rates(sp=sp, chemistry=chemistry, model_params=model_params,
                                            wall_fluxes_frame=wall_fluxes_frame, annotate=True)
        all_rates = volumetric_rates.append(surface_rates, verify_integrity=True)
        consumption_rates = all_rates[all_rates < 0]
        # capping
        if n is None:
            n = len(consumption_rates)
        # print the stuff:
        print('\nProcesses and rates of consumption for {} [{}]:'.format(sp, unit))
        series_to_print = \
            consumption_rates.sort_values(ascending=True).iloc[:n] * ResultsParser.unit_conversion_factors[unit]
        print(series_to_print.to_string())
        if n < len(consumption_rates):
            print('{: ^50}'.format('...'))
        print()

    # ********************************* METHODS POST-PROCESSING SOLUTION DATA **************************************** #

    @staticmethod
    def compress_solution(full_sol, diff_threshold=0.05, min_timestep=0.001):
        """Method to return a compressed solution from a full solution. WARNING: the points of power discontinuity
        might be a bit dodgy in the compressed solution, so use with care!
        :param full_sol: (pd.DataFrame) solution loaded from the .csv log. See GlobalModel.get_solution
                         for explanation of the solution DataFrame structure.
        :param diff_threshold: (float) if relative difference between two timeframes is lower than diff_threshold,
                               the second frame is removed from the compressed solution
        :param min_timestep: (float) if relative time difference between two timeframes is lower than min_timestep,
                             second frame is removed from the compressed solution removed
        :return: (pd.DataFrame) solution dataframe with the same columns as the full_sol, but compressed (not
                 containing all the original timesteps)
        """
        drop_index = set([])  # all the indices to be dropped from the full solution!
        min_step_abs = full_sol['t'].iloc[-1]*min_timestep  # minimal allowed step between two subsequent timeframes

        front_index = 0

        while True:
            if front_index >= len(full_sol.index) - 1:
                break

            next_index = front_index + 1
            while True:
                if next_index >= len(full_sol.index) - 1:
                    break
                diff = abs(full_sol.loc[next_index] - full_sol.loc[front_index])
                rel_diff = diff/full_sol.loc[front_index]
                if diff['t'] < min_step_abs:
                    drop_index.add(next_index)
                    next_index += 1
                elif (rel_diff < diff_threshold).drop('t').all():
                    drop_index.add(next_index)
                    next_index += 1
                else:
                    break
            front_index = next_index

        compressed_sol = full_sol.drop(drop_index, axis=0)

        return compressed_sol

    # ******************************* Methods plotting time-dependent solutions ************************************** #

    def plot_solution(self, run_id, *to_plot, backend=None, results_dir=None, file_name=None, log=True):
        """Method to quickly plot chosen parameters (in time from solution) for a chosen run_id. For most of the
        arguments documentation, see self.fetch_solution method.
        :param run_id: (str) name of a run (in the pygmo_fwork framework) or a directory in the results_dir holding
                       all the logs (in case of fetching from a custom results_dir)
        :param to_plot: (str) 0-or-more parameters to plot, might be eg. 'Te', or 'e', 'Ar+'. If nothing passed, all
                        densities are plotted.
        :param backend: (str) identifier of the backend model. Must be from Config.accepted_backends
        :param results_dir: (str) path to the results directory. Mutually exclusive with backend.
        :param file_name: (str) optional custom name of the model solution csv dump file. if not passed,
                          defaults to the default log name.
        :param log: (bool) if True, logscale used (default).
        :return: None
        """
        sol = self.fetch_solution(run_id, backend=backend, results_dir=results_dir, file_name=file_name)
        if not len(to_plot):
            to_plot = list(sol.columns)[1:-4]  # get rid of time and non-density parameters
        for param in to_plot:
            plt.plot(sol['t'], sol[param], label='{}'.format(param))
        plt.xlabel('Time (s)')
        if len({'p', 'Te', 'Tg'}.intersection(set(to_plot))):
            ylabel = '$n$ (m$^{-3}$), $p$ (Pa), $T_{\\mathrm{e}}$ (eV), $T$ (K)'
        else:
            ylabel = 'Particle Density (m$^{-3}$)'
        plt.ylabel(ylabel)
        if log:
            plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.show()

    # **************************** Methods dealing with multiple logged solutions ************************************ #

    def compare(self, runs, relative_to_first=False, printout=True):
        """Method to compare several previously logged runs. Each run from the passed runs iterable describes one
        run by the means of one of the following:
            'run_id',
            ('run_id', 'backend'),
            ('run_id', 'results_dir'),
            ('run_id', 'solution_file_name'),
            ('run_id', 'backend', 'solution_file_name'),
            ('run_id', 'results_dir', 'solution_file_name')
        Only str (for run_id) or tuples of strings are supported, with all parameters not supplied in tuples being
        defaulted or using the nominal ones.
        The method decides itself which parameter in the run tuple is which. If it cannot decide, raises
        ResultsAttributeError.
        In the normal situation for the runs logged into config-default workspace-based result directories,
        either 'run_id' or ('run_id', 'backend') will be used to identify runs, with 'run_id' defining runs inside
        the nominal backend (passed into ResultsParser __init__) and with the latter specifying backend explicitly.
        Will build and return a comparison dataframe with all the final solutions from all the passed runs.
        If called in verbose mode, will pretty-print the dataframe as well.
        Also an option to build the comparison frame filled with relative differences against the first passed run, as
        opposed to absolute values (default).
        :param runs: iterable of one of {str, (tuple of str)}
        :param relative_to_first: (bool)
        :param printout: (bool)
        :return: (pd.DataFrame)
        """
        final_solutions = []  # list filled with "final" solutions (solutions for the final time)
        for run in runs:
            run_identifiers = self._parse_run_identifiers(run)
            solution = self.fetch_solution(**run_identifiers)  # solution dataframe
            # build the solution name (will be index in the final comparison dataframe)
            run_id = run_identifiers['run_id']
            if run_identifiers['backend'] is not None:
                parent_dir = run_identifiers['backend']
            elif run_identifiers['results_dir'] is not None:
                parent_dir = os.path.split(run_identifiers['results_dir'])[-1]
            else:
                parent_dir = os.path.split(self.nominal_results_dir)[-1]
            sol_name = os.path.join('...{}'.format(parent_dir), run_id)
            if len(sol_name) > 40:
                sol_name = '...{}'.format(sol_name[-40:])
            # that's solution name done
            final_solution = pd.Series(solution.iloc[-1], name=sol_name)
            final_solutions.append(final_solution)
        comp_df = pd.DataFrame(final_solutions)  # build the final comparison dataframe.

        if relative_to_first:  # relative differences rather than absolute values
            comp_df = (comp_df - comp_df.iloc[0]) / comp_df.iloc[0] * 100.

        if printout:  # pretty printout
            print()
            format_str = '{:.2E}' if not relative_to_first else '{:.2f}'
            with pd.option_context('display.float_format', format_str.format):
                print(comp_df)
            print()

        return comp_df
