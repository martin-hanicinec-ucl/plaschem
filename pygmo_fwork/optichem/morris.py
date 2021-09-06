
import glob
import multiprocessing
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Union, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import yaml

from plaschem.chemistry import Chemistry
from plaschem.species import RP
from pygmo_fwork.config import Config
from pygmo_fwork.global_model import GlobalModel
from pygmo_fwork.optichem.exceptions import \
    MorrisTrajectoryError, MorrisInputsError, MorrisLogError, MorrisCoherenceError
from pygmo_fwork.pygmol.model_parameters import ModelParameters
from pygmo_fwork.results import ResultsParser


class MorrisMethod(object):
    """
    This class handles Morris Method of sensitivity/uncertainty analysis. The class instantiates with Chemistry
    instance, dictionary of model parameters and model backend identifier. All of these have strict mandatory
    structures, which need to be satisfied. Not much validation is done at this level, so careful with
    what you're passing in. Study the docs for __init__ and for classes which' instances should be passed, if passing
    different classes' instances.
    The basic unit of the Morris Method analysis in this framework is a Morris Run, identified by it's
    morris_run_id. Each morris run compose of several trajectories. All the results are logged in the  following
    directory structure:

    - MORRIS_RESULTS_DIR
      - MORRIS_RUN_ID  # identifier for a single morris run
        - META_DIR  # directory with morris run meta files - used to verify coherence when adding trajectories to m. run
          - META_FILE.yaml  # various files used to verify the coherence
          - ...
        - LOGS_DIR  # only applicable if full-logging. Contains directories with full GlobalModel run logs
          - LOGS_TRAJ_DIR  # directory containing all full logs for one trajectory
            - 00_INITIAL_RUN  # directory with GlobalModel full logs for each GM run in each directory
              - GLOBAL_MODEL_LOG_FILE.yaml
              - RESULTS.csv
              - RATES.csv
            - 01_ELEMENTARY_EFFECT_01
              - ...
            - ...
          - LOGS_ANOTHER_TRAJ_DIR
            - ...
      - another_morris_run_id
        - ...
        - ...
        - ...

    If the run_morris method is called with full_log==False, the run subdirs inside the LOG_TRAJ_DIR do not
    contain all the backend model output files but rather only a compressed solution.
    """

    # some default names for files and directories in the logging directory structure:
    meta_dir_name = 'meta'

    traj_id = 'traj{:03d}'  # to be formatted by an integer trajectory index

    logs_dir_name = 'logs'  # name of directory storing all the model run logs, if needed.
    logs_traj_dir_name = '{}'  # to be formatted by the trajectory id (or self.traj_id)

    def __init__(self, chemistry, model_parameters, model_backend='pygmol'):
        """Instantiates the Morris Method class
        :param chemistry: (Chemistry instance) - see doc for Chemistry class
        :param model_parameters: (dict-like) - see docs for GlobalModel class, or any of it's backend models.
        :param model_backend: (str) must be in Config.accepted_backends
        """
        self.chemistry = chemistry  # never to be touched, only copied
        self.model_params = model_parameters  # never to be touched, only copied
        self.model_backend = model_backend  # either pygmol or pygkin, or others from Config.accepted_backends

        # building the morris space and some other morris method parameters:
        self.delta_magnitude = 0.5  # magnitude of the single elementary change
        self.grid_density = 10  # how many points in the 0-1 1D grid of possible factor values.
        # grid of possible factor values in 1D:
        self.grid = 1.0 / self.grid_density / 2 + np.arange(0., 1.0, 1.0 / self.grid_density)
        self.grid = np.round(self.grid, decimals=5)

        # some consistency validations - only here for the case if the above settings are ever made parameters...
        if self.delta_magnitude <= 0:
            raise MorrisInputsError('Delta needs to be a positive number!')
        if self.grid_density <= 0:
            raise MorrisInputsError('Density needs to be a positive number!')
        # verify that I can move to at least one direction from each point of the grid:
        if not (((self.grid + self.delta_magnitude) < 1.) + ((self.grid - self.delta_magnitude) > 0.)).all():
            raise MorrisInputsError('Delta is too large to stay in the unit hypercube at all time!')
        # verify that delta_magnitude is divisible by the grid cell size:
        if not \
                round(self.delta_magnitude / (1.0 / self.grid_density), 10) == \
                int(round(self.delta_magnitude / (1.0 / self.grid_density), 10)):
            raise MorrisInputsError(
                'Delta {} needs to be dividable by the grid cell size {}!'.format(
                    self.delta_magnitude, 1. / self.grid_density)
            )

        # directory tree (for logging only):
        self.morris_run_dir = None
        self.meta_dir = None
        self.logs_dir = None
        self.logs_traj_dir = None
        # Those will all be passed to methods handling the preparation of the directory structure, or taken
        # as defaults from the class attributes.

        # debugging:
        self.debug = False

    # ***************************************** HELPER METHODS ******************************************************* #

    def _get_seed_x(self, x_cont):
        """Method to return the seed (first) factors vector for a morris trajectory. Takes a vector of "continuous"
        values x_cont and discretizes it into the Morris space.
        :param x_cont: (iterable of float) - preferably with values in [0 - 1]
        :return: (np.array) with all the values discretized into the morris space in [0, 1] and on self.grid.
        """
        seed_x = np.empty(len(x_cont))
        for i, val in enumerate(x_cont):
            deltas = abs(self.grid - val)  # absolute differences between the continuous value and all the grid points
            min_delta = min(deltas)
            mask = deltas == min_delta  # pick the "closest" grid point
            seed_x[i] = self.grid[mask][0]  # assign the closest grid point to the continuous value.
        return seed_x

    def _get_seed_x_random(self, num_factors):
        """Method returning a uniform-random valid initial vector of factors for a morris trajectory.
        :param num_factors: (int) number of factors in the seed_x vector
        :return: (np.array) with all the values discretized into the morris space in [0, 1] and on self.grid.
        """
        return np.random.choice(self.grid, size=num_factors, replace=True)

    def _trajectory(self, seed_x):
        """A method generating vectors of factors X for one morris trajectory. It's a generator generating tuples
        of (i, delta, X), where i is the index of the last changed factor, and delta = +- delta_magnitude
        :param seed_x: (np.array) Seed for the first vector of factors X in the Morris trajectory. All
                       values need to be belong into the discretized morris space (in [0, 1] and on one of the
                       grid points.)
        :return: N times (i, delta, X), where i is index of the changed factor, delta is delta with a sign
                 and X is the vector of factors (with X[i] being X'[i] + delta, where X' is a previous vector
                 of factors.) This is a generator.
        """
        seed_x = np.array(seed_x)
        min_dim = 2
        if len(seed_x) < min_dim:
            raise MorrisInputsError('The vector of factors needs to be at least {}D!'.format(min_dim))
        # check if the seed lies in the Morris space:
        for val in seed_x:
            if val not in self.grid:
                raise MorrisInputsError('The vector of factors needs to be at least {}D!'.format(min_dim))

        x = seed_x
        num_factors = len(x)

        for i in np.random.permutation(np.arange(num_factors)):
            x = x.copy()
            for delta in np.round(np.random.permutation([-self.delta_magnitude, self.delta_magnitude]), decimals=5):
                if 0. < x[i] + delta < 1.:
                    x[i] = np.round(x[i] + delta, decimals=5)
                    yield i, delta, x
                    break
            else:
                raise MorrisTrajectoryError('Vector of factors outside a unit hypercube! Should never happen!')

    @staticmethod
    def _map_reactions_factors_to_k(x, nominal_k):
        """Method to map the vector of factors for reactions to a vector of values and return the new values.
        In the current implementation, x_i = 0.5 maps to the nominal k_i, while x_1 = 0.0, 1.0 maps to 0.0 and
        2*nominal k_i respectively, with linear distribution in between.
        :param x: (np.array) vector of factors (confined to a unit hypercube of D=len(x)). Needs to be the same length
                  as the number of reactions in the chemistry.
        :param nominal_k: (np.array or pd.Series) vector of the nominal values of Arrhenius pre-exponential values
                          (or arrh_a in the pygmo_fwork nomenclature), which should be changed to new values
                          by mapping the factors vector X onto them. Needs to be the same length as x and as number
                          of reactions in the chemistry.
        :return: (np.array or pd.Series) vector of new values. If nominal_k was pd.Series, the index has been
                 preserved.
        """
        # type checks:
        allowed_types = (np.ndarray, pd.core.series.Series)
        for argument in [x, nominal_k]:
            if not isinstance(argument, allowed_types):
                raise TypeError('Arguments to this method need to be either np.array, or pd.Series!')

        # consistency checks:
        if not len(x) == len(nominal_k):
            raise MorrisInputsError('Cannot map vector of factors X to the vector of nominal arrh_a values!')
        if hasattr(x, 'index') and hasattr(nominal_k, 'index'):
            if list(x.index) != list(nominal_k.index):
                raise MorrisInputsError('Mismatching index of x vector and values vector!')

        return 2*nominal_k * x

    def _map_species_factors_to_k(self, x, nominal_k, chemistry):
        """Method to map the vector of factors x for species to a vector of new Arrhenius pre-exponential values for
        all the reactions in the chemistry. Currently implemented in the way that if all x_i == 0.5, the returned
        vector of reaction coefficients will remain the same as nominal, while x_i > 0.5 will increase the
        reaction coefficients for all the reactions involving the species_i (on LHS or RHS, irrespective of
        the stoichiometric coefficients), and x_i < 0.5 will decrease the same reaction coefficients.
        The passed chemistry instance is NOT modified in this method, rather the new_k returned needs to be "loaded"
        into the chemistry instance at a higher level!
        :param x: (np.array) vector of factors (confined to a unit hypercube of D=len(x)). Needs to be the same
                  length as a number of species in the chemistry.
        :param nominal_k: (np.array or pd.Series) vector of the nominal values of Arrhenius pre-exponential values
                          (or arrh_a in the pygmo_fwork nomenclature), which should be changed to new values
                          by mapping the factors vector X onto them. Needs to be the same length as number of
                          reactions in the chemistry
        :param chemistry: (Chemistry instance) copy of self.chemistry object.
        :return: (np.array or pd.Series) vector of new values. If nominal_k was pd.Series, the index has been
                 preserved.
        """
        assert chemistry is not self.chemistry  # the passed chemistry instance cannot be the saved attribute object!

        stoichiomatrix_lhs = chemistry.get_stoichiomatrix(method='lhs')
        stoichiomatrix_rhs = chemistry.get_stoichiomatrix(method='rhs')
        # a matrix flagging if a species_i is involved at all in a reaction_j:
        involvement_matrix = pd.DataFrame(stoichiomatrix_lhs + stoichiomatrix_rhs, dtype='?')  # boolean type

        # type checks:
        allowed_types = (np.ndarray, pd.core.series.Series)
        for argument in [x, nominal_k]:
            if not isinstance(argument, allowed_types):
                raise TypeError('Arguments to this method need to be either np.array, or pd.Series!')
        # some consistency checks:
        if hasattr(x, 'index') and list(x.index) != list(involvement_matrix.columns):
            raise MorrisInputsError('Mismatching index of x vector and chemistry species!')
        if hasattr(nominal_k, 'index') and list(nominal_k.index) != list(involvement_matrix.index):
            raise MorrisInputsError('Mismatching index of x vector and chemistry reactions!')
        if not len(x) == chemistry.num_species():
            raise MorrisInputsError('Cannot map vector of factors X to the vector of nominal arrh_a values!')
        if not len(nominal_k) == chemistry.num_reactions():
            raise MorrisInputsError('Cannot map vector of factors X to the vector of nominal arrh_a values!')

        new_k = nominal_k.copy()  # seed the perturbed arrh_a
        for i, sp_id in enumerate(involvement_matrix.columns):
            new_k[involvement_matrix[sp_id]] *= 2*np.array(x)[i]

        return new_k

    def _map_a_factor_to_ret_coef(self,
                                  x_i: float,
                                  stuck_sp: Union[str, int, RP],
                                  ret_sp: Union[str, int, RP],
                                  nominal_ret_coef: float,
                                  chemistry: Chemistry,
                                  rel_range: float
                                  ) -> Dict[str, Dict[str, float]]:
        """
        Method to map a single morris factor x_i for a single return coefficient (a single return species for a
        single stuck species) into the new return coefficient. The stuck species and return species cannot be
        the same, since the return coefficient for the stuck species with itself as a return species is actually
        a dependent parameter, which will be changed together with the return coefficient for the specified return
        species, to conserve the number of returned species per one stuck species.
        If x_i == 0.5, the return coefficient will stay the same as nominal,
        and if x_i < 0.5 or > 0.5, this will decrease or increase the return coefficient from the nominal value.
        The passed chemistry instance IS modified in this method (which is different from the species or reactions
        mapping methods)
        :param x_i: this is a single element of the morris vector of factors, with values between (0, 1)
        :param stuck_sp: a name of the species getting stuck and converted to the return species
        :param ret_sp: a name of the return species
        :param nominal_ret_coef: the nominal return coefficient for the ret_sp. Used as a base for mapping from
                                 the morris factor x_i. x_i = 0.5 will map directly to nominal_ret_coef
        :param chemistry: a chemistry instance, used to read the current ret_coef to calculate the compensation
                          to the return coefficient for the "itself" return species. The chemistry instance IS MODIFIED
                          by this method, i.e. the return coefficients are changed for
                          stuck_sp -> ret_sp and for stuck_sp -> stuck_sp (the latter is the compensation to conserve
                          total number of returned species).
        :param rel_range: Maximal range of change of the nominal return coefficient. the change is mapped from the
                          factor x_i to the return coefficient in a logarithmic fashion. x_i = 0.5 maps the return
                          coefficient to the nominal one, x_i = 0 maps it on nominal_ret_coef/rel_range and
                          x_i = 1 maps it to the nominal_ret_coef*rel_range.
        :return: dict of dicts for all the CHANGED return coefficients.
        """
        assert chemistry is not self.chemistry  # the passed chemistry instance cannot be the saved attribute object!
        # strictly not necessary, since this method does not change the chemistry object in any way, but as a precaution

        # the range needs to be more than 1.0:
        if rel_range <= 1.0:
            raise MorrisInputsError('The relative range parameter needs to be more than 1.0!')
        # instances of stuck and return species:
        stuck_species: RP = chemistry.get_rp(stuck_sp)
        return_species: RP = chemistry.get_rp(ret_sp)
        # coefficients for special species cannot be factors!
        if stuck_species.is_special() or return_species.is_special():
            raise MorrisInputsError('Coefficients for special species cannot be Morris factors!')
        # the nominal return coefficient needs to be positive!
        if not nominal_ret_coef > 0:
            raise MorrisInputsError('Only defined return coefficients can be Morris factors!')
        # stuck species cannot equal return species! the return ratio of itself is a dependent parameter, keeping the
        # total number of return species per one stuck species constant. Normally it will be one, but not enforced.
        if stuck_species.id == return_species.id:
            raise MorrisInputsError('Return coefficient of itself cannot be a Morris factor!')

        stick_coef: float = chemistry.get_species_stick_coef()[stuck_species.id]
        ret_matrix: pd.DataFrame = chemistry.get_return_matrix()
        original_ret_coef: float = ret_matrix.at[return_species.id, stuck_species.id]
        original_ret_coef_self: float = ret_matrix.at[stuck_species.id, stuck_species.id]

        # what is the new return coefficient?
        assert 0 <= x_i <= 1, 'Unexpected value of the X_i Morris factor!'
        exponent: float = 2*x_i - 1  # x_i = 0.5 maps to 0, x_i = 0 maps to -1, x_i = 1.0 maps to 1
        new_ret_coef: float = nominal_ret_coef*rel_range**exponent
        # border cases x_i = 0, 1 map to r_nom/range, r_nom*range

        # sticking coefficient for the stuck species must be defined in the chemistry instance:
        if not stick_coef > 0:
            raise MorrisInputsError('Only return coefficients for species which GET STUCK can be Morris factors!')

        # the change in return coefficient needs to be compensated by the return coefficient for itself!
        d_ret_coef: float = new_ret_coef - original_ret_coef
        new_ret_coef_self: float = original_ret_coef_self - d_ret_coef
        # the new return coefficient for itself needs to be >= 0:
        if new_ret_coef_self < 0:
            raise MorrisInputsError('The return coefficient cannot be changed by that much while preserving the '
                                    'number of returned particles! Try lowering the rel_range!')

        # apply the new return coefficients to the chemistry:
        new_ret_coefs: dict = chemistry.get_species_ret_coefs()[stuck_species.id]
        new_ret_coefs.update({return_species.get_name(): new_ret_coef, stuck_species.get_name(): new_ret_coef_self})
        chemistry.set_adhoc_species_attributes('ret_coefs', stuck_species, new_ret_coefs)  # finally set it

        # return the changed return coefficients
        return {
            stuck_species.get_name():
                {return_species.get_name(): new_ret_coef, stuck_species.get_name(): new_ret_coef_self}
        }

    @staticmethod
    def _map_factors_to_model_params(x, model_param_ranges):
        raise NotImplementedError

    # ********************************* BUILDING THE DIRECTORY STRUCTURE ********************************************* #

    def _prepare_logging(self, morris_run_id, morris_results_dir=None, traj_id=None):
        """A method to prepare all the paths and names for logging the morris run results and also to build the
        whole directory structure (at the time of the first call). See the __init__ docstring for the documentation
        of the directory structure.
        :param morris_run_id: (str) Unique identifier for a Morris Run (contains multiple trajectories)
        :param morris_results_dir: (path) Full path to the parent folder for the morris_run_id folder.
                                   If not passed, defaults to the Config default path.
        :param traj_id: (str) this name identifies a single trajectory for the Morris Run.
                        If not passed, defaults to self.traj_id.format(index), with index increasing from trajectories
                        already saved
        :return: None
        """
        if morris_results_dir is None:
            morris_results_dir = Config().get_morris_results_dir()  # Parent results directory

        self.morris_run_dir = os.path.join(morris_results_dir, morris_run_id)  # dir for the current Morris run
        self.meta_dir = os.path.join(self.morris_run_dir, self.meta_dir_name)  # directory for meta files

        self.logs_dir = os.path.join(self.morris_run_dir, self.logs_dir_name)
        if traj_id is None:
            traj_id = self._get_next_traj_id(self.logs_dir)  # either passed or first available name for a single traj.

        # directories where either full logs or compressed solution for each run for the trajectory will be stored
        self.logs_traj_dir = os.path.join(self.logs_dir, self.logs_traj_dir_name.format(traj_id))
        # full path to the trajectory meta.yaml file
        meta_path = os.path.join(self.logs_traj_dir, 'meta.yaml')
        if os.path.isfile(meta_path):
            raise MorrisLogError('Trajectory directory already exists: {}'.format(meta_path))

        # make the directories if they are not yet existing:
        for path in [self.morris_run_dir, self.meta_dir, self.logs_dir, self.logs_traj_dir]:
            if not os.path.isdir(path):
                os.mkdir(path)

    def _get_next_traj_id(self, logs_dir):
        """Method to automatically generate trajectory id identifier. Defaults to MorrisMethod.traj_id.format(i)
        :param logs_dir: (path) directory with previous trajectory runs logs - needed to generate the next name
                         with lowest unused index i.
        :return: (str) Trajectory ID identifier.
        """
        for i in range(1, 999):  # if changed to more than 999, also change the self.traj_id formatting!
            traj_id = self.traj_id.format(i)
            if not os.path.isfile(os.path.join(logs_dir, traj_id, 'meta.yaml')):
                return traj_id
        else:
            raise MorrisLogError('Could not find a suitable trajectory id!')

    # ********************************** RUNNING THE MORRIS TRAJECTORIES ********************************************* #

    def _assert_coherence(self, meta_dir, traj_type):
        """Method asserting the coherence between previously run morris trajectories (for a given Morris Run) and a
        new trajectory about to run. The method will load the meta data from the meta_dir and check, if the chemistry
        and model_params attached to self are coherent with the meta data. This check is performed for each (but first)
        trajectory being added to any Morris Run. Will raise an error if the coherence assertion fails.
        :param meta_dir: (path) to the meta_dir in any particular Morris Run
        :param traj_type: (str) supported are 'reactions', 'species', 'return_coefficients',
                          'model_parameters' each for one supported (coded) Morris Run types
        :return: None
        """
        if len(os.listdir(meta_dir)):
            if not self.chemistry.check_coherence(os.path.join(meta_dir, 'chemistry_attributes.yaml')):
                raise MorrisCoherenceError(
                    'Chemistry is not coherent with previous morris trajectories for this Morris Run ID')
            with open(os.path.join(meta_dir, 'model_parameters.yaml')) as stream:
                meta_model_params = dict(ModelParameters(yaml.load(stream, Loader=yaml.FullLoader)))
                if not meta_model_params == dict(ModelParameters(self.model_params)):
                    raise MorrisCoherenceError(
                        'Model parameters are not coherent with previous morris trajectories for this Morris Run ID')
            with open(os.path.join(meta_dir, 'meta.yaml')) as stream:
                meta_meta = yaml.load(stream, Loader=yaml.FullLoader)
                if not meta_meta['trajectory_type'] == traj_type:
                    raise MorrisCoherenceError(
                        'Trajectory type is not coherent with previous morris trajectories for this Morris Run ID')
                if not meta_meta['model_backend'] == self.model_backend:
                    raise MorrisCoherenceError(
                        'Model backend is not coherent with previous morris trajectories for this Morris Run ID')

    def _dump_meta(self, meta_dir, traj_type):
        """Method to dump meta data to a passed meta_dir (for one particular Morris Run). The traj_type has several
        options:
        'reactions' if called from self.run_trajectory(traj_type='reactions')
        'species' if called from self.run_trajectory(traj_type='species')
        'return_coefficients' if called from self.run_trajectory(traj_type='return_coefficients')
        'model_parameters' if called from self.run_trajectory(traj_type='model_parameters')
        :param meta_dir: (path) full path towards the meta dir, where the meta data files will be dumped.
        :param traj_type: (str) 'reactions', 'species', 'return_coefficients' or 'model_parameters'
        :return: None
        """
        self.chemistry.dump_chemistry_attributes(os.path.join(meta_dir, 'chemistry_attributes.yaml'))

        model_params = dict(ModelParameters(self.model_params))  # this cleans up the dict, so ready for yaml dump
        with open(os.path.join(meta_dir, 'model_parameters.yaml'), 'w') as stream:
            yaml.dump(model_params, stream)

        with open(os.path.join(meta_dir, 'meta.yaml'), 'w') as stream:
            yaml.dump({'trajectory_type': traj_type, 'model_backend': self.model_backend.lower()}, stream)

    @staticmethod
    def _log_solution(solution, run_id, logs_traj_dir, compress):
        """Logs a compressed solution into logs_traj_dir/run_id/solution.csv
        :param solution: (pd.DataFrame)
        :param run_id: (str)
        :param logs_traj_dir: (str)
        :param compress: (bool)
        :return: None
        """
        run_dir = os.path.join(logs_traj_dir, run_id)
        os.mkdir(run_dir)
        if compress:
            solution = ResultsParser.compress_solution(solution, diff_threshold=0.05, min_timestep=0.001)
        solution.to_csv(os.path.join(run_dir, 'solution.csv'))

    def run_trajectory(self, morris_run_id, traj_type, morris_results_dir=None, traj_id=None,
                       full_log=False, verbosity=2, random_seed=None, compress_solution=True,
                       ret_species: Union[None, Sequence[Tuple[str, str]]] = None,
                       rel_ranges: Union[None, float, Sequence[float]] = None
                       ) -> None:
        """Method to run a single morris trajectory under a particular Morris Run (identified by it's morris_run_id).
        The chemistry and model_params instance attributes must be coherent with the same ones for any previous
        trajectories already run for the same morris_run_id. Output of the method is a saved trajectory .csv file
        with all the elementary effects between factors and observables for this particular trajectory in a relevant
        folder within the directory structure. The same matrix is also returned as a pandas.DataFrame.
        If none of the optional parameters are passed, the directory structure is built in the config-specified
        workspace directory and all the log-file names are defaulted to sensible names.
        Recommended way to call this method is only with the morris_run_id argument.
        :param morris_run_id: (str) identifier for a morris run
        :param traj_type: (str) one of {'species', 'results', 'return_coefficients', 'model_params'}.
        :param morris_results_dir: (path) of the morris_run_id parent dir. Normally defaulted to a config-default,
                                   no need to pass anything.
        :param traj_id: (str) identifier for a single trajectory. Normally defaulted, so no need to pass anything.
        :param full_log: (bool) if True, full logs for every single GlobalModel run are saved. This means e.g. for
                         each trajectory, Ns GlobalModel evaluations are made and all logs saved, where Ns is number
                         of reactions in the chemistry. If False, only a solution is logged into the
                         same run_id subdirectory as the full logs. Defaults to False.
        :param verbosity: (int) Describes different level of verbosity => what gets printed out to stdout.
                          0: nothing is printed
                          1: only lines about start and finish of a trajectory are printed
                          2: also prints out lines signaling start for each step in the trajectory
                          3: on top of above, also runs the backend model in the verbose mode
        :param random_seed: (int) seed for the random generator. Leave at default if not debugging!
        :param compress_solution: (bool) only in effect, if full_log is False. If compress_solution is True,
                                  solution is logged compressed, if False, copy is saved to csv. If full_log is True,
                                  the full solution is logged together with all the other log files. Defaults to True.
        :param ret_species: Only relevant if traj_type == 'return_coefficients', otherwise None by default.
                            Ordered sequence of stuck species and their return species, on which to map the morris
                            factors. An example is:
                            [('NH2+', 'NH2'), ('NH2+', 'NH3'), ('NH2+', 'NH'), ('Ar+', 'Ar')],
                            where
                            in total 4 return coefficients will be varied in the Morris run:
                                NH2+   ->   NH2
                                NH2+   ->   NH3
                                NH2+   ->   NH
                                Ar+    ->   Ar
                            Stuck species and return species NEED to be different, since the return coefficient for the
                            stuck species itself gets changed as a dependent parameter, to preserve the number of
                            returned species. Also, all the specified return coefficients must have non-zero value to
                            start with. This is checked for and asserted downstream.
        :param rel_ranges: Only relevant if traj_type == 'return_coefficients', otherwise None by default.
                           This parameters specifies how much should the nominal values of the specified return
                           coefficients change with different morris factors. Supported are either a single float
                           (the same relative ranges for each coefficient) or a simple 1D sequence (the same length
                           as number of specified return coefficients). As an example, if rel_ranges == 2, then
                           for morris factor x_i = 0.0 the return coefficient will become nominal_r/2 and for x_i = 1.0,
                           the return coefficient will be mapped to nominal_r*2. Morris factor x_i = 0.5 ALWAYS gets
                           mapped to the nominal value nominal_r. The mapping distribution is logarithmic.
        :return: None
        """
        # make the copies of everything for this particular run... This way multiple trajectories can run parallel
        chemistry = self.chemistry.copy()
        model_params = self.model_params.copy()
        model = GlobalModel(chemistry, model_params, backend=self.model_backend)

        # make sure that all the relevant attributes are passed:
        if traj_type == 'return_coefficients':
            assert ret_species is not None
            assert len(ret_species) == len(set(ret_species))  # ensure the same pair is not passed twice!
            assert rel_ranges is not None
            if not isinstance(rel_ranges, Sequence):
                assert isinstance(rel_ranges, Number)
                rel_ranges = len(ret_species) * [rel_ranges, ]  # convert to a sequence - each ret coef has it's ranges
        else:
            assert ret_species is None
            assert rel_ranges is None

        # seed the random number generator. this needs to be done so that parallel executions are not identical
        np.random.seed(random_seed)

        # make the directory tree and prepare all the paths to directories and files needed for logging:
        self._prepare_logging(morris_run_id, morris_results_dir, traj_id)

        self._assert_coherence(self.meta_dir, traj_type=traj_type)

        if verbosity >= 1:
            msg = 'MORRIS RUN "{}": TRAJECTORY "{}" STARTED '.format(
                morris_run_id, os.path.split(self.logs_traj_dir)[-1])
            print('{:.<100}'.format(msg))

        # dimension of the morris vector:
        if traj_type == 'species':
            factors_dimension = model.num_species()
        elif traj_type == 'reactions':
            factors_dimension = model.num_reactions()
        elif traj_type == 'return_coefficients':
            factors_dimension = len(ret_species)
        elif traj_type == 'model_params':
            raise NotImplementedError
        else:
            raise ValueError('Trajectory type {} not recognised!'.format(traj_type))

        # RUN THE MORRIS TRAJECTORY:
        meta_dict = {}  # dictionary with all the meta_data for this one trajectory
        traj_data = pd.DataFrame(dtype='object')  # dataframe holding any relevant trajectory data - mainly for debug

        # prepare for the first run...
        # initial vector of factors:
        x0 = self._get_seed_x_random(factors_dimension)
        # initial vector of arrh_a or model_params:
        if traj_type == 'species':
            nominal_values = chemistry.get_reactions_arrh_a(si_units=False).copy()  # pd.Series with r_ids index
            values_ids = np.array(nominal_values.index)
            factors_ids = np.array(chemistry.get_species_name())
            arrh_a0 = self._map_species_factors_to_k(x0, nominal_values, chemistry)
            for r_id in values_ids:
                chemistry.set_adhoc_reactions_attributes('arrh_a', r_id, float(arrh_a0[r_id]))
        elif traj_type == 'reactions':
            nominal_values = chemistry.get_reactions_arrh_a(si_units=False).copy()  # pd.Series with r_ids index
            values_ids = np.array(nominal_values.index)
            factors_ids = values_ids
            arrh_a0 = self._map_reactions_factors_to_k(x0, nominal_values)
            for r_id in values_ids:
                chemistry.set_adhoc_reactions_attributes('arrh_a', r_id, float(arrh_a0[r_id]))
        elif traj_type == 'return_coefficients':
            ret_matrix: pd.DataFrame = chemistry.get_return_matrix()
            if self.debug:
                # initialise the trajectory data DataFrame:
                for stuck_sp_id in ret_matrix.columns:
                    if chemistry.get_species_stick_coef()[stuck_sp_id] > 0:
                        for ret_sp_id in ret_matrix.index:
                            nominal_value = ret_matrix.at[ret_sp_id, stuck_sp_id]
                            # columns are labels for all non-zero return coefficients
                            if nominal_value > 0 or ret_sp_id == stuck_sp_id:
                                col_label = f'r_{ret_sp_id}_{stuck_sp_id}'
                                traj_data.at['stuck species', col_label] = chemistry.get_rp(stuck_sp_id).get_name()
                                traj_data.at['returned species', col_label] = chemistry.get_rp(ret_sp_id).get_name()
                                traj_data.at['nominal values', col_label] = nominal_value
            # just a sanity check: the total number of returned species per stuck species needs to be conserved!
            original_total_returned = ret_matrix.sum(axis=0).values.copy()
            factors_ids = [f'r({s}>{r})' for s, r in ret_species]
            # map all the morris factors to return coefficients:
            nominal_values = []
            values_ids = None  # not needed here, just to silence the "referenced before assignment" warning
            for (stuck_sp, ret_sp), x_i, rel_range in zip(ret_species, x0, rel_ranges):
                stuck_rp: RP = chemistry.get_rp(stuck_sp)
                ret_rp: RP = chemistry.get_rp(ret_sp)
                nominal_ret_coef: float = ret_matrix.at[ret_rp.id, stuck_rp.id]
                nominal_values.append(nominal_ret_coef)
                self._map_a_factor_to_ret_coef(x_i, stuck_sp, ret_sp, nominal_ret_coef, chemistry, rel_range)
            # assert that the total number of returned species is preserved for each species reaching the surface:
            new_ret_matrix: pd.DataFrame = chemistry.get_return_matrix()
            new_total_returned = new_ret_matrix.sum(axis=0).values.copy()
            assert np.isclose(original_total_returned, new_total_returned, rtol=1e-10).all(), \
                f'\n{original_total_returned}\n!=\n{new_total_returned}'
            if self.debug:
                # log in all the morris factors:
                for (stuck_sp, ret_sp), x_i in zip(ret_species, x0):
                    stuck_rp: RP = chemistry.get_rp(stuck_sp)
                    ret_rp: RP = chemistry.get_rp(ret_sp)
                    col_label = f'r_{ret_rp.id}_{stuck_rp.id}'
                    traj_data.at['base X', col_label] = x_i
                # log in all the new return coefficients values:
                row_label = 'base r'
                # noinspection DuplicatedCode
                for stuck_sp_id in new_ret_matrix.columns:
                    if chemistry.get_species_stick_coef()[stuck_sp_id] > 0:
                        for ret_sp_id in new_ret_matrix.index:
                            new_value = new_ret_matrix.at[ret_sp_id, stuck_sp_id]
                            if new_value > 0 or ret_sp_id == stuck_sp_id:
                                col_label = f'r_{ret_sp_id}_{stuck_sp_id}'
                                assert col_label in traj_data.columns
                                traj_data.at[row_label, col_label] = new_value
        elif traj_type == 'model_params':
            raise NotImplementedError
        else:
            raise ValueError('Trajectory type {} not recognised!'.format(traj_type))

        factors_vectors = self._trajectory(seed_x=x0)  # generator of all the vectors in the trajectory

        if verbosity >= 2:
            print('{}\t..::: EVALUATING MODEL WITH INITIAL FACTORS :::..'.format(time.strftime('%H:%M:%S')))
        run_id = '{:0{}d}_initial_run'.format(0, len(str(factors_dimension)))
        model.run(
            verbose=verbosity >= 3, log_results=full_log, log_to=self.logs_traj_dir,
            run_id=run_id
        )
        if not full_log:
            self._log_solution(model.get_solution(), run_id, self.logs_traj_dir, compress=compress_solution)

        meta_dict['base_run_id'] = run_id

        # run the trajectory:
        traj_run_ids = []  # for the meta dump
        traj_deltas = []  # for the meta dump
        traj_factors_ids = []  # for the meta dump
        for j, (i, delta, x) in enumerate(factors_vectors):
            factor_id = factors_ids[i]  # which factor x[i] belongs to? (x[i] is the one being changed in this step)
            if traj_type == 'species':
                arrh_a = self._map_species_factors_to_k(x, nominal_values, chemistry)
                for r_id in values_ids:
                    chemistry.set_adhoc_reactions_attributes('arrh_a', r_id, float(arrh_a[r_id]))
            elif traj_type == 'reactions':
                arrh_a = self._map_reactions_factors_to_k(x, nominal_values)
                for r_id in values_ids:
                    chemistry.set_adhoc_reactions_attributes('arrh_a', r_id, float(arrh_a[r_id]))
            elif traj_type == 'return_coefficients':
                x_i: float = x[i]
                nominal_ret_coef: float = nominal_values[i]
                stuck_sp, ret_sp = ret_species[i]
                stuck_rp: RP = chemistry.get_rp(stuck_sp)
                total_returned: float = sum(chemistry.get_species_ret_coefs()[stuck_rp.id].values())
                rel_range = rel_ranges[i]
                self._map_a_factor_to_ret_coef(x_i, stuck_sp, ret_sp, nominal_ret_coef, chemistry, rel_range)
                assert np.isclose(total_returned, sum(chemistry.get_species_ret_coefs()[stuck_rp.id].values()),
                                  rtol=1e-10)
                if self.debug:
                    new_ret_matrix: pd.DataFrame = chemistry.get_return_matrix()
                    # log in all the morris factors:
                    row_label = f'X (step {j+1})'
                    for (stuck_sp, ret_sp), x_i in zip(ret_species, x):
                        stuck_rp: RP = chemistry.get_rp(stuck_sp)
                        ret_rp: RP = chemistry.get_rp(ret_sp)
                        col_label = f'r_{ret_rp.id}_{stuck_rp.id}'
                        traj_data.at[row_label, col_label] = x_i
                    # log in all the return coefficients values:
                    row_label = f'r (step {j+1})'
                    # noinspection DuplicatedCode
                    for stuck_sp_id in new_ret_matrix.columns:
                        if chemistry.get_species_stick_coef()[stuck_sp_id] > 0:
                            for ret_sp_id in new_ret_matrix.index:
                                new_value = new_ret_matrix.at[ret_sp_id, stuck_sp_id]
                                if new_value > 0 or ret_sp_id == stuck_sp_id:
                                    col_label = f'r_{ret_sp_id}_{stuck_sp_id}'
                                    assert col_label in traj_data.columns
                                    traj_data.at[row_label, col_label] = new_value
            elif traj_type == 'model_params':
                raise NotImplementedError
            else:
                raise ValueError('Trajectory type {} not recognised!'.format(traj_type))

            if verbosity >= 2:
                print('{}\t..::: EVALUATING ELEMENTARY EFFECT OF {} ({}/{}) :::..'.format(
                    time.strftime('%H:%M:%S'), factor_id, j+1, factors_dimension))
            run_id = '{:0{}d}_elem_effect_of_{}'.format(j+1, len(str(factors_dimension)), factor_id)
            model.run(
                verbose=verbosity >= 3, log_results=full_log, log_to=self.logs_traj_dir,
                run_id=run_id
            )
            if not full_log:
                self._log_solution(model.get_solution(), run_id, self.logs_traj_dir, compress=compress_solution)

            traj_run_ids.append(run_id)
            traj_deltas.append(float(delta))
            logged_factor_id = str(factor_id)
            if logged_factor_id.isnumeric():
                logged_factor_id = int(logged_factor_id)  # assumes that it's either non-numeric str (name) or int (id)
            traj_factors_ids.append(logged_factor_id)

        # dump the trajectory meta data:
        meta_dict['run_ids'] = traj_run_ids
        meta_dict['deltas'] = traj_deltas
        meta_dict['factors_ids'] = traj_factors_ids
        with open(os.path.join(self.logs_traj_dir, 'meta.yaml'), 'w') as stream:
            yaml.dump(meta_dict, stream)
        # dump the morris run meta data:
        self._dump_meta(self.meta_dir, traj_type=traj_type)

        if verbosity >= 1:
            msg = ' MORRIS RUN "{}": TRAJECTORY "{}" FINISHED'.format(
                morris_run_id, os.path.split(self.logs_traj_dir)[-1])
            print('{:.>100}'.format(msg))

        if self.debug:
            # get rid of all the columns where nothing is happening (no changes, not mapped from factors):
            cols_to_drop = []
            for col in traj_data.columns:
                if len(set(traj_data[col].iloc[4:len(traj_data[col])+1:2])) == 1:
                    if traj_data[col].iloc[3:len(traj_data[col])+1:2].isnull().all():
                        cols_to_drop.append(col)
            traj_data = traj_data.drop(columns=cols_to_drop)
            # get rid of values which were not changed in each run:
            for col in traj_data.columns:
                for i in range(1, 2*factors_dimension + 1, 2):
                    if traj_data[col].iloc[-i] == traj_data[col].iloc[-i-2]:
                        traj_data[col].iloc[-i] = np.nan
                for i in range(2, 2*factors_dimension + 1, 2):
                    if not np.isnan(traj_data[col].iloc[-i]):
                        if traj_data[col].iloc[-i] == traj_data[col].iloc[-i-2]:
                            traj_data[col].iloc[-i] = np.nan
            # nicer printout:
            traj_data = traj_data.fillna('')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print(traj_data)

    def run_morris(self, morris_run_id, traj_type, num_trajs, num_cores=None, **kwargs):
        """This method runs required number of morris trajectories (either species, reactions or model_params type)
        and save all under the morris_run_id. The number of trajectories are divided between num_cores cores and
        run in parallel. This method calls the methods running single trajectories all with the default arguments.
        Non-default arguments to self.run_trajectory_* can be passed in **kwargs.
        :param morris_run_id: (str) identifier of this Morris Run (Morris Analysis). Also the name of a folder
                              compiling all the morris run results directly in the morris_results_dir.
        :param traj_type: (str) which type of morris trajectories to run. Must be from
                          {'reactions', 'species', 'return_coefficients', 'model_params'}
        :param num_trajs: (int) number of trajectories to run in this morris analysis
        :param num_cores: (int) defaults to number of cores on PC minus 1.
        :param kwargs: see self.run_trajectory docs...
        :return: None
        """
        # some necessary validity checks:
        if 'traj_id' in kwargs:
            raise ValueError('Cannot specify traj_id for parallel batch of trajectories!')
        if traj_type == 'return_coefficients':
            if 'ret_species' not in kwargs or 'rel_ranges' not in kwargs:
                raise ValueError('"ret_species" and "rel_ranges" keyword arguments need to be passed (to be forwarded '
                                 'to the self.run_trajectory method)!')

        # prepare the most basic directory structure:
        morris_results_dir = None
        if 'morris_results_dir' in kwargs:
            morris_results_dir = kwargs['morris_results_dir']
        self._prepare_logging(morris_run_id, morris_results_dir=morris_results_dir)  # this populates self.trajs_dir
        # build the queue of traj_ids to be run:
        traj_ids = []
        i = 1  # start index
        while len(traj_ids) < num_trajs:
            traj_id = self.traj_id.format(i)
            if not os.path.isfile(os.path.join(self.logs_dir, traj_id, 'meta.yaml')):
                traj_ids.append(traj_id)  # if not file already existing, append to the queue
            i += 1  # increase index
        # build the queue of the keyword arguments for the run_trajectory method to be run num_trajs times in parallel
        kwds_q = []
        for traj_id in traj_ids:
            kwds = {'traj_id': traj_id, 'verbosity': 1}
            kwds.update(kwargs)
            kwds_q.append(kwds)
        # create thee pool of workers:
        if num_cores is None:
            num_cores = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(num_cores)  # pool of workers, by default using all but one PC's CPUs

        for kwds in kwds_q:  # run all the trajectories in the pool.
            pool.apply_async(self.run_trajectory, args=(morris_run_id, traj_type), kwds=kwds)

        pool.close()
        pool.join()

    # ********************************* POST-PROCESSING TRAJECTORIES ************************************************* #

    @staticmethod
    def evaluate_trajectory(traj_dir, times):
        """Extracts the elementary effects and mean outputs for the passed model times and for previously run and
        logged morris trajectory.
        :param traj_dir: (str) full path to the subdirectory with all the runs for a single morris trajectory.
        :param times: (array) of float times - must all be inside the solution time range - asserted!
        :return: (namedtuple) with the following structure:
                 times: tuple of the same float times as passed
                 elementary_effects: tuple of pd.DataFrame with elementary effects for each time. Each dataframe
                                     has columns of model outputs and rows of factors ids. These are species names
                                     for species morris run, reaction ids for reactions.
                 mean_outputs: tuple of pd.Series with model outputs as index. These are mean model outputs across all
                               the model evaluations for this morris trajectory.
        """
        rp = ResultsParser()
        with open(os.path.join(traj_dir, 'meta.yaml')) as stream:
            meta_dict = yaml.load(stream, Loader=yaml.FullLoader)  # extract run_ids and deltas for all the runs
        base_run_id = meta_dict['base_run_id']
        run_ids, factors_ids = meta_dict['run_ids'], meta_dict['factors_ids']
        deltas = np.array(meta_dict['deltas'])

        base_sol = rp.fetch_solution(base_run_id, results_dir=traj_dir)  # 00_base solution
        base_frames = rp.get_results_frames(base_sol, times)  # model results only at times
        traj_frames = [base_frames.values]
        for run_id in run_ids:
            sol = rp.fetch_solution(run_id, results_dir=traj_dir)
            frames = rp.get_results_frames(sol, times)
            traj_frames.append(frames.values)
        # all the relevant results from all the runs at the times in one super-array
        traj_frames = np.moveaxis(np.array(traj_frames), 1, 0)  # shape (#times, #runs, #outputs)

        elementary_effects_times = (traj_frames[:, 1:, :] - traj_frames[:, :-1, :]) / deltas[np.newaxis, :, np.newaxis]
        # shape (#times, #factors, #outputs) - processed elementary effects for this trajectory in bare np.array
        elementary_effects_list = []  # this will go into returned namedtuple
        # mean outputs across all the runs in the trajectory and for all the times in bare np.array
        mean_outputs_times = traj_frames.mean(axis=1)  # shape (#times, #outputs) - mean values across all the runs
        mean_outputs_list = []  # this will go into returned namedtuple

        TrajectoryStats = namedtuple('TrajectoryStats', 'times elementary_effects mean_outputs')  # result named tuple
        for elementary_effects in elementary_effects_times:
            # make it into annotated DataFrame
            elem_effects_df = pd.DataFrame(elementary_effects, columns=base_sol.columns, index=factors_ids)
            elementary_effects_list.append(elem_effects_df)  # this will go into returned namedtuple
        for mean_outputs in mean_outputs_times:
            # make it into annotated Series
            mean_outputs_series = pd.Series(mean_outputs, index=base_sol.columns)
            mean_outputs_list.append(mean_outputs_series)  # this will go into returned namedtuple
        trajectory_stats = TrajectoryStats(tuple(times), tuple(elementary_effects_list), tuple(mean_outputs_list))
        return trajectory_stats

    @staticmethod
    def get_morris_stats(morris_run_id=None, logs_dir=None, times=None):
        """Method to calculate the statistics from all the trajectories previously run under the passed
        morris_run_id. Alternatively, one can pass explicitly the directory, where all the trajectories .csv results
        files reside. Statistics calculated are:
        - mean global model outputs (across all steps in all trajectories)
        - mean elementary effects (across all the trajectories) - absolute values
        - std dev of elementary effects (across all the trajectories) - absolute values
        - mean elementary effects relative to the mean model outputs.
        - std dev of elementary effects relative to the mean model outputs.
        :param morris_run_id: (str) only if trajs_dir not passed (in which case the morris_run_dir is identified
                              with default directories and names in the workspace dir.)
        :param logs_dir: (path) only if morris_run_id not passed. Full path towards the directory with all the
                         trajectories logs
        :param times: (array) of float times - must all be inside the solution time range - asserted!
        :return: (namedtuple) object with various morris method statistics and metrics. The attributes are:
                 mean_outputs: tuple[pd.Series] with mean model outputs across all the runs in all the trajectories in
                               this morris run, one for each imp_time
                 mu: tuple[pd.DataFrame] with absolute means of elementary effects across all the trajectories,
                     one for each imp_time
                 sigma: tuple[pd.DataFrame] with absolute standard deviations of elementary effects across all the
                        trajectories, one for each imp_time
                 mu_rel: tuple[pd.DataFrame] with mu relative to the mean outputs, one for each imp_time
                 sigma_rel: tuple[pd.DataFrame] with sigma relative to the mean outputs, one for each imp_time
        """
        # input validity:
        if (morris_run_id is None and logs_dir is None) or (morris_run_id is not None and logs_dir is not None):
            raise ValueError('Exactly one of morris_run_id or trajs_dir arguments must be passed!')
        if times is None:
            raise ValueError('Times array needs to be specified!')
        if logs_dir is None:
            morris_results_dir = Config().get_morris_results_dir()  # Parent results directory
            logs_dir = os.path.join(morris_results_dir, str(morris_run_id), MorrisMethod.logs_dir_name)

        # full paths to all the trajectory directories (the ones containing run logs for all the trajectory runs)
        traj_dirs_paths = list(
            glob.glob(os.path.join(logs_dir, '*'))
        )
        traj_dirs_paths.sort()

        trajectories_stats = []  # the TrajectoryStats object for each of the found trajectories
        for traj_dir in traj_dirs_paths:
            trajectories_stats.append(MorrisMethod.evaluate_trajectory(traj_dir, times))

        # prepare the resulting named tuple to be returned:
        MorrisStats = namedtuple('MorrisStats', 'times mean_outputs mu sigma mu_rel sigma_rel')
        # the following will be saved as attributes to the MorrisStats instance
        ms_times = times
        ms_mean_outputs = []
        ms_mu = []
        ms_sigma = []
        ms_mu_rel = []
        ms_sigma_rel = []

        # mean outputs across all the trajectories for each time:
        for i in range(len(times)):
            mean_outputs_single_time = []
            for ts in trajectories_stats:
                mean_outputs_single_time.append(ts.mean_outputs[i])
            mean_outputs_single_time_df = pd.DataFrame(mean_outputs_single_time)
            mean_outputs_single_time_mean = mean_outputs_single_time_df.mean(axis=0)
            ms_mean_outputs.append(mean_outputs_single_time_mean)

        # again for each time:
        for i in range(len(times)):
            # elementary effects mean (mu) and std dev (sigma) for all the trajectories
            elem_effects_dataframes = [ts.elementary_effects[i] for ts in trajectories_stats]
            elem_effects_concat = pd.concat(elem_effects_dataframes)
            mu_single_time = elem_effects_concat.groupby(elem_effects_concat.index).mean()
            sigma_single_time = elem_effects_concat.groupby(elem_effects_concat.index).std()
            # add to the resulting arrays:
            ms_mu.append(mu_single_time)
            ms_sigma.append(sigma_single_time)

            # relative mu and sigma (normalised to the mean output values...)
            mu_rel_single_time = mu_single_time/ms_mean_outputs[i]
            sigma_rel_single_time = sigma_single_time/ms_mean_outputs[i]
            # add to the resulting arrays:
            ms_mu_rel.append(mu_rel_single_time)
            ms_sigma_rel.append(sigma_rel_single_time)

        morris_stats = MorrisStats(
            times=tuple(ms_times), mean_outputs=tuple(ms_mean_outputs),
            mu=tuple(ms_mu), sigma=tuple(ms_sigma), mu_rel=tuple(ms_mu_rel), sigma_rel=tuple(ms_sigma_rel)
        )

        return morris_stats

    @staticmethod
    def get_ranking(important_outputs, important_times, morris_run_id=None, logs_dir=None, method='max'):
        """Method to create a ranking of parameters corresponding to the morris factors (depending on the morris
        run types, these might be species, reactions or model parameters), where each parameter scores according how
        important it is for the passed outputs of interest
        :param important_outputs: (iterable) of str - global model outputs. See GlobalModel.get_final_solution() docs
        :param important_times: (iterable) of float - times where to evaluate the Morris Method sum stats from.
        :param morris_run_id: (str) identification of the morris run. Only if trajs_dir not passed
        :param logs_dir: (str) explicit full path to the dir with morris trajectories. Only if morris_run_id not passed
        :param method: (str) from {'mean', 'max'}. If mean, scores are mean values of mu_rel and sigma_rel over all
                       important times and all important outputs. If 'max', maximal mu_rel and sigma_rel over all
                       important times and outputs are considered. For both methods, mu and sigma are added together.
        :return: pd.Series - keys of morris run parameters (corresponding to factors) - these are species names or
                             reactions ids
                           - values of arbitrary scores - higher the score, higher the significance of the parameter
                             (factor) in key towards the specified global model outputs of interest.
                           The Series is sorted starting with least significant parameters.
        """
        morris_stats = \
            MorrisMethod.get_morris_stats(morris_run_id=morris_run_id, logs_dir=logs_dir, times=important_times)
        if not set(important_outputs).issubset(set(morris_stats.mean_outputs[0].index)):
            raise ValueError('Some of the passed outputs of interest are not among the model outputs!')

        if method == 'mean':
            # Scores are simply mean values of mu_rel and sigma_rel over all important times both added together and
            # also added for all important outputs - effectively scaled means over important outputs and important
            # times.
            # noinspection PyTypeChecker
            scores = pd.Series(index=morris_stats.mu[0].index, dtype='float64').fillna(0)  # seed the scores Series
            for i in range(len(important_times)):
                scores += abs(morris_stats.mu_rel[i])[important_outputs].sum(axis=1)
                scores += morris_stats.sigma_rel[i][important_outputs].sum(axis=1)
            # normalise it to the number of sampling points an number of important outputs:
            scores /= len(important_times)
            scores /= len(important_outputs)
        elif method == 'max':
            # Scores are maximal values of mu_rel and sigma_rel over all important times and all important outputs - in
            # contrast with mean values.
            scores = pd.DataFrame(columns=morris_stats.mu[0].index)  # seed to scores DF (lines for different times)
            for i in range(len(important_times)):
                scores_for_time = \
                    abs(morris_stats.mu_rel[i])[important_outputs].max(axis=1) + \
                    morris_stats.sigma_rel[i][important_outputs].max(axis=1)
                scores.loc[i, :] = scores_for_time
                # max mu and sigma for each time (on each line)
            scores = scores.max(axis=0)  # only leave maximums across all the important times...
        else:
            raise ValueError('The method argument must be either "mean" or "max"!')

        return scores.sort_values(ascending=True)
