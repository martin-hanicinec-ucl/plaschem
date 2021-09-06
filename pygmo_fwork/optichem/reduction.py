import os
from typing import List, Union, Callable

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from plaschem.chemistry import ChemistryDisableError
from pygmo_fwork.config import Config
from pygmo_fwork.global_model import GlobalModel
from pygmo_fwork.optichem.errors import SolutionsDiff
from pygmo_fwork.optichem.exceptions import ReductionInitError, ReductionRunError
from pygmo_fwork.pygmol.exceptions import ModelSolutionError
from pygmo_fwork.results import ResultsParser


class ReductionIterator(object):
    """A class for running an iterative reduction method. A supplied GlobalModel instance is run iteratively while
    removing species after species from a supplied species ranking Series. The loop terminated as soon as a certain
    number of successive runs result in larger reduction error, than allowed.
    """
    base_run_id = '{:02d}_full_chemistry'.format(0)
    model_run_id = '{:02d}_disabled_{}'  # to be formatted by index and species_name for each removed species.
    reduction_history_filename = 'reduction_history.csv'
    meta_filename = 'meta.yaml'

    def __init__(self,
                 model: GlobalModel, reduction_run_id: str, important_outputs: List[str], important_times: List[float],
                 max_error: float, reduction_results_dir: str = None, verbose_level: int = 2):
        """Initializer of the reduction iterator class.
        WARNING: the model instance passed should NOT BE TOUCHED after the ReductionIterator is instantiated!
        :param model: a GlobalModel instance - this will be used to run all the calculations.
        :param reduction_run_id: sting identifier of this reduction run. This will be a subdirectory in the
                                 reduction_results_dir and all the individual iteration runs of the GlobalModel instance
                                 will be saved in this subdir.
        :param important_outputs: List of important outputs (str names of species).
        :param important_times: List of important times (just floats)
        :param max_error: The maximal reduction error in [%].
        :param reduction_results_dir: The path where the reduction_run_id subdirectory should be saved. If not passed,
                                      it's defaulted from the Config class.
        :param verbose_level: (int) level of verbosity of the reduction run. 0 - nothing gets printed, 2: even the
                              results of all the iterations are printed out.
        """
        # WARNING: It's crucial that the model instance IS NOT TOUCHED manually from instantiation of the
        # ReductionIterator until THE LAST remove_one_species_and_run call!
        self.model = model  # this is a GlobalModel instance, with it's chemistry, model params and backend!
        self.reduction_run_id = str(reduction_run_id)
        self.important_outputs = [str(io) for io in important_outputs]  # want it explicitly in list of strings
        self.important_times = [float(round(it, 10)) for it in important_times]  # want it explicitly in list of floats
        self.max_error = max_error
        self.meta = {
            'important_outputs': self.important_outputs, 'important_times': self.important_times,
            'max_error': self.max_error
        }
        if reduction_results_dir is None:
            reduction_results_dir = Config().get_reduction_results_dir()  # Parent results directory
        self.reduction_results_dir = reduction_results_dir
        self.verbose_level = verbose_level
        # reduction results dir is either default one from config, or custom full path, which NEEDS TO EXIST!
        self.reduction_run_dir = os.path.join(self.reduction_results_dir, self.reduction_run_id)
        # reduction run dir is the one storing all the model runs in this reduction run id.
        self.make_dir_structure_if_clear()

        # keeping a ledger of species/reactions removing history
        self.all_eligible_species = \
            set(self.model.model.chemistry.get_species_name(special=False, disabled=False, protected=False))
        # exclude the important species from eligible:
        self.all_eligible_species = set(sp for sp in self.all_eligible_species if str(sp) not in self.important_outputs)
        if not len(self.all_eligible_species):
            raise ReductionInitError('The chemistry of the passed model does not have any species that can be removed!')
        self.all_species = set(self.model.model.chemistry.get_species_name())
        assert len(self.all_species) == self.model.model.chemistry.num_species()
        self.species_remove_attempted = set([])
        self.left_eligible_species = self.all_eligible_species.copy()
        self.reduction_history = pd.DataFrame(
            columns=[
                'run_id', 'species', 'ranking_score', 'rm_success', 'rm_additional', 'total_removed', 'total_left',
                'error_base_max', 'error_base_rms', 'error_last_max', 'error_last_rms'
            ],
        )
        # index of the dataframe will go from 0 (for base case) to number of base species - each for each remove run.
        # the removed column will signal if the species was successfully removed in which run_id...

        # keeping the solution diff instance to log the errors of reduced models compared to base and to last
        # successful run:
        self.sol_dif_base = None
        self.sol_dif_last = None

        # logging settings:
        self.compress_logs = True
        self.log_solution = True
        self.log_rates = False
        self.log_wall_fluxes = False
        # the ReductionIterator only NEEDS chemistry_attributes.yaml for backtracking

    def can_be_removed(self, species: str) -> bool:
        """Method deciding if a passed species is eligible for removal. some species cannot be removed, such as
        the passed important species, feed gas species, or last positive ion.
        :param species: (str) species name
        :return: bool if species can be removed
        """
        return species in self.left_eligible_species

    def make_dir_structure_if_clear(self) -> None:
        """This method builds the directory structure for logging all the data for this reduction run.
        :return: None
        """
        if not os.path.isdir(self.reduction_results_dir):
            raise ReductionInitError('Results parent directory {} does not exist!'.format(self.reduction_results_dir))
        if not os.path.isdir(self.reduction_run_dir):
            os.mkdir(self.reduction_run_dir)
            # log the meta data (important outputs and times)
            with open(os.path.join(self.reduction_run_dir, self.meta_filename), 'w') as stream:
                yaml.dump(self.meta, stream)
        # it's allowed that the run dir already exists, but it needs to be empty!
        elif len(os.listdir(self.reduction_run_dir)):
            raise ReductionInitError('{} already exists!'.format(self.reduction_run_dir))

    def _run_model(self, run_id: str) -> None:
        """This method simply runs the self.model instance with all the settings defined in the set of the instance
        attributes.
        :param run_id: (str) run id for this single model run iteration.
        :return: None
        """
        # run the self.model as it currently is and log under run_id with all the logging settings defined in __init__
        self.model.run(run_id=run_id, log_to=self.reduction_run_dir, verbose=(self.verbose_level >= 2))
        # selectively remove or compress certain logs, if needed to:
        for log_flag, file_name in zip(
                [self.log_solution, self.log_rates, self.log_wall_fluxes],
                ['solution.csv', 'rates.csv', 'wall_fluxes.csv']
        ):
            log_path = os.path.join(self.reduction_run_dir, run_id, file_name)
            if log_flag and self.compress_logs:
                # Compress the log. Faster than calling model.run with compress=True, if not saving all logs.
                log = pd.read_csv(log_path, index_col=0)
                log = ResultsParser.compress_solution(log, diff_threshold=0.05, min_timestep=0.001)
                log.to_csv(log_path)
            elif not log_flag:  # remove the log if logging turned off for this file...
                os.remove(log_path)

    def run_base_case(self) -> None:
        """Method running the base-case - the zeroth reduction iteration. All the reduction errors will be assessed
        against this base case. The run id of this iteration is defined by self.base_run_id
        :return: None
        """
        # some sanity checks:
        if self.reduction_history.shape[0] != 0:  # only can be run of the base case has not been...
            raise ReductionRunError('The base run needs to be run only once and before any species are removed!')
        assert not len(self.species_remove_attempted)
        assert self.sol_dif_base is None
        assert self.sol_dif_last is None

        # run the base case:
        run_id = self.base_run_id
        self._run_model(run_id)

        # build up the reduction history log and log it:
        self.reduction_history.loc[0, ['run_id', 'total_removed', 'total_left']] = [run_id, 0, len(self.all_species)]
        self.reduction_history.loc[0, ['error_base_max', 'error_base_rms']] = [0., 0.]
        self.reduction_history.loc[0, ['error_last_max', 'error_last_rms']] = [0., 0.]
        self.reduction_history.at[0, 'rm_success'] = True
        self.log()

        # prepare the solution diff instances for quantifying the errors:
        base_sol = self.model.get_solution()
        self.sol_dif_base = SolutionsDiff(base_sol)
        self.sol_dif_last = SolutionsDiff(base_sol)

    def remove_one_species_and_run(
            self, species: str, sp_ranking_score: float = np.nan, backtrack_if_over_max_error: bool = False):
        """The method will try to remove passed species from the chemistry and run the reduced model. The species
        ranking score is just for the logs. If the species cannot be disabled from the chemistry (either it cannot be
        disabled or the model does not successfully run after disabling), then it is
        automatically reinstalled (chemistry is restored to the last "working" state) and appropriate logs are entered.
        If the model run hangs, user can KeyboardInterrupt it and the method continues exactly as if the species could
        not be disabled.
        If the species is disabled and the reduced run yields higher max_error (for self.important_outputs and
        self.important_times), then it depends on the backtrack_if_over_max_error parameter. If True, the species is
        reentered to the chemistry (or rather the chemistry is restored to the last "working" state with errors
        lower than max_error). If the parameter is True, than nothing happens. In both cases, the same entries are
        entered into the log, showing success to remove and the induced error.

        :param species: (str) species to be removed
        :param sp_ranking_score: (float) ranking score of the species being removed - for logging purpose.
        :param backtrack_if_over_max_error: (bool) if this is True, the species will be re-added to the chemistry
                                            if the reduction error shoots above the limit after the species is removed.
        :return: line being logged to the reduction history
        """
        # some sanity checks:
        if self.reduction_history.shape[0] == 0:  # only can be run after the base run has been...
            raise ReductionRunError('The base run needs to be run first!')
        if species not in self.left_eligible_species:
            raise ReductionRunError('Species {} cannot be removed!'.format(species))
        assert species not in self.species_remove_attempted
        assert self.meta == yaml.load(
            open(os.path.join(self.reduction_run_dir, self.meta_filename)), Loader=yaml.FullLoader
        )
        self.species_remove_attempted.add(species)
        index = len(self.species_remove_attempted)
        assert index == list(self.reduction_history.index)[-1] + 1  # check that everything makes sense

        # if not successful (either cannot disable, or ModelSolutionError or if max_error defined and
        # backtrack_if_over_max_error is True), then the chemistry needs to be backtracked to the last "working" run id:
        restore_mask = self.reduction_history['rm_success'].copy()
        if backtrack_if_over_max_error:
            restore_mask &= self.reduction_history['error_base_max'] <= self.max_error / 100
        restore_run_id = self.reduction_history.loc[restore_mask, 'run_id'].dropna().iloc[-1]
        restore_chem_attrib_path = os.path.join(self.reduction_run_dir, restore_run_id, 'chemistry_attributes.yaml')

        # now do the removal and run the reduced model:
        run_id = self.model_run_id.format(index, species)
        # add the entry:
        self.reduction_history.loc[index, ['species', 'ranking_score']] = [species, sp_ranking_score]
        species_left_before_remove = set(self.model.model.chemistry.get_species_name())
        try:
            self.model.model.chemistry.disable(species=[species])
            removed_successfully = True
        except ChemistryDisableError:  # species could not even be disabled... will be skipped.
            removed_successfully = False
        if removed_successfully:
            try:
                try:
                    self._run_model(run_id)
                except KeyboardInterrupt:
                    raise ModelSolutionError('Model solution aborted by KeyboardInterrupt')
                self.left_eligible_species.remove(species)  # species definitely removed
            except ModelSolutionError:
                removed_successfully = False
                # restore the chemistry:
                self.model.model.chemistry.load_chemistry_attributes(restore_chem_attrib_path)
                if self.verbose_level >= 2:
                    print('PyGMol solution failed, restoring chemistry to {}\n'.format(restore_run_id))

        # irrespective of the success of removing or running the reduced model, log these:
        species_left = set(self.model.model.chemistry.get_species_name())
        total_left = len(species_left)
        total_removed = len(self.all_species) - total_left
        self.reduction_history.at[index, 'rm_success'] = removed_successfully
        self.reduction_history.loc[index, ['total_removed', 'total_left']] = [total_removed, total_left]

        if removed_successfully:
            # record all the data and update the previous solution diff instance.
            species_removed = species_left_before_remove - species_left
            assert species in species_removed, '{} not in {}'.format(species, species_removed)  # sanity check
            implicitly_removed = species_removed - {species}
            if len(implicitly_removed):  # one or more species were removed implicitly on the back of species:
                self.reduction_history.at[index, 'rm_additional'] = ', '.join(implicitly_removed)
                for sp in implicitly_removed:
                    self.left_eligible_species.remove(sp)

            self.reduction_history.at[index, 'run_id'] = run_id
            sol = self.model.get_solution()
            error_base_max = self.sol_dif_base.get_error_max(sol, self.important_outputs, self.important_times)
            error_base_rms = self.sol_dif_base.get_error_rms(sol, self.important_outputs, self.important_times)
            error_last_max = self.sol_dif_last.get_error_max(sol, self.important_outputs, self.important_times)
            error_last_rms = self.sol_dif_last.get_error_rms(sol, self.important_outputs, self.important_times)
            self.reduction_history.loc[index, ['error_base_max', 'error_base_rms']] = [error_base_max, error_base_rms]
            self.reduction_history.loc[index, ['error_last_max', 'error_last_rms']] = [error_last_max, error_last_rms]

            # if the error is higher than error_max, restore the chemistry:
            if backtrack_if_over_max_error and error_base_max > self.max_error / 100:
                # restore the chemistry:
                self.model.model.chemistry.load_chemistry_attributes(restore_chem_attrib_path)
                if self.verbose_level >= 2:
                    print('Maximum error exceeded, restoring chemistry to {}\n'.format(restore_run_id))
                # add all the removed species back to the set of eligible species
                # (since they may need to be removed implicitly later on with a different species...)
                self.left_eligible_species.add(species)
                self.left_eligible_species |= implicitly_removed
            else:
                # update the last successful solution:
                self.sol_dif_last = SolutionsDiff(sol)

        # log the reduction history:
        self.log()
        return self.reduction_history.loc[index].copy()

    def log(self) -> None:
        """Logs the reduction history dataframe held in self.reduction_history to the csv table in the reduction_run_id
        directory. The name of the logged file is controlled by self.reduction_history_filename
        """
        self.reduction_history.to_csv(
            os.path.join(self.reduction_run_dir, self.reduction_history_filename),
            header=True, index=True, index_label='run_index'
        )

    def run_reduction(
            self,
            ranking: Union[pd.Series, Callable[[GlobalModel, ], pd.Series]], max_unsuccess_streak: int = 0) -> None:
        """Method to run a species reduction routine.
        For the static reduction, it takes the ranking pd.Series, indexed by species names
        with values of species ranking scores.
        For the dynamic reduction, it takes callable, accepting a GlobalModel instance (which has been solved already!)
        and returning a species ranking pd.Series in the same form as used by the static reduction.
        The iterator than removes one species after another until the
        max_error is reached max_unsuccess_streak times in a row, which stops the reduction run.
        In the case of the Dynamic reduction - the species ranking is recalculated after each successful species
        elimination and only the first species from ranking is removed (attempted to) in the next iteration.
        If a species cannot be removed (would violate neutrality for example) or if the model does not solve
        successfully, the species is reinstalled, skipped and the reduction goes on with another species in the
        ranking. If a species removal induces error more than max_error, the species is reinstalled and skipped,
        but only up to max_unsuccess_streak times.
        :param ranking: two options here - reduction with a static ranking, or with a dynamic ranking (recalculated
                        after each species elimination).
                        - For a static ranking, ranking parameter needs to be
                          (pd.Series) or (float) rankings indexed by (str) species names
                        - For a dynamic ranking, the ranking parameter needs to be a callable, accepting a GlobalModel
                          instance (which has been solved!) as an attribute and returning a (pd.Series) or
                          (float) rankings indexed by (str) species names for all the species left in the
                          global_model.model.chemistry, such as ranking = func, func(self.model) -> pd.Series
        :param max_unsuccess_streak: (int) the reduction run is finished after n = max_unsuccess_streak species
                                     in a row induce error more than max_error.
        :return: None
        """
        # check if the chemistry does not have any disabled species or reactions!
        if len(self.model.model.chemistry.get_disabled_species()) or \
                len(self.model.model.chemistry.get_disabled_reactions()):
            raise AssertionError('Chemistry has some disabled reactions or species, cannot run the reduction!')

        # check if static or dynamic reduction method?
        dynamic = callable(ranking)

        if self.verbose_level >= 1:
            print()
            print('Running Reduction Run: {}'.format(self.reduction_run_id))
            print()

        # run the reduction routine:
        unsuccess_streak = 0
        self.run_base_case()

        if not dynamic:
            ranking_series = ranking.copy()
        else:
            ranking_series = ranking(self.model)

        while len(ranking_series):
            out_of_species = False
            # check if the ranking Series is sorted ascending order:
            assert list(ranking_series.values) == list(sorted(ranking_series.values)), \
                'Supplied ranking Series is not sorted!'
            # get the next species which should be attempted to remove
            sp = ranking_series.index[0]
            ranking_score = ranking_series.pop(sp)
            while sp in self.species_remove_attempted:
                if len(ranking_series):
                    sp = ranking_series.index[0]
                    ranking_score = ranking_series.pop(sp)
                else:
                    out_of_species = True
                    break
            if out_of_species:  # means there are no species not tried yet...
                break

            if self.can_be_removed(sp):
                reduction_run_results = self.remove_one_species_and_run(
                    sp, sp_ranking_score=ranking_score, backtrack_if_over_max_error=True
                )
                reduction_run_max_error = reduction_run_results['error_base_max']
                if not np.isnan(reduction_run_max_error) and reduction_run_max_error > self.max_error / 100:
                    unsuccess_streak += 1  # the chemistry was reset to the one before this run in remove_one_and_run()
                elif np.isnan(reduction_run_max_error):
                    pass  # only if the species could not have been disabled or the model solution faulted, in that
                    # case do not reset the unsuccess_streak counter!
                else:
                    unsuccess_streak = 0  # reset the streak if reduced run has low enough error.
                    # update the ranking if running the dynamic reduction:
                    if dynamic:
                        ranking_series = ranking(self.model)
                if unsuccess_streak > max_unsuccess_streak or not len(self.left_eligible_species):
                    break  # terminate the reduction as soon as I hit the limit appropriate number of times in a row...
        if self.verbose_level >= 1:
            print()
            print('Reduction Run {} Finished...'.format(self.reduction_run_id))
            print()
        self.model.model.chemistry.reset()  # reset the chemistry. This should only reset disabled spcs and reactions!

    def run_dynamic_reduction(self, ranking_func, max_unsuccess_streak=0):
        # ranking is dynamically recalculated after each species elimination ()
        raise NotImplementedError


# Analyses of the completed reduction runs
# TODO: this needs to be factored out to a separate class
def compare_reduction_runs(*reduction_runs_paths,
                           line_styles_success=None, line_styles_unsuccess=None, labels=None,
                           annotate=False, dead_ends=True, plot_endpoints=False, plot_max_error=True,
                           fig_size=None, xlim=None, ylim=None,
                           spcs_annotations=None, annotations_shift=None,
                           save_to=None):
    if line_styles_success is None:
        line_styles_success = 10 * ['b-o', 'g-o', 'r-o', 'c-o', 'm-o', 'y-o', 'k-o']
    if line_styles_unsuccess is None:
        line_styles_unsuccess = 10 * ['b:X', 'g:X', 'r:X', 'c:X', 'm:X', 'y:X', 'k:X']
    if labels is None:
        labels = [os.path.split(path)[-1] for path in reduction_runs_paths]
    if spcs_annotations is None:
        spcs_annotations = {}
    if annotations_shift is None:
        annotations_shift = [0.05, -0.45]

    for passed in line_styles_success, line_styles_unsuccess, labels:
        assert len(passed) >= len(reduction_runs_paths), 'inconsistency in number of attributes!'

    linewidth = 0.75

    history_frames = []
    max_allowed_error = None
    for path in reduction_runs_paths:
        rh = pd.read_csv(os.path.join(path, 'reduction_history.csv'), header=0, index_col=0)
        history_frames.append(rh)
        with open(os.path.join(os.path.join(path, 'meta.yaml')), 'r') as meta:
            delta = yaml.load(meta, Loader=yaml.FullLoader)['max_error']
            if max_allowed_error is None:
                max_allowed_error = delta
            else:
                assert max_allowed_error == delta, 'inconsistent maximal errors in the passed reduction runs!'

    def prettify_species_name(raw_name):
        if raw_name in spcs_annotations:
            return spcs_annotations[raw_name]
        name = raw_name
        for symbol in ['+', '-']:
            name = name.replace(symbol, '$^{%s}$' % symbol)
        for num in '1 2 3 4 5 6 7 8 9'.split():
            name = name.replace(num, '$_{%s}$' % num)
        return name

    def get_text_pos(point_pos):
        pos = np.array(point_pos) + np.array(annotations_shift)
        return pos

    fig, ax = plt.subplots(figsize=fig_size)
    max_eliminated = 0  # maximum number of eliminated species across reduction runs
    max_error = 0  # maximum error of plotted points across reduction runs

    # plot all the reduction runs together:
    for i, rh in enumerate(history_frames):

        # plot the successful iterations as a continual line:
        mask = rh.loc[:, 'rm_success'] & (rh.loc[:, 'error_base_max'] <= max_allowed_error / 100)
        errors = rh.loc[mask, 'error_base_max'] * 100
        if errors.iloc[-1] > max_error:
            max_error = errors.iloc[-1]
        num_eliminated = rh.loc[mask, 'total_removed']
        # plot endpoint as a vertical line with same formatting as dead-ends:
        if plot_endpoints:
            plt.plot(2 * [num_eliminated.iloc[-1]], [-1e10, 1e10], line_styles_unsuccess[i])
        if num_eliminated.iloc[-1] > max_eliminated:
            max_eliminated = num_eliminated.iloc[-1]
        sp_eliminated = []
        for _, row in rh.loc[mask, :].iterrows():
            # print(row['species'], row['rm_additional'])
            if pd.isna(row['species']):
                sp_eliminated.append('')
            elif pd.isna(row['rm_additional']):
                sp_eliminated.append(row['species'])
            else:
                sp_eliminated.append('{}, {}'.format(row['species'], row['rm_additional']))
        sp_eliminated = [prettify_species_name(sp) for sp in sp_eliminated]
        ax.plot(num_eliminated, errors, line_styles_success[i], label=labels[i], linewidth=linewidth)
        # points annotation
        if annotate:
            for j, point in enumerate(zip(num_eliminated, errors)):
                ax.annotate(sp_eliminated[j], get_text_pos(point), fontsize='x-small', color=line_styles_success[i][0])

        # plot the unsuccessful iterations as dead ends:
        if dead_ends:
            mask_unsuccess = rh.loc[:, 'rm_success'] & (rh.loc[:, 'error_base_max'] > max_allowed_error / 100)
            for unsucc_index in rh.index[mask_unsuccess]:
                if rh.at[unsucc_index, 'error_base_max'] * 100 > max_error:
                    max_error = rh.at[unsucc_index, 'error_base_max'] * 100
                if rh.at[unsucc_index, 'total_removed'] > max_eliminated:
                    max_eliminated = rh.at[unsucc_index, 'total_removed']
                prev_index = rh.index[mask][rh.index[mask] < unsucc_index][-1]
                # plot line
                ax.plot([rh.at[prev_index, 'total_removed'], rh.at[unsucc_index, 'total_removed']],
                        [rh.at[prev_index, 'error_base_max'] * 100, rh.at[unsucc_index, 'error_base_max'] * 100],
                        line_styles_unsuccess[i][:-1], linewidth=linewidth)
                # plot point
                ax.plot(rh.at[unsucc_index, 'total_removed'], rh.at[unsucc_index, 'error_base_max'] * 100,
                        line_styles_unsuccess[i])
                # points annotation
                sp_not_eliminated = rh.at[unsucc_index, 'species']
                point = (rh.at[unsucc_index, 'total_removed'], rh.at[unsucc_index, 'error_base_max'] * 100)
                if annotate:
                    ax.annotate(prettify_species_name(sp_not_eliminated), get_text_pos(point), fontsize='x-small',
                                color=line_styles_unsuccess[i][0])

    if xlim is None:
        xlim = (-0.5, max_eliminated + 0.5)
    if ylim is None:
        ylim = (-0.9, max(max_error, max_allowed_error) + 0.9)

    # plot the maximal reduced error:
    if plot_max_error:
        ax.plot(xlim, (max_allowed_error, max_allowed_error), 'r--', label='maximal allowed reduction error')

    # pimp up the plot
    ax.grid()
    ax.set_xlabel('Number of eliminated species')
    ax.set_ylabel('Reduction error $\\delta$ (%)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)
