
# import os
import yaml
from pathlib import Path

from pygmo_fwork.exceptions import ConfigError, GlobalModelAttributeError


class Config(dict):
    """
    This is a custom dict loading the configuration for the PyGMo_Fwork package. It also takes care of building
    the whole workspace directory structure. The folders in the workspace directory are only created upon request,
    it they are not present already. These requests are represented by the implemented self.get_xxx() methods.
    The keys of the Config directory itself are only the ones saved in the yaml config file.
    """
    accepted_backends = (
        'pygmol',
        'pygkin'
    )

    def __init__(self, config_file_path=Path.home().joinpath('.pygmofwork')):
        """Instantiates the Config class. Takes a path to the config file, which is supposed to be a yaml dump file.
        After loading, if should yield a following dict:
        {'workspace': 'path/to/workspace', 'globalkin_bin': 'path/to/gk_binary_file'}
        ConfigError exception will be raised if the file does not exist or if it cannot be loaded by yaml lib.
        :param config_file_path: (str) path to the config file. Defaults to '~/.pygmofwork'
        """
        # some error raising:
        if not Path(config_file_path).is_file():
            raise ConfigError('Config file not found at {}!'.format(config_file_path))
        with open(config_file_path, 'r') as stream:
            try:  # try yaml loading the config into a dict
                config_dict = dict(yaml.load(stream, yaml.FullLoader))
            except (TypeError, ValueError) as _:
                raise ConfigError('Failed to parse the config file! Needs to follow the prescribed format & structure!')

        # instantiate with the parsed dictionary
        super().__init__(config_dict)

    def __getitem__(self, key):
        """Custom getitem method - only amended by raising the custom ConfigError exception whenever getting
        non-existing key. Except that, the same as parent dict.
        """
        if key not in self.keys():
            raise ConfigError('{} not found in the Config keys!'.format(key))
        return super().__getitem__(key)

    # ******************************** DEFAULT PATHS ***************************************************************** #

    def _get_results_root(self):
        """Query method for the 'config-default' results root directory.
        :return: (Path) path to the results root directory.
        """
        results_root = Path(self['workspace']) / 'results'
        return results_root

    def get_results_dir(self, backend_id):
        """Query method for the 'config-default' results directory for a particular backend model.
        The structure of the workspace directory is hardwired here in the Config class and the results directories
        will be built inside the workspace dir at the time of the first query for them.
        :param backend_id: (str) whatever string identifier for the backend model. Use pygmol or pygkin!
        :return: (Path) path to the results directory for the backend model.
        """
        backend_id = backend_id.lower()
        if backend_id not in self.accepted_backends:
            raise GlobalModelAttributeError(
                'Unsupported global model backend! Only {} allowed'.format(self.accepted_backends))
        results_dir = self._get_results_root() / backend_id
        # build results directory (and parents) if not existing
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir  # return the path to the results directory

    def get_globalkin_stage_dir(self):
        """Query method for the 'config-default' global_kin staging directory.
        The structure of the workspace directory is hardwired here in the Config class and the directories
        will be built at the time of the first query.
        :return: (Path) path to the config-default global_kin staging directory.
        """
        # want to have a valid global_kin bin file before I'm returning the staging directory!
        if not Path(self['globalkin_bin']).is_file():
            raise ConfigError('Invalid path to the Global_Kin binary file: {}!'.format(self['globalkin_bin']))
        globalkin_stage_dir = Path(self['workspace']) / 'globalkin_stage'
        # make the staging directory (and parents) if not existing
        globalkin_stage_dir.mkdir(parents=True, exist_ok=True)
        return globalkin_stage_dir  # return the staging directory path.

    def get_morris_results_dir(self):
        """Query method for the 'config-default' results directory for morris method runs' results.
        The structure of the workspace directory is hardwired here in the Config class and the results directories
        will be built inside the workspace dir at the time of the first query for them.
        :return: (Path) path to the morris method results directory.
        """
        morris_results_dir = self._get_results_root() / 'morris_sensitivity'
        # make the morris results directory (and parents) if not existing
        morris_results_dir.mkdir(parents=True, exist_ok=True)
        return morris_results_dir

    def get_reduction_results_dir(self):
        """Query method for the 'config-default' results directory for reduction runs' results.
        The structure of the workspace directory is hardwired here in the Config class and the results directories
        will be built inside the workspace dir at the time of the first query for them.
        :return: (Path) path to the reduction results directory.
        """
        reduction_results_dir = self._get_results_root() / 'reduction'
        # make the reduction results directory (and parents) if not existing
        reduction_results_dir.mkdir(parents=True, exist_ok=True)
        return reduction_results_dir

    # ******************************** DEFAULT NAMES ***************************************************************** #

    # noinspection PyUnusedLocal
    @staticmethod
    def default_log_name(log, **kwargs):
        """Returns a default log file name for each logged file.
        :param kwargs: other possible parameters, added for future compatibility
        :param log: (str) what it is it's logged - e.g. 'solution', 'rates', 'model_attributes', 'chemistry_attributes'
        :return: (str) default name of a log file
        """
        if log in {'solution', 'rates', 'wall_fluxes'}:
            return '{}.csv'.format(log)
        elif log in {'model_attributes', 'chemistry_attributes', 'solver_log'}:
            return '{}.yaml'.format(log)
        else:
            raise AssertionError('Unrecognised log identifier!')
