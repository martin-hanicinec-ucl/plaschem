
import unittest
import os
import shutil

from pygmo_fwork.config import Config
from pygmo_fwork.exceptions import \
    ConfigError, GlobalModelAttributeError


class TestConfig(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))
    temp = os.path.join(context, '.temp')  # temp dir path
    config_path = os.path.join(temp, '.config')
    workspace_dir = os.path.join(temp, 'workspace')

    def make_temp_dir(self):
        self.remove_temp_dir()
        os.mkdir(self.temp)

    def remove_temp_dir(self):
        if os.path.isdir(self.temp):
            shutil.rmtree(self.temp)

    def build_dummy_config(self, empty=False, dummy_content=None):
        with open(self.config_path, 'w') as f:
            if empty:
                pass
            elif dummy_content is not None:
                f.write(dummy_content)
            else:
                # build a valid config file
                f.write('workspace: {}\n'.format(self.workspace_dir))
                f.write('globalkin_bin: {}\n'.format(os.path.join(self.temp, 'global_kin')))

    def test_init(self):
        with self.assertRaises(ConfigError):
            _ = Config('not/existing/path/.pygmofwork')

        self.make_temp_dir()
        self.build_dummy_config(empty=True)
        with self.assertRaises(ConfigError):
            _ = Config(self.config_path)
        invalid_content = '- key: blah blah'
        self.build_dummy_config(dummy_content=invalid_content)
        with self.assertRaises(ConfigError):
            _ = Config(self.config_path)
        # valid config file:
        self.build_dummy_config()
        self.assertFalse(os.path.isdir(self.workspace_dir))  # the workspace directory not existing yet
        config = Config(self.config_path)
        self.assertEqual(config['workspace'], os.path.join(self.temp, 'workspace'))
        self.assertFalse(os.path.isdir(self.workspace_dir))  # workspace still has not been built
        self.remove_temp_dir()

        with self.assertRaises(ConfigError):
            _ = config['unsupported']  # test the custom KeyError...

    def test_config_getters(self):
        self.make_temp_dir()
        self.build_dummy_config()
        config = Config(self.config_path)

        backend = 'unsupported'
        with self.assertRaises(GlobalModelAttributeError):
            _ = config.get_results_dir(backend_id=backend)  # unsupported backend id

        self.assertFalse(os.path.isdir(self.workspace_dir))  # the workspace directory not existing yet
        self.assertFalse(os.path.isdir(os.path.join(self.workspace_dir, 'results')))
        backend = 'pygmol'
        results_dir = config.get_results_dir(backend_id=backend)
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', backend)))
        self.assertEqual(os.path.join(self.workspace_dir, 'results', backend), str(results_dir))

        with self.assertRaises(ConfigError):
            _ = config.get_globalkin_stage_dir()  # illegal, since the global_kin binary non existing
        with open(os.path.join(self.temp, 'global_kin'), 'w'):
            pass  # build an empty global_kin file
        self.assertFalse(os.path.isdir(os.path.join(self.workspace_dir, 'globalkin_stage')))
        gk_stage = config.get_globalkin_stage_dir()  # should be legal now
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'globalkin_stage')))
        self.assertEqual(os.path.join(self.workspace_dir, 'globalkin_stage'), str(gk_stage))

        # add some dummy files into the results/staging directories:
        solution_path = os.path.join(self.workspace_dir, 'results', backend, 'solution.csv')
        with open(solution_path, 'w'):
            pass  # build an empty solution file
        self.assertTrue(os.path.isfile(solution_path))
        # check I can get the directories even after they are already existing:
        self.assertEqual(results_dir, config.get_results_dir(backend_id=backend))
        self.assertTrue(os.path.isfile(solution_path))  # the solution file has not been deleted...

        # morris results directory:
        self.assertFalse(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'morris_sensitivity')))
        morris_results_dir = config.get_morris_results_dir()
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'morris_sensitivity')))
        # add a dummy run:
        os.mkdir(os.path.join(morris_results_dir, 'tmp'))
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'morris_sensitivity', 'tmp')))
        morris_results_dir = config.get_morris_results_dir()
        # still there?:
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'morris_sensitivity', 'tmp')))
        self.assertTrue(os.path.isdir(os.path.join(morris_results_dir, 'tmp')))

        # reduction results directory:
        self.assertFalse(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'reduction')))
        reduction_results_dir = config.get_reduction_results_dir()
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'reduction')))
        # add a dummy run:
        os.mkdir(os.path.join(reduction_results_dir, 'tmp'))
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'reduction', 'tmp')))
        reduction_results_dir = config.get_reduction_results_dir()
        # still there?:
        self.assertTrue(os.path.isdir(os.path.join(self.workspace_dir, 'results', 'reduction', 'tmp')))
        self.assertTrue(os.path.isdir(os.path.join(reduction_results_dir, 'tmp')))

        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
