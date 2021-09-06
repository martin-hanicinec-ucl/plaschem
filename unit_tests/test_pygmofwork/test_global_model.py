
import unittest
import os
import shutil
import glob
import yaml

from scipy import constants
import pandas as pd
import numpy as np

from plaschem.chemistry import Chemistry
from pygmo_fwork.global_model import GlobalModel
from pygmo_fwork.config import Config
from pygmo_fwork.exceptions import \
    GlobalModelInitialGuessError, BackendNoRunError, BackendRunUnsuccessfulError, \
    GlobalModelLogError, GlobalModelAttributeError
from unit_tests.test_plaschem.utils import draw_test_species, draw_test_reactions


class TestGlobalModel(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        self.chem = Chemistry(species=draw_test_species('e Ar Ar+'), reactions=draw_test_reactions(3, 7, 48))
        self.mp = {'feeds': {'Ar': 1}, 'Tg': 1, 'r': 1, 'z': 1, 'p': 1, 'P': (1, 1), 't_P': (-9, 9), 't_end': 1}
        self.temp = os.path.join(self.context, '.temp')  # temp dir path

    def make_temp_dir(self):
        self.remove_temp_dir()
        os.mkdir(self.temp)

    def remove_temp_dir(self):
        if os.path.isdir(self.temp):
            shutil.rmtree(self.temp)

    def test_init(self):
        with self.assertRaises(GlobalModelAttributeError):
            gm = GlobalModel(None, None, 'unsupported')
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        self.assertIs(gm.model.chemistry, self.chem)
        new_mp = self.mp.copy()
        self.assertEqual(self.mp, new_mp)
        new_mp['p'] = 42
        self.assertNotEqual(new_mp['p'], gm.model.model_params['p'])
        gm.reload_model_parameters(new_mp)
        self.assertEqual(new_mp['p'], gm.model.model_params['p'])
        self.assertIs(gm.model.chemistry, self.chem)
        self.assertIsNot(gm.model.model_params, self.mp)

    def test_initial_guess(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        initial_n = gm.generate_initial_n()
        total_n = sum(initial_n.values())
        pressure = total_n * constants.k * self.mp['Tg']
        self.assertAlmostEqual(pressure, self.mp['p'])

        # test the validator
        gm.validate_initial_n(initial_n)
        invalid_init_n = {key: 0 for key in initial_n}
        with self.assertRaises(GlobalModelInitialGuessError):
            gm.validate_initial_n(invalid_init_n)
        invalid_init_n = {key: 1 for key in initial_n}
        invalid_init_n.popitem()
        with self.assertRaises(GlobalModelInitialGuessError):
            gm.validate_initial_n(invalid_init_n)
        invalid_init_n = {key.upper(): initial_n[key] for key in initial_n}
        with self.assertRaises(GlobalModelInitialGuessError):
            gm.validate_initial_n(invalid_init_n)
        invalid_init_n = {key: -1e-10 for key in initial_n}
        with self.assertRaises(GlobalModelInitialGuessError):
            gm.validate_initial_n(invalid_init_n)

    def test_run(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        with self.assertRaises(GlobalModelAttributeError):
            gm.run()  # need to specify the run_id
        with self.assertRaises(GlobalModelLogError):
            gm.run(run_id='dummy_run', log_to='non/existing/path')  # path is non-existing
        with self.assertRaises(GlobalModelInitialGuessError):
            gm.run(run_id='dummy_run', init_el_temp=-42, log_to='.')  # inconsistent initial Te
        with self.assertRaises(GlobalModelInitialGuessError):
            # incons initial n:
            gm.run(run_id='dummy run', init_n={name: -42 for name in self.chem.get_species_name()}, log_to='.')

    def test_success(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        with self.assertRaises(BackendNoRunError):
            gm.success()
        try:
            gm.run(log_results=False, init_el_temp=-42)  # this will throw an exception!
        except GlobalModelInitialGuessError:
            pass
        self.assertEqual(gm.success(), False)
        gm.reload_model_parameters(self.mp)
        with self.assertRaises(BackendNoRunError):
            gm.success()

    def test_log_results(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        with self.assertRaises(BackendNoRunError):
            gm.log_results(None, None, None)
        gm.last_run_success = False
        with self.assertRaises(BackendRunUnsuccessfulError):
            gm.log_results(None, None, None)

        # test the actual logging:
        # pimp up the gm so I have something to log:
        dummy_solution = pd.DataFrame(index=range(0, 5), columns=range(5, 10)).fillna(4.2)
        dummy_rates = pd.DataFrame(index=range(10, 15), columns=range(15, 20)).fillna(42.0)
        dummy_fluxes = pd.DataFrame(index=range(20, 25), columns=range(25, 30)).fillna(0.42)
        gm.get_solution = lambda: dummy_solution
        gm.get_rates = lambda: dummy_rates
        gm.get_wall_fluxes = lambda: dummy_fluxes
        # no backend-specific logs will be saved here, these should be tested solo:
        gm.model.dump_logs = lambda run_results_dir: None
        gm.last_run_success = True
        # log the stuff into the temp dir:
        run_id = 'dummy_run'
        self.remove_temp_dir()
        with self.assertRaises(GlobalModelLogError):
            gm.log_results(run_id=run_id, results_dir=self.temp, init_params={})  # ilegal, results_dir not existent
        self.make_temp_dir()
        gm.log_results(run_id=run_id, results_dir=self.temp, init_params={})
        # read it back:
        read_solution = pd.read_csv(
            os.path.join(self.temp, run_id, Config.default_log_name('solution', run_id=run_id)), index_col=0)
        read_rates = pd.read_csv(
            os.path.join(self.temp, run_id, Config.default_log_name('rates', run_id=run_id)), index_col=0)
        read_fluxes = pd.read_csv(
            os.path.join(self.temp, run_id, Config.default_log_name('wall_fluxes', run_id=run_id)), index_col=0)
        self.assertTrue((dummy_solution.values == read_solution.values).all())
        self.assertEqual([int(a) for a in dummy_solution.index], [int(a) for a in read_solution.index])
        self.assertEqual([int(a) for a in dummy_solution.columns], [int(a) for a in read_solution.columns])
        self.assertTrue((dummy_rates.values == read_rates.values).all())
        self.assertEqual([int(a) for a in dummy_rates.index], [int(a) for a in read_rates.index])
        self.assertEqual([int(a) for a in dummy_rates.columns], [int(a) for a in read_rates.columns])
        self.assertTrue((dummy_fluxes.values == read_fluxes.values).all())
        self.assertEqual([int(a) for a in dummy_fluxes.index], [int(a) for a in read_fluxes.index])
        self.assertEqual([int(a) for a in dummy_fluxes.columns], [int(a) for a in read_fluxes.columns])

        # check all the other log files exist:
        for log_file in [
            Config.default_log_name('chemistry_attributes', run_id=run_id), Config.default_log_name('model_attributes', run_id=run_id)
        ]:
            file_path = os.path.join(self.temp, run_id, log_file.format(run_id))
            self.assertTrue(os.path.isfile(file_path))
        # check none other files were logged!
        self.assertEqual(len(glob.glob(os.path.join(self.temp, run_id, '*'))), 5, msg='Unexpected log files!')

        with self.assertRaises(GlobalModelLogError):
            gm.log_results(run_id=run_id, results_dir=self.temp, init_params={})  # ilegal, the run_id exists
        init = {'init_n': gm.generate_initial_n(), 'init_el_temp': 3.0}
        gm.log_results(run_id=run_id, results_dir=self.temp, init_params=init, overwrite=True)  # legal - overwrite
        # check the model attributes yaml dump:
        with open(os.path.join(
                self.temp, run_id, Config.default_log_name('model_attributes', run_id=run_id))) as stream:
            loaded_model_attributes = dict(yaml.load(stream, Loader=yaml.FullLoader))
        self.assertEqual(init, loaded_model_attributes['initial_params'])

        self.remove_temp_dir()

    def test_backend_getters(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        for getter in ['solution', 'solution_final', 'rates', 'rates_final', 'wall_fluxes', 'wall_fluxes_final']:
            with self.assertRaises(BackendNoRunError):
                _ = getattr(gm, 'get_{}'.format(getter))()
        gm.last_run_success = False
        for getter in ['solution', 'solution_final', 'rates', 'rates_final', 'wall_fluxes', 'wall_fluxes_final']:
            with self.assertRaises(BackendRunUnsuccessfulError):
                _ = getattr(gm, 'get_{}'.format(getter))()

        self.assertEqual(gm.num_species(), len(self.chem.get_species()))
        self.assertEqual(gm.num_reactions(), len(self.chem.get_reactions()))
        self.chem.disable(reactions=[48, ])
        self.assertEqual(gm.num_species(), len(self.chem.get_species()))
        self.assertEqual(gm.num_reactions(), len(self.chem.get_reactions()))
        self.chem.reset()

    def test_convergency_check(self):
        gm = GlobalModel(self.chem, self.mp, backend='pygmol')
        # build a dummy solution:
        time = pd.Series([0, 1], name='t')
        par01 = pd.Series([0, 1.1], name='p01')  # only just non-converging
        par02 = pd.Series([0.1, 1.2], name='p02')  # only just converging
        par03 = pd.Series([1, 1], name='p03')  # definitely convergent
        par04 = pd.Series([2.2, 1.1], name='p04')  # only just converging
        par05 = pd.Series([2.1, 1.0], name='p05')  # only just non-converging
        solution = pd.DataFrame([time, par01, par02, par03, par04, par05]).T
        convergency = gm.check_convergency(solution, self.mp, eps=0.1, timewindow=0.1, verbose=False)
        self.assertTrue((convergency.values == np.array([False, True, True, True, False])).all())


if __name__ == '__main__':
    unittest.main()
