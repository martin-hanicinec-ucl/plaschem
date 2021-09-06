
import unittest
import os
import shutil
import yaml

import numpy as np

from pygmo_fwork.config import Config
from plaschem.chemistry import Chemistry
from pygmo_fwork.global_model import GlobalModel
from pygmo_fwork.results import ResultsParser


class TestPyGMoFwork(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))
    # file path to the chemistry xml file:
    chemistry_path = os.path.realpath(os.path.join(context, '..', 'shared_resources', 'run_test_chemistry.xml'))
    # temporary results directory, where the test run logs will be saved to:
    temp_results_dir = os.path.join(context, '.temp')  # temp dir path

    def setUp(self):
        self.chem = Chemistry(xml_path=self.chemistry_path)
        self.model_params = \
            {
                'feeds': {}, 'pressure': 100, 'power': (1000, 1000), 't_power': (-1e3, 1e3),
                'radius': 0.1, 'length': 0.1, 'gas_temp': 500, 't_end': 0.001
            }
        # add some random surface coefficients:
        self.chem.set_adhoc_species_attributes('stick_coef', 'O', 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O', {'O': 0.9, 'O2': 0.05})
        self.chem.set_adhoc_species_attributes('stick_coef', 'O-', 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O-', {'O': 1.0})
        self.chem.set_adhoc_species_attributes('stick_coef', 'O--', 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O--', {'O': 1.0})
        self.chem.set_adhoc_species_attributes('stick_coef', 'Ar+', 0.1)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar': 0.5})
        self.chem.set_adhoc_species_attributes('stick_coef', 'Ar++', 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar++', {'Ar': 0.5, 'Ar+': 0.5})
        self.chem.set_adhoc_species_attributes('stick_coef', 'Ar*', 0.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar*', {'Ar': 1.0})
        self.chem.set_adhoc_species_attributes('stick_coef', 'Ar**', 0.1)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar**', {'Ar': 0.5, 'Ar*': 0.25, 'Ar+': 0.2, 'Ar++': 0.05})

    def make_temp_dir(self):
        self.remove_temp_dir()
        os.mkdir(self.temp_results_dir)

    def remove_temp_dir(self):
        if os.path.isdir(self.temp_results_dir):
            shutil.rmtree(self.temp_results_dir)

    def test_run_pygmol(self):
        gm = GlobalModel(chemistry=self.chem, model_parameters=self.model_params, backend='PyGMol')
        self.make_temp_dir()
        run_id = 'run_id'
        gm.run(run_id=run_id, init_el_temp=1.0, log_results=True, log_to=self.temp_results_dir, verbose=False)

        rp = ResultsParser(nominal_results_dir=self.temp_results_dir)
        logged_solution = rp.fetch_solution(run_id)
        logged_rates = rp.fetch_rates(run_id)
        logged_wall_fluxes = rp.fetch_wall_fluxes(run_id)
        logged_model_params = rp.fetch_model_params(run_id)
        logged_initial_params = rp.fetch_initial_params(run_id)

        with open(
                os.path.join(
                    self.temp_results_dir, run_id, Config.default_log_name('chemistry_attributes', run_id=run_id)), 'r'
        ) as stream:
            logged_chemistry_attributes = yaml.load(stream, yaml.FullLoader)
        with open(os.path.join(
                self.temp_results_dir, run_id, Config.default_log_name('solver_log', run_id=run_id)), 'r') as stream:
            logged_solver_log = yaml.load(stream, yaml.FullLoader)

        # check if all the logs agree to the values from GlobalModel methods:
        self.assertTrue(np.isclose(logged_solution.values, gm.get_solution().values).all())
        self.assertTrue(np.isclose(logged_wall_fluxes.values, gm.get_wall_fluxes().values).all())
        self.assertTrue(np.isclose(logged_rates.values, gm.get_rates().values).all())
        self.assertEqual(logged_model_params['pressure'], gm.model.model_params['p'])
        self.assertEqual(logged_initial_params['init_n'], gm.generate_initial_n())
        self.assertEqual(logged_chemistry_attributes['xml_path'], self.chemistry_path)
        self.assertEqual(logged_solver_log['nfev'], gm.model.sol_raw.nfev)

        # the next block ensures correct behaviour of the ResultsParser methods for production and consumption rates:
        self.assertEqual(list(self.chem.get_species_name()), list(logged_solution.columns)[1:-4])
        self.assertEqual(list(self.chem.get_reactions().index), list(logged_rates.columns)[1:])
        self.assertEqual(list(self.chem.get_species_name()), list(logged_wall_fluxes.columns)[1:])
        self.assertTrue((logged_solution['t'].values == logged_rates['t'].values).all())
        self.assertTrue((logged_solution['t'].values == logged_wall_fluxes['t'].values).all())

        # ensure the production/consumption rates are correct and consistent with the equation class:
        rates_frame = logged_rates.iloc[-1]  # last time frame
        wall_fluxes_frame = logged_wall_fluxes.iloc[-1]  # last time frame
        eq_volumetric_source_rates = gm.model.diagnose('volumetric_source_rates').iloc[-1]
        eq_diffusion_source_rates = gm.model.diagnose('diffusion_source_rates').iloc[-1]
        for sp in eq_diffusion_source_rates.index:
            if sp != 't':
                self.assertEqual(
                    '{:.5E}'.format(eq_volumetric_source_rates[sp]),
                    '{:.5E}'.format(ResultsParser.get_volumetric_rates(sp, self.chem, rates_frame).sum())
                )
                self.assertEqual(
                    '{:.5E}'.format(eq_diffusion_source_rates[sp]),
                    '{:.5E}'.format(
                        ResultsParser.get_surface_rates(sp, self.chem, self.model_params, wall_fluxes_frame).sum()
                    )
                )

        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
