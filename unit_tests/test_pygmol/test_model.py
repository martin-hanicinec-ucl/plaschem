
import unittest
from collections import namedtuple
import os
import shutil
import yaml

import numpy as np
import pandas as pd
from scipy import constants

from pygmo_fwork.pygmol.model import ModelParameters, Model
from pygmo_fwork.pygmol.equations import Equations
from pygmo_fwork.pygmol.exceptions import \
    ModelParametersTypeError, ModelParametersError, InitialSolutionError, ModelSolutionError, ModelInitError
from unit_tests.test_plaschem.utils import draw_test_chemistry


class TestModel(unittest.TestCase):
    chem04 = draw_test_chemistry(4)
    chem05 = draw_test_chemistry(5)
    chem06 = draw_test_chemistry(6)

    context = os.path.dirname(os.path.realpath(__file__))
    temp = os.path.join(context, '.temp')  # temp dir path

    def setUp(self):
        self.mp_dict = {'feeds': {'H': 1}, 'Tg': 1, 'r': 1, 'z': 1, 'p': 1, 'P': (1, 1), 't_P': (0, 1), 't_end': 0.1}
        self.model_params = ModelParameters(self.mp_dict)

    def make_temp_dir(self):
        if os.path.isdir(self.temp):
            self.remove_temp_dir()
        os.mkdir(self.temp)

    def remove_temp_dir(self):
        shutil.rmtree(self.temp)

    def test_init(self):
        model = Model(self.chem04, self.model_params)
        self.assertIs(self.model_params, model.model_params)
        # simulate what will happen in solve():
        self.assertIsNotNone(model.equations)  # these should be already built from the initializer
        model._build_equations()
        self.assertTrue(isinstance(model.equations, Equations))
        self.assertIs(self.model_params, model.equations.model_params)

        self.assertEqual(model.num_species(), model.equations.num_species)
        self.assertEqual(model.num_reactions(), model.equations.num_reactions)
        self.assertEqual(model.dimension(), model.equations.num_unknowns)

        test_dict = self.mp_dict.copy()
        test_dict['z'] = '1.0'
        with self.assertRaises(ModelParametersTypeError):
            _ = Model(self.chem04, test_dict)
        del(test_dict['z'])
        with self.assertRaises(ModelParametersError):
            _ = Model(self.chem04, test_dict)

        with self.assertRaises(ModelInitError):  # chemistry does not have the required structures
            _ = Model(None, self.model_params)

    def test_build_y0(self):
        mp_dict = self.mp_dict.copy()
        mp_dict['feeds'] = {'SF6': 420, 'O2': 42}
        model = Model(self.chem06, mp_dict)

        model._build_equations()
        # check the electron temperature in the initial solution builder
        self.assertAlmostEqual(3.0, model.equations.get_electron_temperature(model._build_y0(el_temp=3.0)))
        self.assertAlmostEqual(1.0, model.equations.get_electron_temperature(model._build_y0()))  # default
        # the same with pressure:
        self.assertAlmostEqual(mp_dict['p'], model.equations.get_total_pressure(model._build_y0()))

    def test_y0_from_init_params(self):
        mp_dict = self.mp_dict.copy()
        mp_dict['feeds'] = {'SF6': 420, 'O2': 42}
        model = Model(self.chem06, mp_dict)

        init_el_temp = 4.2
        species_names = np.array(model.chemistry.get_species_name())
        init_n = dict(zip(species_names, np.random.random(len(species_names))))
        init_p = sum(init_n.values()) * constants.k * mp_dict['Tg']

        init_n_ilegal = init_n.copy()
        init_n_ilegal.pop(list(init_n.keys())[0])
        with self.assertRaises(InitialSolutionError):
            model._y0_from_init_params(init_el_temp, init_n_ilegal)

        init_n_ilegal = init_n.copy()
        init_n_ilegal['unsupported'] = 42.
        with self.assertRaises(InitialSolutionError):
            model._y0_from_init_params(init_el_temp, init_n_ilegal)

        model._build_equations()
        y0 = model._y0_from_init_params(init_el_temp, init_n=None)
        self.assertAlmostEqual(init_el_temp, model.equations.get_electron_temperature(y0))
        y0 = model._y0_from_init_params(init_el_temp, init_n)
        self.assertAlmostEqual(init_el_temp, model.equations.get_electron_temperature(y0))
        self.assertEqual(list(pd.Series(init_n)[species_names]), list(model.equations.get_density_vector(y0)))
        # pressure:
        self.assertAlmostEqual(init_p, model.equations.get_total_pressure(y0))

    def test_validate_y0(self):
        model = Model(self.chem05, self.model_params)
        try:
            y0 = np.ones(model.dimension())  # this should be legal y0, since there are more +ions than -ions
            model._validate_y0(y0)
            y0 = model._build_y0()
            model._validate_y0(y0)
        except InitialSolutionError:
            self.fail('InitialSolutionError raised unexpectedly!')
        with self.assertRaises(InitialSolutionError):
            model._validate_y0(-y0)  # to simulate negative charges
        with self.assertRaises(InitialSolutionError):
            model._validate_y0(np.zeros(model.dimension()))  # to simulate no charges
        with self.assertRaises(InitialSolutionError):
            model._validate_y0(np.ones(model.dimension() + 1))  # wrong dimmension
        with self.assertRaises(InitialSolutionError):
            model._validate_y0(np.ones(model.dimension() - 1))  # wrong dimmension

    def test_results(self):
        model = Model(self.chem05, self.model_params)
        with self.assertRaises(ModelSolutionError):
            model.get_solution()
        with self.assertRaises(ModelSolutionError):
            model.get_rates()
        with self.assertRaises(ModelSolutionError):
            model.get_solution_final()
        with self.assertRaises(ModelSolutionError):
            model.get_rates_final()

    def test_dump_logs(self):
        model = Model(self.chem05, self.model_params)
        SolRaw = namedtuple('SolRaw', 'nfev njev nlu status message success')
        model.sol_raw = SolRaw(nfev=42, njev=42, nlu=42, status='status', message='message', success=1)
        self.make_temp_dir()
        model.dump_logs(self.temp)  # dump the dummy raw solution attributes into self.temp directory
        # try to load them back from the yaml:
        with open(os.path.join(self.temp, 'solver_log.yaml'), 'r') as stream:
            solver_log = yaml.load(stream, yaml.FullLoader)
        self.assertEqual(solver_log['message'], model.sol_raw.message)
        self.assertEqual(model.sol_raw.message, 'message')
        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
