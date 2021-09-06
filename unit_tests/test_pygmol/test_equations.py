
import unittest

import numpy as np

from unit_tests.test_plaschem.utils import draw_test_chemistry
from pygmo_fwork.pygmol.equations import Equations
from pygmo_fwork.pygmol.model_parameters import ModelParameters
from pygmo_fwork.pygmol.exceptions import EquationsParametersConsistencyError


class TestEquations(unittest.TestCase):
    chem04 = draw_test_chemistry(4)
    chem05 = draw_test_chemistry(5)
    chem06 = draw_test_chemistry(6)

    def test_init(self):
        # first check that instantiation is without problems for various power definitions and feeds:
        # instantiating directly with the ModelParametes instance:
        test_params = ModelParameters(
            feeds={'Ar': 42}, Tg=300, r=0.1, z=0.1, p=100, P=(1000, 1000), t_P=(0, 1), t_end=0.1
        )
        _ = Equations(chemistry=self.chem04, model_params=test_params)
        test_params = ModelParameters(
            feeds={}, Tg=300, r=0.1, z=0.1, p=100, P=(1000, 1000), t_P=(0, 1), t_end=0.1
        )
        _ = Equations(chemistry=self.chem05, model_params=test_params)

        # instantiating with dict model parameters:
        test_params = {
            'feeds': {'O2': 42, 'SF6': 42}, 'Tg': 300, 'r': 0.1, 'z': 0.1, 'p': 100,
            'P': [1000, 1000, 0, 0, 1000, 1000],
            't_P': [-1e10, 0.01, 0.01, 0.02, 0.02, 1e10],
            't_end': 0.1
        }
        _ = Equations(chemistry=self.chem06, model_params=test_params)

        # feed species consistency:
        model_params = \
            ModelParameters(feeds={'O2': 42}, Tg=300, r=0.1, z=0.1, p=100, P=(1000, 1000), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(EquationsParametersConsistencyError):
            _ = Equations(self.chem04, model_params)  # chemistry_04 does not containt O2

    def test_getters(self):
        model_params = ModelParameters(
            feeds={'Ar': 42}, temp_gas=300, radius=0.1, length=0.1, pressure=100, t_end=1,
            power=(0, 1000, 1000, 0, 0, 2000, 2000),
            t_power=(0, 0.4, 0.5, 0.5, 0.6, 0.6, 1e6),
        )  # ramp-up from 0 to 1kW for 0.4 sec, then constant 1kW for 0.1 sec, then turn off for 0.1 sec and than
        # turn on to 2kW for the rest...

        eq = Equations(self.chem04, model_params)

        # following should be legal
        _ = eq.get_objective_functions()

        # time dependent power recall:
        for t, power in (
                (0, 0), (0.2, 500), (0.49999, 1000), (0.5, 500), (0.50001, 0), (0.55, 0),
                (0.6, 1000), (0.60001, 2000), (1.0, 2000)
        ):
            self.assertAlmostEqual(eq.get_power_ext(t), power)

        # feeds:
        self.assertEqual(len(eq.feed_flows[eq.feed_flows == 0]), eq.num_species - 1)

        # some random partial result:
        y = np.array([1e10]*eq.num_unknowns)

        _ = eq.get_drho_dt(t=0.45, y=y)  # numeric time derivative of electron density

        self.assertEqual(eq.num_unknowns - 1, eq.num_species)


if __name__ == '__main__':
    unittest.main()
