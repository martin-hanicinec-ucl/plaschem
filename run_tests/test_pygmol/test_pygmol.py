
import unittest
import os
from itertools import combinations
import glob

import numpy as np
import pandas as pd
from scipy import integrate

from pygmo_fwork.pygmol.model import Model
from pygmo_fwork.pygmol.equations import Equations
from plaschem.chemistry import Chemistry


class TestPyGMol(unittest.TestCase):

    file_dir = os.path.dirname(os.path.realpath(__file__))
    chem_xml_path = os.path.realpath(os.path.join(file_dir, '..', 'shared_resources', 'run_test_chemistry.xml'))
    solutions_dir = os.path.join(file_dir, 'solutions')

    def setUp(self):
        self.chem = Chemistry(xml_path=self.chem_xml_path)
        self.model_params = \
            {'feeds': {}, 'p': 100, 'P': (1000, 1000), 't_P': (-1e3, 1e3), 'r': 0.1, 'z': 0.1, 'Tg': 500, 't_end': 1.0}
        # TODO: this is a quick dirty fix - I want to have the new diffusion model (or both) in run tests!
        Equations.diffusion_model = 0

    @staticmethod
    def matching_solutions(sol1, sol2, verbose=True):
        # solutions_matched = np.isclose(sol1[-1], sol2[-1], ).all()
        sol1_final = sol1[-1]
        sol2_final = sol2[-1]

        if not len(sol1_final) == len(sol2_final):
            if verbose:
                print()
                print('Solutions dimensions do not match!')
            return False

        # don't want to care of elements with practically zero densities:
        zero_mask = np.c_[sol1_final, sol2_final].max(axis=1) < 1.e2
        non_zero1 = sol1_final[~zero_mask]
        non_zero2 = sol2_final[~zero_mask]
        solutions_matched = (2*abs(non_zero1 - non_zero2)/(non_zero1 + non_zero2) < 5.e-4).all()
        if not solutions_matched and verbose:
            print()
            print(pd.DataFrame([pd.Series(sol1_final, name='solution vector #1'),
                                pd.Series(sol2_final, name='solution vector #2')]).T)
        return solutions_matched

    # ***************************************** RUN TESTS AGAINSt STORED SOLUTIONS *********************************** #

    def test_all_stored_solutions_unique(self):
        stored_solutions = \
            [np.loadtxt(path, delimiter=',') for path in glob.glob(os.path.join(self.solutions_dir, '*.csv'))]
        for sol1, sol2 in combinations(stored_solutions, 2):
            self.assertFalse(self.matching_solutions(sol1, sol2, verbose=False))

    def test_nominal(self):
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '01_nominal.csv'), delimiter=',')

        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

        # test the diagnose method:
        el_temp = model.diagnose('electron_temperature')['value'].values
        el_dens = model.diagnose('electron_density')['value'].values
        el_en_dens = 3/2*el_dens*el_temp
        self.assertTrue(np.isclose(el_en_dens[-50:], model.sol_primary[-50:, -1]).all())  # only compare last 50 vals

    def test_feeds(self):
        # set some feeds to non-zero values:
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '02_feeds.csv'), delimiter=',')

        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_sticking1(self):
        # set some sticking coefficients to 1.0 and keep all the return coefs to 0.0:
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '03_sticking1.csv'), delimiter=',')

        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 1.0)
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_sticking2(self):
        # set some sticking coefficients to 0.5 and keep all the return coefs to 0.0:
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '04_sticking2.csv'), delimiter=',')

        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 0.5)
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_return1(self):
        # set ion and exc sticking coefficients to 0.5 and all return coefficient to 1.0 for ground state neutrals:
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '05_return1.csv'), delimiter=',')

        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 0.5)
            self.chem.set_adhoc_species_attributes('ret_coefs', sp, {sp.strip('+-*'): 1.0})
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

        # reset the ad-hoc changes and check if it matched the solution without them...
        self.chem.reset_adhoc_species_attributes('stick_coef')
        self.chem.reset_adhoc_species_attributes('ret_coefs')

        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '02_feeds.csv'), delimiter=',')
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_return2(self):
        # set ion and exc sticking coefficients to 1.0 and some more interesting return coefficients:
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '06_return2.csv'), delimiter=',')

        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar': 0.4, 'Ar*': 0.3, 'Ar**': 0.3})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar++', {'Ar': 3.0})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar*', {'Ar': 0.5})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar**', {'Ar*': 0.1, 'Ar+': 0.1, 'Ar': 0.8})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O-', {'O--': 0.5, 'O': 0.5})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O--', {'O-': 0.1, 'O': 0.9})

        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_disable(self):
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '07_disable.csv'), delimiter=',')

        self.chem.disable(reactions=[3, 4, 12])
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

        self.chem.reset()
        model._solve(y0=y0, method='BDF', jacobian=False)
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '01_nominal.csv'), delimiter=',')
        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    def test_adhoc_k(self):
        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '08_adhoc_k.csv'), delimiter=',')

        orig_arrh_a = self.chem.get_reactions_arrh_a(si_units=False)
        for r_id in [10, 20, 30]:
            self.chem.set_adhoc_reactions_attributes('arrh_a', r_id, 1.1*orig_arrh_a[r_id])
        model = Model(self.chem, self.model_params)
        y0 = stored_primary_solution[0].copy()
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

        # reset the ad-hoc changes and check if it matched the solution without them...
        self.chem.reset_adhoc_reactions_attributes('arrh_a')

        stored_primary_solution = np.loadtxt(os.path.join(self.solutions_dir, '01_nominal.csv'), delimiter=',')
        model._solve(y0=y0, method='BDF', jacobian=False)

        self.assertTrue(self.matching_solutions(stored_primary_solution, model.sol_primary))

    # ************************************** COMPARE RUNS WHICH SHOULD BE EQUIVALENT ********************************* #

    def test_equivalence(self):
        # prepare chemistry and parameters:
        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 1.0)
            self.chem.set_adhoc_species_attributes('ret_coefs', sp, {sp.strip('+-*'): 1.0})
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        # run the nominal solution
        model = Model(self.chem, self.model_params)
        y0 = model._build_y0()
        model._solve(y0=y0, method='BDF', jacobian=False)
        nominal_sol = model.sol_primary.copy()

        equivalent_solutions = []

        model._solve(y0=y0, method='Radau', jacobian=False)
        equivalent_solutions.append(model.sol_primary.copy())

        for equivalent_sol in equivalent_solutions:
            self.assertTrue(self.matching_solutions(nominal_sol, equivalent_sol))

    # **************************************** REGENERATE ALL THE TEST SOLUTIONS ************************************* #

    def regenerate_solutions(self):
        """Method re-generating all the test solutions. To be used if the solutions for the test
        runs change in a controlled way (by changing the simulated physics for example).
        :return: None
        """
        # nominal:
        self.setUp()
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '01_nominal.csv'), model.sol_primary, delimiter=',')

        # feeds:
        self.setUp()
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '02_feeds.csv'), model.sol_primary, delimiter=',')

        # sticking1:
        self.setUp()
        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 1.0)
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '03_sticking1.csv'), model.sol_primary, delimiter=',')

        # sticking2:
        self.setUp()
        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 0.5)
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '04_sticking2.csv'), model.sol_primary, delimiter=',')

        # return1:
        self.setUp()
        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 0.5)
            self.chem.set_adhoc_species_attributes('ret_coefs', sp, {sp.strip('+-*'): 1.0})
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '05_return1.csv'), model.sol_primary, delimiter=',')

        # return2:
        self.setUp()
        for sp in ['O-', 'O--', 'Ar+', 'Ar++', 'Ar*', 'Ar**']:
            self.chem.set_adhoc_species_attributes('stick_coef', sp, 1.0)
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar': 0.4, 'Ar*': 0.3, 'Ar**': 0.3})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar++', {'Ar': 3.0})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar*', {'Ar': 0.5})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'Ar**', {'Ar*': 0.1, 'Ar+': 0.1, 'Ar': 0.8})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O-', {'O--': 0.5, 'O': 0.5})
        self.chem.set_adhoc_species_attributes('ret_coefs', 'O--', {'O-': 0.1, 'O': 0.9})
        self.model_params['feeds'] = {'Ar': 100, 'O': 100}
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '06_return2.csv'), model.sol_primary, delimiter=',')

        # disable:
        self.setUp()
        self.chem.disable(reactions=[3, 4, 12])
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '07_disable.csv'), model.sol_primary, delimiter=',')

        # adhoc_k:
        self.setUp()
        orig_arrh_a = self.chem.get_reactions_arrh_a(si_units=False)
        for r_id in [10, 20, 30]:
            self.chem.set_adhoc_reactions_attributes('arrh_a', r_id, 1.1 * orig_arrh_a[r_id])
        model = Model(self.chem, self.model_params)
        model._solve(method='BDF')
        np.savetxt(os.path.join(self.solutions_dir, '08_adhoc_k.csv'), model.sol_primary, delimiter=',')


if __name__ == '__main__':
    unittest.main()
