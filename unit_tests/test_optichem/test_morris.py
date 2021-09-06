
import unittest
import os
import shutil

import numpy as np
import pandas as pd

from plaschem.chemistry import Chemistry
from pygmo_fwork.optichem.morris import MorrisMethod
from pygmo_fwork.optichem.exceptions import MorrisInputsError, MorrisCoherenceError
from unit_tests.test_plaschem.utils import draw_test_species, draw_test_reactions


# noinspection PyUnresolvedReferences
class TestMorrisMethod(unittest.TestCase):
    context = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(context, '.temp')

    # noinspection DuplicatedCode
    def setUp(self):
        np.random.seed(42)
        self.chem = Chemistry(species=draw_test_species('e Ar Ar+'), reactions=draw_test_reactions(3, 7, 48))
        # species ids: 1, 2, 35
        self.mp = {'feeds': {'Ar': 1}, 'Tg': 1, 'r': 1, 'z': 1, 'p': 1, 'P': (1, 1), 't_P': (-9, 9), 't_end': 1}
        self.morris_method = MorrisMethod(self.chem, self.mp)  # morris method class

    def tearDown(self):
        # noinspection PyArgumentList
        np.random.seed()

    def build_temp(self):
        self.remove_temp()
        os.mkdir(self.temp_dir)

    def remove_temp(self):
        # remove the .temp dir
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_seed_x(self):
        mm = self.morris_method
        self.assertTrue((mm._get_seed_x(mm.grid) == mm.grid).all())  # grid maps to itself
        self.assertTrue((mm._get_seed_x(mm.grid + 0.001) == mm.grid).all())
        self.assertTrue((mm._get_seed_x(mm.grid - 0.001) == mm.grid).all())
        with self.assertRaises(TypeError):
            mm._get_seed_x('abc')
        with self.assertRaises(TypeError):
            mm._get_seed_x(0.5)

    def test_trajectory(self):
        mm = self.morris_method
        valid_seed_x = mm._get_seed_x_random(10)
        self.assertTrue((mm._get_seed_x(valid_seed_x) == valid_seed_x).all())

        with self.assertRaises(MorrisInputsError):
            seed_x = np.random.choice(mm.grid, size=1)
            next(mm._trajectory(seed_x))  # not enough values
        with self.assertRaises(MorrisInputsError):
            next(mm._trajectory([1, 2]))  # enough values but not in morris space

        traj = mm._trajectory(seed_x=valid_seed_x)
        x_prev = None
        for i, d, x in traj:
            self.assertEqual(abs(d), mm.delta_magnitude)
            self.assertTrue((mm._get_seed_x(x) == x).all())
            if x_prev is not None:
                x_new = x_prev.copy()
                x_new[i] += d
                x_new = np.round(x_new, decimals=5)
                self.assertTrue((x_new == x).all())
            x_prev = x.copy()

        # dimension:
        self.assertEqual(len(list(mm._trajectory(seed_x=mm._get_seed_x_random(10)))), 10)

    def test_map_reactions_factors(self):
        mm = MorrisMethod
        # non-compatible sizes and types:
        with self.assertRaises(MorrisInputsError):
            mm._map_reactions_factors_to_k(np.array([1, 2, 3]), np.array([1, 2]))
        with self.assertRaises(TypeError):
            mm._map_reactions_factors_to_k(np.array([1, 2, 3]), 3)
        with self.assertRaises(TypeError):
            mm._map_reactions_factors_to_k([1, 2, 3], [1, 2, 3])
        with self.assertRaises(TypeError):
            mm._map_reactions_factors_to_k((1, 2, 3), (1, 2, 3))
        with self.assertRaises(TypeError):
            mm._map_reactions_factors_to_k(np.array([1, 2, 3]), {'a': 1, 'b': 2, 'c': 3})
        with self.assertRaises(TypeError):
            mm._map_reactions_factors_to_k(np.array([1, 2, 3]), {1, 2, 3})
        with self.assertRaises(MorrisInputsError):
            mm._map_reactions_factors_to_k(pd.Series([0, 1, 2], index=[0, 1, 2]), pd.Series([2, 1, 0], index=[1, 2, 3]))

        # these should be legal:
        old_vals = np.arange(10)
        # 0.5 should always map to the old value!
        new_vals = mm._map_reactions_factors_to_k(np.ones(len(old_vals))*0.5, old_vals)
        self.assertTrue((old_vals == new_vals).all())
        old_vals = pd.Series(old_vals, index=reversed(range(10)))
        # check if it preserves the index of pd.Series
        new_vals = mm._map_reactions_factors_to_k(np.ones(len(old_vals))*0.5, old_vals)
        self.assertEqual(list(old_vals.index), list(new_vals.index))

    def test_map_species_factors(self):
        mm = self.morris_method
        chem = self.chem.copy()
        num_sp = mm.chemistry.num_species()
        num_react = mm.chemistry.num_reactions()

        test_nominal_arrh_a = np.random.random(num_react)
        self.assertEqual(
            list(mm._map_species_factors_to_k(0.5*np.ones(num_sp), test_nominal_arrh_a, chem)),
            list(test_nominal_arrh_a)
        )
        self.assertNotEqual(
            list(mm._map_species_factors_to_k(0.*np.ones(num_sp), test_nominal_arrh_a, chem)),
            list(test_nominal_arrh_a)
        )
        self.assertNotEqual(
            list(mm._map_species_factors_to_k(1.*np.ones(num_sp), test_nominal_arrh_a, chem)),
            list(test_nominal_arrh_a)
        )
        self.assertNotEqual(
            list(mm._map_species_factors_to_k(np.random.random(num_sp), test_nominal_arrh_a, chem)),
            list(test_nominal_arrh_a)
        )

        # dimensions mismatch:
        with self.assertRaises(MorrisInputsError):
            mm._map_species_factors_to_k(np.random.random(num_sp + 1), np.random.random(num_react), chem)
        with self.assertRaises(MorrisInputsError):
            mm._map_species_factors_to_k(np.random.random(num_sp), np.random.random(num_react + 1), chem)
        with self.assertRaises(MorrisInputsError):
            mm._map_species_factors_to_k(np.random.random(num_sp + 1), np.random.random(num_react + 1), chem)

        # index mismatch:
        nominal_k_index = np.arange(42, 42+num_react)  # these is definitely not the reactions index
        with self.assertRaises(MorrisInputsError):
            mm._map_species_factors_to_k(
                np.random.random(num_sp),
                pd.Series(test_nominal_arrh_a, index=nominal_k_index),
                chem
            )

        # type mismatch:
        test_x = np.random.random(num_sp)
        test_k = test_nominal_arrh_a
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(list(test_x), test_k, chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(list(test_x), list(test_k), chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(test_x, tuple(test_k), chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(set(test_x), test_k, chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(test_x[0], test_k, chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(test_x, test_k[0], chem)
        with self.assertRaises(TypeError):
            mm._map_species_factors_to_k(test_x[0], test_k[0], chem)

    def test_map_a_factor_to_ret_coef(self):
        mm = self.morris_method

        with self.assertRaises(AssertionError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar', 'Ar+', 0.5, self.chem, 1.0)

        chem = self.chem.copy()

        # nominal coefficients: Ar:  s = 0.0, r = 0.0
        #                       Ar+: s = 1.0, r = 1.0
        assert chem.get_species_stick_coef()[2] == 0.0
        assert chem.get_species_stick_coef()[35] == 1.0
        assert chem.get_species_ret_coefs()[2] == {}
        assert chem.get_species_ret_coefs()[35] == {'Ar': 1.0}
        # test for wrong values:
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar', 'Ar+', 0.5, chem, 2.0)
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'e', 'Ar', 0.5, chem, 2.0)
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar+', 'e', 0.5, chem, 2.0)
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar+', 'Ar+', 0.5, chem, 2.0)
        # wrong relative range:
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar+', 'Ar', 0.5, chem, 1.0)
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar+', 'Ar', 0.5, chem, 0.0)
        with self.assertRaises(MorrisInputsError):
            mm._map_a_factor_to_ret_coef(0.5, 'Ar+', 'Ar', 0.5, chem, -1)

        # set some meaningful nominal surface coefficients:
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar': 0.5, 'Ar+': 0.5})
        assert chem.get_species_ret_coefs()[35] == {'Ar': 0.5, 'Ar+': 0.5}

        # check the compensation:
        x_i = 0.5
        nom_value = 0.5
        ret_coefs = mm._map_a_factor_to_ret_coef(x_i, 'Ar+', 'Ar', nom_value, chem, 2.0)
        # x_i = 0.5 and the same nominal value as the original value => no change!
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 0.5, 'Ar+': 0.5}})
        nom_value_2 = 0.6
        ret_coefs = mm._map_a_factor_to_ret_coef(x_i, 'Ar+', 'Ar', nom_value_2, chem, 2.0)
        # x_i = 0.5, but not the same nominal value as the original value => change, and return as self compensated!
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 0.6, 'Ar+': 0.4}})
        # and back again:
        ret_coefs = mm._map_a_factor_to_ret_coef(x_i, 'Ar+', 'Ar', nom_value, chem, 2.0)
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 0.5, 'Ar+': 0.5}})
        # different x_i:
        ret_coefs = mm._map_a_factor_to_ret_coef(1.0, 'Ar+', 'Ar', nom_value, chem, 2.0)
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 1.0, 'Ar+': 0.0}})
        ret_coefs = mm._map_a_factor_to_ret_coef(0.0, 'Ar+', 'Ar', nom_value, chem, 2.0)
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 0.25, 'Ar+': 0.75}})
        ret_coefs = mm._map_a_factor_to_ret_coef(0.0, 'Ar+', 'Ar', nom_value, chem, 10.0)
        self.assertEqual(ret_coefs, {'Ar+': {'Ar': 0.05, 'Ar+': 0.95}})
        with self.assertRaises(MorrisInputsError):  # this would increase the Ar return coefficient to more than 1, so
            # Ar+ cannot compensate
            _ = mm._map_a_factor_to_ret_coef(1.0, 'Ar+', 'Ar', nom_value, chem, 2.01)

        # if I actually implement the new ret_coefs in between, the results should not change!
        ret_coefs = mm._map_a_factor_to_ret_coef(1.0, 'Ar+', 'Ar', nom_value, chem, 2.0)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', ret_coefs['Ar+'])
        self.assertEqual(chem.get_species_ret_coefs()[35], {'Ar': 1.0, 'Ar+': 0.0})
        ret_coefs = mm._map_a_factor_to_ret_coef(0.0, 'Ar+', 'Ar', nom_value, chem, 2.0)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', ret_coefs['Ar+'])
        self.assertEqual(chem.get_species_ret_coefs()[35], {'Ar': 0.25, 'Ar+': 0.75})
        ret_coefs = mm._map_a_factor_to_ret_coef(0.0, 'Ar+', 'Ar', nom_value, chem, 10.0)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar+', ret_coefs['Ar+'])
        self.assertEqual(chem.get_species_ret_coefs()[35], {'Ar': 0.05, 'Ar+': 0.95})

        # just for fun, add some surface coefficients for Ar as well:
        chem.set_adhoc_species_attributes('stick_coef', 'Ar', 0.5)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar', {'Ar': 1.0, 'Ar+': 1e-3})  # 0.1% of Ar gets ionised
        total_ret_coef = 1 + 1e-3
        ret_coefs = mm._map_a_factor_to_ret_coef(0.5, 'Ar', 'Ar+', 1e-3, chem, 1e2)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar', ret_coefs['Ar'])
        self.assertEqual(chem.get_species_ret_coefs()[2], {'Ar+': 1e-3, 'Ar': 1.0})
        ret_coefs = mm._map_a_factor_to_ret_coef(0, 'Ar', 'Ar+', 1e-3, chem, 1e2)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar', ret_coefs['Ar'])
        self.assertEqual(chem.get_species_ret_coefs()[2]['Ar+'], 1e-5)
        self.assertEqual(round(sum(chem.get_species_ret_coefs()[2].values()), 10), total_ret_coef)
        ret_coefs = mm._map_a_factor_to_ret_coef(1, 'Ar', 'Ar+', 1e-3, chem, 1e2)
        chem.set_adhoc_species_attributes('ret_coefs', 'Ar', ret_coefs['Ar'])
        self.assertEqual(chem.get_species_ret_coefs()[2]['Ar+'], 1e-1)
        self.assertEqual(round(sum(chem.get_species_ret_coefs()[2].values()), 10), total_ret_coef)

        # just as a sanity check, resetting the surface coefs:
        chem.reset_adhoc_species_attributes('stick_coef')
        chem.reset_adhoc_species_attributes('ret_coefs')
        assert chem.get_species_stick_coef()[2] == 0.0
        assert chem.get_species_stick_coef()[35] == 1.0
        assert chem.get_species_ret_coefs()[2] == {}
        assert chem.get_species_ret_coefs()[35] == {'Ar': 1.0}

    def test_dir_structure(self):
        mm = self.morris_method
        self.assertIs(mm.meta_dir, None)

        self.build_temp()
        morris_results_dir = self.temp_dir

        morris_run_id = 'mrid'
        traj_id = 'tid'
        mm._prepare_logging(morris_run_id, morris_results_dir, traj_id)

        path = os.path.join(self.temp_dir, morris_run_id, 'logs', traj_id)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(mm.logs_traj_dir, path)

        path = os.path.join(self.temp_dir, morris_run_id, 'meta')
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(mm.meta_dir, path)

        self.remove_temp()

    def test_coherence_check(self):
        mm = self.morris_method
        self.build_temp()
        mm._prepare_logging('id', self.temp_dir)

        meta_dir = str(mm.meta_dir)

        mm._dump_meta(meta_dir, 'method')

        mm._assert_coherence(meta_dir, 'method')

        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'different_method')

        self.chem.disable(reactions=(48, ))
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')
        self.chem.reset()
        mm._assert_coherence(meta_dir, 'method')

        self.chem.set_adhoc_reactions_attributes('arrh_a', 48, 42.42)
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')
        self.chem.reset_adhoc_reactions_attributes('all')
        mm._assert_coherence(meta_dir, 'method')

        self.chem.set_adhoc_species_attributes('stick_coef', 'Ar', 0.42)
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')
        self.chem.reset_adhoc_species_attributes('all')
        mm._assert_coherence(meta_dir, 'method')

        self.chem.disable(reactions=(48, ))
        mm = MorrisMethod(self.chem, self.mp)  # new instance
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')
        self.chem.reset()
        mm._assert_coherence(meta_dir, 'method')

        mm = MorrisMethod(self.chem, self.mp)  # new instance
        mm._assert_coherence(meta_dir, 'method')
        new_mp = self.mp.copy()
        new_mp['p'] = 42
        mm = MorrisMethod(self.chem, new_mp)  # new instance with different pressure
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')

        self.chem.disable(reactions=(48, ))
        mm._dump_meta(meta_dir, 'method')
        mm._assert_coherence(meta_dir, 'method')
        self.chem.reset()
        with self.assertRaises(MorrisCoherenceError):
            mm._assert_coherence(meta_dir, 'method')

        self.remove_temp()

    def test_evaluate_trajectory(self):
        # TODO: Implement unit test for trajectory evaluation!
        # self.fail('Not Implemented!')
        pass

    def test_get_morris_stats(self):
        # TODO: Implement unit test for morris stats!
        # self.fail('Not Implemented!')
        pass

    def test_get_ranking(self):
        # TODO: Implement unit test for morris ranking!
        # self.fail('Not Implemented!')
        pass


if __name__ == '__main__':
    unittest.main()
