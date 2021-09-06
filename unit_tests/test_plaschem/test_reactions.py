
import unittest
from collections import OrderedDict

import numpy as np

from plaschem.reactions import Reaction
from plaschem.exceptions import ReactionInitError, ReactionAttributeError, ReactionValueError
from unit_tests.test_plaschem.utils import draw_test_species, draw_test_reactions


class TestReaction(unittest.TestCase):

    def test_initialisation(self):
        with self.assertRaises(ReactionInitError):
            r = Reaction(0, draw_test_species('M'), draw_test_species('M'), attributes={})
        with self.assertRaises(ReactionInitError):
            r = Reaction(-1, draw_test_species('M'), draw_test_species('M'), special_number=-1)
        with self.assertRaises(ReactionInitError):
            r = Reaction(1, [], draw_test_species('M'), special_number=-1)
        with self.assertRaises(ReactionInitError):
            r = Reaction(2, draw_test_species('M'), [], special_number=-1)
        # too many reactants:
        with self.assertRaises(ReactionInitError):
            species = draw_test_species('M M M M')
            r = Reaction(3, reactants=species, products=species, special_number=-1)
        try:
            r = Reaction(4, draw_test_species('M'), draw_test_species('M'), arrh_a=1.e-20, units={'arrh_a': 's-1'})
        except (ReactionInitError, ReactionAttributeError) as e:
            self.fail('Reaction() raised ReactionInitError unexpectedly!')
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(5, draw_test_species('M'), draw_test_species('M'), arrh_a=-1.e-20,units={'arrh_a': 's-1'})
        # number of 'M not conserved':
        with self.assertRaises(ReactionValueError):
            r = Reaction(6, draw_test_species('M M'), draw_test_species('M'), special_number=45)

    def test_mandatory_attributes(self):
        m = draw_test_species('M')[0]
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(1, reactants=[m], products=[m], attributes={})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(2, reactants=[m], products=[m], attributes={'arrh_a': 1.0})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(3, reactants=[m], products=[m], attributes={'arrh_c': 1.0})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(4, reactants=[m], products=[m], attributes={'arrh_a': 1.0}, units={'arrh_c': 'eV'})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(5, reactants=[m], products=[m], attributes={'arrh_c': 1.0}, units={'arrh_a': 'cm3.s-1'})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(6, reactants=[m], products=[m], attributes={'special_number': -1, 'unsupported': None})
        try:
            r = Reaction(7, reactants=[m], products=[m],
                         attributes={'special_number': -1, 'arrh_a': 1.0, 'arrh_c': 1.0},
                         units={'arrh_a': 's-1', 'arrh_c': 'K'})
            r = Reaction(8, reactants=[m], products=[m],
                         attributes={'arrh_a': 1.0},
                         units={'arrh_a': 's-1'})
            r = Reaction(9, reactants=[m], products=[m], attributes={'special_number': -1})
        except (ReactionInitError, ReactionAttributeError) as e:
            self.fail('Reaction() raised ReactionInitError or ReactionAttributeError unexpectedly!')

    def test_units(self):
        m, e = draw_test_species('M e')
        # activation energy unit:
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(1, [m, m], [m, m], {'special_number': 1, 'arrh_c': 1.0}, units={'arrh_c': 'eV'})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(2, [e, m], [e, m], {'special_number': 1, 'arrh_c': 1.0}, units={'arrh_c': 'K'})
        # pre-exponential factor unit:
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(3, [m], [m], {'special_number': 1, 'arrh_a': 1.0}, units={'arrh_a': 'cm3.s-1'})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(4, [m, e], [m, e], {'special_number': 1, 'arrh_a': 1.0}, units={'arrh_a': 's-1'})
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(4, [m, e, e], [m, e, e], {'special_number': 1, 'arrh_a': 1.0}, units={'arrh_a': 'cm3.s-1'})
        try:
            r = Reaction(5, [m, e, e], [m, e, e], special_number=1, arrh_a=1.0, units={'arrh_a': 'cm6.s-1'})
            r = Reaction(6, [m, e], [m, e], {'special_number': 1}, arrh_a=1.0, units={'arrh_a': 'm3.s-1'})
            r = Reaction(7, [e], [e], {'special_number': 1, 'arrh_a': 1.0}, units={'arrh_a': 's-1'})
            r = Reaction(8, [m, m], [m, m], {'special_number': 1, 'arrh_c': 1.0}, units={'arrh_c': 'K'})
            r = Reaction(9, [e, m], [m, e], {'special_number': 1, 'arrh_c': 1.0}, units={'arrh_c': 'eV'})
        except (ReactionInitError, ReactionAttributeError) as e:
            self.fail('Reaction() raised ReactionInitError or ReactionAttributeError unexpectedly!')

        # test missing or unsupported units:
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(10, [m, e], [m, e], {'special_number': 1, 'arrh_a': 1.0})  # missing mandatory unit
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(11, [m, e], [m, e], {'special_number': 1, 'arrh_a': 1.0},
                         units={'arrh_a': 'm3.s-1', 'arrh_c': 'eV'})  # extra unit passed
        with self.assertRaises(ReactionAttributeError):
            r = Reaction(12, [m, e], [m, e], {'special_number': 1}, units={'arrh_a': 'm3.s-1', 'unsup': 'eV'})  # usup
        with self.assertRaises(ReactionAttributeError):  # wrong unit type:
            r = Reaction(13, [m, e], [m, e], {'special_number': 1}, units={'arrh_a': 1.})
        with self.assertRaises(ReactionAttributeError):  # wrong unit value:
            r = Reaction(14, [m, e], [m, e], {'special_number': 1}, units={'arrh_a': 'eV'})
        try:
            r = Reaction(
                15, [e, m], [m, e], arrh_a=1., arrh_b=1., arrh_c=1.,
                units={'arrh_a': 'cm3.s-1', 'arrh_c': 'eV'}
            )
            r = Reaction(
                16, [e, m], [m, e], arrh_a=1., arrh_b=1., arrh_c=1., el_en_loss=1., gas_heat_contrib=1.,
                units={'arrh_a': 'cm3.s-1', 'arrh_c': 'eV'}
            )
            r = Reaction(
                17, [e, m], [m, e], arrh_a=1., arrh_b=1., arrh_c=1., el_en_loss=1., gas_heat_contrib=1.,
                units={'arrh_a': 'cm3.s-1', 'arrh_c': 'eV', 'el_en_loss': 'eV', 'gas_heat_contrib': 'eV'}
            )
            r = Reaction(18, [e, m], [m, e], special_number=-1)
        except (ReactionInitError, ReactionAttributeError) as e:
            self.fail('Reaction() raised ReactionInitError or ReactionAttributeError unexpectedly!')

    def test_stoich_coefs(self):
        reactants = draw_test_species('O2 M O-')  # sp_id: (14, 293, 16)
        products = draw_test_species('e e O O O+ M')  # sp_id: (1, 1, 15, 15, 9, 293)
        r = Reaction(1, reactants=reactants, products=products, special_number=-1)

        self.assertEqual(r.get_stoich_coefs('lhs'), OrderedDict([(14, 1), (293, 1), (16, 1)]))
        self.assertEqual(r.get_stoich_coefs('rhs'), OrderedDict([(1, 2), (15, 2), (9, 1), (293, 1)]))
        self.assertEqual(r.get_stoich_coefs('net'), OrderedDict([(14, -1), (293, 0), (16, -1), (1, 2), (15, 2), (9, 1)]))

        with self.assertRaises(ReactionValueError):
            sc = r.get_stoich_coefs('unsupported')

    def test_sets(self):
        reactants = draw_test_species('O- O M')
        products = draw_test_species('O O e M')
        r = Reaction(1, reactants, products, special_number=-1)

        self.assertEqual(r.get_rp_names_set('lhs'), ('O-', 'O', 'M'))
        self.assertEqual(r.get_rp_names_set('rhs'), ('O', 'e', 'M'))
        self.assertEqual(r.get_rp_names_set('all'), ('O-', 'O', 'M', 'e'))
        with self.assertRaises(ReactionValueError):
            r.get_rp_names_set('unsupported')

        self.assertEqual(r.get_rp_ids_set('lhs'), (16, 15, 293))
        self.assertEqual(r.get_rp_ids_set('rhs'), (15, 1, 293))
        self.assertEqual(r.get_rp_ids_set('all'), (16, 15, 293, 1))
        with self.assertRaises(ReactionValueError):
            r.get_rp_ids_set('unsupported')

    def test_process_type(self):
        r = Reaction(1, draw_test_species('e M'), draw_test_species('e M'), special_number=-1)
        self.assertTrue(r.is_electron_process())
        self.assertTrue(r.is_elastic_process())
        self.assertFalse(r.is_ion_process())

        r = Reaction(2, draw_test_species('e O2 M'), draw_test_species('O2- M'), special_number=-1)
        self.assertTrue(r.is_electron_process())
        self.assertFalse(r.is_elastic_process())
        self.assertFalse(r.is_ion_process())

        r = Reaction(3, draw_test_species('O2- M'), draw_test_species('e O2 M'), special_number=-1)
        self.assertFalse(r.is_electron_process())
        self.assertFalse(r.is_elastic_process())
        self.assertTrue(r.is_ion_process())

        r = Reaction(4, draw_test_species('O2+ e'), draw_test_species('O O+ e'), special_number=-1)
        self.assertTrue(r.is_electron_process())
        self.assertFalse(r.is_elastic_process())
        self.assertTrue(r.is_ion_process())

        r = Reaction(5, draw_test_species('O2+ e'), draw_test_species('O O+ e'), special_number=-1, elastic=True)
        self.assertTrue(r.is_elastic_process())

        r = Reaction(6, draw_test_species('e M'), draw_test_species('e M'), special_number=-1, elastic=False)
        self.assertFalse(r.is_elastic_process())

    def test_consistency(self):
        with self.assertRaises(ReactionValueError):
            # constituent atoms conservation:
            r = Reaction(1, draw_test_species('H+'), draw_test_species('O2+'), special_number=-1)
        with self.assertRaises(ReactionValueError):
            # charge conservation
            r = Reaction(2, draw_test_species('H+'), draw_test_species('H'), special_number=-1)
        with self.assertRaises(ReactionValueError):
            r = Reaction(3, draw_test_species('e H+ M'), draw_test_species('H'), special_number=-1)
        try:
            r = Reaction(4, draw_test_species('e H+'), draw_test_species('H'), special_number=-1)
            r = Reaction(5, draw_test_species('e H+ M'), draw_test_species('H M'), special_number=-1)
        except ReactionValueError:
            self.fail('Unexpected ReactionValueError raised!')

    def test_attribute_getters(self):
        # Arrhinius A factor (SI units conversions)
        species = draw_test_species('e M H')
        r = Reaction(1, species, species, arrh_a=1., units={'arrh_a': 'cm6.s-1'})
        self.assertEqual(r.get_arrh_a(), 1.e-12)
        self.assertEqual(r.get_arrh_a(si_units=False), 1.)
        r = Reaction(2, species[:-1], species[:-1], arrh_a=1., units={'arrh_a': 'cm3.s-1'})
        self.assertEqual(r.get_arrh_a(), 1.e-6)
        self.assertEqual(r.get_arrh_a(si_units=False), 1.)
        r = Reaction(3, species[:-2], species[:-2], arrh_a=1., units={'arrh_a': 's-1'})
        self.assertEqual(r.get_arrh_a(), 1.)
        self.assertEqual(r.get_arrh_a(si_units=False), 1.)
        # test the default values:
        r = Reaction(4, species, species, special_number=-1)
        self.assertEqual(r.get_arrh_a(), None)
        self.assertEqual(r.get_unit('arrh_a'), None)
        self.assertEqual(r.get_arrh_b(), None)
        self.assertEqual(r.get_arrh_c(), None)
        self.assertEqual(r.get_unit('arrh_c'), None)
        self.assertEqual(r.get_el_en_loss(), 0.)
        self.assertEqual(r.get_gas_heat_contrib(), 0.)
        self.assertEqual(r.get_special_number(), -1)
        self.assertEqual(r.get_qdb_r_id(), None)
        self.assertEqual(r.get_qdb_ds_id(), None)
        self.assertEqual(r.get_doi(), None)
        self.assertEqual(r.get_comments(), None)
        self.assertEqual(r.get_elastic(), True)
        r = Reaction(5, species, species, arrh_a=1., units={'arrh_a': 'cm6.s-1'})
        self.assertEqual(r.get_special_number(), 10)
        self.assertEqual(r.get_arrh_a(), 1.e-12)
        self.assertEqual(r.get_unit('arrh_a'), 'cm6.s-1')
        self.assertEqual(r.get_arrh_b(), 0.)
        self.assertEqual(r.get_arrh_c(), 0.)
        self.assertEqual(r.get_unit('arrh_c'), None)
        r = Reaction(6, species[1:], species[1:], arrh_a=1., units={'arrh_a': 'cm3.s-1'})
        self.assertEqual(r.get_special_number(), 20)

        # latex representation:
        reactants = draw_test_species('e O O')
        products = draw_test_species('O O-')
        r = Reaction(7, reactants, products, arrh_a=1., units={'arrh_a': 'cm6.s-1'})
        self.assertEqual(
            r.get_latex(),
            r'\mathrm{e} + 2 \mathrm{O} \rightarrow \mathrm{O} + \mathrm{O}^-'
        )

    def test_collision_partner(self):
        r = Reaction(1, draw_test_species('e'), draw_test_species('e'), arrh_a=1., units={'arrh_a': 's-1'})
        with self.assertRaises(ReactionValueError):
            r.get_electron_collision_partner()
        r = Reaction(2, draw_test_species('e M'), draw_test_species('e M'), arrh_a=1., units={'arrh_a': 'cm3.s-1'})
        with self.assertRaises(ReactionValueError):
            r.get_electron_collision_partner()
        r = Reaction(3, draw_test_species('M M'), draw_test_species('M M'), arrh_a=1., units={'arrh_a': 'cm3.s-1'})
        with self.assertRaises(ReactionValueError):
            r.get_electron_collision_partner()
        r = Reaction(
            4, draw_test_species('e H H+'), draw_test_species('e H H+'), arrh_a=1., units={'arrh_a': 'cm6.s-1'}
        )
        self.assertIs(r.get_electron_collision_partner(), draw_test_species('H')[0])

    def test_rate_coefficient(self):
        r = Reaction(1, draw_test_species('e M'), draw_test_species('e M'),
                     arrh_a=1., arrh_b=1., arrh_c=1., units={'arrh_a': 'm3.s-1', 'arrh_c': 'eV'})
        self.assertEqual(r.get_rate_coefficient(1.), np.exp(-1))
        r = Reaction(2, draw_test_species('e M'), draw_test_species('e M'),
                     arrh_a=1., arrh_b=1., arrh_c=1., units={'arrh_a': 'cm3.s-1', 'arrh_c': 'eV'})
        self.assertEqual(r.get_rate_coefficient(1.), np.exp(-1) * 1.e-6)
        r = Reaction(3, draw_test_species('M M'), draw_test_species('M M'),
                     arrh_a=1., arrh_b=1., arrh_c=300., units={'arrh_a': 'm3.s-1', 'arrh_c': 'K'})
        self.assertEqual(r.get_rate_coefficient(300.), np.exp(-1))
        with self.assertRaises(ReactionValueError):
            r = Reaction(4, draw_test_species('e M'), draw_test_species('e M'),
                         special_number=-1)
            r.get_rate_coefficient(1.)

    def test_attribute_setters(self):
        r = draw_test_reactions(137)[0]
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('unsupported', 0.)  # unsupported attribute
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_c_unit', 0.)  # unsupported value type
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('qdb_ds_id', 1.)  # unsupported value type
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('elastic', 1.)  # unsupported value type
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('elastic', 'untrue')
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_a', '-1e-15')
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_a', -1e-15)
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_a', 1e-15)  # missing unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_a', 1e-15, unit='foo')  # incorrect unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_a', 1e-15, unit=None)  # incorrect unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_c', 42)  # missing unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('arrh_c', 1e-15, unit='foo')  # incorrect unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('el_en_loss', 42, unit='K')  # incorrect unit
        with self.assertRaises(ReactionAttributeError):
            r.set_reaction_attribute('comments', 'foo', unit='foo')  # does not support unit
        self.assertIs(r.get_elastic(), False)
        r.set_reaction_attribute('elastic', 'true')
        self.assertIs(r.get_elastic(), True)
        r.set_reaction_attribute('elastic', 'false')
        self.assertIs(r.get_elastic(), False)

        # this should be fine:
        r.set_reaction_attribute('el_en_loss', 42)  # missing unit which is not mandatory
        r.set_reaction_attribute('comments', 'foo')  # no unit


if __name__ == '__main__':
    unittest.main()

