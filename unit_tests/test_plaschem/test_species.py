import unittest

from plaschem.species import RP
from plaschem.exceptions import RPInitError, RPAttributeError
from unit_tests.test_plaschem.utils import draw_test_species


class TestRP(unittest.TestCase):

    def test_initialisation(self):
        with self.assertRaises(RPInitError):
            sp = RP(-1, name='Ar', mass=1.)  # invalid id
        with self.assertRaises(RPAttributeError):
            sp = RP(0, mass=1.)  # missing mandaroty attributes
        with self.assertRaises(RPAttributeError):
            sp = RP(0, name='e')  # missing mandaroty attributes
        with self.assertRaises(RPAttributeError):
            sp = RP(0, name='Ar*', mass=1., state='*')  # missing mandatory attribute
        with self.assertRaises(RPAttributeError):
            sp = RP(0, name='M', unsupported=1.)
        with self.assertRaises(RPAttributeError):
            sp = RP(0, name='<Ar>', mass=1.)  # invalid name characters
        try:
            # these should be legal:
            sp = RP(0, name='Ar', mass=1.)
            sp = RP(0, name='Ar', mass='1.')
            sp = RP(999, name='e', mass=1.e-5)
            sp = RP(0, name='M')
            sp = RP(1, name='Ar*', mass=1., state='*', stateless='Ar')
        except (RPInitError, RPAttributeError) as e:
            self.fail('Unexpected RPInitError or RPAttributeError raised!')

    def test_units(self):
        with self.assertRaises(RPAttributeError):
            sp = RP(1, name='Ar', mass=1., units={'charge': 'e'})  # extra unit
        with self.assertRaises(RPAttributeError):
            sp = RP(1, name='Ar', mass=1., charge=0, units={'charge': 'J'})  # wrong value
        with self.assertRaises(RPAttributeError):
            sp = RP(1, name='Ar', mass=1., charge=0, h_form=0., lj_epsilon=0., lj_sigma=0.,
                    units={'charge': 'e', 'mass': 'amu', 'h_form': 'eV', 'lj_sigma': 'A', 'lj_epsilon': 'K', 'u': 'u'})
        try:
            # these should be legal:
            sp = RP(1, name='Ar', mass=1., charge=0, h_form=0., lj_epsilon=0., lj_sigma=0.,
                    units={'charge': 'e', 'mass': 'amu', 'h_form': 'eV', 'lj_sigma': 'A', 'lj_epsilon': 'K'})
            sp = RP(1, name='Ar', mass=1., charge=0, h_form=0., lj_epsilon=0., lj_sigma=0.,)
        except (RPInitError, RPAttributeError) as e:
            self.fail('Unexpected RPInitError or RPAttributeError raised!')

    def test_special(self):
        self.assertTrue(RP(1, name='e', mass=1.).is_special())
        self.assertTrue(RP(1, name='M').is_special())
        self.assertFalse(RP(0, name='Ar', mass=1.).is_special())
        self.assertFalse(RP(999, name='Ar', mass=1.).is_special())
        for sp in draw_test_species('e M'):
            for method in ['get_state', 'get_h_form', 'get_lj_epsilon', 'get_lj_sigma',
                           'get_stick_coef', 'get_ret_coefs']:
                self.assertIs(getattr(sp, method)(), None)

        sp = draw_test_species('M')[0]
        self.assertIs(sp.get_charge(), None)
        self.assertIs(sp.get_mass(), None)

    def test_is_stateless(self):
        self.assertTrue(RP(1, name='Ar', mass=1.).is_stateless())
        self.assertTrue(RP(1, name='Ar', stateless='Ar', mass=1.).is_stateless())
        self.assertTrue(RP(1, name='Ar*', stateless='Ar', mass=1.).is_stateless())
        self.assertFalse(RP(1, name='Ar*', state='*', stateless='Ar', mass=1.).is_stateless())
        self.assertFalse(RP(1, name='Ar', state='*', stateless='Ar', mass=1.).is_stateless())

    def test_charge_strip(self):
        for stateless_name in ['Ar2-2', 'Ar2++', 'Ar-6', 'Ar--', 'Ar+1', 'H2SO4++', 'H2SO4-7']:
            self.assertEqual(RP.strip_charge(stateless_name), stateless_name[:-2])
        for stateless_name in ['Ar2-', 'H2SO4+', 'O-']:
            self.assertEqual(RP.strip_charge(stateless_name), stateless_name[:-1])

    def test_get_atoms(self):
        sp = draw_test_species('Ar**')[0]
        self.assertEqual(sp.get_atoms(), {'Ar': 1})
        sp = draw_test_species('ArH+')[0]
        self.assertEqual(sp.get_atoms(), {'Ar': 1, 'H': 1})
        sp = draw_test_species('C3H8+')[0]
        self.assertEqual(sp.get_atoms(), {'C': 3, 'H': 8})
        sp = draw_test_species('COF2')[0]
        self.assertEqual(sp.get_atoms(), {'C': 1, 'O': 1, 'F': 2})
        sp = draw_test_species('SOF4')[0]
        self.assertEqual(sp.get_atoms(), {'S': 1, 'O': 1, 'F': 4})
        sp = draw_test_species('SF6-')[0]
        self.assertEqual(sp.get_atoms(), {'S': 1, 'F': 6})
        sp = draw_test_species('e')[0]
        self.assertEqual(sp.get_atoms(), {})
        sp = draw_test_species('M')[0]
        self.assertEqual(sp.get_atoms(), {'M': 1})
        # double ionisation
        for name in ['F+2*', 'F*+2', 'F++**']:
            sp = draw_test_species(name)[0]
            self.assertEqual(sp.get_atoms(), {'F': 1})
        for name in ['SO2F2(v=5)--', 'SO2F2-2(v=5)']:
            sp = draw_test_species(name)[0]
            self.assertEqual(sp.get_atoms(), {'S': 1, 'F': 2, 'O': 2})
        for name in ['Ar+2', 'Ar--']:
            sp = draw_test_species(name)[0]
            self.assertEqual(sp.get_atoms(), {'Ar': 1})
        for name in ['Ar2++', 'Ar2-2']:
            sp = draw_test_species(name)[0]
            self.assertEqual(sp.get_atoms(), {'Ar': 2})
        # large number of atoms:
        sp = draw_test_species('Si5H11')[0]
        self.assertEqual(sp.get_atoms(), {'Si': 5, 'H': 11})

    def test_attribute_getters(self):
        # test the default values:
        sp = RP(1, name='Ar', mass=1.)
        self.assertEqual(sp.get_stateless(), 'Ar')
        self.assertEqual(sp.get_state(), None)
        self.assertEqual(sp.get_charge(), 0)
        self.assertEqual(sp.get_h_form(), 0.)
        self.assertEqual(sp.get_lj_sigma(), 3.)
        self.assertEqual(sp.get_lj_epsilon(), 0.)
        self.assertEqual(sp.get_hpem_name(), None)
        self.assertEqual(sp.get_qdb_id(), None)
        self.assertEqual(sp.get_stick_coef(), 0.)
        self.assertEqual(sp.get_ret_coefs(), {})
        self.assertEqual(sp.get_comments(), None)

        sp = RP(1, name='Ar+', mass=1., charge=1)
        self.assertEqual(sp.get_stateless(), 'Ar+')
        self.assertEqual(sp.get_stick_coef(), 1.)
        self.assertEqual(sp.get_ret_coefs(), {'Ar': 1.})

        sp = RP(1, name='O-', mass=1., charge=-1)
        self.assertEqual(sp.get_stateless(), 'O-')
        self.assertEqual(sp.get_stick_coef(), 1.)
        self.assertEqual(sp.get_ret_coefs(), {'O': 1.})

        sp = RP(1, name='Ar*', mass=1., charge=0, state='*', stateless='Ar')
        self.assertEqual(sp.get_stateless(), 'Ar')
        self.assertEqual(sp.get_stick_coef(), 1.)
        self.assertEqual(sp.get_ret_coefs(), {'Ar': 1.})

        # latex representation:
        for sp_name, latex in [
            ('O', r'\mathrm{O}'),
            ('O2-', r'\mathrm{O}_2^-'),
            ('O2(StaTe)', r'\mathrm{O}_2 (StaTe)'),
            ('O2++(StaTe)', r'\mathrm{O}_2^{2+} (StaTe)'),
            ('O2N2', r'\mathrm{O}_2 \mathrm{N}_2'),
            ('Ar', r'\mathrm{Ar}'),
            ('Ar++', r'\mathrm{Ar}^{2+}'),
            ('Ar*', r'\mathrm{Ar}^*'),
            ('O***', r'\mathrm{O}^{***}'),
            ('O---', r'\mathrm{O}^{3-}'),
            ('Al2O3', r'\mathrm{Al}_2 \mathrm{O}_3'),
            ('Al2O3(St42)', r'\mathrm{Al}_2 \mathrm{O}_3 (St42)'),
            ('C19', r'\mathrm{C}_{19}'),
            ('C19+', r'\mathrm{C}_{19}^+'),
            ('Al3C19--', r'\mathrm{Al}_3 \mathrm{C}_{19}^{2-}'),
            ('e', r'\mathrm{e}'),
            ('M', r'\mathrm{M}')
        ]:
            sp = RP(1, name=sp_name, mass=1.)
            self.assertEqual(sp.get_latex(), latex)
        sp = RP(1, name='Ar', latex='42', mass=1.)
        self.assertEqual(sp.get_latex(), '42')

    def test_attribute_setters(self):
        sp = draw_test_species('O2')[0]
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('unsupported', 0.)  # unsupported attribute
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('stateless', 0.)  # unsupported value type
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('charge', 1.)  # unsupported value type
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('ret_coefs', '{}')  # unsupported value type
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('mass', 42, unit='km')  # incorrect unit
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('charge', 42, unit='foo')  # incorrect unit
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('h_form', 42, unit='foo')  # incorrect unit
        with self.assertRaises(RPAttributeError):
            sp.set_species_attribute('stick_coef', 42, unit='foo')  # does not support units

        # these should be fine:
        sp.set_species_attribute('mass', 42)
        sp.set_species_attribute('mass', 42, unit='amu')
        sp.set_species_attribute('charge', 42)
        sp.set_species_attribute('charge', 42, unit='e')
        sp.set_species_attribute('h_form', 42)
        sp.set_species_attribute('h_form', 42, unit='eV')
        sp.set_species_attribute('stick_coef', 42)


if __name__ == '__main__':
    unittest.main()
