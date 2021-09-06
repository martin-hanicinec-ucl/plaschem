
import unittest
import os
import shutil

from plaschem.chemistry import Chemistry
from plaschem.species import RP
from plaschem.reactions import Reaction
from plaschem.exceptions import \
    ChemistryConsistencyError, RPAttributeError, ReactionAttributeError, ChemistrySpeciesNotPresentError, \
    ChemistryReactionNotPresentError, ChemistryDisableError, ChemistryInitError, XmlError, \
    ChemistryAttributeError
from plaschem.xml_inputs import XmlParser
from unit_tests.test_plaschem.utils import draw_test_species, draw_test_reactions


class TestChemistry(unittest.TestCase):

    def setUp(self):
        species = draw_test_species('e Ar Ar+ Ar* Ar2+ M')
        # species ids =            ( 1 2  35  3   36   293)
        reactions = draw_test_reactions(3, 48, 13, 4, 9, 7, 11, 2539, 28, 13677, 2455)
        """
        reactions = {
            3: e + Ar > e + Ar, 
            48: e + Ar+ > e + Ar+, 
            13: e + Ar+ > Ar
            4: e + Ar > e + Ar*, 
            9: e + Ar* > e + Ar, 
            7: e + Ar > e + e + Ar+, 
            11: e + Ar* > e + e + Ar+, 
            2539: Ar* > Ar, 
            28: Ar+ + Ar + Ar > Ar + Ar2+
            13677: Ar + Ar+ + M > Ar2+ + M
            2455: Ar + Ar2+ > Ar + Ar + Ar+
        }
        """
        self.test_chem_01 = Chemistry(species=species, reactions=reactions)

        self.context = os.path.dirname(os.path.realpath(__file__))
        self.xml_resources = os.path.join(self.context, 'resources', 'xml_files')

    def make_temp_dir(self):
        self.temp_dir = os.path.join(self.context, 'resources', '.temp')
        if os.path.isdir(self.temp_dir):
            raise ValueError('Temporary directory already exists!')
        else:
            os.mkdir(self.temp_dir)

    def remove_temp_dir(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # try different combinations of parameters and see if invalid ones raise ChemistryInitError
        with self.assertRaises(ChemistryInitError):
            ch = Chemistry(reactions=[])
        with self.assertRaises(ChemistryInitError):
            ch = Chemistry(species=[])
        with self.assertRaises(ChemistryInitError):
            ch = Chemistry(reactions=[], xml_path='path/to/xml')
        with self.assertRaises(ChemistryInitError):
            ch = Chemistry(species=[], xml_path='path/to/xml')
        with self.assertRaises(ChemistryInitError):
            ch = Chemistry()

        reactions = list(self.test_chem_01._reactions)
        species = list(self.test_chem_01._species)
        # following should be legal even though they are raising errors for not finding the xml file and for chem cons.:
        try:
            with self.assertRaises(XmlError):
                ch = Chemistry(xml_path='path/to/xml')
            ch = Chemistry(reactions=reactions, species=species, xml_path='path/to/xml')
            self.assertEqual(ch.xml_path, 'path/to/xml')
            ch = Chemistry(reactions=reactions, species=species)
            self.assertEqual(ch.xml_path, None)
        except (ChemistryInitError, ChemistryConsistencyError) as f:
            self.fail('ChemistryConsistencyError or ChemistryInitError raised unexpectedly!')

    def test_consistency(self):
        # try initialising the chemistry without electron:
        species = draw_test_species('Ar Ar+ Ar* Ar2+ M')
        reactions = draw_test_reactions(2539, 28, 13677, 2455)
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(reactions=reactions, species=species)
        # initialisation without any positive ion:
        species = draw_test_species('e Ar Ar*')
        reactions = draw_test_reactions(3, 4, 9, 11, 2539)
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(reactions=reactions, species=species)
        species = draw_test_species('e Ar Ar+ Ar* Ar2+ M')
        reactions = draw_test_reactions(3, 48, 13, 4, 9, 7, 11, 2539, 28, 13677, 2455)
        try:
            ch = Chemistry(reactions=reactions, species=species)
            ch.assert_consistency()
        except ChemistryConsistencyError:
            self.fail('ChemistryConsistencyError raised unexpectedly!')
        species[-1] = RP(rp_id=1, name='M')  # change id to the same one as electron
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
            ch.assert_consistency()
        species[-1] = draw_test_species('M')[0]
        species.append(RP(rp_id=9999999999, name='M'))  # add a second M with the same name
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
            ch.assert_consistency()
        species.pop(-1)
        # add a specie which is not in reactions:
        species.extend(draw_test_species('Ar**'))
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
            ch.assert_consistency()
        species.pop(-1)
        # swap M for another copy of M:
        species[-1] = RP(rp_id=293, name='M', hpem_name='M', qdb_id=293)
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
            ch.assert_consistency()
        species[-1] = draw_test_species('M')[0]
        # two species having the same id?
        orig_M = species.pop(-1)
        species.append(RP(rp_id=2, name='M', hpem_name='M', qdb_id=293))
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
        species[-1] = orig_M  # return back
        # two reactions having the same id need to trigger error!
        reactions.append(Reaction(
            r_id=reactions[0].id, reactants=3*[species[1], ], products=3*[species[1],], arrh_a=0.0,
            units={'arrh_a': 'cm6.s-1'}
        ))
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
        reactions.pop(-1)
        # two same reactions with different ids cannot trigger error!
        reactions.append(Reaction(
            r_id=99999, reactants=list(reactions[0]._reactants), products=list(reactions[0]._products), arrh_a=0.0,
            units={'arrh_a': 'cm3.s-1'}
        ))
        try:
            ch = Chemistry(reactions=reactions, species=species)
            ch.assert_consistency()
        except ChemistryConsistencyError:
            self.fail('ChemistryConsistencyError raised unexpectedly!')
        reactions.pop(-1)
        # try again that everything works:
        try:
            ch = Chemistry(reactions=reactions, species=species)
            ch.assert_consistency()
        except ChemistryConsistencyError:
            self.fail('ChemistryConsistencyError raised unexpectedly!')
        # add a reaction containing specie not supplied by the species:
        reactions.extend(draw_test_reactions(15))
        with self.assertRaises(ChemistryConsistencyError):
            ch = Chemistry(species=species, reactions=reactions)
            ch.assert_consistency()
        reactions.pop(-1)
        # ignore specie and leave its reactions:
        ch = Chemistry(species=species, reactions=reactions)
        ch._enabled_sp.iat[4] = False
        with self.assertRaises(ChemistryConsistencyError):
            ch.assert_consistency()
        ch._enabled_sp.iat[4] = True
        # ignore all reactions for a specie and leave the specie:
        ch._enabled_r[[28, 13677, 2455]] = False  # only ones using up M and Ar2+
        with self.assertRaises(ChemistryConsistencyError):
            ch.assert_consistency()
        ch._enabled_r[[28, 13677, 2455]] = True
        # define return species not in chemistry:
        orig = species[4]._ret_coefs
        species[4].set_species_attribute('ret_coefs', {'C60': 1.})
        with self.assertRaises(ChemistryConsistencyError):
            ch.assert_consistency()
        with self.assertRaises(ChemistryConsistencyError):  # needs to raise exception if initialising with incons. ret
            ch = Chemistry(species=species, reactions=reactions)
        species[4]._ret_coefs = orig
        # try again that everything works:
        try:
            ch.assert_consistency()
        except ChemistryConsistencyError:
            self.fail('ChemistryConsistencyError raised unexpectedly!')

    def test_get_species_and_reactions(self):
        ch = self.test_chem_01

        # test species getter
        ch._enabled_sp.iat[4] = False  # explicitly ignore 5th specie (Ar2+)
        self.assertEqual(list(ch.get_species()), draw_test_species('Ar Ar+ Ar*'))
        self.assertEqual(list(ch.get_species(disabled=True)), draw_test_species('Ar Ar+ Ar* Ar2+'))
        self.assertEqual(list(ch.get_species(special=True)), draw_test_species('e Ar Ar+ Ar* M'))
        self.assertEqual(list(ch.get_species(disabled=True, special=True)), draw_test_species('e Ar Ar+ Ar* Ar2+ M'))
        ch._enabled_sp.iat[-1] = False  # explicitly ignore 'M' which is a special species:
        self.assertEqual(list(ch.get_species()), draw_test_species('Ar Ar+ Ar*'))
        self.assertEqual(list(ch.get_species(disabled=True)), draw_test_species('Ar Ar+ Ar* Ar2+'))
        self.assertEqual(list(ch.get_species(special=True)), draw_test_species('e Ar Ar+ Ar*'))
        self.assertEqual(list(ch.get_species(disabled=True, special=True)), draw_test_species('e Ar Ar+ Ar* Ar2+ M'))
        ch._enabled_sp.iat[-1] = True
        ch._enabled_sp.iat[4] = True

        # test the correct level of copying in the returned array: needs to return a copy of _species array containing
        # the same RP objects:
        ch_species = ch.get_species(disabled=True, special=True)
        ch_species = ch_species[[True, True, True, True, False, True]]
        self.assertNotEqual(len(ch_species), len(ch.get_species(disabled=True, special=True)))
        ch_species = ch.get_species(disabled=True, special=True)
        self.assertEqual(ch_species.iat[-1], ch.get_species(disabled=True, special=True).iat[-1])
        ch_species.iat[-1] = RP(rp_id=273, name='M')
        self.assertNotEqual(ch_species.iat[-1], ch.get_species(disabled=True, special=True).iat[-1])

        ch_species.iat[0].set_species_attribute('comments', 'let us get schwifty...')
        self.assertEqual(ch_species.iloc[0].get_comments(), 'let us get schwifty...')
        self.assertEqual(ch.get_species(special=True).iloc[0].get_comments(), 'let us get schwifty...')
        ch_species.iat[0].set_species_attribute('comments', None)

        # test reactions getter...
        ch._enabled_r.at[2539] = False  # ignore the radiative transition explicitly
        self.assertEqual(list(ch.get_reactions().index), [3, 48, 13, 4, 9, 7, 11, 28, 13677, 2455])
        self.assertEqual(list(ch.get_reactions(disabled=True).index), [3, 48, 13, 4, 9, 7, 11, 2539, 28, 13677, 2455])
        ch._enabled_r.at[2539] = True  # reset disabled mask

        # copy level:
        ch_reactions = ch.get_reactions(disabled=True)
        ch_reactions.at[2539] = None
        self.assertIs(ch_reactions[2539], None)
        self.assertNotEqual(ch.get_reactions(disabled=True)[2539], None)
        ch_reactions.at[2455].set_reaction_attribute('comments', 'let us get schwifty...')
        self.assertEqual(ch_reactions[2455].get_comments(), 'let us get schwifty...')
        self.assertEqual(ch.get_reactions(disabled=True)[2455].get_comments(), 'let us get schwifty...')
        ch_reactions.at[2455].set_reaction_attribute('comments', None)

    def test_protected_species(self):
        ch = self.test_chem_01
        # test the protected species in getters:
        self.assertEqual(len(ch.get_protected_species()), 1)  # only electron
        # protect explicitly Ar:
        ch.set_protected_species('Ar')
        self.assertEqual(len(ch.get_protected_species()), 2)
        self.assertEqual(list(ch.get_species_name()), 'Ar Ar+ Ar* Ar2+'.split())
        self.assertEqual(list(ch.get_species_name(protected=False)), 'Ar+ Ar* Ar2+'.split())
        # disable one of the ions, the second should be set as protected implicitly (adhoc_protected)
        ch.disable(species=['Ar2+'])  # also disables M
        self.assertEqual(list(ch.get_species_name()), 'Ar Ar+ Ar*'.split())
        self.assertEqual(list(ch.get_species_name(protected=False, special=True)), 'Ar*'.split())
        self.assertEqual(list(ch.get_species_name(protected=False, special=False)), 'Ar*'.split())
        self.assertEqual(ch.num_species(special=True), 4)  # two are disabled
        self.assertEqual(ch.num_species(disabled=True, special=True), 6)
        self.assertEqual(ch.num_species(disabled=True), 4)
        self.assertEqual(ch.num_species(protected=False, special=True), 1)
        self.assertEqual(ch.num_species(protected=False, special=False), 1)  # one is disabled
        # check the chemistry reset does not reset the (explicitly) protected species
        ch.reset()  #  electron needs to stay protected!
        self.assertEqual(list(ch.get_species_name(protected=False, special=True)), 'Ar+ Ar* Ar2+ M'.split())
        # this was defined in another method originaly:
        self.setUp()
        ch = self.test_chem_01
        self.assertTrue(ch._protected_sp.at[1])  # check that e is protected
        self.assertTrue(1 in set(ch.get_protected_species().index))  # check that e is protected
        for sp_id in (2, 3, 293):  # check that Ar, Ar* and M are not protected
            self.assertFalse(ch._protected_sp.at[sp_id])
            self.assertFalse(sp_id in set(ch.get_protected_species().index))
        ch.set_protected_species('Ar', 3, ch._species.at[293])  # set protected Ar, Ar* and M, each passed different way
        for sp_id in (1, 2, 3, 293):
            self.assertTrue(ch._protected_sp.at[sp_id])
            self.assertTrue(sp_id in set(ch.get_protected_species().index))
        # check if the sole positive ion gets protected:
        species = draw_test_species('e Ar Ar+')
        reactions = draw_test_reactions(3, 48, 13, 7)
        ch = Chemistry(reactions=reactions, species=species)
        self.assertTrue(ch._protected_sp.at[35])  # check if Ar+ is protected

    def test_species_attributes_getters(self):
        ch = self.test_chem_01
        na = 42

        # without any tinkering:
        self.assertEqual(list(ch.get_species_name()), ['Ar', 'Ar+', 'Ar*', 'Ar2+'])
        self.assertEqual(list(ch.get_species_charge()), [0, 1, 0, 1])
        self.assertEqual(list(ch.get_species_mass()), [39.948, 39.948, 39.948, 79.896])
        self.assertEqual(list(ch.get_species_h_form()), [0., 0., 0., 0.])
        self.assertEqual(list(ch.get_species_lj_epsilon()), [93.3, 93.3, 93.3, 93.3])
        self.assertEqual(list(ch.get_species_lj_sigma()), [3.542, 3.542, 3.542, 3.54])
        self.assertEqual(list(ch.get_species_stick_coef()), [0., 1., 1., 1.])
        self.assertEqual(list(ch.get_species_ret_coefs()), [{}, {'Ar': 1.}, {'Ar': 1.}, {'Ar': 2.}])

        self.assertEqual(list(ch.get_species_name(special=True)), ['e', 'Ar', 'Ar+', 'Ar*', 'Ar2+', 'M'])
        self.assertEqual(list(ch.get_species_charge(special=True).fillna(na)), [-1, 0, 1, 0, 1, na])
        self.assertEqual(list(ch.get_species_mass(special=True).fillna(na)),
                         [5.46e-4, 39.948, 39.948, 39.948, 79.896, na])
        self.assertEqual(list(ch.get_species_h_form(special=True).fillna(na)), [na, 0., 0., 0., 0., na])
        self.assertEqual(list(ch.get_species_lj_epsilon(special=True).fillna(na)), [na, 93.3, 93.3, 93.3, 93.3, na])
        self.assertEqual(list(ch.get_species_lj_sigma(special=True).fillna(na)), [na, 3.542, 3.542, 3.542, 3.54, na])
        self.assertEqual(list(ch.get_species_stick_coef(special=True).fillna(na)), [na, 0., 1., 1., 1., na])
        self.assertEqual(list(ch.get_species_ret_coefs(special=True).fillna(na)),
                         [na, {}, {'Ar': 1.}, {'Ar': 1.}, {'Ar': 2.}, na])

        # ignore explicitly Ar2+, M
        ch._enabled_sp.iloc[[4, 5]] = False

        self.assertEqual(list(ch.get_species_name()), ['Ar', 'Ar+', 'Ar*'])
        self.assertEqual(list(ch.get_species_charge()), [0, 1, 0])
        self.assertEqual(list(ch.get_species_mass()), [39.948, 39.948, 39.948])
        self.assertEqual(list(ch.get_species_h_form()), [0., 0., 0.])
        self.assertEqual(list(ch.get_species_lj_epsilon()), [93.3, 93.3, 93.3])
        self.assertEqual(list(ch.get_species_lj_sigma()), [3.542, 3.542, 3.542])
        self.assertEqual(list(ch.get_species_stick_coef()), [0., 1., 1.])
        self.assertEqual(list(ch.get_species_ret_coefs()), [{}, {'Ar': 1.}, {'Ar': 1.}])

        self.assertEqual(list(ch.get_species_name(special=True)), ['e', 'Ar', 'Ar+', 'Ar*'])
        self.assertEqual(list(ch.get_species_charge(special=True)), [-1, 0, 1, 0])
        self.assertEqual(list(ch.get_species_mass(special=True)), [5.46E-04, 39.948, 39.948, 39.948])
        self.assertEqual(list(ch.get_species_h_form(special=True).fillna(na)), [na, 0., 0., 0.])
        self.assertEqual(list(ch.get_species_lj_epsilon(special=True).fillna(na)), [na, 93.3, 93.3, 93.3])
        self.assertEqual(list(ch.get_species_lj_sigma(special=True).fillna(na)), [na, 3.542, 3.542, 3.542])
        self.assertEqual(list(ch.get_species_stick_coef(special=True).fillna(na)), [na, 0., 1., 1.])
        self.assertEqual(list(ch.get_species_ret_coefs(special=True).fillna(na)), [na, {}, {'Ar': 1.}, {'Ar': 1.}])

        self.assertEqual(list(ch.get_species_name(disabled=True)), ['Ar', 'Ar+', 'Ar*', 'Ar2+'])
        self.assertEqual(list(ch.get_species_charge(disabled=True)), [0, 1, 0, 1])
        self.assertEqual(list(ch.get_species_mass(disabled=True)), [39.948, 39.948, 39.948, 79.896])
        self.assertEqual(list(ch.get_species_h_form(disabled=True)), [0., 0., 0., 0.])
        self.assertEqual(list(ch.get_species_lj_epsilon(disabled=True)), [93.3, 93.3, 93.3, 93.3])
        self.assertEqual(list(ch.get_species_lj_sigma(disabled=True)), [3.542, 3.542, 3.542, 3.54])
        self.assertEqual(list(ch.get_species_stick_coef(disabled=True)), [0., 1., 1., 1.])
        self.assertEqual(list(ch.get_species_ret_coefs(disabled=True)), [{}, {'Ar': 1.}, {'Ar': 1.}, {'Ar': 2.}])

        self.assertEqual(list(ch.get_species_name(special=True, disabled=True)), ['e', 'Ar', 'Ar+', 'Ar*', 'Ar2+', 'M'])
        self.assertEqual(list(ch.get_species_charge(special=True, disabled=True).fillna(na)), [-1, 0, 1, 0, 1, na])
        self.assertEqual(list(ch.get_species_mass(special=True, disabled=True).fillna(na)),
                         [5.46e-4, 39.948, 39.948, 39.948, 79.896, na])
        self.assertEqual(list(ch.get_species_h_form(special=True, disabled=True).fillna(na)), [na, 0., 0., 0., 0., na])
        self.assertEqual(list(ch.get_species_lj_epsilon(special=True, disabled=True).fillna(na)),
                         [na, 93.3, 93.3, 93.3, 93.3, na])
        self.assertEqual(list(ch.get_species_lj_sigma(special=True, disabled=True).fillna(na)),
                         [na, 3.542, 3.542, 3.542, 3.54, na])
        self.assertEqual(list(ch.get_species_stick_coef(special=True, disabled=True).fillna(na)),
                         [na, 0., 1., 1., 1., na])
        self.assertEqual(list(ch.get_species_ret_coefs(special=True, disabled=True).fillna(na)),
                         [na, {}, {'Ar': 1.}, {'Ar': 1.}, {'Ar': 2.}, na])

    def test_species_attributes(self):
        ch = self.test_chem_01
        ch._enabled_sp.at[293] = False  # ignore M
        # check if the prettypring is working without raising exceptions:
        attribs = ch.species_attributes(printout=False)
        self.assertEqual(attribs.shape, (6, 13))
        attribs = ch.species_attributes(printout=False, units=True)
        self.assertEqual(attribs.shape, (6, 18))  # adding units for 5 attributes that define them.
        ch._enabled_sp.at[3] = False  # ignore Ar*
        attribs = ch.species_attributes(special=False, disabled=False, printout=False, pretty_print=False, sort=True)
        self.assertEqual(attribs.shape, (3, 13))
        ch._enabled_sp.at[293] = True  # reset disabled M
        ch._enabled_sp.at[3] = True  # reset disabled Ar*

        # check the disabled return species gets removed from ret coefs:
        ch.get_species()[35].set_species_attribute('ret_coefs', {'Ar*': 1.0})  # set different return species to Ar+
        attribs = ch.species_attributes(special=True, disabled=True, printout=False, pretty_print=False)
        self.assertEqual(attribs['ret_coefs'][35], {'Ar*': 1.0})
        ch.disable(species=['Ar*'])
        attribs = ch.species_attributes(special=True, disabled=True, printout=False, pretty_print=False)
        self.assertEqual(attribs['ret_coefs'][35], {})
        ch.reset()
        ch.get_species()[35].set_species_attribute('ret_coefs', {'Ar': 1.0})  # revert back

    def test_reactions_attributes_getters(self):

        ch = self.test_chem_01
        tested_methods = [
            ch.get_reactions_arrh_a, ch.get_reactions_arrh_b, ch.get_reactions_arrh_c,
            ch.get_reactions_el_en_loss, ch.get_reactions_ges_heat_contrib, ch.get_reactions_elastic
        ]

        # only test lengths of the returned arrays
        for getter in tested_methods:
            self.assertEqual(len(getter()), 11)
        for getter in tested_methods:
            self.assertEqual(len(getter(disabled=True)), 11)
        ch._enabled_r[[2539, 28, 13677, 2455]] = False  # ignore the last 4 reactions
        for getter in tested_methods:
            self.assertEqual(len(getter()), 7)
        for getter in tested_methods:
            self.assertEqual(len(getter(disabled=True)), 11)
        ch._enabled_r[[2539, 28, 13677, 2455]] = True

        # test correct copy level of the returned arrays:
        for getter in tested_methods:
            res_array = getter()
            res_array.iat[0] = 42
            self.assertTrue(res_array.iat[0] in {42, '42', True})  # depending on the dtype of the array...
            self.assertNotEqual(getter().iat[0], 42)

    def test_reactions_attributes(self):
        ch = self.test_chem_01
        ch._enabled_r[[2539, 28, 13677, 2455]] = False  # ignore the last 4 reactions
        attribs = ch.reactions_attributes(printout=False, sort=True, classify=True)
        self.assertEqual(attribs.shape, (11+10, 12))  # classify adds 10 lines of type and categories
        self.assertEqual(
            ch.reactions_attributes(disabled=False, printout=False, pretty_print=False).shape, (7, 12)
        )
        # columns for units:
        self.assertEqual(
            ch.reactions_attributes(disabled=False, printout=False, pretty_print=False, units=True).shape, (7, 12+4)
        )
        ch._enabled_r[[2539, 28, 13677, 2455]] = True

    def test_return_matrix(self):
        ch = self.test_chem_01
        # test correct copy level
        ret_mat = ch.get_return_matrix()
        ret_mat.iat[0, 0] = 42
        self.assertEqual(ret_mat.iat[0, 0], 42)
        self.assertNotEqual(ch.get_return_matrix().iat[0, 0], 42)
        # test the shapes
        ch._enabled_r[[2539, 28, 13677, 2455]] = False  # disable the last 4 reactions
        ch._enabled_sp[[36, 293]] = False  # disable 'M and 'Ar2+'
        self.assertEqual(ch.get_return_matrix().shape, (3, 3))
        self.assertEqual(ch.get_return_matrix(disabled=True).shape, (4, 4))
        self.assertEqual(ch.get_return_matrix(special=True).shape, (4, 4))
        self.assertEqual(ch.get_return_matrix(special=True, disabled=True).shape, (6, 6))
        # reset disabled species and reactions
        ch._enabled_r[[2539, 28, 13677, 2455]] = True
        ch._enabled_sp[[36, 293]] = True

    def test_stoichiomatrix(self):
        ch = self.test_chem_01
        # test correct copy level
        stoich_mat = ch.get_stoichiomatrix()
        stoich_mat.iat[0, 0] = 42
        self.assertEqual(stoich_mat.iat[0, 0], 42)
        self.assertNotEqual(ch.get_stoichiomatrix().iat[0, 0], 42)
        # test the shapes
        ch._enabled_r[[2539, 28, 13677, 2455]] = False  # disable the last 4 reactions
        ch._enabled_sp[[36, 293]] = False  # disable 'M and 'Ar2+'
        self.assertEqual(ch.get_stoichiomatrix().shape, (7, 3))
        self.assertEqual(ch.get_stoichiomatrix(disabled=True).shape, (11, 4))
        self.assertEqual(ch.get_stoichiomatrix(special=True).shape, (7, 4))
        self.assertEqual(ch.get_stoichiomatrix(special=True, disabled=True).shape, (11, 6))
        # reset disabled species and reactions
        ch._enabled_r[[2539, 28, 13677, 2455]] = True
        ch._enabled_sp[[36, 293]] = True
        # test lhs, rhs, net:
        stoich_mat_lhs = ch.get_stoichiomatrix(method='lhs')
        stoich_mat_rhs = ch.get_stoichiomatrix(method='rhs')
        stoich_mat_net = ch.get_stoichiomatrix(method='net')
        self.assertTrue(list(stoich_mat_lhs.index) == list(stoich_mat_rhs.index) == list(stoich_mat_net.index))
        self.assertEqual(list(stoich_mat_net), list(stoich_mat_rhs - stoich_mat_lhs))

    def test_adhoc_attributes(self):
        # adhoc changes need to be reflected in the chemistry-level getters but cannot change the attributes of the
        # RP and Reaction instances
        ch = self.test_chem_01
        self.assertEqual(ch.get_species_stick_coef().at[35], 1.)
        self.assertEqual(ch.get_species().at[35].get_stick_coef(), 1.)
        ch.set_adhoc_species_attributes('stick_coef', 35, 0.42)
        self.assertEqual(ch.get_species().at[35].get_stick_coef(), 1.)
        self.assertEqual(ch.get_species_stick_coef().at[35], 0.42)
        ch.set_adhoc_species_attributes('ret_coefs', 35, {'Ar*': 1.})
        self.assertEqual(ch.get_species_ret_coefs().at[35], {'Ar*': 1.})
        with self.assertRaises(RPAttributeError):  # unsupported attribute
            ch.set_adhoc_species_attributes('unsupported', 35, 1.)
        with self.assertRaises(RPAttributeError):  # unsupported attribute value type
            ch.set_adhoc_species_attributes('stick_coef', 35, [1., ])
        with self.assertRaises(ChemistrySpeciesNotPresentError):  # species with such an id not in the chemistry...
            ch.set_adhoc_species_attributes('stick_coef', 999, 1.)

        self.assertEqual(ch.get_reactions_arrh_a().at[2539], 1.)
        self.assertEqual(ch.get_reactions().at[2539].get_arrh_a(), 1.)
        ch.set_adhoc_reactions_attributes('arrh_a', 2539, 42)
        self.assertEqual(ch.get_reactions().at[2539].get_arrh_a(), 1.)
        self.assertEqual(ch.get_reactions_arrh_a().at[2539], 42)
        with self.assertRaises(ReactionAttributeError):  # unsupported attribute
            ch.set_adhoc_reactions_attributes('unsupported', 2539, 1.)
        with self.assertRaises(ReactionAttributeError):  # unsupported attribute value type
            ch.set_adhoc_reactions_attributes('arrh_a', 2539, [1., ])
        with self.assertRaises(ChemistryReactionNotPresentError):  # species with such an id not in the chemistry...
            ch.set_adhoc_reactions_attributes('arrh_a', 999, 1.)

        # resetting:
        ch.reset_adhoc_species_attributes('ret_coefs')
        self.assertNotEqual(ch.get_species_stick_coef().at[35], ch.get_species().at[35].get_stick_coef())
        self.assertEqual(ch.get_species_ret_coefs().at[35], ch.get_species().at[35].get_ret_coefs())
        self.assertNotEqual(ch.get_reactions_arrh_a().at[2539], ch.get_reactions().at[2539].get_arrh_a())
        ch.reset_adhoc_species_attributes('all')
        ch.reset_adhoc_reactions_attributes('all')
        self.assertEqual(ch.get_species_stick_coef().at[35], ch.get_species().at[35].get_stick_coef())
        self.assertEqual(ch.get_species_ret_coefs().at[35], ch.get_species().at[35].get_ret_coefs())
        self.assertEqual(ch.get_reactions_arrh_a().at[2539], ch.get_reactions().at[2539].get_arrh_a())
        with self.assertRaises(RPAttributeError):
            ch.reset_adhoc_species_attributes('unsupported')
        with self.assertRaises(ReactionAttributeError):
            ch.reset_adhoc_reactions_attributes('unsupported')

        # check that the getters using ad-hoc attributes do not change the index of the returned arrays:
        # disable Ar2+ and M:
        ch._enabled_sp[[36, 293]] = False
        ch._enabled_r[[28, 13677, 2455]] = False
        stick_coef_1 = ch.get_species_stick_coef()
        # change stick coef of an disabled species:
        ch.set_adhoc_species_attributes('stick_coef', 36, 0.42)  # changing stick coef of Ar2+
        self.assertEqual(list(stick_coef_1.index), list(ch.get_species_stick_coef().index))

        # check that setting an invalid ret_coefs will trigger the consistency error
        with self.assertRaises(ChemistryConsistencyError):
            ch.set_adhoc_species_attributes('ret_coefs', 35, {'Ar2+': 1.})
        self.assertEqual(ch.get_species_ret_coefs().at[35], {'Ar': 1.})

        # check the units with setting ad-hoc arrh_a:
        ch.set_adhoc_reactions_attributes('arrh_a', 3, 1.e-42)  # set adhoc coefficient in nominal units (cm3.s-1)
        self.assertEqual(ch.get_reactions_arrh_a(si_units=False)[3], 1.e-42)
        self.assertEqual(ch.get_reactions_arrh_a()[3], 1.e-48)
        # hack the unit directly in the Reaction instance:
        r = ch.get_reaction(3)
        r._units['arrh_a'] = 'm3.s-1'
        self.assertEqual(ch.get_reactions_arrh_a(si_units=False)[3], 1.e-42)
        self.assertEqual(ch.get_reactions_arrh_a()[3], 1.e-42)
        # reset all changes:
        r._units['arrh_a'] = 'cm3.s-1'
        ch.reset_adhoc_reactions_attributes('arrh_a')

        # test going in circles:
        for attr in ch.adhoc_species_attributes:
            ch.reset_adhoc_species_attributes(attr)
            nominal_values = getattr(ch, 'get_species_{}'.format(attr))()
            for sp_id in nominal_values.index:
                ch.set_adhoc_species_attributes(attr, sp_id, nominal_values[sp_id])
            new_values = getattr(ch, 'get_species_{}'.format(attr))()
            self.assertTrue((nominal_values == new_values).all())
        for attr in ch.adhoc_reactions_attributes:
            ch.reset_adhoc_reactions_attributes(attr)
            nominal_values = getattr(ch, 'get_reactions_{}'.format(attr))()
            if attr == 'arrh_a':
                nominal_values_nominal_units = getattr(ch, 'get_reactions_{}'.format(attr))(si_units=False)
            else:
                nominal_values_nominal_units = getattr(ch, 'get_reactions_{}'.format(attr))()
            for r_id in nominal_values_nominal_units.index:
                ch.set_adhoc_reactions_attributes(attr, r_id, nominal_values_nominal_units[r_id])
            new_values = getattr(ch, 'get_reactions_{}'.format(attr))()
            self.assertTrue((nominal_values == new_values).all())

    def test_adhoc_ret_coefs(self):
        # build a test chemistry with empty ret coefs:
        e = RP(rp_id=0, name='e', charge=-1, mass=0.00055)
        ar = RP(rp_id=1, name='Ar', charge=0, mass=39.95, stick_coef=0.0, ret_coefs={})
        ar_i = RP(rp_id=2, name='Ar+', charge=1, mass=39.95, stick_coef=1.0, ret_coefs={})
        reactions = [
            Reaction(r_id=1, reactants=(e, ar), products=(e, e, ar_i), special_number=-1),
            Reaction(r_id=2, reactants=(e, ar_i), products=(ar, ), special_number=-2),
            Reaction(r_id=3, reactants=(ar, ar), products=(ar, ar_i, e), special_number=-3)
        ]
        chem = Chemistry([e, ar, ar_i], reactions)

        # check that the return matrix is empty
        self.assertEqual(list(chem.get_species_ret_coefs()), [{}, {}])
        self.assertFalse(chem.get_return_matrix().values.any())

        # set some adhoc return coefficients and check that the return matrix is not empty:
        chem.set_adhoc_species_attributes('ret_coefs', ar_i, {'Ar': 1.0})
        self.assertEqual(ar_i.get_ret_coefs(), {})
        self.assertEqual(list(chem.get_species_ret_coefs()), [{}, {'Ar': 1.0}])
        self.assertTrue(chem.get_return_matrix().values.any())

        # build a test chemistry with non-empty ret coefs:
        ar_i.set_species_attribute('ret_coefs', {'Ar': 1.0})

        # check that the return matrix is NOT empty
        self.assertEqual(list(chem.get_species_ret_coefs()), [{}, {'Ar': 1.0}])
        self.assertTrue(chem.get_return_matrix().values.any())

        # set adhoc ret coefs to {} and check the return matrix is empty:
        chem.set_adhoc_species_attributes('ret_coefs', ar_i, {})
        self.assertEqual(ar_i.get_ret_coefs(), {'Ar': 1.0})
        self.assertEqual(list(chem.get_species_ret_coefs()), [{}, {}])
        self.assertFalse(chem.get_return_matrix().values.any())

    def test_attributes_dump(self):
        xml_path_04 = os.path.join(self.xml_resources, 'chem04.xml')
        xml_path_05 = os.path.join(self.xml_resources, 'chem05.xml')
        ch0 = Chemistry(*XmlParser().get_species_and_reactions(xml_path_04), xml_path=xml_path_04)

        # tweak the chemistry attributes:
        ch0.set_protected_species('Ar', 8)  # setting Ar and H2 as protected species
        ch0.disable(species=('Ar**', 13), reactions=(1, 2, 3))  # ignore Ar** and ArH+ and first three reactions
        ch0.set_adhoc_species_attributes('stick_coef', 8, 1.0)  # change species adhoc param
        ch0.set_adhoc_species_attributes('ret_coefs', 8, {'H': 2.0})  # change species adhoc param
        ch0.set_adhoc_species_attributes('ret_coefs', 'H+', {})  # change species adhoc param
        ch0.set_adhoc_species_attributes('stick_coef', 'H+', 0)  # change species adhoc param
        ch0.set_adhoc_reactions_attributes('arrh_a', 4, 1.e-42)  # change reaction adhoc param
        ch0.set_adhoc_reactions_attributes('arrh_a', 5, 1.e-42)  # change reaction adhoc param

        # dump the chemistry attributes into a .yaml file
        self.make_temp_dir()
        yaml_path = os.path.join(self.temp_dir, 'ch0_dump.yaml')
        ch0.dump_chemistry_attributes(yaml_path)

        # trying to load attributes to chemistry with different xml_path:
        ch1 = Chemistry(*XmlParser().get_species_and_reactions(xml_path_04))
        with self.assertRaises(ChemistryAttributeError):
            ch1.load_chemistry_attributes(yaml_path)
        # trying to load attributes to chemistry with different xml file content (hash):
        ch1 = Chemistry(*XmlParser().get_species_and_reactions(xml_path_04), xml_path=xml_path_05)
        with self.assertRaises(ChemistryAttributeError):
            ch1.load_chemistry_attributes(yaml_path)

        # test if the attribute load actually works:
        ch1 = Chemistry(*XmlParser().get_species_and_reactions(xml_path_04), xml_path=xml_path_04)
        self.assertNotEqual(ch0._protected_sp.at[1], ch1._protected_sp.at[1])
        self.assertNotEqual(ch0._enabled_sp.at[13], ch1._enabled_sp.at[13])
        self.assertNotEqual(ch0._enabled_r.at[1], ch1._enabled_r.at[1])
        self.assertNotEqual(ch0.get_species_stick_coef(disabled=True)[8], ch1.get_species_stick_coef(disabled=True)[8])
        self.assertNotEqual(ch0.get_species_ret_coefs()[9], ch1.get_species_ret_coefs()[9])
        self.assertNotEqual(ch0.get_reactions_arrh_a()[4], ch1.get_reactions_arrh_a()[4])
        ch1.load_chemistry_attributes(yaml_path)
        self.assertEqual(ch0._protected_sp.at[1], ch1._protected_sp.at[1])
        self.assertEqual(ch0._enabled_sp.at[13], ch1._enabled_sp.at[13])
        self.assertEqual(ch0._enabled_r.at[1], ch1._enabled_r.at[1])
        self.assertEqual(ch0.get_species_stick_coef(disabled=True)[8], ch1.get_species_stick_coef(disabled=True)[8])
        self.assertEqual(ch0.get_species_ret_coefs()[9], ch1.get_species_ret_coefs()[9])
        self.assertEqual(ch0.get_reactions_arrh_a()[4], ch1.get_reactions_arrh_a()[4])

        # dump chemistry without an xml path to a yaml file:
        ch0 = Chemistry(*XmlParser().get_species_and_reactions(xml_path_04))
        ch0.dump_chemistry_attributes(yaml_path)
        with self.assertRaises(ChemistryAttributeError):
            ch1.load_chemistry_attributes(yaml_path)

        self.remove_temp_dir()

    def test_disable(self):
        ch = self.test_chem_01
        ch.disable(species=(36, 293))  # disabling Ar2+ and M:
        self.assertEqual(list(ch.get_species(special=True).index), [1, 2, 35, 3])
        self.assertEqual(list(ch.get_reactions().index), [3, 48, 13, 4, 9, 7, 11, 2539])

        # disabling Ar on top of Ar2+ and M should only leave two reactions and remove Ar from return coefs:
        self.assertEqual(ch.get_species_ret_coefs().at[35], {'Ar': 1.})
        self.assertEqual(ch.get_species_ret_coefs().at[3], {'Ar': 1.})
        ch.disable(species=('Ar', ))
        self.assertEqual(list(ch.get_species(special=True).index), [1, 35, 3])
        self.assertEqual(list(ch.get_reactions().index), [48, 11])
        self.assertEqual(ch.get_species_ret_coefs().at[35], {})
        self.assertEqual(ch.get_species_ret_coefs().at[3], {})

        # disabling the Ar+ should be prohibited, since it's the last positive ion:
        with self.assertRaises(ChemistryDisableError):
            ch.disable(species=('Ar+', ))
        # the same with electron:
        with self.assertRaises(ChemistryDisableError):
            ch.disable(species=('e', ))
        # check that e and Ar+ not disabled:
        self.assertTrue(35 not in ch.get_disabled_species().index)
        self.assertTrue(1 not in ch.get_disabled_species().index)
        # disabling all but one reactions with Ar+:
        ch.disable(reactions=(48, 13, 7))
        with self.assertRaises(ChemistryDisableError):
            ch.disable(reactions=(11, ))

        # reseting the chemistry:
        ch.reset()
        self.assertEqual(list(ch.get_species(special=True).index), [1, 2, 35, 3, 36, 293])
        self.assertEqual(list(ch.get_reactions().index), [3, 48, 13, 4, 9, 7, 11, 2539, 28, 13677, 2455])

        # disabling the Ar2+ should also disable the M:
        ch.disable(species=('Ar2+', ))
        self.assertEqual(list(ch.get_disabled_species().index), [36, 293])

        # disabling both positive ions at the same time:
        ch.reset()
        with self.assertRaises(ChemistryDisableError):
            ch.disable(species=(35, 36))
        with self.assertRaises(ChemistryDisableError):
            ch.disable(species=(35,), reactions=(28, 13677, 2455))

        # disabling protected species:
        ch.reset()
        ch.set_protected_species('Ar*')
        with self.assertRaises(ChemistryDisableError):
            ch.disable(reactions=(4, 9, 11, 2539))
        self.assertEqual(list(ch.get_disabled_species().index), [])

        # test once more the return species alteration when disabling:
        ch.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar': 0.5, 'Ar*': 0.5})
        self.assertEqual(ch.get_species_ret_coefs().at[35], {'Ar': 0.5, 'Ar*': 0.5})
        ch.disable(species=('Ar', ))
        self.assertEqual(ch.get_species_ret_coefs().at[35], {'Ar*': 0.5})

    def test_reactions_stoich_coefs(self):
        ch = self.test_chem_01
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron()), [0, 0, -1, 0, 0, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_arbitrary()), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron(method='lhs')), [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_arbitrary(method='lhs')), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron(method='rhs')), [1, 1, 0, 1, 1, 2, 2, 0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_arbitrary(method='rhs')), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        ch.disable(species=['Ar*', 'Ar2+', 'M'])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron()), [0, 0, -1, 1])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_arbitrary()), [0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron(method='lhs')), [1, 1, 1, 1])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_arbitrary(method='rhs')), [0, 0, 0, 0])
        self.assertEqual(list(ch.get_reactions_stoich_coefs_electron(disabled=True)), [0, 0, -1, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_copy(self):
        chem = self.test_chem_01
        ch1 = chem.copy()
        ch2 = chem.copy()

        self.assertEqual(list(ch1.get_species().index), list(ch2.get_species().index))
        self.assertEqual(list(ch1.get_reactions().index), list(ch2.get_reactions().index))
        self.assertEqual(list(ch1._protected_sp.index), list(ch2._protected_sp.index))
        self.assertEqual(ch1.get_species_stick_coef()[2], ch2.get_species_stick_coef()[2])
        self.assertEqual(ch1.get_species_ret_coefs()[2], ch2.get_species_ret_coefs()[2])
        self.assertEqual(ch1.get_reactions_arrh_a()[11], ch2.get_reactions_arrh_a()[11])

        ch1.disable(species=('Ar2+', ))
        ch1.set_protected_species('Ar*')
        ch1.set_adhoc_species_attributes('stick_coef', 'Ar', 1.0)
        ch1.set_adhoc_species_attributes('ret_coefs', 'Ar', {'Ar+': 0.5})
        ch1.set_adhoc_reactions_attributes('arrh_a', 11, 42)

        self.assertNotEqual(list(ch1.get_species().index), list(ch2.get_species().index))
        self.assertNotEqual(list(ch1.get_reactions().index), list(ch2.get_reactions().index))
        self.assertNotEqual(list(ch1._protected_sp.values), list(ch2._protected_sp.values))
        self.assertNotEqual(ch1.get_species_stick_coef()[2], ch2.get_species_stick_coef()[2])
        self.assertNotEqual(ch1.get_species_ret_coefs()[2], ch2.get_species_ret_coefs()[2])
        self.assertNotEqual(ch1.get_reactions_arrh_a()[11], ch2.get_reactions_arrh_a()[11])

    def test_coherence_check(self):
        chem = self.test_chem_01
        chem_copy = self.test_chem_01.copy()
        self.make_temp_dir()
        yaml_path = os.path.join(self.temp_dir, 'chemistry_attributes.yaml')
        chem.dump_chemistry_attributes(yaml_path)

        self.assertTrue(chem.check_coherence(yaml_path))
        self.assertTrue(chem_copy.check_coherence(yaml_path))

        chem.disable(species=('Ar2+', ))
        self.assertFalse(chem.check_coherence(yaml_path))
        self.assertTrue(chem_copy.check_coherence(yaml_path))
        chem.reset()
        self.assertTrue(chem.check_coherence(yaml_path))
        chem.disable(reactions=(48, ))
        self.assertFalse(chem.check_coherence(yaml_path))
        self.assertTrue(chem_copy.check_coherence(yaml_path))
        chem.reset()
        self.assertTrue(chem.check_coherence(yaml_path))
        chem.set_adhoc_reactions_attributes('arrh_a', 48, 42.0)
        self.assertFalse(chem.check_coherence(yaml_path))
        self.assertTrue(chem_copy.check_coherence(yaml_path))
        chem.reset_adhoc_reactions_attributes('arrh_a')
        self.assertTrue(chem.check_coherence(yaml_path))
        chem.set_adhoc_species_attributes('stick_coef', 'Ar', 1.0)
        self.assertFalse(chem.check_coherence(yaml_path))
        self.assertTrue(chem_copy.check_coherence(yaml_path))
        chem.reset_adhoc_species_attributes('all')
        self.assertTrue(chem.check_coherence(yaml_path))

        chem.disable(species=('Ar2+', ), reactions=(48, ))
        chem.set_adhoc_reactions_attributes('arrh_a', 48, 42.0)
        chem.set_adhoc_species_attributes('stick_coef', 'Ar', 1.0)
        chem.dump_chemistry_attributes(yaml_path)

        self.assertTrue(chem.check_coherence(yaml_path))
        self.assertFalse(chem_copy.check_coherence(yaml_path))

        chem_copy = chem.copy()
        self.assertTrue(chem_copy.check_coherence(yaml_path))

        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
