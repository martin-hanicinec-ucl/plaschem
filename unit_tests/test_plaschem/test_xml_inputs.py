
import unittest
import os
import shutil

from plaschem.classification import ReactionClassifier
from plaschem.chemistry import Chemistry
from plaschem.xml_inputs import XmlParser, XmlBuilder
from plaschem.exceptions import XmlError
from unit_tests.test_plaschem.utils import draw_test_species, draw_test_reactions, draw_test_chemistry


class TestXmlLoader(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))
    xml_resources = os.path.join(context, 'resources', 'xml_files')
    xml_path = os.path.join(xml_resources, 'chem04.xml')

    def test_loader(self):
        xml_parser = XmlParser()

        species, reactions = xml_parser.get_species_and_reactions(xml_path=self.xml_path)
        # check both are ordered itterables:
        try:
            sp0_id = species[0].id
            r0_id = reactions[0].id
        except TypeError:
            self.fail('Species or reactions returned appear not to be ordered iterables!')

        # check lens of both arrays:
        self.assertEqual(len(species), 14)
        self.assertEqual(len(reactions), 64)


class TestXmlBuilder(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))
    xml_resources = os.path.join(context, 'resources', 'xml_files')

    def make_temp_dir(self):
        self.temp_dir = os.path.join(self.context, 'resources', '.temp')
        if os.path.isdir(self.temp_dir):
            raise ValueError('Temporary directory already exists!')
        else:
            os.mkdir(self.temp_dir)

    def remove_temp_dir(self):
        shutil.rmtree(self.temp_dir)

    def get_dummy_qdb_rp(self, ordinary_formula, states_strs=()):
        class State:
            def __init__(self, state_str):
                self.state_str = state_str

        class States:
            def __init__(self, strs):
                self.states = [State(state_str) for state_str in strs]

            def all(self):
                return self.states

        class QdbRP:
            charge, hform, lj_epsilon, lj_sigma, hpem, pk = None, None, None, None, None, 42
            mass = 1.

            def __init__(self):
                self.ordinary_formula = ordinary_formula
                self.state = States(states_strs)

        return QdbRP()

    def get_dummy_qdb_reaction(self, reactants, products):
        class Citations:
            def __init__(self):
                self.doi = "It's a trap!"

            def all(self):
                return [self, ]

        class RPSet:
            def __init__(self, rp_set):
                self.rp_set = rp_set

            def all(self):
                return self.rp_set

        class QdbDS:
            entry_id = -42
            pk = 42
            citations = Citations()

            def __init__(self):
                pass

            def all(self):
                return [self, ]

            @staticmethod
            def is_xsec():
                return True  # means will not look for arrhenius parameters, but will call fit_arrhenius()

            @staticmethod
            def is_arrhenius():
                return True  # means will not look for el_en_loss in the datatable, but instead will return arrh_c

            @staticmethod
            def fit_arrhenius():
                return (0.042, 0.), (0.42, 0.), (4.2, 0.)

        class ParentReaction:
            @ staticmethod
            def all():
                return []

        class QdbReaction:
            pk = 4242

            def __init__(self):
                self.reactants = RPSet(reactants)
                self.products = RPSet(products)
                self.reactiondataset_set = QdbDS()
                self.parent_reaction = ParentReaction()  # need to have .all() method returning empty iterable
                self.child_reactions = ParentReaction()  # need to have .all() method returning empty iterable

            def get_all_rps(self):
                return self.reactants.rp_set + self.products.rp_set

        return QdbReaction()

    def setUp(self):
        self.xml_builder = XmlBuilder()

    def test_qdb_renaming_conventions(self):
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='e-')
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'e')
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='Ar+2')
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'Ar++')

        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='M')
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'M')
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='Ar')
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'Ar')
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='Ar', states_strs=['*'])
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'Ar*')
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='Ar', states_strs=['*', '**'])
        self.assertEqual(self.xml_builder._get_rp_name_from_qdb(dummy_qdb_rp), 'Ar(*,**)')

    def test_add_rp(self):
        self.xml_builder.add_rp_from_attributes(rp_id=1, name='Ar')
        self.assertEqual(self.xml_builder.species[-1].attrib['id'], '1')
        self.xml_builder.add_rp_from_attributes(name='Ar+', mass=1., ret_coefs={'Ar': 1.})
        self.assertEqual(self.xml_builder.species[-1].attrib['id'], '999')
        with self.assertRaises(XmlError):
            self.xml_builder.add_rp_from_attributes(name='Ar', mass=1., ret_coefs={'Ar+': 1})  # re-adding the same rp
        self.xml_builder.add_rp_from_instance(draw_test_species('Ar*')[0])  # add Ar* from instance
        self.assertEqual(self.xml_builder.species[-1].attrib['id'], '3')  # check if the RP id was propagated
        with self.assertRaises(XmlError):
            self.xml_builder.add_rp_from_instance(draw_test_species('Ar')[0])  # re-add Ar this thime from instance
        dummy_qdb_rp = self.get_dummy_qdb_rp(ordinary_formula='C4F8', states_strs=['*', '**'])
        self.xml_builder.add_rp_from_qdb(dummy_qdb_rp)
        self.assertIn('C4F8(*,**)', self.xml_builder.species_dict)  # that should be the dummy qdb rp name
        self.assertEqual(self.xml_builder.species[-1].attrib['id'], '42')  # check the RP.pk was propagated as id

        self.assertEqual(len(self.xml_builder.species), 4)
        self.assertEqual(len(self.xml_builder.species_dict), 4)
        self.assertEqual(len(self.xml_builder.reactions), 0)

    def test_add_reaction(self):
        self.assertEqual(len(self.xml_builder.species), 0)
        self.assertEqual(len(self.xml_builder.species_dict), 0)
        self.assertEqual(len(self.xml_builder.reactions), 0)
        self.xml_builder.add_reaction_from_instance(draw_test_reactions(382)[0])  # add e + H2 > e + H + H* from inst.
        self.assertEqual(self.xml_builder.reactions[-1].attrib['id'], '382')  # check the Reaction id was propagated
        self.assertEqual(len(self.xml_builder.species), 4)
        self.assertEqual(len(self.xml_builder.species_dict), 4)
        self.assertEqual(len(self.xml_builder.reactions), 1)
        self.xml_builder.add_reaction_from_attributes(('H*', 'e'), ('H', 'e'))  # addig from attributes
        self.assertEqual(self.xml_builder.reactions[-1].attrib['id'], '999')  # check the Id was assigned 999
        self.assertEqual(len(self.xml_builder.species), 4)
        self.assertEqual(len(self.xml_builder.species_dict), 4)
        self.assertEqual(len(self.xml_builder.reactions), 2)
        # test adding from QDB: very limited test, constructing crude dummy QDB Reaction class...
        dummy_qdb_reaction = self.get_dummy_qdb_reaction(
            (self.get_dummy_qdb_rp('e-'), self.get_dummy_qdb_rp('e'), self.get_dummy_qdb_rp('H2')),
            (self.get_dummy_qdb_rp('H2-2'), )
        )  # added 1 new reaction (double attachment :)) and one new specie (H2--)
        self.xml_builder.add_reaction_from_qdb(dummy_qdb_reaction)
        self.assertEqual(self.xml_builder.reactions[-1].attrib['id'], '4242')  # check the Reaction.pk saved as ID
        self.assertEqual(len(self.xml_builder.species), 5)
        self.assertEqual(len(self.xml_builder.species_dict), 5)
        self.assertEqual(len(self.xml_builder.reactions), 3)

    def test_add_chemistry(self):
        ch04 = draw_test_chemistry(4)
        # add reactions from chemistry instance:
        self.xml_builder.add_chemistry_from_instance(ch04)
        ch04_species, ch04_reactions = ch04.get_species(special=True), ch04.get_reactions()
        self.assertEqual(len(self.xml_builder.species), len(ch04_species))
        self.assertEqual(len(self.xml_builder.species_dict), len(ch04_species))
        self.assertEqual(len(self.xml_builder.reactions), len(ch04_reactions))
        # add species and reactions from previously saved xml of chemistry 6:
        ch6_xml_path = os.path.join(self.xml_resources, 'chem06.xml')
        self.xml_builder.add_chemistry_from_xml(ch6_xml_path)
        ch06_species, ch06_reactions = XmlParser().get_species_and_reactions(ch6_xml_path)
        # chemistry04 and chemistry06 only share electron
        self.assertEqual(len(self.xml_builder.species), len(ch04_species) + len(ch06_species) - 1)
        self.assertEqual(len(self.xml_builder.species_dict), len(ch04_species) + len(ch06_species) - 1)
        self.assertEqual(len(self.xml_builder.reactions), len(ch04_reactions) + len(ch06_reactions))
        # I'm not goiong to unit_test adding QDB Chemistry... That's too much work with coding dummy classes...

    def test_assign_ids(self):
        ch04 = draw_test_chemistry(4)
        # add reactions from chemistry instance:
        self.xml_builder.add_chemistry_from_instance(ch04)
        old_sp_ids = [sp_xml.attrib['id'] for sp_xml in self.xml_builder.species]
        old_r_ids = [r_xml.attrib['id'] for r_xml in self.xml_builder.reactions]
        self.xml_builder.assign_ids()
        new_sp_ids = [sp_xml.attrib['id'] for sp_xml in self.xml_builder.species]
        new_r_ids = [r_xml.attrib['id'] for r_xml in self.xml_builder.reactions]
        self.assertNotEqual(old_sp_ids, new_sp_ids)
        self.assertNotEqual(old_r_ids, new_r_ids)
        # check they are both ascending (species from 0(e) and reactions from 1)
        self.assertEqual(new_sp_ids, [str(sp_id) for sp_id in range(len(self.xml_builder.species))])
        self.assertEqual(new_r_ids, [str(r_id) for r_id in range(1, len(self.xml_builder.reactions) + 1)])

    def test_reset_chemistry(self):
        self.assertEqual(len(self.xml_builder.species), 0)
        self.assertEqual(len(self.xml_builder.species_dict), 0)
        self.assertEqual(len(self.xml_builder.reactions), 0)
        self.xml_builder.add_reaction_from_instance(draw_test_reactions(382)[0])
        self.assertNotEqual(len(self.xml_builder.species), 0)
        self.assertNotEqual(len(self.xml_builder.species_dict), 0)
        self.assertNotEqual(len(self.xml_builder.reactions), 0)
        self.xml_builder.reset_chemistry()
        self.assertEqual(len(self.xml_builder.species), 0)
        self.assertEqual(len(self.xml_builder.species_dict), 0)
        self.assertEqual(len(self.xml_builder.reactions), 0)

    def test_sort_elements(self):
        chem04 = draw_test_chemistry(4)
        reactions = chem04.get_reactions()
        for r in reactions:
            self.xml_builder.add_reaction_from_instance(r)

        reaction_classifier = ReactionClassifier(chem04.get_species(special=True), chem04.get_reactions())
        reaction_classifier.sort_species()
        sorted_sp_names = [sp.get_name() for sp in reaction_classifier.species]
        reaction_classifier.sort_reactions()

        xml_sp_names = [rp_xml.find('name').text for rp_xml in self.xml_builder.species]

        self.assertNotEqual(xml_sp_names, sorted_sp_names)

        self.xml_builder.sort_elements()
        xml_sp_names = [rp_xml.find('name').text for rp_xml in self.xml_builder.species]

        self.assertEqual(xml_sp_names, sorted_sp_names)  # checks if the species are sorted same way as in ReactionsCls.

        # not going to test for sorting of reactions - too much work since ids differ and no other label to define eq.

    def test_dump_xml(self):
        ch = draw_test_chemistry(4)
        self.xml_builder.add_chemistry_from_instance(ch)
        self.xml_builder.sort_elements()

        self.make_temp_dir()

        xml_path = os.path.join(self.temp_dir, 'test_chemistry.xml')
        self.xml_builder.dump_xml(xml_path)
        ch1 = Chemistry(xml_path=xml_path)

        self.assertEqual(len(ch.get_reactions()), len(ch1.get_reactions()))
        self.assertEqual(len(ch.get_species(special=True)), len(ch1.get_species(special=True)))

        with self.assertRaises(XmlError):
            self.xml_builder.dump_xml(xml_path)  # dumping to the path which already exists
        try:
            self.xml_builder.dump_xml(xml_path, overwrite=True)  # this shoud be legal
        except XmlError:
            self.fail('XmlError raised unexpectadly!')

        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
