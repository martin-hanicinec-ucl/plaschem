
import unittest

import numpy as np

from plaschem.classification import ReactionClassifier
from unit_tests.test_plaschem.utils import draw_test_chemistry


class TestReactionClassifier(unittest.TestCase):

    def test_classification(self):
        ch = draw_test_chemistry(5)
        rc = ReactionClassifier(reactions=ch.get_reactions(), species=ch.get_species(special=True))
        # just a few randomly selected processes:
        r1 = ch._reactions[420]  # e + H3+  >  e + H3+
        r2 = ch._reactions[625]  # e + O3  >  O  + O2-
        r3 = ch._reactions[1011]  # O3  + OH    >  O2  + HO2
        r4 = ch._reactions[1347]  # H3+ + O2-  >  H + H2 + O2
        self.assertEqual(rc._classify(r1), ('ELECTRON IMPACT COLLISIONS', 'elastic and momentum transfer'))
        self.assertEqual(rc._classify(r2), ('ELECTRON IMPACT COLLISIONS', 'dissociative attachment'))
        self.assertEqual(rc._classify(r3), ('NEUTRAL - NEUTRAL COLLISIONS', 'interchange'))
        self.assertEqual(rc._classify(r4), ('ION - ION COLLISIONS', 'dissociative neutralization'))

    def test_sort_and_classify(self):
        ch = draw_test_chemistry(5)
        reactions = list(np.random.permutation(ch.get_reactions()))  # shuffled reactions
        np.random.seed(42)  # need a controlled permut., preserving the order of exc. spcs., since these are not sorted
        species = list(np.random.permutation(ch.get_species(special=True)))  # shuffled species
        rc = ReactionClassifier(reactions=reactions, species=species)
        rc.sort_species()
        # test sorting species:
        self.assertEqual(
            [sp.get_name() for sp in rc.species],
            ['e', 'H', 'H+', 'H*', 'H2', 'H2+', 'H3+', 'O', 'O+', 'O-', 'O*', 'O2', 'O2+', 'O2-', 'O2(v)', 'O2*', 'O3',
             'O3-', 'HO2', 'H2O', 'H2O+', 'H2O2', 'OH', 'M']
        )
        rc.sort_reactions()
        # test sorting reactions (checking that some random subarrays equal)
        r_ids = [r.id for r in rc.reactions]
        self.assertEqual(r_ids[:5], [391, 387, 388, 392, 393])
        self.assertEqual(r_ids[-5:], [1386, 1019, 1013, 1014, 1012])
        # test classification (checking that counts of random categories are what they should be):
        classified_reactions = rc.get_classified_reactions()
        self.assertEqual(len(classified_reactions['ELECTRON IMPACT COLLISIONS']['dissociative excitation']), 2)
        self.assertEqual(len(classified_reactions['ELECTRON IMPACT COLLISIONS']['three body attachment']), 3)
        self.assertEqual(len(classified_reactions['ION - ION COLLISIONS']['neutralization']), 21)

    def test_aligned_reaction_string(self):
        ch = draw_test_chemistry(5)
        reactions = ch.get_reactions()
        species = ch.get_species(special=True)
        r1 = reactions.at[626]  # e + O3-  >  e + O3-  # this is the reaction with the longest aligned string
        rc = ReactionClassifier(reactions=reactions, species=species)
        # reactions_classified = rc.get_classified_reactions()
        with self.assertRaises(KeyError):
            rc.get_aligned_reaction_string(r1)
        # need to first build the classified dicts:
        rc.get_classified_reactions()
        self.assertEqual(rc.get_aligned_reaction_string(r1), 'e + O3- > e + O3-')
        # change name of a O* species to something longer than O3- (to change O* elastic r. in the same cat as O3-):
        species.at[141]._name = 'O*****'
        # reset the chemistry in the classifier class:
        rc.set_new_chemistry(reactions=reactions, species=species)
        with self.assertRaises(KeyError):
            rc.get_aligned_reaction_string(r1)
        # need to first build the classified dicts:
        rc.get_classified_reactions()
        self.assertEqual(rc.get_aligned_reaction_string(r1), 'e + O3-    > e + O3-   ')  # three more spaces bcs O*****
        species.at[141]._name = 'O*'  # set back, since other methods use this


if __name__ == '__main__':
    unittest.main()
