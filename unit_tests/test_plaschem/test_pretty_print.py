
import unittest

from plaschem.pretty_print import PrettyPrint
from unit_tests.test_plaschem.utils import draw_test_chemistry


class TestPrettyPrint(unittest.TestCase):

    def test_pretty_print(self):
        ch = draw_test_chemistry(5)
        ch.disable(species=('H*', 1134, 'H2O+'), reactions=(6214, ))
        pp = PrettyPrint(chemistry=ch)
        pp._get_pretty_dataframe(ch.get_stoichiomatrix(disabled=True, special=True), annotation=True)
        pp._get_pretty_dataframe(ch.get_return_matrix(), annotation=False)


if __name__ == '__main__':
    unittest.main()
