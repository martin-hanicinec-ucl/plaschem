
import unittest

import pandas as pd
import numpy as np

from pygmo_fwork.optichem.errors import SolutionsDiff


class TestSolutionsDiff(unittest.TestCase):

    def setUp(self):
        # dummy full solution: time in [0, 4], species A, B, C, D
        self.sol_full = pd.DataFrame(np.arange(25).reshape((5, 5)).T, columns=['t', 'A', 'B', 'C', 'D'])
        # dummy reduced solution, time in [0, 3], species A, B, C
        self.sol_red = pd.DataFrame(np.arange(16).reshape((4, 4)).T, columns=['t', 'A', 'B', 'C'])
        self.sol_red_neg = self.sol_red.copy()
        for col in ['A', 'B', 'C']:
            self.sol_red_neg.at[:, col] = -self.sol_red[col]

    def test_validate(self):
        with self.assertRaises(AssertionError):
            SolutionsDiff._validate_solution('t')
        with self.assertRaises(AssertionError):
            SolutionsDiff._validate_solution(pd.Series({'t': 0.42, 'O': 42}))
        # following should go through:
        SolutionsDiff._validate_solution(pd.DataFrame(columns=['t', ], index=['42', ]))

    def test_get_result_in_time(self):
        with self.assertRaises(AssertionError):
            SolutionsDiff._get_result_in_time(self.sol_full, 'E', 1)  # 'E' not in outputs
        with self.assertRaises(AssertionError):
            SolutionsDiff._get_result_in_time(self.sol_full, 'B', 5)  # time t=5 outside of the range
        with self.assertRaises(AssertionError):
            SolutionsDiff._get_result_in_time(self.sol_full, 'A', -1)  # time t=-1 outside of the range
        # following should ho through:
        self.assertEqual(SolutionsDiff._get_result_in_time(self.sol_full, 'A', 0), 5)  # on grid point
        self.assertEqual(SolutionsDiff._get_result_in_time(self.sol_full, 'D', 4), 24)  # on grid point
        self.assertEqual(SolutionsDiff._get_result_in_time(self.sol_full, 'A', 0.5), 5.5)  # actually interpolating

    def test_get_errors(self):
        sol_diff = SolutionsDiff(self.sol_full)
        self.assertEqual(
            list(sol_diff.get_output_errors(self.sol_red, 'B', [0, 2.5])),
            [-4/(10 + 12.5), -4/(12.5 + 12.5)]
        )
        # vals_full = [10, 12.5],  vals_red = [8, 10.5]
        with self.assertRaises(AssertionError):
            sol_diff.get_output_errors(self.sol_red, 'D', [0, 2.5])  # 'D' is not in sol_red
        with self.assertRaises(AssertionError):
            sol_diff.get_output_errors(self.sol_red, 'B', [0, 3.5])  # sol_red ends at time=3.0

    def test_get_error(self):
        # test suite for the maximum error and RMS across outputs:
        sol_diff = SolutionsDiff(self.sol_full)
        important_outputs = ['A', 'C']
        times = [1, 3]
        self.assertEqual(sol_diff.get_error_max(self.sol_red_neg, important_outputs, times), 1.875)
        self.assertAlmostEqual(
            sol_diff.get_error_rms(self.sol_red_neg, important_outputs, times),
            (((1.57142857**2 + 1.875**2)/2 + (1.70588235**2 + 1.83333333**2)/2)/2)**0.5
        )


if __name__ == '__main__':
    unittest.main()
