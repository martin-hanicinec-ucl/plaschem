
import unittest
import os
import shutil
import yaml

import pandas as pd
from numpy import pi
import numpy as np

from pygmo_fwork.config import Config
from pygmo_fwork.results import ResultsParser
from plaschem.chemistry import Chemistry
from plaschem.reactions import Reaction
from pygmo_fwork.exceptions import \
    ResultsAttributeError
from unit_tests.test_plaschem.utils import draw_test_species


class TestResultsParser(unittest.TestCase):

    context = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(context, '.temp')
    workspace_dir = os.path.join(temp_dir, 'workspace')
    config_dict = {'workspace': workspace_dir}
    config_path = os.path.join(temp_dir, '.config')

    def get_custom_results_parser(self):
        """Method returning the ResultsParser class only with a pimped-up config, so the workspace directory is not
        built in the location specified in the actual config file...
        """
        class ResPars(ResultsParser):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _fetch_results_dir(self, backend=None, results_dir=None):
                # this is the only ResultsParser method instantiating the Config...
                if backend is None and results_dir is not None:
                    return results_dir
                elif backend is not None and results_dir is None:
                    return Config(config_file_path=TestResultsParser.config_path).get_results_dir(backend)
                elif backend is None and results_dir is None:
                    return self.nominal_results_dir
                else:
                    raise ResultsAttributeError('Invalid combination of attributes!')
        return ResPars

    def setUp(self):
        # make the temporary directory
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.mkdir(self.temp_dir)
        # build the config file:
        with open(self.config_path, 'w') as stream:
            yaml.dump(self.config_dict, stream=stream)

        self.ResultsParser = self.get_custom_results_parser()  # pimped-up ResultsParser class with .temp config file

    def tearDown(self):
        # remove the .temp dir
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        rp = self.ResultsParser()
        self.assertEqual(str(rp.nominal_results_dir), os.path.join(self.config_dict['workspace'], 'results', 'pygmol'))
        rp = self.ResultsParser(nominal_backend='pygmol')  # passing backend
        self.assertEqual(str(rp.nominal_results_dir), os.path.join(self.config_dict['workspace'], 'results', 'pygmol'))
        rp = self.ResultsParser(nominal_results_dir='results/directory')  # passing path
        self.assertEqual(str(rp.nominal_results_dir), 'results/directory')
        with self.assertRaises(ResultsAttributeError):
            self.ResultsParser(nominal_results_dir='some/thing', nominal_backend='other_thing')  # cannot pass both!

    def test_fetch_run_dir(self):
        rp = self.ResultsParser(nominal_results_dir='res_dir')
        self.assertEqual(rp._fetch_run_dir('run_id'), os.path.join('res_dir', 'run_id'))
        self.assertEqual(
            rp._fetch_run_dir('run_id', backend='pygkin'),
            os.path.join(self.config_dict['workspace'], 'results', 'pygkin', 'run_id')
        )
        self.assertEqual(
            rp._fetch_run_dir('run_id', results_dir='another/res_dir'),
            os.path.join('another/res_dir', 'run_id')
        )
        with self.assertRaises(ResultsAttributeError):
            rp._fetch_run_dir('run_id', results_dir='some/thing', backend='other_thing')  # cannot pass both!

    def test_fetching(self):
        test_results_dir = os.path.join(self.context, 'resources')
        rp = self.ResultsParser(nominal_results_dir=test_results_dir)
        run_id = 'run_00'
        sol = rp.fetch_solution(run_id)
        rates = rp.fetch_rates(run_id, results_dir=test_results_dir)
        model_params = rp.fetch_model_params(run_id, attributes_dump_name='model_attributes.yaml')
        initial_params = rp.fetch_initial_params(
            run_id, results_dir=test_results_dir, attributes_dump_name='model_attributes_copy.yaml')
        self.assertEqual(
            sol.iloc[-1, 1],
            pd.read_csv(os.path.join(test_results_dir, run_id, 'solution.csv'), index_col=0).iloc[-1, 1]
        )
        self.assertEqual(
            rates.iloc[-1, 1],
            pd.read_csv(os.path.join(test_results_dir, run_id, 'rates.csv'), index_col=0).iloc[-1, 1]
        )
        self.assertEqual(model_params['pressure'], 100)
        self.assertEqual(initial_params['init_n']['O'], 7242973.034150649)

    def test_compare(self):
        test_results_dir = os.path.join(self.context, 'resources')
        rp = self.ResultsParser(nominal_results_dir=test_results_dir)
        comp = rp.compare(
            [
                'run_00',
                ('run_00', test_results_dir),
                ('run_00', 'solution_copy.csv'),
                ('run_00', test_results_dir, 'solution.csv')
            ],
            printout=False
        )
        self.assertEqual(comp.shape, (4, 27))
        comp = rp.compare(
            [
                'run_00',
                ('run_00', test_results_dir),
                ('run_00', 'solution_copy.csv'),
                ('run_00', test_results_dir, 'solution.csv')
            ],
            relative_to_first=True,
            printout=False
        )
        self.assertTrue((comp.values == 0.).all())  # all the values from one solution

    def test_compress_solution(self):
        full_sol = pd.DataFrame(
            # ['t', 'A', 'B', 'C']
            [[0, 1, 1, 1],
             [1, 10, 10, 10],
             [2, 20, 20, 20],
             [10, 21, 21, 21],
             [100, 22, 22, 22]],
            columns=['t', 'A', 'B', 'C']
        )
        orig_t = [0, 1, 2, 10, 100]
        rp = ResultsParser()

        compressed_sol = rp.compress_solution(full_sol, diff_threshold=0.04, min_timestep=0.009)
        self.assertEqual(list(compressed_sol['t']), orig_t)  # no compression, thresholds too tight

        compressed_sol = rp.compress_solution(full_sol, diff_threshold=0.04, min_timestep=0.011)
        self.assertEqual(list(compressed_sol['t']), [0, 2, 10, 100])

        compressed_sol = rp.compress_solution(full_sol, diff_threshold=1.1, min_timestep=0.009)
        self.assertEqual(list(compressed_sol['t']), [0, 1, 10, 100])

        compressed_sol = rp.compress_solution(full_sol, diff_threshold=1.1, min_timestep=0.011)
        self.assertEqual(list(compressed_sol['t']), [0, 2, 100])

        full_sol.at[:, 'A'] = [1, 1, 1, 1, 1]
        full_sol.at[:, 'B'] = [1, 1, 1, 1, 1]
        # should not affect the compression, the 'C' column still the same
        compressed_sol = rp.compress_solution(full_sol, diff_threshold=1.1, min_timestep=0.011)
        self.assertEqual(list(compressed_sol['t']), [0, 2, 100])

        full_sol.at[:, 'C'] = [1, 1, 1, 1, 1]  # now the solution is not changing in time, should compress to first,last
        compressed_sol = rp.compress_solution(full_sol, diff_threshold=1.1, min_timestep=0.011)
        self.assertEqual(list(compressed_sol['t']), [0, 100])

        self.assertEqual(list(full_sol['t']), orig_t)  # check that the original solution is untouched

    def test_get_solution_frames(self):
        solution_orig = pd.DataFrame(columns=['t', 'A', 'B'])
        solution_orig.loc[:, 't'] = [0, 1, 3]
        solution_orig.loc[:, 'A'] = [0, 0, 2]
        solution_orig.loc[:, 'B'] = [2, 0, 0]
        solution = solution_orig.copy()

        frames = ResultsParser.get_results_frames(solution, times=[2])
        self.assertEqual(frames['t'].iloc[0], 2)
        self.assertEqual(frames['A'].iloc[0], 1)
        self.assertEqual(frames['B'].iloc[0], 0)

        frames = ResultsParser.get_results_frames(solution, times=[0.5, 2])
        self.assertEqual(frames['t'].iloc[0], 0.5)
        self.assertEqual(frames['A'].iloc[0], 0)
        self.assertEqual(frames['B'].iloc[0], 1)
        self.assertEqual(list(frames.columns), ['t', 'A', 'B'])
        self.assertEqual(list(frames['t']), [0.5, 2])

        for times in ([-1], [-1, 1], [1, 3.1]):
            with self.assertRaises(AssertionError):
                ResultsParser.get_results_frames(solution, times=times)

        self.assertTrue((solution_orig.values == solution.values).all())  # ensure that the original solution untouched

    def test_get_volumetric_rates(self):
        # build test chemistry and inputs:
        sps = draw_test_species('e Ar Ar+')
        reactions = [
            Reaction(reactants=(sps[0], sps[1]), products=(sps[0], sps[1]), r_id=1, special_number=42),
            Reaction(reactants=(sps[0], sps[1]), products=(sps[0], sps[0], sps[2]), r_id=2, special_number=42),
            Reaction(reactants=(sps[1], sps[1]), products=(sps[0], sps[0], sps[2], sps[2]), r_id=3, special_number=42),
            Reaction(reactants=(sps[1], sps[1]), products=(sps[0], sps[1], sps[2]), r_id=4, special_number=42)
        ]
        # e + Ar > e + Ar
        # e + Ar > e + e + Ar+
        # Ar + Ar > e + e + Ar+ + Ar+
        # Ar + Ar > e + Ar + Ar+

        ch = Chemistry(sps, reactions)

        for rf in [
            pd.Series([1, 2, 3, 4], index=[r.id for r in ch.get_reactions()]),   # different rates_frames
            pd.Series([42, 1, 2, 3, 4], index=['t'] + [r.id for r in ch.get_reactions()])
        ]:
            # test for correct output:
            vol_rates = ResultsParser.get_volumetric_rates('Ar+', ch, rf)
            expected = pd.Series(
                [2, 6, 4],
                index=[2, 3, 4]
            )
            self.assertTrue(np.allclose(vol_rates, expected[vol_rates.index]))
            self.assertTrue(np.allclose(expected, vol_rates[expected.index]))

            vol_rates = ResultsParser.get_volumetric_rates('Ar', ch, rf)
            expected = pd.Series(
                [-2, -6, -4],
                index=[2, 3, 4]
            )
            self.assertTrue(np.allclose(vol_rates, expected[vol_rates.index]))
            self.assertTrue(np.allclose(expected, vol_rates[expected.index]))

            vol_rates = ResultsParser.get_volumetric_rates('e', ch, rf)
            expected = pd.Series(
                [2, 6, 4],
                index=[2, 3, 4]
            )
            self.assertTrue(np.allclose(vol_rates, expected[vol_rates.index]))
            self.assertTrue(np.allclose(expected, vol_rates[expected.index]))

    def test_get_surface_rates(self):
        # build test chemistry and inputs:
        sps = draw_test_species('e Ar Ar+')
        reactions = [
            Reaction(reactants=(sps[0], sps[1]), products=(sps[0], sps[1]), r_id=1, special_number=42),
            Reaction(reactants=(sps[0], sps[1]), products=(sps[0], sps[0], sps[2]), r_id=2, special_number=42),
        ]

        ch = Chemistry(sps, reactions)
        sp_ids = [sp.id for sp in ch.get_species()]
        sp_names = ['Ar', 'Ar+']
        mp = {'radius': 42, 'length': 4.2}
        V = pi*42**2*4.2
        A = 2*pi*42*(42 + 4.2)

        wf1, wf2 = 2, 3
        for wff in [
            pd.Series([-wf1, -wf2], index=sp_names),  # different wall fluxes frames
            pd.Series([42, -wf1, -wf2], index=['t'] + sp_names)
        ]:
            # test for correct outputs:
            # default stick and ret:

            surf_rates = ResultsParser.get_surface_rates('Ar', ch, mp, wff)
            expected = pd.Series([-wf1*A/V, wf2*A/V], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            surf_rates = ResultsParser.get_surface_rates('Ar+', ch, mp, wff)
            expected = pd.Series([0, -wf2*A/V], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            with self.assertWarns(UserWarning):
                sr = ResultsParser.get_surface_rates('e', ch, mp, wff)
            self.assertEqual(len(sr), 0)

        # tweak surface coefs (no effect on wall fluxes, since those already have stick_coef in them):
        ch.set_adhoc_species_attributes('stick_coef', 'Ar', 1.0)

        for wff in [
            pd.Series([-wf1, -wf2], index=sp_names),  # different wall fluxes frames
            pd.Series([42, -wf1, -wf2], index=['t'] + sp_names)
        ]:
            surf_rates = ResultsParser.get_surface_rates('Ar', ch, mp, wff)
            expected = pd.Series([-wf1*A/V, wf2*A/V], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            surf_rates = ResultsParser.get_surface_rates('Ar+', ch, mp, wff)
            expected = pd.Series([0, -wf2*A/V], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            with self.assertWarns(UserWarning):
                sr = ResultsParser.get_surface_rates('e', ch, mp, wff)
            self.assertEqual(len(sr), 0)

        # tweak surface return coefs:
        r1 = 0.25
        r2 = 0.5
        ch.set_adhoc_species_attributes('ret_coefs', 'Ar+', {'Ar+': r2, 'Ar': r1})

        for wff in [
            pd.Series([-wf1, -wf2], index=sp_names),  # different wall fluxes frames
            pd.Series([42, -wf1, -wf2], index=['t'] + sp_names)
        ]:
            surf_rates = ResultsParser.get_surface_rates('Ar', ch, mp, wff)
            expected = pd.Series([-wf1*A/V, wf2*A/V*r1], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            surf_rates = ResultsParser.get_surface_rates('Ar+', ch, mp, wff)
            expected = pd.Series([0, -wf2*A/V + wf2*A/V*r2], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            with self.assertWarns(UserWarning):
                sr = ResultsParser.get_surface_rates('e', ch, mp, wff)
            self.assertEqual(len(sr), 0)

        # tweak surface coefs:
        r3, r4 = 0.1, 0.42
        ch.set_adhoc_species_attributes('ret_coefs', 'Ar', {'Ar+': r4, 'Ar': r3})

        for wff in [
            pd.Series([-wf1, -wf2], index=sp_names),  # different wall fluxes frames
            pd.Series([42, -wf1, -wf2], index=['t'] + sp_names)
        ]:
            surf_rates = ResultsParser.get_surface_rates('Ar', ch, mp, wff)
            expected = pd.Series([-wf1*A/V + wf1*A/V*r3, wf2*A/V*r1], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            surf_rates = ResultsParser.get_surface_rates('Ar+', ch, mp, wff)
            expected = pd.Series([wf1*A/V*r4, -wf2*A/V + wf2*A/V*r2], index=sp_ids)
            expected = expected[expected != 0]
            self.assertTrue(np.allclose(surf_rates, expected[surf_rates.index]))
            self.assertTrue(np.allclose(expected, surf_rates[expected.index]))

            with self.assertWarns(UserWarning):
                sr = ResultsParser.get_surface_rates('e', ch, mp, wff)
            self.assertEqual(len(sr), 0)


if __name__ == '__main__':
    unittest.main()
