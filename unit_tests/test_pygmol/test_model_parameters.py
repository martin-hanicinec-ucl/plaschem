
import unittest

from pygmo_fwork.pygmol.model_parameters import \
    ModelParameters, _ModelParametersFeeds, _ModelParametersPower
from pygmo_fwork.pygmol.exceptions import \
    ModelParametersError, ModelParametersTypeError, ModelParametersConsistencyError


class TestModelParameters(unittest.TestCase):

    def test_feeds(self):
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': '1'})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({1: 1})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': 1, 1: 1e0})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': 1, 'Ar2': '1', 1: 1.})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': None})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({None: 1.})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': float('nan')})
        with self.assertRaises(ModelParametersTypeError):
            _ = _ModelParametersFeeds({'Ar': True})
        with self.assertRaises(TypeError):
            _ = _ModelParametersFeeds()
        # following should be legal:
        feeds = _ModelParametersFeeds({})
        feeds = _ModelParametersFeeds({'Ar': 100, 'O2': 300.5})
        self.assertEqual(set(feeds.values()), {100, 300.5})
        self.assertEqual(set(feeds.keys()), {'Ar', 'O2'})
        self.assertEqual(len(feeds), 2)
        feeds['He'] = 42.
        self.assertIn('He', feeds)
        with self.assertRaises(ModelParametersTypeError):
            feeds[42] = 42
        with self.assertRaises(ModelParametersTypeError):
            feeds['H2'] = '42'
        self.assertNotIn(42, feeds)
        self.assertNotIn('H2', feeds)

    def test_power(self):
        power_cls = _ModelParametersPower()
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_power([1])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_power([])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_time([1])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_time([])
        with self.assertRaises(ModelParametersTypeError):
            power_cls.set_power(['0', 1])
        with self.assertRaises(ModelParametersTypeError):
            power_cls.set_time(['0', 1])
        self.assertEqual(power_cls.get_power(), None)
        self.assertEqual(power_cls.get_time(), None)

        power_cls.set_time([0, 1])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_power([100, 200, 300])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_power([100])
        with self.assertRaises(ModelParametersTypeError):
            power_cls.set_power([100, '200'])
        self.assertEqual(power_cls.get_power(), None)
        self.assertEqual(power_cls.get_time(), [0, 1])
        power_cls.set_power((1000, 500.5))
        self.assertEqual(power_cls.get_power(), [1000, 500.5])
        self.assertEqual(power_cls.get_time(), [0, 1])

        power_cls._time_array = None
        power_cls.set_power([1000, 500.5])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_time([0, 1, 2])
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.set_time([0])
        with self.assertRaises(ModelParametersTypeError):
            power_cls.set_time([0, '1'])
        self.assertEqual(power_cls.get_time(), None)
        self.assertEqual(power_cls.get_power(), [1000, 500.5])
        power_cls.set_time((0, 1.))
        self.assertEqual(power_cls.get_power(), [1000, 500.5])
        self.assertEqual(power_cls.get_time(), [0, 1.])

        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.assert_consistency(t_end=1.)
        power_cls.assert_consistency(t_end=0.9)  # this should be legal
        power_cls.set_time((0.1, 1.))
        with self.assertRaises(ModelParametersConsistencyError):
            power_cls.assert_consistency(t_end=0.9)

    def test_init(self):
        test_params = {'feeds': {}, 'Tg': 300, 'r': 0.1, 'z': 0.1, 'p': 100, 'P': [9, 9], 't_P': [0, 0.2], 't_end': 0.1}
        # this is legal:
        _ = ModelParameters(test_params)
        for key in test_params:
            ilegal_test_params = test_params.copy()
            ilegal_test_params.pop(key)
            with self.assertRaises(ModelParametersError):
                _ = ModelParameters(ilegal_test_params)

        ilegal_test_params = test_params.copy()
        ilegal_test_params['unsupported'] = 42
        with self.assertRaises(ModelParametersError):
            _ = ModelParameters(ilegal_test_params)  # this should be ilegal

        # legal:
        _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(0, 1), t_P=(-1, 1), t_end=0.1)
        # these should be ilegal:
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={}, Tg=300, r=0.1, z=0.1, p=100, P=1, t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={}, Tg=300, r=0.1, z=0.1, p=100, P=1, t_P=0, t_end=0.1)
        with self.assertRaises(ModelParametersError):
            _ = ModelParameters(feeds={}, Tg=300, r=0.1, z=0.1, p=100, P=[1, 1], t_end=0.1)
        with self.assertRaises(ModelParametersError):
            _ = ModelParameters(feeds={}, Tg=300, r=0.1, z=0.1, p=100, P=[1, 1], t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds=[], Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={}, Tg='300', r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={}, Tg=300, r=None, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=True, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersConsistencyError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersConsistencyError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=1)
        with self.assertRaises(ModelParametersConsistencyError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0.1, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': -1.}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=-300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=0, P=(1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(-1, 1), t_P=(0, 1), t_end=0.1)
        with self.assertRaises(ModelParametersTypeError):
            _ = ModelParameters(feeds={'Ar': 1.}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0)

    def test_set_del(self):
        model_params = ModelParameters(
            feeds={'O': 1}, temp_gas=300, radius=0.1, length=0.1, pressure=100, power=(1, 1), t_power=(0, 1), t_end=0.1
        )
        with self.assertRaises(TypeError):
            del(model_params['feeds'])
        with self.assertRaises(TypeError):
            model_params['feeds'] = {'Ar': 100.}

    def test_get(self):
        model_params = ModelParameters(
            feeds={'O': 1}, temp_gas=300, radius=0.1, length=0.1, pressure=100, power=(1, 1), t_power=(0, 1), t_end=0.1
        )
        self.assertEqual(model_params['r'], 0.1)
        self.assertEqual(model_params['P'], [1, 1])
        self.assertEqual(model_params['radius'], 0.1)
        self.assertEqual(model_params['power'], [1, 1])
        model_params = ModelParameters(
            feeds={'O': 1}, Tg=300, r=0.1, z=0.1, p=100, P=(1, 1), t_P=(0, 1), t_end=0.1
        )
        self.assertEqual(model_params['z'], 0.1)
        self.assertEqual(model_params['length'], 0.1)
        self.assertEqual(model_params['p'], 100)
        self.assertEqual(model_params['pressure'], 100)
        self.assertEqual(model_params['t_end'], 0.1)

    def test_special(self):
        model_params = ModelParameters(
            feeds={'O': 1}, temp_gas=300, radius=0.1, length=0.1, pressure=100, power=(1, 1), t_power=(0, 1), t_end=0.1
        )
        self.assertEqual(len(model_params), 8)
        self.assertEqual(len(model_params.keys()), 8)
        self.assertEqual(len(model_params.values()), 8)
        self.assertEqual(len(model_params.items()), 8)
        self.assertEqual(str(model_params), str(model_params.parameters))

    def test_dictify(self):
        mp_dict = {
            'feeds': {'Ar': 1.},
            'temp_gas': 300, 'radius': 0.1, 'length': 0.1, 'pressure':100, 't_end': 0.1,
            'power': [0, 1], 't_power': [-1, 1]
        }
        mp = ModelParameters(mp_dict)
        self.assertEqual(dict(mp), mp_dict)


if __name__ == '__main__':
    unittest.main()
