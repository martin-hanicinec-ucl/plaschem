import numbers

import numpy as np

from pygmo_fwork.pygmol.exceptions import \
    ModelParametersError, ModelParametersConsistencyError, ModelParametersTypeError


class TypeCheck(object):
    """
    Only a base method for inheritance defining some useful methods to check types of objects.
    """

    # noinspection PyTypeChecker
    @staticmethod
    def assert_is_number(item, number_type='positive'):
        # special cases, which ARE instances of numbers.Number
        if isinstance(item, bool):
            raise ModelParametersTypeError('The parameter {} needs to be a Number type!'.format(item))
        if isinstance(item, numbers.Number) and np.isnan(item):
            raise ModelParametersTypeError('The parameter {} needs to be a Number type!'.format(item))
        # ordinary cases - just check if is instance of numbers.Number:
        if not isinstance(item, numbers.Number):
            raise ModelParametersTypeError('The parameter {} needs to be a Number type!'.format(item))
        # on top of verifying item is a number, also check if it is positive/non-negative
        if number_type == 'positive':
            if item <= 0:
                raise ModelParametersTypeError('The parameter {} needs to be positive!'.format(item))
        elif number_type == 'non-negative':
            if item < 0:
                raise ModelParametersTypeError('The parameter {} needs to be positive!'.format(item))
        elif number_type == 'any':
            pass
        else:
            raise ValueError('This should never happen!')

    @staticmethod
    def assert_is_string(item):
        if not isinstance(item, str):
            raise ModelParametersTypeError('The parameter {} needs to be a string type!'.format(item))

    @staticmethod
    def assert_is_dict(item):
        if not isinstance(item, dict):
            raise ModelParametersTypeError('The parameter {} needs to be a dict type!'.format(item))

    @staticmethod
    def is_collection(item):
        return hasattr(item, '__iter__') and not isinstance(item, (str, bytes, bytearray))

    def assert_is_collection(self, item):
        if self.is_collection(item):
            pass  # is a collection
        else:
            raise ModelParametersTypeError('The parameter {} needs to be an iterable type!'.format(item))


class _ModelParametersFeeds(dict, TypeCheck):
    """
    This is a custom dictionary defined for validation of the feeds parameter (in model parameters)
    """

    def __init__(self, d):
        """Initialisation of this custom dict works only with dict.
        In the __init__ call, types of keys and values are checked to be consistent with what feeds need to be.
        The same goes for the setitem method, so model_parameters_feeds[key] = value will only work for string keys and
        Number values.
        :param d: (dict) of (str): (Number) pairs...
        """
        for key, value in d.items():
            self.assert_is_string(key)
            self.assert_is_number(value)
        super().__init__(d)

    def __setitem__(self, key, value):
        self.assert_is_string(key)
        self.assert_is_number(value)
        super().__setitem__(key, value)


class _ModelParametersPower(TypeCheck):
    """
    This is a helper object used to validate the power parameter for the GlobalModel. Both 'power' and 't_power' need
    to be present in the ModelParams, both have to be iterables of Numbers, both have to have the same length (min 2)
    and the time needs to be defined over the whole time domain, starting with 0 and ending with t_end, where also
    at least 10% excess over t_end is compulsory. This class ensures all of the above is in place.
    """

    def __init__(self):
        self._power_array = None
        self._time_array = None

    def set_power(self, power_array):
        """Sets the array of power values. If time values already set, lengths need to match.
        The length needs to be at least 2. If any condition violated, ModelParametersConsistencyError is raised.
        :param power_array: (iterable) of (Number)
        :return: None
        """
        if len(power_array) < 2:
            raise ModelParametersConsistencyError('"power" parameter needs to have length at least 2!')
        for power in power_array:
            self.assert_is_number(power, number_type='non-negative')
        if self._time_array is not None and len(self._time_array) != len(power_array):
            raise ModelParametersConsistencyError('"power" and "t_power" parameters are inconsistent sizes!')
        self._power_array = power_array

    def get_power(self):
        """Returns the power values.
        :return: (iterable) of (Number)
        """
        if self.is_collection(self._power_array):
            return list(self._power_array)
        else:
            return self._power_array

    def set_time(self, time_array):
        """Sets the array of times corresponding to the power values. If power values already set, the length needs to
        match! The length needs to be at least 2. If any condition violated, ModelParametersConsistencyError is raised.
        :param time_array: (iterable) of (Number)
        :return: None
        """
        if len(time_array) < 2:
            raise ModelParametersConsistencyError('"t_power" parameter needs to have length at least 2!')
        for time in time_array:
            self.assert_is_number(time, number_type='any')
        if self._power_array is not None and len(self._power_array) != len(time_array):
            raise ModelParametersConsistencyError('"power" and "t_power" parameters are inconsistent sizes!')
        self._time_array = time_array

    def get_time(self):
        """Returns the time values corresponding to the power values.
        :return: (iterable) of (Number)
        """
        if self.is_collection(self._time_array):
            return list(self._time_array)
        else:
            return self._time_array

    def assert_consistency(self, t_end):
        """This method is to ensure that both t_power and power has been set and that the t_power covers the whole
        time domain with some overlap (defined by 0 and t_end). If any inconsistency is found,
        ModelParametersConsistencyError is raised
        :param t_end: (Number) length of the simulation domain.
        :return: None
        """
        if self._power_array is None or self._time_array is None:
            raise ModelParametersConsistencyError('Either "power" or "t_power" parameter not defined!')
        # check it starts with <= 0 and ends with >= 1.1*t_end:
        if self._time_array[0] > 0.:
            raise ModelParametersConsistencyError('The "t_power" parameter needs to start with t_i <= 0.0 s!')
        if self._time_array[-1] < 1.1 * t_end:
            raise ModelParametersConsistencyError('The "t_power" parameter needs to end with t_f >= 1.1*t_end!')


class _ModelParametersBase(TypeCheck):
    """
    This is a base class for the ModelParameters classes. It defined a scheme of alternative keys
    for getting items saved under nominal keys. It also disabled deleting of elements as well as setting of elements
    (all ModelParameters classes have their keys/values set at the __init__ call and they cannot be changed later!
    """
    # dict of 'nominal_key': (*alternative_keys)
    nominal2alternative_map = {
        'radius': ('r', 'R',),
        'length': ('z', 'Z',),
        'pressure': ('p',),
        'power': ('P',),
        't_power': ('t_P', 'time_P', 'time_power',),
        'temp_gas': ('Tg', 'T_g', 'gas_temp',),
        'feeds': ('feed_flows',),
        't_end': ('time_end', 'end_time',)
    }
    # build the mapping dict mapping all the alternative names to their nominal name
    alternative2nominal_map = {key: key for key in nominal2alternative_map.keys()}
    for key in nominal2alternative_map:
        alt_keys = nominal2alternative_map[key]
        keys = len(alt_keys) * (key,)
        alternative2nominal_map.update(dict(zip(alt_keys, keys)))

    def __setitem__(self, key, value):
        raise TypeError('ModelParameters instance does not allow to set items after instantiation!')

    def __delitem__(self, key):
        raise TypeError('ModelParameters instance does not allow to delete items!')


class ModelParameters(_ModelParametersBase):
    """
    This is a class dedicated to compile and validate the consistency of the model parameters, passed to the
    GlobalModel or Equations instances. It can be instantiated from a dictionary or with **kwargs.
    It acts very much like a dictionary, but once it's instantiated, items cannot be set nor deleted.
    Also, upon instantiation, all the consistency checks are performed, such as if all the parameters required are
    present and with the correct types, etc.
    The instance will raise ModelParametersError or ModelParametersConsistencyError, if it's being instantiated with
    unknown parameter keys, or if the parameters are in any way inconsistent. It will raise the ModelParametersTypeError
    if the passed values are of incorrect types.
    The instance will raise KeyError if one attempts to set a new item (model_parameters[key] = value) or delete item.
    The only way to set items in the ModelParameters instance is during __init__ call.
    Apart from validations, other functionality defined in this class includes alternative keys to the parameters.
    For example the power value might be got by model_params['power'] or by model_params['P']. This makes future
    changes and backwards compatibility easier.
    """

    def __init__(self, model_params=None, **additional_model_params):
        """Initialiser for the ModelParameters. The class is pretty much a wrapper for a dict, with some validation
        functionality and with mapping between nominal keys and alternative keys for the same values.
        The model parameters passed (either in dict or in kwargs) need to include:
        - radius (R, r) => number in [m]
        - length (Z, z) => number in [m]
        - pressure (p) => number in [Pa]
        - power (P) => iterable of at least 2 values in [W]
        - t_power (t_P, time_P, time_power) => iterable of len(power) values in [s] - need to span whole time domain
        - temp_gas (Tg, T_g, gas_temp) => number in [K]
        - feeds (feed flows) => dict of str(species_name): number(value in [sccm])
        - t_end (time_end, end_time) => number in [s]
        If the keys differ from prescribed, if the types are mismatched, or if the power and t_power arrays are
        inconsistent, appropriate errors are raised.
        :param model_params: (dict) of model parameters
        :param additional_model_params: additional model parameters
        """
        if model_params is None:
            all_passed_params = {}
        else:
            all_passed_params = model_params.copy()
        all_passed_params.update(additional_model_params)

        self.parameters = {}  # this is the "master" dict holding all the values under their nominal keys

        # check the passed parameters for completeness and if not including any unknown keys:
        passed_nominal_keys = set()
        for key in all_passed_params:
            if key in self.alternative2nominal_map:
                passed_nominal_keys.add(self.alternative2nominal_map[key])
            else:
                raise ModelParametersError('Passed parameters include unsupported key: {}'.format(key))
        required_nominal_keys = set(self.nominal2alternative_map)
        if len(passed_nominal_keys - required_nominal_keys):
            raise ModelParametersError(
                'Passed parameters include unsupported keys: {}'.format(passed_nominal_keys - required_nominal_keys)
            )
        if len(required_nominal_keys - passed_nominal_keys):
            raise ModelParametersError(
                'Passed parameters missing some required keys: {}'.format(required_nominal_keys - passed_nominal_keys)
            )

        # set all the parameters:
        power_cls = _ModelParametersPower()
        for key, value in all_passed_params.items():
            nom_key = self.alternative2nominal_map[key]
            if nom_key in {'radius', 'length', 'pressure', 'temp_gas', 't_end'}:
                # for the number-like parameters, set check type and set values straight away...
                self.assert_is_number(value)
                self.parameters[nom_key] = value
            elif nom_key == 'power':
                # for the power parameters, validate them by in a custom class
                self.assert_is_collection(value)
                power_cls.set_power(value)
                self.parameters[nom_key] = power_cls
            elif nom_key == 't_power':
                # for the power parameters, validate them by in a custom class
                self.assert_is_collection(value)
                power_cls.set_time(value)
                self.parameters[nom_key] = power_cls
            elif nom_key == 'feeds':
                # for the feeds parameter, validate it in a custom dict
                self.assert_is_dict(value)
                self.parameters[nom_key] = _ModelParametersFeeds(value)  # turn into a custom dict, which checks types
            else:
                raise ModelParametersError('Unexpected nominal key "{}": Inconsistent __init__ method!'.format(nom_key))
        # check the consistency of the power class:
        if self.parameters['power'] is not self.parameters['t_power']:
            raise ModelParametersError('This should never happen!')
        self.parameters['power'].assert_consistency(t_end=self.parameters['t_end'])

    def __getitem__(self, key):
        if key not in self.alternative2nominal_map:
            raise ModelParametersError('Trying to get an unsupported parameter {}!'.format(key))
        nom_key = self.alternative2nominal_map[key]
        # special parameters (non-Number):
        if nom_key == 'feeds':
            return dict(self.parameters[nom_key])  # convert it back from custom dict to regular dict
        elif nom_key == 'power':
            return self.parameters[nom_key].get_power()  # extract the iterable from the custom power class
        elif nom_key == 't_power':
            return self.parameters[nom_key].get_time()  # extract the iterable from the custom power class
        # regular parameters (Numbers):
        return self.parameters[nom_key]

    # all the other special methods come from the self.parameters dictionary:
    def __repr__(self):
        return self.parameters.__repr__()

    def __str__(self):
        return self.parameters.__str__()

    def __len__(self):
        return self.parameters.__len__()

    def __iter__(self):
        return self.parameters.__iter__()

    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()

    def items(self):
        return self.parameters.items()
