
from collections import OrderedDict

import numpy as np

from plaschem.exceptions import ReactionInitError, ReactionAttributeError, ReactionValueError


# noinspection PyUnresolvedReferences
class Reaction(object):
    """
    Class representing reactions. Each reaction is uniquely identified by it's id (int). The class implements
    a number of methods helping with description and categorisation of the reactions.
    """
    mandatory_attributes = set()  # currently empty - everything can be defaulted!

    default_attributes = OrderedDict([
        ('arrh_a', None),  # do not have a default
        ('arrh_b', None),  # default depends on arrh_a
        ('arrh_c', None),  # default depends on arrh_a
        ('elastic', None),  # default depends on reactants and products
        ('el_en_loss', 0.),
        ('gas_heat_contrib', 0.),
        ('special_number', None),  # default depends on is electron process or not
        ('qdb_r_id', None),  # do not have a default
        ('qdb_ds_id', None),  # do not have a default
        ('doi', None),  # do not have a default
        ('comments', None),  # do not have a default
        ('latex', None)  # do not have a default
    ])
    # these are the only allowed values for certain attributes:
    allowed_attribute_values = {
        'elastic': {None, 'true', 'True', 'TRUE', 'false', 'False', 'FALSE', True, False}
    }
    # these are allowed types looked at when trying to set new attributes values with self.set_reaction_attribute
    nonetype = type(None)
    allowed_attribute_types = {
        'arrh_a': [float, np.float64, str, int, nonetype],
        'arrh_b': [float, np.float64, str, int, nonetype],
        'arrh_c': [float, np.float64, str, int, nonetype],
        'el_en_loss': [float, np.float64, str, int, nonetype],
        'gas_heat_contrib': [float, np.float64, str, int, nonetype],
        'special_number': [int, str, nonetype],
        'qdb_r_id': [int, str, nonetype],
        'qdb_ds_id': [int, str, nonetype],
        'doi': [str, nonetype],
        'comments': [str, nonetype],
        'elastic': [bool, str, nonetype],
        'latex': [str, nonetype]
    }

    attributes_with_mandatory_units = {'arrh_a', 'arrh_c'}  # if their attributes present, units need to be passed
    default_units = OrderedDict([
        ('arrh_a', None),  # mandatory - does not have a default
        ('arrh_c', None),  # mandatory - does not have a default
        ('el_en_loss', 'eV'),
        ('gas_heat_contrib', 'eV'),
    ])
    allowed_unit_values = {
        'arrh_a': {None, 's-1', 'cm3.s-1', 'm3.s-1', 'cm6.s-1', 'm6.s-1'},
        'arrh_c': {None, 'eV', 'K'},
        'el_en_loss': {None, 'eV', },
        'gas_heat_contrib': {None, 'eV', },
    }  # if I add others, need to implement conversion factors into attributes getters!
    si_conversion_factors = {
        'arrh_a': {
            's-1': 1.,
            'm3.s-1': 1., 'm6.s-1': 1.,
            'cm3.s-1': 1e-6, 'cm6.s-1': 1e-12,
        },
    }  # dicts of conversion factors from non-SI allowed units to SI allowed units.

    def __init__(self, r_id, reactants, products, attributes=None, units=None, **additional_attributes):
        """Instantiates the Reaction object.
        The ID must be positive integer uniquely identifying the reaction inside it's chemistry.
        The products and reactants are lists of RP instances, which need to define following methods and attributes:
            RP.id
            RP.get_name()  # unique name of the species (str)
            RP.get_charge()  # in elementary charges (int)
            RP.get_hpem_name()  # HPEM/Global_Kin name (str)
            RP.get_atoms()  # returns (dict) of constituent atom names (str) as keys and stoich. coefs as values
            RP.is_special()  # returning (bool) True if the species is named 'e' or 'M' (M being arb. heavy species)
        The attributes is a dict of values to be saved as instance attributes. Allowed attributes and their allowed
        types are specified in some class attributes. The additional_attributes kwargs will amend/overwrite the
        attributes parameter.

        :param r_id: (int) unique integer id of the reaction. Needs to be positive!
        :param reactants: (list) of RP objects representing reactants
        :param products: (list) of RP objects representing products
        :param attributes: (dict) with keys and values all to be saved as instance attributes. The attributes dict
                           might look like this:
                           attributes = {
                                'arrh_a': 1.,  # mandatory if no special number
                                'arrh_b': 0.,
                                'arrh_c': 1.,
                                'el_en_loss': 13.6,
                                'gas_heat_contrib': 10.,
                                'special_number': -1165,  # mandatory if no arrh_a
                                'qdb_r_id': 158,
                                'qdb_ds_id': 42,
                                'doi': 'aoi.boi/coi',
                                'comments': 'foo',
                                'elastic': True
                           }
                           All the values are explicitly converted into final types, so several different input types
                           are possible (defined in self.allowed_attribute_types.
        :param units: (dict) with keys of unitful attributes and str values from self.allowed_unit_values. If the
                      attribute is in the Reaction.attributes_with_mandatory_units, then the unit needs to be passed
                      if the attribute is passed.
        :param additional_attributes: amending or overwriting the attributes passed.
        """
        if units is None:
            units = {}

        if attributes is None:
            all_attributes = {}
        else:
            all_attributes = attributes.copy()
        all_attributes.update(additional_attributes)

        # check the id consistency
        if type(r_id) is not int or r_id <= 0:
            raise ReactionInitError('Reaction ID must be a positive integer')
        self.id = r_id

        # check for the correct number of reactants and products:
        if len(reactants) < 1 or len(products) < 1:
            raise ReactionInitError('Reaction R{} does not have enough reactants or products!'.format(self.id))
        if len(reactants) > 3:
            raise ReactionInitError('Reaction R{} has too many reactants!'.format(self.id))
        self._reactants = tuple(reactants)  # I want this immutable
        self._products = tuple(products)  # I want this immutable

        # check if all the mandatory attributes are there:
        mandatory_attributes = self.mandatory_attributes.copy()
        if 'special_number' not in all_attributes:
            mandatory_attributes |= {'arrh_a', }
        if not mandatory_attributes.issubset(set(all_attributes)):
            raise ReactionAttributeError(
                'Missing some mandatory attributes: {}!'.format(mandatory_attributes - set(all_attributes))
            )

        # check if any unsupported attributes are not there:
        if not set(all_attributes).issubset(set(self.default_attributes)):
            raise ReactionAttributeError(
                'Passed some unsupported attributes: {}!'.format(set(all_attributes) - set(self.default_attributes))
            )

        # check the units (more checks are performed in set_reaction_attribute method):
        supported_unitful_attributes = set(self.default_units) & set(all_attributes)
        if not set(units).issubset(supported_unitful_attributes):
            raise ReactionAttributeError(
                'R{}: Passed some unsupported unitful attributes in units: {}.'.format(
                    self.id, set(units) - supported_unitful_attributes
                )
            )
        mandatory_units = self.attributes_with_mandatory_units & set(all_attributes)
        if not mandatory_units.issubset(set(units)):
            raise ReactionAttributeError('R{}: Missing some mandatory unitful attributes in units'.format(r_id))

        # save the units into self._units - populated in self.set_reaction_attribute
        self._units = {}  # this dict values are the ones passed in units param, or None, if not passed

        # set all the attributes as instance attributes filling in the Nones for all not defined. sets units as well:
        for key in self.default_attributes:
            unit = units[key] if key in units else None
            if key in all_attributes:
                self.set_reaction_attribute(key, all_attributes[key], unit=unit)
            else:
                self.set_reaction_attribute(key, None, unit=unit)

        # check the consistency of the reaction...
        self.assert_consistency()

    def __str__(self):
        return '%s > %s' % \
               (' + '.join([r.get_name() for r in self.get_reactants()]),
                ' + '.join([p.get_name() for p in self.get_products()]))

    def __repr__(self):
        return self.__str__()

    def get_reactants(self):
        """Returns the tuple of the reactants RP objects...
        """
        return self._reactants

    def get_products(self):
        """Returns the tuple of the products RP objects...
        """
        return self._products

    def get_rp(self, rp):
        """Returns an RP instance from the reaction. Raises a ValueError if the species is not found in the reaction.
        :param rp: (int or str) species id or name.
        :return: RP instance from the reaction
        """
        if rp in self.get_rp_names_set('all'):
            for sp in self.get_reactants() + self.get_products():
                if sp.get_name() == rp:
                    return sp
            else:
                raise ValueError('Species not found in the reaction!')
        if rp in self.get_rp_ids_set('all'):
            for sp in self.get_reactants() + self.get_products():
                if sp.id == rp:
                    return sp
            else:
                raise ValueError('Species not found in the reaction!')
        raise ValueError('Species not found in the reaction!')

    def get_string(self):
        """Wrapper around __str__ method, only for Chemistry compatibility...
        :return: self.__str__()
        """
        return str(self)

    @staticmethod
    def unique_key(r_string):
        """Returns a unique hashable key from the reaction string. This needs to be equivalent for reactions
        with different order of reactants/products.

        :param r_string: (str) string representing a reaction
        :return: (tuple) in a form of ((r1, r2, ..., rn), (p1, p2, ..., pm)) where all r's and p's are ABC sorted.
        """
        lhs, rhs = r_string.split(' > ')
        lhs_tuple = tuple(sorted([r.strip() for r in lhs.split(' + ')]))
        rhs_tuple = tuple(sorted([p.strip() for p in rhs.split(' + ')]))
        return lhs_tuple, rhs_tuple

    def get_stoich_coefs(self, method='net'):
        """Builds and returns an ordered dict of stoichiometric coefficients for the reaction.
        Can build either 'lhs' or 'rhs' stoichiometric coefficients of 'net' (which is rhs - lhs).
        The data structure is dict with RP.id's as keys. All species are present, including electron
        and 'M'.
        Example: for the reaction 'O + O+ + e -> O + O*', the Dict will be:
            method='net': OD({RP(O).id: 0, RP(O+).id: -1, RP(e).id: -1, RP(O*).id: 1})
            method='lhs': OD({RP(O).id: 1, RP(O+).id: 1, RP(e).id: 1})
            method='rhs': OD({RP(O).id: 1, RP(O*).id: 1})

        :param method: (str) either 'lhs', 'rhs' or 'net'. Defaults to 'net'
        :return: (OrderedDict)
        """
        if method == 'lhs':
            stoich_coefs_dict = OrderedDict([(sp_id, 0) for sp_id in self.get_rp_ids_set(method='lhs')])
            for sp in self.get_reactants():
                stoich_coefs_dict[sp.id] += 1
        elif method == 'rhs':
            stoich_coefs_dict = OrderedDict([(sp_id, 0) for sp_id in self.get_rp_ids_set(method='rhs')])
            for sp in self.get_products():
                stoich_coefs_dict[sp.id] += 1
        elif method == 'net':
            stoich_coefs_dict = OrderedDict([(sp_id, 0) for sp_id in self.get_rp_ids_set(method='all')])
            for sp in self.get_products():
                stoich_coefs_dict[sp.id] += 1
            for sp in self.get_reactants():
                stoich_coefs_dict[sp.id] -= 1
        else:
            raise ReactionValueError('Unsupported value of the method argument: {}'.format(method))

        return stoich_coefs_dict
        # return pd.Series(stoich_coefs_dict)

    def get_rp_names_set(self, method='all'):
        """Returns an ordered set of species names. For method = 'lhs'/'rhs' it returns names of reactants and products,
        for method = 'all', returns set of names of all species.

        :param method: (str) from {'lhs', 'rhs', 'all'}. Defaults to 'all'
        :return: (tuple) of str species names, ordered according to index in reactants+products.
        """
        if method == 'lhs':
            names = [rp.get_name() for rp in self.get_reactants()]
        elif method == 'rhs':
            names = [rp.get_name() for rp in self.get_products()]
        elif method == 'all':
            names = [rp.get_name() for rp in self.get_reactants()] + [rp.get_name() for rp in self.get_products()]
        else:
            raise ReactionValueError('Unsupported value of the method argument: {}'.format(method))

        names_set = set(names)
        # sort:
        names_set = tuple(sorted(names_set, key=lambda x: names.index(x)))

        return names_set

    def get_rp_ids_set(self, method='all'):
        """Returns an ordered set of species ids. For method = 'lhs'/'rhs' it returns ids of reactants and products,
        for method = 'all', returns set of ids of all species.

        :param method: (str) from {'lhs', 'rhs', 'all'}. Defaults to 'all'
        :return: (tuple) of int species ids, ordered according to index in reactants+products.
        """
        if method == 'lhs':
            ids = [rp.id for rp in self.get_reactants()]
        elif method == 'rhs':
            ids = [rp.id for rp in self.get_products()]
        elif method == 'all':
            ids = [rp.id for rp in self.get_reactants()] + [rp.id for rp in self.get_products()]
        else:
            raise ReactionValueError('Unsupported value of the method argument: {}'.format(method))

        ids_set = set(ids)
        # sort:
        ids_set = tuple(sorted(ids_set, key=lambda x: ids.index(x)))

        return ids_set

    def is_electron_process(self):
        """Returns True if this is an electron process. Returns False if this is a heavy particle process

        :return: (bool)
        """
        return 'e' in self.get_rp_names_set('lhs')

    def is_ion_process(self):
        """Returns True if this reaction involves ions on the LHS.

        :return: (bool)
        """
        return np.array([r.get_charge() for r in self.get_reactants() if not r.is_special()]).any()

    def is_elastic_process(self):
        """Returns True if this is an elastic process. Only a wrapper around self.get_elastic, for consistency.
        self.get_elastic is defined rather than self.is_elastic_process, since it is a getter method getting the
        'elastic' attribute, and defaulting to the usual decision method if not finding the elastic attribute value.
        This way I can force a reaction to not be elastic, even though having the same reactants and products, which
        is of a great importance.

        :return: (bool)
        """
        return self.get_elastic()

    def get_hpem_string(self, backward=False):
        """Method building and returning a HPEM (Global_Kin) reaction string

        :param backward: (bool) if this is to be a backwards reaction (swapped reactants and products)
        :return: (str) e.g. 'O^ + O2- > O3^ + e'
        """
        if not backward:
            r_str = '%s > %s' % (' + '.join([rp.get_hpem_name() for rp in self.get_reactants()]),
                                 ' + '.join([rp.get_hpem_name() for rp in self.get_products()]))
        else:
            r_str = '%s > %s' % (' + '.join([rp.get_hpem_name() for rp in self.get_products()]),
                                 ' + '.join([rp.get_hpem_name() for rp in self.get_reactants()]))
        return r_str

    def assert_consistency(self):
        """Checks if the reaction conserves charge and atom counts. If not, raises ReactionValueError error.

        :return: None
        """
        # check charge conservation:
        charge_lhs = sum([r.get_charge() for r in self.get_reactants() if not r.get_name() == 'M'])
        charge_rhs = sum([p.get_charge() for p in self.get_products() if not p.get_name() == 'M'])
        if charge_lhs != charge_rhs:
            raise ReactionValueError('Charge conservation in Reaction {} is violated'.format(self.id))

        # check conservation of atom counts:
        lhs_atoms, rhs_atoms = {}, {}
        for sp in self.get_reactants():
            atom_counts = sp.get_atoms()
            for atom in atom_counts:
                try:
                    lhs_atoms[atom] += atom_counts[atom]
                except KeyError:
                    lhs_atoms[atom] = atom_counts[atom]
        for sp in self.get_products():
            atom_counts = sp.get_atoms()
            for atom in atom_counts:
                try:
                    rhs_atoms[atom] += atom_counts[atom]
                except KeyError:
                    rhs_atoms[atom] = atom_counts[atom]
        if lhs_atoms != rhs_atoms:
            raise ReactionValueError('Atoms counts conservation of Reaction {} is violated!'.format(self.id))

    def get_rate_coefficient(self, temp, si_units=True):
        """Method returning a rate coefficient from the arrhenius data. Expects the arrhenius data to exist, will
        raise an exception if they do not exist. Option to get the coefficient either in SI units or in the nominal
        units defined in arrh_a_unit attribute. The SI units regard only to the pre-expo factor, the activation energy
        is always in eV for electron collisions!

        :param temp: (float) temperature in [eV] if electron process, or in [K] heavy species process
        :param si_units: (bool) if True, rate coefficient will be returned in SI, if False, in nominal units
                         (corresponding to self.arrh_a_unit)
        :return: (float) reaction rate coefficient
        """
        a = self.get_arrh_a(si_units=si_units)
        if a is None:
            raise ReactionValueError('Cannot get a coefficient for a reaction without Arrhenius data!')
        b = self.get_arrh_b()
        c = self.get_arrh_c()

        if self.is_electron_process():
            return a * temp**b * np.exp(-c/temp)
        else:
            return a * (temp/300.)**b * np.exp(-c/temp)

    def get_electron_collision_partner(self):
        """Gets a heavy species collision partner for electron process. Only makes sense for electron processes.
        Raises a ReactionValueError for non-electron processes. Returns the first reactant which is not electron or
        raises ReactionValueError if it does not find such a reactant.

        :return: (RP instance) of a heavy specie collision partner in electron process
        """
        if not self.is_electron_process():
            raise ReactionValueError('Requesting collision partner for {} '
                                     'which appears not to be an electron process'.format(self.__str__()))
        for sp in self.get_reactants():
            if not sp.is_special():
                return sp
        else:
            raise ReactionValueError('When requesting a collision partner for electron process {}, '
                                     'no heavy species was found'.format(self.__str__()))

    # ************************************* ATTRIBUTES SETTER METHODS ************************************************ #

    def set_reaction_attribute(self, attribute, value, unit=None):
        """Method to be called whenever need to set a new attribute to the Reaction instance...
        Takes care of verifications of correct types, allowed attributes etc. ALWAYS use for setting the attributes!
        Also checks the units consistency and populates the self._units!

        :param attribute: (str) attribute
        :param value: (various supported types) value
        :param unit: (str) mandatory only for self.attributes_with_mandatory_units and needs to be among allowed
        :return: None
        """

        # check the units consistency:
        if value is not None and unit is not None:
            if attribute not in self.allowed_unit_values.keys():
                raise ReactionAttributeError('Passing unit for attribute which does not support it: ',
                                             attribute)
            elif unit not in self.allowed_unit_values[attribute]:
                raise ReactionAttributeError(
                    'Passing unsupported unit {} for attribute {}!'.format(unit, attribute))
        # is the unit there for attributes which need units?
        if value is not None and attribute in self.attributes_with_mandatory_units and unit is None:
            raise ReactionAttributeError('Mandatory unit not passed for attribute', attribute, '!')

        # check if the units are consistent with the reactants:
        if attribute == 'arrh_c' and value is not None:
            if self.is_electron_process() and unit != 'eV':
                raise ReactionAttributeError('Activation energy for electron collisions needs to be in (eV)!')
            elif not self.is_electron_process() and unit != 'K':
                raise ReactionAttributeError('Activation energy for non-electron process needs to be in (K)!')
        if attribute == 'arrh_a' and value is not None:
            if len(self.get_reactants()) == 1 and unit != 's-1':
                raise ReactionAttributeError('Incorrect unit for the Arrhenius pr-exponential factor!')
            elif len(self.get_reactants()) == 2 and not unit.endswith('3.s-1'):
                raise ReactionAttributeError('Incorrect unit for the Arrhenius pr-exponential factor!')
            elif len(self.get_reactants()) == 3 and not unit.endswith('6.s-1'):
                raise ReactionAttributeError('Incorrect unit for the Arrhenius pr-exponential factor!')
            elif len(self.get_reactants()) > 3:
                raise ReactionInitError('Reaction {} has too many reactants!'.format(self))

        # check value and type consistency
        if attribute not in self.default_attributes:
            raise ReactionAttributeError('Unsupported attributes passed: ', attribute)
        if attribute in self.allowed_attribute_values and value not in self.allowed_attribute_values[attribute]:
            raise ReactionAttributeError('Unsupported value for {} attribute: {}'.format(attribute, value))
        if type(value) not in self.allowed_attribute_types[attribute]:
            raise ReactionAttributeError(
                'Unsupported value type for {} attribute: {}'.format(attribute, type(value)))
        if attribute == 'arrh_a' and value is not None:
            if float(value) < 0:
                raise ReactionAttributeError('Arrhenius parameter arrh_a needs to be non-negative!')

        # if all in order, set the value and unit:
        setattr(self, '_{}'.format(attribute), value)  # set it as a private attribute
        self._units[attribute] = unit

    # ************************************* ATTRIBUTES GETTER METHODS ************************************************ #

    def get_arrh_a(self, si_units=True):
        """Method returning a float of the Arrhenius pre-exponential (a) parameter. Returns float or None,
        if not defined.

        :param si_units: (bool), if in SI units. This gets decided from arrh_a_unit value, consistency of which is
                         ensured elsewhere.
        :return: (float) or None
        """
        arrh_a = self._arrh_a
        if arrh_a is not None:
            arrh_a = float(arrh_a)  # explicit conversion
            if si_units:
                arrh_a *= self.si_conversion_factors['arrh_a'][self.get_unit('arrh_a')]
        return arrh_a

    def get_arrh_b(self):
        """Method returning the Arrhenius exponent factor n (or b). If arrh_a is defined, always returns float
        defaulting to 0. if not defined. If on the other hand arrh_a is not defined, defaults to 0.

        :return: (float)
        """
        arrh_b = self._arrh_b
        if arrh_b is not None:
            arrh_b = float(arrh_b)
        elif self.get_arrh_a() is not None:
            arrh_b = 0.
        return arrh_b

    def get_arrh_c(self):
        """Method returning the Arrhenius activation energy factor E_a (or c).
        If arrh_a is defined, always returns float, defaulting to 0. if not defined. If on the other hand arrh_a is
        not defined, defaults to 0. Return values is in 'eV' if electron process or in
        'K' if heavy species process. The arrh_c_unit needs to correspond to those, which is asserted elsewhere.

        :return: (float)
        """
        arrh_c = self._arrh_c
        if arrh_c is not None:
            arrh_c = float(arrh_c)
        elif self.get_arrh_a() is not None:
            arrh_c = 0.
        return arrh_c

    def get_elastic(self):
        """Returns True if this is an elastic process. If self._elastic attribute value not defined, then it
        decides based on reactants and products names.

        :return: (bool)
        """
        elastic = self._elastic
        if elastic is None:  # if not explicitly declared
            return set(self.get_rp_names_set('lhs')) == set(self.get_rp_names_set('rhs'))
        else:
            if elastic in {'true', 'True', 'TRUE', True}:
                return True
            else:
                return False

    def get_el_en_loss(self):
        """Method returning the electron energy loss for the reaction.
        Always returns float, defaults to 0. if not defined. Return values is in 'eV'.

        :return: (float)
        """
        el_en_loss = self._el_en_loss
        if el_en_loss is not None:
            return float(el_en_loss)
        else:
            return self.default_attributes['el_en_loss']

    def get_gas_heat_contrib(self):
        """Returns the change of enthalpy manifesting as the gas heating contribution. Always returns float,
        defaulting to 0. if not defined.

        :return: (float)
        """
        gas_heat_contrib = self._gas_heat_contrib
        if gas_heat_contrib is not None:
            return float(gas_heat_contrib)
        else:
            return self.default_attributes['gas_heat_contrib']

    def get_special_number(self):
        """Returns integer HPEM/Global_Kin special number of the reaction
        Defaults to 10/20 for arrhenius data or -N for cross sections in the hpem/global_kin database.

        :return: (int) special number of the reaction relating to hpem/global_kin
        """
        special_number = self._special_number
        if special_number is not None:
            special_number = int(special_number)
        else:
            if self.is_electron_process():
                special_number = 10
            else:
                special_number = 20
        return special_number

    def get_qdb_r_id(self):
        """Returns integer QDB Reaction id (primary key) of the reaction in QDB database. Returns None if not defined
        (does not have a default value)
        :return: (int) or None
        """
        qdb_r_id = self._qdb_r_id
        if qdb_r_id is not None:
            qdb_r_id = int(qdb_r_id)
        return qdb_r_id

    def get_qdb_ds_id(self):
        """Returns integer QDB ReactionDataSet id (primary key) of the reaction in QDB database.
        Returns None if not defined (does not have a default value)
        :return: (int) or None
        """
        qdb_ds_id = self._qdb_ds_id
        if qdb_ds_id is not None:
            qdb_ds_id = int(qdb_ds_id)
        return qdb_ds_id

    def get_doi(self):
        """Gets the DOI string if it defined, or None
        :return: (str) or None
        """
        doi = self._doi
        if doi is not None:
            doi = str(doi)
        return doi

    def get_comments(self):
        """Gets the comments string if it defined, or None
        :return: (str) or None
        """
        comments = self._comments
        if comments is not None:
            comments = str(comments)
        return comments

    # noinspection PyTypeChecker
    def get_latex(self):
        """Gets the LaTeX representation string. If it's not defined explicitly, builds it up from LaTeX representations
        of species. The LaTeX string needs to work in the math-mode but does NOT contain the '$' characters.
        :return: (str)
        """
        latex = self._latex
        if latex is not None:
            latex = str(latex)
        else:
            species_dicts = [OrderedDict(), OrderedDict()]
            for sp_dict, id_stoich in zip(species_dicts, (self.get_stoich_coefs('lhs'), self.get_stoich_coefs('rhs'))):
                for sp_id in id_stoich:
                    sp_latex = self.get_rp(sp_id).get_latex()
                    if id_stoich[sp_id] == 1:
                        sp_dict[sp_latex] = ''
                    elif id_stoich[sp_id] > 1:
                        sp_dict[sp_latex] = '{} '.format(id_stoich[sp_id])
            # now species_dicts is list with two dicts for reactants and products, where each dict is keyed by
            # latex repr of species and value is string of the stoich coefficient (or empty string if it's 1)
            sides = [
                ' + '.join(['{}{}'.format(stoich, sp_latex) for sp_latex, stoich in species_dict.items()])
                for species_dict in species_dicts
            ]
            latex = r'{} \rightarrow {}'.format(*sides)

        return latex

    # ************************************** UNITS GETTER METHODS **************************************************** #

    def get_unit(self, attribute):
        """Gets a unit for an attribute
        :param attribute: (str)
        :return: (str or NoneType)
        """
        if attribute not in self.default_units:
            raise ReactionAttributeError('Querying a unit for an unsupported unitful attribute: {}!'.format(attribute))
        unit = self._units[attribute]
        if unit is not None:
            return str(unit)
        else:
            return self.default_units[attribute]
