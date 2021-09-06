import re
from collections import OrderedDict

import numpy as np
from pyvalem.formula import Formula

from plaschem.exceptions import RPInitError, RPAttributeError
from plaschem.latex import Texify


# noinspection PyUnresolvedReferences
class RP(object):
    """
    Class representing species in general (Reactant or Product). Each RP is uniquely identified by its
    id. Species with identical ids will be treated as the same species in the Reaction class. On top of that,
    each species needs to have a name, which can be arbitrary, but electron needs to have a "e" name and an
    arbitrary heavy species needs to have a "M" name.
    """
    mandatory_attributes = {'name', 'mass'}

    default_attributes = OrderedDict([
        ('name', None),  # mandatory
        ('stateless', None),  # might be mandatory
        ('state', None),  # does not have a default - depends
        ('mass', None),  # mandatory
        ('charge', 0),
        ('h_form', 0.),
        ('lj_epsilon', 0.),
        ('lj_sigma', 3.),
        ('hpem_name', None),  # does not have a default
        ('qdb_id', None),  # does not have a default
        ('stick_coef', None),  # depends
        ('ret_coefs', None),  # depends
        ('comments', None),  # does not have a default
        ('latex', None)  # does not have a default - depends
    ])  # these defaults will be set as instance attributes if some of them not passed into __init__

    nonetype = type(None)
    allowed_attribute_types = {
        'name': [str, nonetype],
        'stateless': [str, nonetype],
        'state': [str, nonetype],
        'mass': [float, np.float64, str, int, nonetype],
        'charge': [int, str, nonetype],
        'h_form': [float, np.float64, str, int, nonetype],
        'lj_epsilon': [float, np.float64, str, int, nonetype],
        'lj_sigma': [float, np.float64, str, int, nonetype],
        'hpem_name': [str, nonetype],
        'qdb_id': [int, str, nonetype],
        'stick_coef': [float, np.float64, str, int, nonetype],
        'ret_coefs': [dict, nonetype],
        'comments': [str, nonetype],
        'latex': [str, nonetype]
    }  # these are allowed types looked at when trying to set new attributes values with self.set_species_attribute

    special_species_names = {'e', 'M'}  # these are names of the species treated as special species (differently)

    invalid_name_characters = {'<', '>'}

    attributes_with_mandatory_units = set()  # currently empty - everything can be defaulted!
    default_units = OrderedDict([
        ('mass', 'amu'),
        ('charge', 'e'),
        ('h_form', 'eV'),
        ('lj_epsilon', 'K'),
        ('lj_sigma', 'A')
    ])
    allowed_unit_values = {
        'mass': {None, 'amu', },
        'charge': {None, 'e', },
        'h_form': {None, 'eV', },
        'lj_epsilon': {None, 'K', },
        'lj_sigma': {None, 'A', }
    }  # if I add others, need to implement conversion of factors into attributes getters!

    def __init__(self, rp_id, attributes=None, units=None, **additional_attributes):
        """Instantiates the RP object. Takes an ID and a set of attributes and their values in a dict

        :param rp_id: (int) unique identifying id of the specie. It is a good practice to assign an electron species
                      an ID=0 and an arbitrary heavy species an ID=999.
        :param attributes: (dict) with keys and values all to be saved as instance attributes. All values are
                           str/int/float, except the ret_coefs key, whose value is a dict.
                           The attributes dict might look like this (some keys are mandatory, some optional):
                           attributes = {
                                'name': 'Ar*',  # mandatory!
                                'stateless': 'Ar',
                                'state': '3P',
                                'mass': '28.0',  # mandatory!
                                'charge': '0',
                                'h_form': '0.0',
                                'lj_epsilon': '1.0',
                                'lj_sigma': '1.0',
                                'hpem_name': 'AR',
                                'qdb_id': '42',
                                'stick_coef': '1.0',
                                'ret_coefs': {'Ar': '0.5', 'Ar+': '0.25', 'Ar**': '0.25'},
                                'comments': 'this data is obviously made up',
                                'latex': 'Ar' wrapped in mathrm etc...
                           }
                           The types of the attributes values are not very important, everything gets explicitly
                           converted in their getter methods. This way the RP can be created from an xml format for
                           example, without explicit conversions.
                           WARNING: name of the electron species MUST be 'e' and name of the arbitrary heavy species
                           MUST be 'M'. How these two species are treated is determined solely by their names!
        :param units: (dict) with keys of unitfull attributes and str values from self.allowed_unit_values. If the
                      attribute is in the RP.attributes_with_mandatory_units, then the unit needs to be passed
                      if the attribute is passed.
        :param additional_attributes: exactly the same logic as attributes, contains either additional attributes
                                      not present in attributes dict, or overwriting values for those or both.
        """
        if units is None:
            units = {}

        if attributes is None:
            all_attributes = {}
        else:
            all_attributes = attributes.copy()
        all_attributes.update(additional_attributes)

        # check the id consistency:
        if type(rp_id) is not int or rp_id < 0:
            raise RPInitError('Species ID must be a non-negative integer!')
        self.id = rp_id

        # check if mandatory attributes are there:
        mandatory_attributes = self.mandatory_attributes.copy()
        if 'state' in all_attributes and all_attributes['state']:
            mandatory_attributes.add('stateless')
        # exception: "M" species does not have to define mass:
        if 'name' in all_attributes and all_attributes['name'] == 'M':
            mandatory_attributes.remove('mass')
        if not mandatory_attributes.issubset(set(all_attributes)):
            raise RPAttributeError(
                'Missing some mandatory attributes: {}!'.format(mandatory_attributes - set(all_attributes))
            )

        # check the validity of name:
        if len(set(all_attributes['name']) & self.invalid_name_characters):
            raise RPAttributeError(
                'Species name contains some of the invalid characters: {}'.format(self.invalid_name_characters)
            )

        # check if any unsupported attributes are not there:
        if not set(all_attributes).issubset(set(self.default_attributes)):
            raise RPAttributeError(
                'Passed some unsupported attributes: {}!'.format(set(all_attributes) - set(self.default_attributes))
            )

        # check the units (more checks are performed in set_reaction_attribute method):
        supported_unitful_attributes = set(self.default_units) & set(all_attributes)
        if not set(units).issubset(supported_unitful_attributes):
            raise RPAttributeError('Passed some unsupported unitfull attributes in units.')
        mandatory_units = self.attributes_with_mandatory_units & set(all_attributes)
        if not mandatory_units.issubset(set(units)):
            raise RPAttributeError('Missing some mandatory unitfull attributes in units')

        # save the units into self._units  - populated at self.set_species_attribute
        self._units = {}  # this dicts values are the ones passed in units param, or None, if not passed

        # set all the attributes as instance attributes filling in the Nones for all not defined. sets units as well:
        for key in self.default_attributes:
            unit = units[key] if key in units else None
            if key in all_attributes:
                self.set_species_attribute(key, all_attributes[key], unit=unit)
            else:
                self.set_species_attribute(key, None, unit=unit)

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.__str__()

    def is_special(self):
        return self.get_name() in self.special_species_names

    def is_stateless(self):
        return self.get_state() is None

    def get_atoms(self):
        """Builds a dictionary of constituent atoms and their counts.

        :return: (dict) {'O': 2, 'N':1} for NO2*+
        """
        if self.get_name() == 'e':
            return {}
        if self.get_name() == 'M':
            return {'M': 1}
        stateless_neutral = self.strip_charge(stateless_name=self.get_stateless())
        return dict(Formula(stateless_neutral).atom_stoich)

    # ************************************* ATTRIBUTES SETTER METHODS ************************************************ #

    def set_species_attribute(self, attribute, value, unit=None):
        """Method to be called whenever need to set a new attribute to the RP instance... Takes care of verifications
        of correct types, allowed attributes etc. ALWAYS use for setting the attributes!
        Also checks the units consistency and populates the self._units!

        :param attribute: (str) attribute
        :param value: (various supported types) value
        :param unit: (str) mandatory only for self.attributes_with_mandatory_units and needs to be among allowed
        :return: None
        """

        # check the units consistency:
        if value is not None and unit is not None:
            if attribute not in self.allowed_unit_values.keys():
                raise RPAttributeError('Passing unit for attribute which does not support it: ', attribute)
            elif unit not in self.allowed_unit_values[attribute]:
                raise RPAttributeError(
                    'Passing unsupported unit {} for attribute {}!'.format(unit, attribute))
        # is the unit there for attributes which need units?
        if value is not None and attribute in self.attributes_with_mandatory_units and unit is None:
            raise RPAttributeError('Mandatory unit not passed for attribute', attribute, '!')

        # check value and type consistency
        if attribute not in self.default_attributes:
            raise RPAttributeError('Unsupported attributes passed: ', attribute)
        if type(value) not in self.allowed_attribute_types[attribute]:
            raise RPAttributeError('Unsupported value type for {} attribute: {}'.format(attribute, type(value)))

        # if all in order, set the value and unit:
        setattr(self, '_{}'.format(attribute), value)  # set it as a private attribute
        self._units[attribute] = unit

    # ************************************* ATTRIBUTES GETTER METHODS ************************************************ #

    def get_name(self) -> str:
        """Returns RP name if defined, or raises the RPAttributeError if not (since the name is a mandatory attribute).
        :return: (str)
        """
        name = self._name
        if name is not None:
            return str(name)
        else:
            raise RPAttributeError

    def get_stateless(self):
        """Stateless name is mandatory if the state is defined, otherwise it defaults to the name. If not
        defined while state is defined, raises RPAttributeError
        :return: (str)
        """
        if self.is_special():
            return self.get_name()
        stateless = self._stateless
        if stateless is None:
            state = self.get_state()
            if state is not None:
                raise RPAttributeError('Stateless name not defined, while state is defined, which is not allowed!')
            else:
                stateless = self.get_name()
        return stateless

    def get_state(self):
        """Returns state. This might be either None if not defined or str if defined. State is an optional parameter.
        :return: (str or NoneType)
        """
        if self.is_special():
            return None
        state = self._state
        if state is not None:
            state = str(state)
        return state

    def get_mass(self):
        """Mass in amu. Mandatory. If not defined, raises an RPAttributeError
        :return: (float or NoneType for M)
        """
        if self.get_name() == 'M':
            return None
        mass = self._mass
        if mass is not None:
            return float(mass)
        else:
            raise RPAttributeError('Species {} does not define mass!'.format(self))

    def get_charge(self):
        """Charge in elementary charges. Optional, if not defined, then defaults 0.
        :return: (int or NoneType for M)
        """
        if self.get_name() == 'M':
            return None
        charge = self._charge
        if charge is not None:
            return int(charge)
        else:
            return self.default_attributes['charge']

    def get_h_form(self):
        """Enthalpy of formation in eV. Optional, if not defined, defaults to 0.0
        :return: (float or NoneType for special species)
        """
        if self.is_special():
            return None
        h_form = self._h_form
        if h_form is not None:
            return float(h_form)
        else:
            return self.default_attributes['h_form']

    def get_lj_epsilon(self):
        """Lennard-Jones potential epsilon in K. Optional, if not defined, defaults to 0.0.
        :return: (float or NoneType for special species)
        """
        if self.is_special():
            return None
        lj_epsilon = self._lj_epsilon
        if lj_epsilon is not None:
            return float(lj_epsilon)
        else:
            return self.default_attributes['lj_epsilon']

    def get_lj_sigma(self):
        """Lennard-Jones sigma parameter - actually used in PyGMol. If not defined, defaults to 3.0.
        :return: (float or NoneType for special species)
        """
        if self.is_special():
            return None
        lj_sigma = self._lj_sigma
        if lj_sigma is not None:
            return float(lj_sigma)
        else:
            return self.default_attributes['lj_sigma']

    def get_hpem_name(self):
        """Mandatory (only if you work with global_kin though!) Raises RPAttributeError if called while not defined.
        :return: (str)
        """
        hpem_name = self._hpem_name
        if hpem_name is not None:
            hpem_name = str(hpem_name)
        return hpem_name

    def get_qdb_id(self):
        """Optional QDB RP primary key. If not defined and called, raises RPAttributeError
        :return: (int)
        """
        qdb_id = self._qdb_id
        if qdb_id is not None:
            qdb_id = int(qdb_id)
        return qdb_id

    def get_stick_coef(self):
        """Optional sticking coefficient for the species surface reactions. If not defined, assumed 1.0 for all
        stateful charged species and 0.0 for neutrals in the ground states.
        :return: (float or NoneType for special species)
        """
        if self.is_special():
            return None
        stick_coef = self._stick_coef
        if stick_coef is not None:
            return float(stick_coef)
        else:  # if none defined, 0. for ground state neutrals, 1. otherwise
            state = self.get_state()
            if state is None and self.get_charge() == 0:
                return 0.
            else:
                return 1.

    def get_ret_coefs(self):
        """Optional return coefficients for the species surface reactions. The dict keys are return species names
        and the values are return coefficients, which denote how many return species particles are produced per
        each one self particle stuck to a surface. The example of what gets returned
        for H2+ might be {'H2': 0.5, 'H': 1.0}, which would mean that per TWO H2+ molecule STUCK to a surface,
        there is ONE H2 particle and TWO H particles returned.
        If not defined explicitly, will default into:
            for ions and stateful species:
                - {ground_state_neutral: 1.0}
            for stateless neutrals:
                - {} (empty dict)
        :return: (dict or NoneType for special species) with species names as keys
        """
        if self.is_special():
            return None
        ret_coefs = self._ret_coefs
        if ret_coefs is not None:
            return {key: float(ret_coefs[key]) for key in ret_coefs}
        else:
            state = self.get_state()
            if state is None and self.get_charge() == 0:
                return {}
            else:
                return {self.strip_charge(self.get_stateless()): 1.0}

    def get_comments(self):
        """Gets the comments string if it defined, or None
        :return: (str or NoneType)
        """
        comments = self._comments
        if comments is not None:
            comments = str(comments)
        return comments

    def get_latex(self):
        """Gets the LaTeX representation string. If it's not defined explicitly, builds it up using default texify func.
        The LaTeX string needs to work in the math-mode but does NOT contain the '$' characters.
        :return: (str)
        """
        latex = self._latex
        if latex is not None:
            latex = str(latex)
        else:
            latex = Texify.texify_species(self.get_name())
        return latex

    # ************************************** UNITS GETTER METHODS **************************************************** #

    def get_unit(self, attribute):
        """Gets a unit for an attribute. If the unit was not passed in __init__, (meaning value None in self._unit), it
        will be substituted with the default from self.default_units.
        :param attribute: (str)
        :return: (str or NoneType)
        """
        if attribute not in self.default_units:
            raise RPAttributeError('Querying a unit for an unsupported unitfull attribute: {}!'.format(attribute))
        unit = self._units[attribute]
        if unit is not None:
            return str(unit)
        else:
            return self.default_units[attribute]

    # ****************************************** HELPER METHODS ****************************************************** #

    @staticmethod
    def strip_charge(stateless_name):
        """Method to strip any charge notation from a stateless name string. WARNING: this will not work on names
        containing states on any notation, since the method ONLY looks at the end of the string. Stateless name
        needs to be passed into the method!
        :param stateless_name: (str)
        """
        chp1 = r'[+-]+$'  # Ar+, Ar--, Ar++++++++,
        chp2 = r'[+-]\d+$'  # Ar+2, Ar-6, Ar+1
        charge_pattern = '(?:{}|{})'.format(chp1, chp2)
        return str(re.sub(charge_pattern, '', stateless_name))
